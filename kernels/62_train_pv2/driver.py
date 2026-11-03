# ---------------- BEETLE own model, Phikon-v2 (DINOv2 ViT-L/16) + last-N-block fine-tuning + CNN-skip decoder, trained on the 512px / 0.5 um patches of beetle-43-pack2 ----------------
# Differences to kernels/32_train/driver.py (the ViT-B Phikon runs): backbone from the Hugging Face checkpoint (beetle-60-phikonv2-fetch), a fresh decoder (no warm start), a RESUME option that spans
# two Kaggle sessions (state saved after every epoch), session controls (STOP_AFTER, TIME_BUDGET_H), a stop rule on the held-out stain-shift score, a short GPU TIMING mode and optional backbone-LR delay.
import os, io, re, glob, json, time, math, random, shutil
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

dev = os.environ.get("BEETLE_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
AMP = (dev == "cuda") and os.environ.get("BEETLE_AMP", "1") == "1"
OUTD = os.environ.get("BEETLE_OUT", "/kaggle/working")
INROOT = os.environ.get("BEETLE_IN", "/kaggle/input")
os.makedirs(OUTD, exist_ok=True)
_T0 = time.time()
def stamp(msg): print(f"[t+{time.time() - _T0:6.0f}s] {msg}", flush=True)


def find_first(pred):
    for r, dirs, files in os.walk(INROOT):
        dirs[:] = [d for d in dirs if d != "patches"]
        if pred(r, dirs, files): return r, files
    return None, None


random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
stamp("start: looking for the patch data")
packed_dirs = []
for r, dirs, files in os.walk(INROOT):
    if r.endswith("packed") and "index.json" in files and any(f.startswith("shard_") for f in files): packed_dirs.append(r)
    dirs[:] = [d for d in dirs if d != "patches"]
packed_dir = packed_dirs[0] if packed_dirs else None
proots = []
if packed_dir is None:
    for r, dirs, files in os.walk(INROOT):
        if r.endswith("patches") and "train" in dirs and "val" in dirs: proots.append(r); dirs[:] = []
    assert proots, "neither a packed patch set nor patches directories were found"
    stamp(f"{len(proots)} patch roots found")
else:
    PACKED_INDEX = []
    for _d in sorted(packed_dirs):
        _ix = json.load(open(f"{_d}/index.json"))
        for _e in _ix: _e["dir"] = _d
        PACKED_INDEX += _ix; stamp(f"packed patch set found: {_d} ({len(_ix)} patches)")
    assert len({(_e["split"], _e["name"]) for _e in PACKED_INDEX}) == len(PACKED_INDEX), "duplicate patch names across the packed sets"
    stamp(f"{len(packed_dirs)} packed set(s), {len(PACKED_INDEX)} patches in total")
WEIGHTS, _ = find_first(lambda r, d, f: "model.safetensors" in f and "config.json" in f)                 # the Phikon-v2 checkpoint folder (beetle-60-phikonv2-fetch output)
assert WEIGHTS or os.environ.get("BEETLE_TINY"), "Phikon-v2 checkpoint (model.safetensors + config.json) not found - kernel source beetle-60-phikonv2-fetch missing"
stamp("weights located"); print("patch roots:", proots or packed_dir, "| weights:", WEIGHTS, "| device", dev, flush=True)


class PatchDS(Dataset):
    def __init__(s, split, limit=0):
        s.items = [(f"{r}/{split}", f[:-4]) for r in proots for f in sorted(os.listdir(f"{r}/{split}/images"))]
        if limit: s.items = s.items[::max(1, len(s.items) // limit)][:limit]
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        d, n = s.items[i]
        x = np.array(Image.open(f"{d}/images/{n}.jpg").convert("RGB")); y = np.array(Image.open(f"{d}/masks/{n}.png"))
        return torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(y.astype(np.int64))


class PackedDS(Dataset):
    """Reads patches from the shard files written by kernels/41_pack (raw jpg / png bytes at known offsets)."""
    def __init__(s, split, limit=0, exclude=(), only=(), cap=0, exclude_slides=()):
        pid = lambda e: int(re.match(r"patient(\d+)", e["name"]).group(1)); sid = lambda e: "_".join(e["name"].split("_")[:2])        # slide id = patientNNN_wsiM
        s.items = [e for e in PACKED_INDEX if e["split"] == split and (not exclude or pid(e) not in exclude) and (not only or pid(e) in only) and (not exclude_slides or sid(e) not in exclude_slides)]
        if cap and len(s.items) > cap: s.items = [s.items[i] for i in np.random.default_rng(0).choice(len(s.items), cap, replace=False)]
        if limit: s.items = s.items[::max(1, len(s.items) // limit)][:limit]
        s.fh = {}
    def __len__(s): return len(s.items)
    def _read(s, d, shard, off, ln):
        f = s.fh.get((d, shard))
        if f is None: f = s.fh[(d, shard)] = open(f"{d}/shard_{shard:03d}.bin", "rb")
        f.seek(off); return f.read(ln)
    def __getitem__(s, i):
        e = s.items[i]
        x = np.array(Image.open(io.BytesIO(s._read(e["dir"], e["shard"], e["io"], e["il"]))).convert("RGB")); y = np.array(Image.open(io.BytesIO(s._read(e["dir"], e["shard"], e["mo"], e["ml"]))))
        return torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(y.astype(np.int64))


def to_target(y):
    t = y - 1; t[y == 0] = 255; return t


HED_M = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]]); HED_M = HED_M / HED_M.norm(dim=1, keepdim=True); HED_INV = torch.linalg.inv(HED_M)


def stain_jitter(x, sigma):
    hm, hi = HED_M.to(x.device), HED_INV.to(x.device)
    od = -torch.log10((x * 255 + 1) / 256)
    c = torch.einsum("bchw,cd->bdhw", od, hi)
    B = x.shape[0]
    a = 1 + (torch.rand(B, 3, 1, 1, device=x.device) * 2 - 1) * sigma; b = (torch.rand(B, 3, 1, 1, device=x.device) * 2 - 1) * sigma
    od2 = torch.einsum("bdhw,dc->bchw", c * a + b, hm)
    return ((256 * 10 ** (-od2) - 1).clamp(0, 255) / 255).to(x.dtype)


def augment(x, y):
    xs, ys = [], []
    for i in range(x.shape[0]):
        xi, yi = x[i:i + 1], y[i:i + 1]; k = random.randint(0, 3)
        xi = torch.rot90(xi, k, (2, 3)); yi = torch.rot90(yi, k, (1, 2))
        if random.random() < 0.5: xi = xi.flip(3); yi = yi.flip(2)
        gain = 1 + 0.08 * torch.randn(1, 3, 1, 1, device=x.device); bright = 0.05 * torch.randn(1, 1, 1, 1, device=x.device); con = 1 + 0.12 * torch.randn(1, 1, 1, 1, device=x.device)
        m = xi.mean((1, 2, 3), keepdim=True); xi = (((xi - m) * con + m) * gain + bright).clamp(0, 1)
        xs.append(xi); ys.append(yi)
    xo = torch.cat(xs)
    if globals().get("STAIN", 0): xo = stain_jitter(xo, float(STAIN))
    return xo, torch.cat(ys)


def dice_loss(logits, t):
    valid = (t != 255); p = torch.softmax(logits.float(), 1)
    oh = F.one_hot(t.clamp(max=3), 4).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
    p = p * valid.unsqueeze(1)
    inter = (p * oh).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    present = oh.sum((0, 2, 3)) > 0
    d = (2 * inter + 1) / (den + 1)
    return 1 - d[present].mean() if present.any() else logits.sum() * 0


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); cm = torch.zeros(4, 4, dtype=torch.long)
    for x, y in loader:
        x = x.to(dev).float() / 255; t = to_target(y.to(dev))
        with torch.autocast(dev, dtype=torch.float16, enabled=AMP): lo = model(x)
        pr = lo.argmax(1); v = t != 255
        idx = t[v] * 4 + pr[v]; cm += torch.bincount(idx, minlength=16).view(4, 4).cpu()
    tp = cm.diag().float(); fp = cm.sum(0).float() - tp; fn = cm.sum(1).float() - tp
    dice = (2 * tp / (2 * tp + fp + fn).clamp(min=1)).tolist()
    return dict(overall=float(tp.sum() / cm.sum().clamp(min=1)), mean_class=float(np.mean(dice)), per_class=[round(d, 4) for d in dice], n_px=int(cm.sum()))


# ---- build ----
model = ViTSeg(build_phikon(WEIGHTS)).to(dev)
set_trainable(model, UNFREEZE_LAST)
SELECT = globals().get("SELECT", "both")
n_all = sum(p.numel() for p in model.parameters()); n_bb = sum(p.numel() for p in model.vit.parameters())
stamp(f"model built: {n_all / 1e6:.1f}M parameters (backbone {n_bb / 1e6:.1f}M), last {UNFREEZE_LAST} blocks trainable"); print("checkpoint selection by:", SELECT, flush=True)
DS = PackedDS if packed_dir else PatchDS
SHIFT = set(globals().get("SHIFT_PATIENTS", []))
SLIDE_EX = set(globals().get("EXCLUDE_SLIDES", [])); KW = dict(exclude_slides=SLIDE_EX) if (SLIDE_EX and DS is PackedDS) else {}          # EXCLUDE_SLIDES: whole slides left out of the TRAINING patches (ablation: back to the original 154 slides)
if SHIFT and DS is PackedDS: train_ds, val_ds, shift_ds = DS("train", LIMIT, exclude=SHIFT, **KW), DS("val", LIMIT), DS("train", LIMIT, only=SHIFT, cap=1500)
else: train_ds, val_ds, shift_ds = DS("train", LIMIT, **KW), DS("val", LIMIT), None
if SLIDE_EX: print(f"excluded {len(SLIDE_EX)} slides from the training patches", flush=True)
N_TRAIN_FULL = len(DS("train", 0, exclude=SHIFT, **KW)) if (SHIFT and DS is PackedDS) else len(DS("train", 0, **KW)); N_VAL_FULL = len(DS("val", 0)); N_SHIFT_FULL = len(DS("train", 0, only=SHIFT, cap=1500)) if (SHIFT and DS is PackedDS) else 0
if shift_ds is not None: print(f"stain-shift validation: {len(SHIFT)} held-out patients, {len(shift_ds)} patches (never trained on)", flush=True)
stamp("patch lists built")
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"train patches {len(train_ds)} (full {N_TRAIN_FULL}) | val patches {len(val_ds)} (full {N_VAL_FULL}) | trainable {n_tr / 1e6:.1f}M", flush=True)


def class_presence(ds):
    if isinstance(ds, PackedDS): return np.array([[bool((e["pres"] >> c) & 1) for c in range(4)] for e in ds.items])
    pres = np.zeros((len(ds), 4), bool)
    for i, (dd, n) in enumerate(ds.items):
        u = np.unique(np.array(Image.open(f"{dd}/masks/{n}.png"))); pres[i] = [(c in u) for c in (1, 2, 3, 4)]
    return pres


stamp("reading masks for the class-rarity sampler" if BALANCE else "no balancing")
if BALANCE:
    pres = class_presence(train_ds); n_c = pres.sum(0).clip(min=1); w = (pres / n_c).sum(1) + 1e-6
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(train_ds), replacement=True)
    exp = (pres * w[:, None]).sum(0) / w.sum() * len(train_ds)
    print("patches containing each class (other, non-inv, invasive, necrosis):", pres.sum(0).tolist(), "-> expected per epoch after balancing:", np.round(exp).astype(int).tolist(), flush=True)
else: sampler = None
NW = int(os.environ.get("BEETLE_WORKERS", 4))
tl = DataLoader(train_ds, batch_size=BS, shuffle=(sampler is None), sampler=sampler, num_workers=NW, drop_last=True, pin_memory=True); vl = DataLoader(val_ds, batch_size=BS, num_workers=NW)
sl = DataLoader(shift_ds, batch_size=BS, num_workers=NW) if shift_ds is not None else None
dec_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("vit.")]
bb_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("vit.")]
opt = torch.optim.AdamW([{"params": dec_params, "lr": LR_DEC}] + ([{"params": bb_params, "lr": LR_BB}] if bb_params else []), weight_decay=1e-2)
total = EPOCHS * len(tl); warm = max(1, int(0.03 * total)); delay = int(BB_DELAY * total)       # BB_DELAY = fraction of all steps during which the backbone LR stays 0 (the fresh decoder trains alone first), then a linear warm-up and the same cosine
lam_dec = lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm)))
lam_bb = lambda t: 0.0 if t < delay else ((t - delay + 1) / warm if t < delay + warm else 0.5 * (1 + math.cos(math.pi * (t - delay - warm) / max(1, total - delay - warm))))
sched = torch.optim.lr_scheduler.LambdaLR(opt, [lam_dec] + ([lam_bb] if bb_params else []))
scaler = torch.amp.GradScaler(enabled=AMP)

best, history, start_ep = -1, [], 0
if RESUME_TAG:                                           # second session: everything (model, optimizer, scheduler, scaler, RNG states, best score, history) comes from the previous session's resume_<tag>.pt
    ck_dir, _ = find_first(lambda r, d, f: f"resume_{RESUME_TAG}.pt" in f)
    assert ck_dir, f"resume_{RESUME_TAG}.pt not found under {INROOT} (attach the previous session's notebook as a kernel source; it must have ended COMPLETE)"
    R = torch.load(f"{ck_dir}/resume_{RESUME_TAG}.pt", map_location="cpu", weights_only=False)
    assert R["total_epochs"] == EPOCHS and R["steps_per_epoch"] == len(tl), f"resume mismatch: epochs {R['total_epochs']} vs {EPOCHS}, steps/epoch {R['steps_per_epoch']} vs {len(tl)}"
    model.load_state_dict(R["model"], strict=True); opt.load_state_dict(R["opt"]); sched.load_state_dict(R["sched"]); scaler.load_state_dict(R["scaler"])
    start_ep, best, history = R["epoch"], R["best"], R["history"]
    random.setstate(R["py_rng"]); np.random.set_state(R["np_rng"]); torch.set_rng_state(R["torch_rng"])
    if R.get("cuda_rng") is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(R["cuda_rng"])
    if os.path.exists(f"{ck_dir}/seg_best_{RESUME_TAG}.pt"): shutil.copy(f"{ck_dir}/seg_best_{RESUME_TAG}.pt", f"{OUTD}/seg_best_{TAG}.pt")        # the best checkpoint so far travels on in this session's output
    print(f"RESUMED from {RESUME_TAG}: {start_ep} epochs done, best {SELECT} {best:.4f}, {len(history)} history rows", flush=True); del R
stamp("sampler ready, training starts" if not RESUME_TAG else "resumed, training continues")


def save_resume(ep_done):                                  # written after every epoch (tmp file + rename: a kill during the write never leaves a broken file)
    R = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(), epoch=ep_done, best=best, history=history, total_epochs=EPOCHS, steps_per_epoch=len(tl),
             py_rng=random.getstate(), np_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    torch.save(R, f"{OUTD}/resume_{TAG}.pt.tmp"); os.replace(f"{OUTD}/resume_{TAG}.pt.tmp", f"{OUTD}/resume_{TAG}.pt")


def finish(reason):
    torch.save(model.state_dict(), f"{OUTD}/seg_last_{TAG}.pt")
    json.dump(dict(tag=TAG, history=history, select=SELECT, best_score=best, best_overall=max([h["overall"] for h in history] or [0]), best_mean_class=max([h["mean_class"] for h in history] or [0]), shift_patients=sorted(SHIFT),
                   unfreeze_last=UNFREEZE_LAST, resumed_from=RESUME_TAG, epochs_done=len(history), total_epochs=EPOCHS, stopped=reason), open(f"{OUTD}/metrics_{TAG}.json", "w"), indent=1)
    print(f"done ({reason}) | best {SELECT} {round(best, 4)} | epochs done {len(history)}/{EPOCHS}", flush=True)
    import sys; sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


t0 = time.time(); stop_reason = "all epochs done"
for ep in range(start_ep, EPOCHS):
    model.train(); run = 0.0; te = time.time()
    for bi, (x, y) in enumerate(tl):
        x = x.to(dev, non_blocking=True).float() / 255; t = to_target(y.to(dev, non_blocking=True)); x, t = augment(x, t)
        with torch.autocast(dev, dtype=torch.float16, enabled=AMP): lo = model(x)
        loss = F.cross_entropy(lo.float(), t, ignore_index=255) + dice_loss(lo, t)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0); scaler.step(opt); scaler.update(); sched.step()
        run += loss.item()
        if bi == 0: stamp("first batch done")
        if bi % 50 == 0: print(f"ep{ep} {bi}/{len(tl)} loss {run/(bi+1):.4f} | {time.time()-te:.0f}s | peak {torch.cuda.max_memory_allocated()/2**30 if dev=='cuda' else 0:.1f} GB", flush=True)
        if TIMING and bi + 1 == TIMING:                    # GPU timing test: a few training steps + a little evaluation, then project the full run and stop
            dt = (time.time() - te) / (bi + 1); steps_full = N_TRAIN_FULL // BS; peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
            te2 = time.time(); evaluate(model, vl); ev = (time.time() - te2) / max(1, len(val_ds)); ep_train_h = dt * steps_full / 3600; ep_eval_h = ev * (N_VAL_FULL + N_SHIFT_FULL) / 3600
            print(f"TIMING: {dt:.2f} s/step (batch {BS}) over {bi + 1} steps (the first steps include warm-up, so this is an upper bound) | peak GPU memory {peak:.1f} GB | eval {ev * 1000:.0f} ms/patch", flush=True)
            print(f"PROJECTION: {steps_full} steps/epoch -> {ep_train_h:.2f} h train + {ep_eval_h:.2f} h eval per epoch | {EPOCHS} epochs = {EPOCHS * (ep_train_h + ep_eval_h):.1f} h | session A (3 epochs) = {3 * (ep_train_h + ep_eval_h):.1f} h", flush=True)
            import sys; sys.stdout.flush(); os._exit(0)
    m = evaluate(model, vl); m.update(epoch=ep, train_loss=round(run / max(1, len(tl)), 4), secs=round(time.time() - te))
    if sl is not None: m["shift"] = evaluate(model, sl); m["both"] = (m["mean_class"] + m["shift"]["mean_class"]) / 2
    else: m["both"] = m["mean_class"]
    history.append(m); print(json.dumps(m), flush=True)
    if m[SELECT] > best:
        best = m[SELECT]; torch.save(model.state_dict(), f"{OUTD}/seg_best_{TAG}.pt"); print("saved best by", SELECT, round(best, 4), "| per-class", m["per_class"], flush=True)
    sh = [h["shift"]["mean_class"] for h in history if "shift" in h]
    if SHIFT_STOP and len(sh) >= SHIFT_STOP + 1 and all(sh[-i] < sh[-i - 1] for i in range(1, SHIFT_STOP + 1)): stop_reason = f"stain-shift score fell {SHIFT_STOP} epochs in a row ({[round(v, 4) for v in sh[-(SHIFT_STOP + 1):]]})"
    elif STOP_AFTER and ep + 1 >= STOP_AFTER and ep + 1 < EPOCHS: stop_reason = f"session stop after epoch {ep + 1} (resume from resume_{TAG}.pt)"
    elif TIME_BUDGET_H and ep + 1 < EPOCHS and (time.time() - t0) + 1.1 * m["secs"] > TIME_BUDGET_H * 3600: stop_reason = f"time budget {TIME_BUDGET_H} h reached after epoch {ep + 1}"
    save_resume(ep + 1)                                    # RNG states are captured here, after all of the epoch's evaluation, so a resumed run continues exactly like an uninterrupted one
    if stop_reason != "all epochs done": print("STOPPING:", stop_reason, flush=True); break
finish(stop_reason)
