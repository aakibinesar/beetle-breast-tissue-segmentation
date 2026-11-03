EPOCHS, BS, LR_DEC, LR_BB, UNFREEZE_LAST, LIMIT, TAG, BALANCE, INIT, SELECT, WATCHDOG, STAIN = 5, 8, 5e-4, 3e-5, 4, 0, "v3", 1, "v2", "both", 0, 0.1
SHIFT_PATIENTS = [25, 51, 87, 100, 107, 173, 181, 185, 199, 216, 219, 223, 224, 225, 227, 233, 251, 303, 306]
"""ViT (torchvision layout, e.g. Phikon ViT-B/16) + light CNN skip branch + conv decoder for 4-class tissue segmentation (classes 1-4 -> logits 0-3)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ViTSeg(nn.Module):
    def __init__(s, vit, feat_layers=(3, 6, 9, 12), n_cls=4, dec=128):
        super().__init__()
        s.vit = vit; s.layers = tuple(feat_layers); d = vit.hidden_dim
        s.proj = nn.ModuleList([nn.Conv2d(d, dec, 1) for _ in s.layers])
        s.fuse = conv_bn(dec * len(s.layers), 256)
        s.c1 = nn.Sequential(conv_bn(3, 32), conv_bn(32, 32))                    # 1/1 skip (sharp boundaries; the ViT grid is only 1/16)
        s.c2 = nn.Sequential(conv_bn(32, 64, 3, 2), conv_bn(64, 64))            # 1/2
        s.c4 = nn.Sequential(conv_bn(64, 96, 3, 2), conv_bn(96, 96))            # 1/4
        s.c8 = nn.Sequential(conv_bn(96, 128, 3, 2), conv_bn(128, 128))         # 1/8
        s.d8 = conv_bn(256 + 128, 192); s.d4 = conv_bn(192 + 96, 128); s.d2 = conv_bn(128 + 64, 96); s.d1 = conv_bn(96 + 32, 64)
        s.out = nn.Conv2d(64, n_cls, 1)
        s.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        s.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def backbone_trainable(s):
        return any(p.requires_grad for p in s.vit.parameters())

    def tokens(s, x):                                                            # x: normalised [B,3,H,W], H and W multiples of 16
        v = s.vit; B, _, H, W = x.shape; h, w = H // 16, W // 16
        t = v.conv_proj(x).flatten(2).transpose(1, 2)
        t = torch.cat([v.class_token.expand(B, -1, -1), t], 1)
        pe = v.encoder.pos_embedding; g = int(math.sqrt(pe.shape[1] - 1))
        pg = pe[:, 1:].reshape(1, g, g, -1).permute(0, 3, 1, 2)
        pg = F.interpolate(pg.float(), size=(h, w), mode="bicubic", align_corners=False).permute(0, 2, 3, 1).reshape(1, h * w, -1)
        t = v.encoder.dropout(t + torch.cat([pe[:, :1], pg.to(pe.dtype)], 1))
        feats = []
        for i, blk in enumerate(v.encoder.layers, 1):
            t = blk(t)
            if i in s.layers: feats.append(v.encoder.ln(t)[:, 1:].transpose(1, 2).reshape(B, -1, h, w))
        return feats

    def forward(s, x01):                                                         # x01: float [B,3,H,W] in 0..1
        x = (x01 - s.mean) / s.std
        f1 = s.c1(x); f2 = s.c2(f1); f4 = s.c4(f2); f8 = s.c8(f4)
        if s.backbone_trainable(): feats = s.tokens(x)
        else:
            with torch.no_grad(): feats = s.tokens(x)
        p = s.fuse(torch.cat([pr(f.float() if f.dtype != x.dtype else f) for pr, f in zip(s.proj, feats)], 1))
        up = lambda a, ref: F.interpolate(a, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        y = s.d8(torch.cat([up(p, f8), f8], 1)); y = s.d4(torch.cat([up(y, f4), f4], 1)); y = s.d2(torch.cat([up(y, f2), f2], 1)); y = s.d1(torch.cat([up(y, f1), f1], 1))
        return s.out(y)


def build_phikon(weights_path):
    from torchvision.models import VisionTransformer
    vit = VisionTransformer(image_size=224, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=4 * 768, num_classes=0)
    vit.heads = nn.Identity()
    if weights_path: vit.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return vit


def set_trainable(model, last_blocks):
    for p in model.vit.parameters(): p.requires_grad = False
    if last_blocks > 0:
        for blk in list(model.vit.encoder.layers)[-last_blocks:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in model.vit.encoder.ln.parameters(): p.requires_grad = True

# ---------------- BEETLE own model: Phikon ViT-B/16 (+ optional last-N-block fine-tuning) + CNN-skip decoder, trained on the 512px / 0.5 um patches of beetle-31-prep ----------------
import os, io, re, glob, json, time, math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

dev = os.environ.get("BEETLE_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
AMP = (dev == "cuda") and os.environ.get("BEETLE_AMP", "1") == "1"      # fp16 autocast (the local MX450 has a broken fp16 cuDNN conv path; Kaggle T4 is fine)
OUTD = os.environ.get("BEETLE_OUT", "/kaggle/working")


INROOT = os.environ.get("BEETLE_IN", "/kaggle/input")


_T0 = time.time()
def stamp(msg): print(f"[t+{time.time() - _T0:6.0f}s] {msg}", flush=True)            # startup stages are timed so a slow mount can be seen in the log


def find_first(pred):
    for r, dirs, files in os.walk(INROOT):
        dirs[:] = [d for d in dirs if d != "patches"]                  # never descend into the (huge) patch folders when looking for weights/checkpoints
        if pred(r, dirs, files): return r, files
    return None, None


stamp("start: looking for the patch data")
packed_dir = None                                    # a notebook output made by kernels/41_pack: packed/shard_XXX.bin + packed/index.json (one mount, few large files)
for r, dirs, files in os.walk(INROOT):
    if r.endswith("packed") and "index.json" in files and any(f.startswith("shard_") for f in files): packed_dir = r; break
    dirs[:] = [d for d in dirs if d != "patches"]
proots = []                                          # every patches/ directory among the attached kernel outputs (beetle-31-prep and beetle-33-prep2)
if packed_dir is None:
    for r, dirs, files in os.walk(INROOT):
        if r.endswith("patches") and "train" in dirs and "val" in dirs: proots.append(r); dirs[:] = []
    assert proots, "neither a packed patch set nor patches directories were found"
    stamp(f"{len(proots)} patch roots found")
else:
    PACKED_INDEX = json.load(open(f"{packed_dir}/index.json")); stamp(f"packed patch set found: {packed_dir} ({len(PACKED_INDEX)} patches)")
wdir, wfiles = find_first(lambda r, d, f: "phikon.pth" in f)
WEIGHTS = os.path.join(wdir, "phikon.pth") if wdir else None
assert WEIGHTS or os.environ.get("BEETLE_IN"), "phikon.pth not found - model source missing"
stamp("weights located"); print("patch roots:", proots or packed_dir, "| weights:", WEIGHTS, "| device", dev, flush=True)


class PatchDS(Dataset):
    def __init__(s, split, limit=0):
        s.items = [(f"{r}/{split}", f[:-4]) for r in proots for f in sorted(os.listdir(f"{r}/{split}/images"))]
        if limit: s.items = s.items[::max(1, len(s.items) // limit)][:limit]            # a spread over all patch roots (old and new), not just the first one
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        d, n = s.items[i]
        x = np.array(Image.open(f"{d}/images/{n}.jpg").convert("RGB")); y = np.array(Image.open(f"{d}/masks/{n}.png"))
        return torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(y.astype(np.int64))


class PackedDS(Dataset):
    """Reads patches from the shard files written by kernels/41_pack (raw jpg / png bytes at known offsets)."""
    def __init__(s, split, limit=0, exclude=(), only=(), cap=0):
        pid = lambda e: int(re.match(r"patient(\d+)", e["name"]).group(1))
        s.items = [e for e in PACKED_INDEX if e["split"] == split and (not exclude or pid(e) not in exclude) and (not only or pid(e) in only)]      # exclude / only: patient numbers (stain-shift validation)
        if cap and len(s.items) > cap: s.items = [s.items[i] for i in np.random.default_rng(0).choice(len(s.items), cap, replace=False)]       # fixed random subset
        if limit: s.items = s.items[::max(1, len(s.items) // limit)][:limit]
        s.fh = {}                                         # shard file handles are opened lazily inside each DataLoader worker
    def __len__(s): return len(s.items)
    def _read(s, shard, off, ln):
        f = s.fh.get(shard)
        if f is None: f = s.fh[shard] = open(f"{packed_dir}/shard_{shard:03d}.bin", "rb")
        f.seek(off); return f.read(ln)
    def __getitem__(s, i):
        e = s.items[i]
        x = np.array(Image.open(io.BytesIO(s._read(e["shard"], e["io"], e["il"]))).convert("RGB")); y = np.array(Image.open(io.BytesIO(s._read(e["shard"], e["mo"], e["ml"]))))
        return torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(y.astype(np.int64))


def to_target(y):                       # mask 0 = unannotated -> ignore (255); 1..4 -> 0..3
    t = y - 1; t[y == 0] = 255; return t


HED_M = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]]); HED_M = HED_M / HED_M.norm(dim=1, keepdim=True); HED_INV = torch.linalg.inv(HED_M)


def stain_jitter(x, sigma):
    """Stain (haematoxylin / eosin / residual) jitter in optical-density space: every patch gets its own random per-stain scale (1 +- sigma) and shift (+- sigma).
    x: float [B,3,H,W] in 0..1. A round trip with scale 1 and shift 0 returns the input (tested). Meant to make the model robust to other labs' stains / scanners."""
    hm, hi = HED_M.to(x.device), HED_INV.to(x.device)
    od = -torch.log10((x * 255 + 1) / 256)                                  # optical density
    c = torch.einsum("bchw,cd->bdhw", od, hi)                               # stain concentrations
    B = x.shape[0]
    a = 1 + (torch.rand(B, 3, 1, 1, device=x.device) * 2 - 1) * sigma; b = (torch.rand(B, 3, 1, 1, device=x.device) * 2 - 1) * sigma
    od2 = torch.einsum("bdhw,dc->bchw", c * a + b, hm)
    return ((256 * 10 ** (-od2) - 1).clamp(0, 255) / 255).to(x.dtype)


def augment(x, y):                      # x float [B,3,H,W] 0..1 ; y long [B,H,W]; per-sample dihedral + colour jitter (rough stain/brightness variation)
    xs, ys = [], []
    for i in range(x.shape[0]):
        xi, yi = x[i:i + 1], y[i:i + 1]; k = random.randint(0, 3)
        xi = torch.rot90(xi, k, (2, 3)); yi = torch.rot90(yi, k, (1, 2))
        if random.random() < 0.5: xi = xi.flip(3); yi = yi.flip(2)
        gain = 1 + 0.08 * torch.randn(1, 3, 1, 1, device=x.device); bright = 0.05 * torch.randn(1, 1, 1, 1, device=x.device); con = 1 + 0.12 * torch.randn(1, 1, 1, 1, device=x.device)
        m = xi.mean((1, 2, 3), keepdim=True); xi = (((xi - m) * con + m) * gain + bright).clamp(0, 1)
        xs.append(xi); ys.append(yi)
    xo = torch.cat(xs)
    if globals().get("STAIN", 0): xo = stain_jitter(xo, float(STAIN))             # STAIN = sigma of the stain jitter (0 = off)
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
sys_path_ok = True
model = ViTSeg(build_phikon(WEIGHTS)).to(dev)
INIT, SELECT = globals().get("INIT", ""), globals().get("SELECT", "overall")          # builds made before these options existed keep the old behaviour
if INIT:                                              # warm start: weights of an earlier run (seg_best_<INIT>.pt among the attached kernel outputs); the whole model incl. the backbone is restored
    ckpt_dir, _ = find_first(lambda r, d, f: f"seg_best_{INIT}.pt" in f)
    assert ckpt_dir, f"warm-start checkpoint seg_best_{INIT}.pt not found under {INROOT}"
    print(model.load_state_dict(torch.load(f"{ckpt_dir}/seg_best_{INIT}.pt", map_location=dev), strict=True), "<- warm start from", INIT, flush=True)
set_trainable(model, UNFREEZE_LAST)
stamp("model built" + (" + warm start loaded" if INIT else "")); print("checkpoint selection by:", SELECT, flush=True)
DS = PackedDS if packed_dir else PatchDS
SHIFT = set(globals().get("SHIFT_PATIENTS", []))                  # patients whose slides are held out of training as a stain-shift validation set (never seen by the warm-start model either, see kernels/52_stain2 + select_shift.py)
if SHIFT and DS is PackedDS: train_ds, val_ds, shift_ds = DS("train", LIMIT, exclude=SHIFT), DS("val", LIMIT), DS("train", LIMIT, only=SHIFT, cap=1500)
else: train_ds, val_ds, shift_ds = DS("train", LIMIT), DS("val", LIMIT), None
if shift_ds is not None: print(f"stain-shift validation: {len(SHIFT)} held-out patients, {len(shift_ds)} patches (never trained on)", flush=True)
stamp("patch lists built (os.listdir of every root)")
print(f"train patches {len(train_ds)} | val patches {len(val_ds)} | trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M", flush=True)
def class_presence(ds):                                  # which of the 4 classes occur in each patch (cheap: reads the small mask PNGs once)
    if isinstance(ds, PackedDS): return np.array([[bool((e["pres"] >> c) & 1) for c in range(4)] for e in ds.items])      # stored in the index by the packer
    pres = np.zeros((len(ds), 4), bool)
    for i, (dd, n) in enumerate(ds.items):
        u = np.unique(np.array(Image.open(f"{dd}/masks/{n}.png"))); pres[i] = [(c in u) for c in (1, 2, 3, 4)]
    return pres


stamp("reading masks for the class-rarity sampler" if BALANCE else "no balancing")
if BALANCE:                                               # sampling weight of a patch = sum over the classes it contains of 1/(#patches containing that class)
    pres = class_presence(train_ds); n_c = pres.sum(0).clip(min=1); w = (pres / n_c).sum(1) + 1e-6
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(train_ds), replacement=True)
    exp = (pres * w[:, None]).sum(0) / w.sum() * len(train_ds)
    print("patches containing each class (other, non-inv, invasive, necrosis):", pres.sum(0).tolist(), "-> expected per epoch after balancing:", np.round(exp).astype(int).tolist(), flush=True)
else: sampler = None
tl = DataLoader(train_ds, batch_size=BS, shuffle=(sampler is None), sampler=sampler, num_workers=int(os.environ.get("BEETLE_WORKERS", 4)), drop_last=True, pin_memory=True); vl = DataLoader(val_ds, batch_size=BS, num_workers=int(os.environ.get("BEETLE_WORKERS", 4)))
sl = DataLoader(shift_ds, batch_size=BS, num_workers=int(os.environ.get("BEETLE_WORKERS", 4))) if shift_ds is not None else None
dec_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("vit.")]
bb_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("vit.")]
opt = torch.optim.AdamW([{"params": dec_params, "lr": LR_DEC}] + ([{"params": bb_params, "lr": LR_BB}] if bb_params else []), weight_decay=1e-2)
total = EPOCHS * len(tl); warm = max(1, int(0.03 * total))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm))))
scaler = torch.amp.GradScaler(enabled=AMP)
stamp("sampler ready, training starts")
best, history, t0 = -1, [], time.time()
if INIT and sl is not None:                                  # reference: the warm-start model itself on the same validation sets (it never saw the held-out patients)
    b0 = evaluate(model, vl); b1 = evaluate(model, sl); print("BASELINE (warm-start model, before training) val:", json.dumps(b0), "| stain-shift:", json.dumps(b1), flush=True)
    json.dump(dict(val=b0, shift=b1), open(f"{OUTD}/baseline_{TAG}.json", "w"))
for ep in range(EPOCHS):
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
    m = evaluate(model, vl); m.update(epoch=ep, train_loss=round(run / max(1, len(tl)), 4), secs=round(time.time() - te))
    if sl is not None: m["shift"] = evaluate(model, sl); m["both"] = (m["mean_class"] + m["shift"]["mean_class"]) / 2          # SELECT=both: average of the mean-class Dice on the normal validation slides and on the held-out stain-shift slides
    else: m["both"] = m["mean_class"]
    history.append(m); print(json.dumps(m), flush=True)
    if m[SELECT] > best:                                  # SELECT = "mean_class" (average of the four per-class Dice) or "overall" (micro Dice, dominated by the biggest class)
        best = m[SELECT]; torch.save(model.state_dict(), f"{OUTD}/seg_best_{TAG}.pt"); print("saved best by", SELECT, round(best, 4), "| per-class", m["per_class"], flush=True)
    if time.time() - t0 > 7.5 * 3600: print("time budget reached"); break
torch.save(model.state_dict(), f"{OUTD}/seg_last_{TAG}.pt")
json.dump(dict(tag=TAG, history=history, select=SELECT, best_score=best, best_overall=max(h["overall"] for h in history), best_mean_class=max(h["mean_class"] for h in history), shift_patients=sorted(SHIFT), unfreeze_last=UNFREEZE_LAST, init=INIT), open(f"{OUTD}/metrics_{TAG}.json", "w"), indent=1)
print(f"done | best {SELECT} {round(best, 4)} | best overall {max(h['overall'] for h in history):.4f} | best mean-class {max(h['mean_class'] for h in history):.4f}")
import sys; sys.stdout.flush(); sys.stderr.flush(); os._exit(0)             # hard exit: DataLoader workers / pinned-memory threads must not keep the session alive after the results are saved

