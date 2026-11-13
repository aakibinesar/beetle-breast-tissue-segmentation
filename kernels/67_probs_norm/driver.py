# ---------------- BEETLE: Phikon-v2 inference on the 170 evaluation ROIs with TEST-TIME COLOUR MATCHING (probabilities saved for CPU-side blends) ----------------
# Same as kernels/66_probs_vitb/driver.py (own model, 4-way flip/rotation TTA, probs/<roi>.npz uint8 'own' [4,H,W], zip of labels), plus ONE pre-processing step that is fixed in advance:
# every ROI is colour-matched (stain_norm.py: Reinhard in CIELAB on tissue pixels, soft tissue mask) to the average colour statistics of the TRAINING patches of the model (target_stats.json).
# No labels and no learning on the test images; nothing in this step is tuned on the test score. Header variables (set by build.sh):
#   NORM_ONLY = 1: CPU only, no model: target statistics + per-ROI statistics before/after matching + montages of original / matched ROIs (to look at before any GPU time is spent)
#   SMOKE = n: pretrained Phikon-v2 + RANDOM decoder on the first n ROIs only (pipeline, speed, memory; predictions are meaningless)       EXCLUDE_SLIDES: whole slides left out of the training patches (as in the trainer)
import os, glob, io, json, re, time, zipfile, shutil, csv
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

dev = os.environ.get("BEETLE_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
INROOT = os.environ.get("BEETLE_IN", "/kaggle/input"); WORK = os.environ.get("BEETLE_OUT", "/kaggle/working"); N_EXPECT = int(os.environ.get("BEETLE_N_ROIS", 170))
T0 = time.time()
def stamp(m): print(f"[t+{time.time() - T0:5.0f}s] {m}", flush=True)
peak_gb = lambda: torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0
roi_dir = next(os.path.join(r, "rois") for r, d, f in os.walk(INROOT) if "rois" in d and "model" in d)
packed_dirs = sorted(r for r, d, f in os.walk(INROOT) if r.endswith("packed") and "index.json" in f and any(x.startswith("shard_") for x in f))
assert packed_dirs, "no packed training patches attached"
os.makedirs(WORK, exist_ok=True)
if NORM_ONLY: model, ck = None, None
elif SMOKE:
    wdir = next((r for r, d, f in os.walk(INROOT) if "model.safetensors" in f and "config.json" in f), None); ck = None
    model = ViTSeg(build_phikon(wdir)).to(dev).eval()
    print("SMOKE MODE: pretrained Phikon-v2 backbone + RANDOM decoder, first", SMOKE, "ROIs | ROIs:", roi_dir, "| weights:", wdir, "| device", dev, flush=True)
else:
    ck = next(os.path.join(r, f) for r, d, fs in os.walk(INROOT) for f in fs if f.startswith("seg_best_") and f.endswith(".pt"))
    print("ROIs:", roi_dir, "| checkpoint:", ck, "| device", dev, flush=True)
    if not os.environ.get("BEETLE_TEST"):
        model = ViTSeg(build_phikon(None)); sd = remap_dino_keys(torch.load(ck, map_location="cpu"), model.state_dict().keys()); model.load_state_dict(sd, strict=True); model = model.to(dev).eval()
if globals().get("model") is not None:
    try:
        import transformers
        stamp(f"transformers {transformers.__version__}, torch {torch.__version__}")
    except Exception: pass
    with torch.no_grad(): _o = model(torch.zeros(1, 3, 512, 512, device=dev)); assert _o.shape[:2] == (1, 4), _o.shape
    stamp("model loaded; forward self-check OK")
target = target_from_packed(packed_dirs, set(globals().get("EXCLUDE_SLIDES", [])), dev=dev); json.dump(target, open(f"{WORK}/target_stats.json", "w"))
stamp(f"target colour statistics from {target['n_slides']} training slides ({target['n_patches']} patches, {target['n_excluded']} slides excluded): L/a/b mean {np.round(target['mean'], 2).tolist()} std {np.round(target['std'], 2).tolist()}")
P, STRIDE = 512, 256
win = torch.hann_window(P, periodic=False).clamp(min=1e-3); WIN2D = (win[:, None] * win[None, :]).to(dev)       # cosine taper: window centres count more than their borders
TRANSFORMS = [(0, 0), (0, 1), (2, 0), (2, 1)][:N_TTA]                                                           # (quarter turns, flip along width)
BATCH = int(os.environ.get("BEETLE_INF_BATCH", 8))


def predict_probs(img):                                   # img uint8 [H,W,3] -> softmax [4,H,W] (float32, cpu)
    H, W = img.shape[:2]; ph, pw = max(P, H), max(P, W)
    ys = list(range(0, ph - P + 1, STRIDE)); xs = list(range(0, pw - P + 1, STRIDE))
    if ys[-1] != ph - P: ys.append(ph - P)
    if xs[-1] != pw - P: xs.append(pw - P)
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255)
    x = F.pad(x[None], (0, pw - W, 0, ph - H), mode="reflect")[0].to(dev) if (ph > H or pw > W) else x.to(dev)
    acc = torch.zeros(4, ph, pw, device=dev); wsum = torch.zeros(1, ph, pw, device=dev)
    coords = [(y, xx) for y in ys for xx in xs]
    for (k, f) in TRANSFORMS:
        for i in range(0, len(coords), BATCH):
            cs = coords[i:i + BATCH]; b = torch.stack([x[:, y:y + P, xx:xx + P] for (y, xx) in cs])
            if k: b = torch.rot90(b, k, (2, 3))
            if f: b = b.flip(3)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")): lo = model(b)
            p = torch.softmax(lo.float(), 1)
            if f: p = p.flip(3)
            if k: p = torch.rot90(p, -k, (2, 3))
            for j, (y, xx) in enumerate(cs): acc[:, y:y + P, xx:xx + P] += p[j] * WIN2D; wsum[:, y:y + P, xx:xx + P] += WIN2D
    return (acc / wsum.clamp(min=1e-6))[:, :H, :W].cpu()


OUT = f"{WORK}/predictions"; os.makedirs(OUT, exist_ok=True)
rois = sorted(glob.glob(f"{roi_dir}/*.png"))
if SMOKE: rois = rois[:SMOKE]
else: assert len(rois) == N_EXPECT, len(rois)
# equivalence check: the first ROI WITHOUT colour matching, saved for a comparison with the earlier run of the same checkpoint (same pixels = same weights loaded under this transformers version)
if globals().get("model") is not None and not SMOKE:
    pr0 = predict_probs(np.array(Image.open(rois[0]).convert("RGB")))
    np.savez_compressed(f"{WORK}/check_raw_{os.path.basename(rois[0])[:-4]}.npz", own=np.round(pr0.numpy() * 255).astype(np.uint8)); stamp("un-matched first ROI predicted for the equivalence check")
dist_before = {}
for p in rois:                                            # pass 1: colour distance of every ROI to the training statistics (also used to pick the ROIs shown in the montages)
    s = lab_stats(np.array(Image.open(p).convert("RGB")), dev=dev); dist_before[p] = None if s is None else colour_distance(s[0], target)
ok = sorted((p for p in rois if dist_before[p] is not None), key=lambda p: dist_before[p])
assert ok, "no ROI has enough tissue"
shown = set(ok[:3] + ok[-3:])
stamp(f"pass 1: {len(ok)} of {len(rois)} ROIs have enough tissue | colour distance to the training statistics (units of the target std): median {np.median([dist_before[p] for p in ok]):.2f}, min {dist_before[ok[0]]:.2f}, max {dist_before[ok[-1]]:.2f}")
t0 = time.time(); counts = np.zeros(5, np.int64); rows = []; os.makedirs(f"{WORK}/montage", exist_ok=True)
for n, p in enumerate(rois):
    nm = os.path.basename(p); img0 = np.array(Image.open(p).convert("RGB")); img, info = normalise(img0, target, dev=dev)
    s1 = lab_stats(img, dev=dev); d1 = None if s1 is None else colour_distance(s1[0], target)
    rows.append(dict(name=nm, tissue_frac=None if info is None else round(info["tissue_frac"], 4), dist_before=None if dist_before[p] is None else round(dist_before[p], 4), dist_after=None if d1 is None else round(d1, 4),
                     **({} if info is None else {f"mean_{c}": round(v, 2) for c, v in zip("Lab", info["src_mean"])}), **({} if info is None else {f"scale_{c}": round(v, 3) for c, v in zip("Lab", info["scale"])})))
    if p in shown: Image.fromarray(np.concatenate([img0[::4, ::4], np.full((img0[::4, ::4].shape[0], 8, 3), 255, np.uint8), img[::4, ::4]], 1)).save(f"{WORK}/montage/{nm}")
    if NORM_ONLY:
        if n % 20 == 0: print(f"{n + 1}/{len(rois)} | {nm} {img0.shape[:2]} | distance {dist_before[p]} -> {d1} | {time.time() - t0:.0f}s", flush=True)
        continue
    pr = predict_probs(img)
    pred = (pr.argmax(0) + 1).numpy().astype(np.uint8); assert pred.shape == img0.shape[:2]
    os.makedirs(f"{WORK}/probs", exist_ok=True); np.savez_compressed(f"{WORK}/probs/{nm[:-4]}.npz", own=np.round(pr.numpy() * 255).astype(np.uint8))
    counts += np.bincount(pred.ravel(), minlength=5)
    Image.fromarray(pred, mode="L").save(f"{OUT}/{nm}")
    if n % 10 == 0 or SMOKE: print(f"{n + 1}/{len(rois)} | {nm} {img0.shape[:2]} | {time.time() - t0:.0f}s ({(time.time() - t0) / (n + 1):.1f} s/ROI) | peak {peak_gb():.1f} GB", flush=True)
cols = list(dict.fromkeys(k for r in rows for k in r))
with open(f"{WORK}/norm_stats.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
da = [r["dist_after"] for r in rows if r["dist_after"] is not None]; db = [r["dist_before"] for r in rows if r["dist_before"] is not None]
stamp(f"colour matching: distance to the training statistics, median before {np.median(db):.2f} -> after {np.median(da):.2f} | ROIs without enough tissue (left unchanged): {len(rows) - len(db)}")
if NORM_ONLY:
    shutil.rmtree(OUT); print("NORM_ONLY OK (no model was run); files: target_stats.json, norm_stats.csv, montage/*.png (left: original, right: colour-matched)", flush=True)
else:
    print(f"done in {time.time() - t0:.0f}s | pixel share other/non-inv/invasive/necrosis:", np.round(counts[1:] / counts.sum(), 4).tolist(), flush=True)
    bad = []; names = sorted(os.path.basename(p) for p in rois); assert sorted(os.listdir(OUT)) == names
    for p in rois:
        a = Image.open(p); b = Image.open(f"{OUT}/{os.path.basename(p)}"); arr = np.array(b)
        if b.mode != "L" or b.size != a.size or not set(np.unique(arr)).issubset({1, 2, 3, 4}): bad.append((os.path.basename(p), b.mode, b.size, a.size))
    print("validator: files", len(names), "| problems", len(bad), bad[:3], flush=True); assert not bad
    if SMOKE: print("SMOKE OK (pipeline runs; predictions are from a random decoder and are meaningless); projected full run for 170 ROIs: %.0f s" % ((time.time() - t0) / len(rois) * 170), flush=True); shutil.rmtree(OUT)
    else:
        with zipfile.ZipFile(f"{WORK}/beetle_own_{TAG}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for nme in names: z.write(f"{OUT}/{nme}", nme)
        print("zip written:", os.path.getsize(f"{WORK}/beetle_own_{TAG}.zip") / 2**20, "MB"); shutil.rmtree(OUT)
