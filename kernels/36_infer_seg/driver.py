# ---------------- BEETLE: run our own segmentation model on the 170 evaluation ROIs (sliding windows with overlap, dihedral TTA) ----------------
import os, glob, io, json, time, zipfile, shutil
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

dev = "cuda" if torch.cuda.is_available() else "cpu"
roi_dir = next(os.path.join(r, "rois") for r, d, f in os.walk("/kaggle/input") if "rois" in d and "model" in d)
ck = next(os.path.join(r, f) for r, d, fs in os.walk("/kaggle/input") for f in fs if f.startswith("seg_best_") and f.endswith(".pt"))
print("ROIs:", roi_dir, "| checkpoint:", ck, flush=True)
if not os.environ.get("BEETLE_TEST"): model = ViTSeg(build_phikon(None)); model.load_state_dict(torch.load(ck, map_location="cpu"), strict=True); model = model.to(dev).eval()
P, STRIDE = 512, 256
win = torch.hann_window(P, periodic=False).clamp(min=1e-3); WIN2D = (win[:, None] * win[None, :]).to(dev)       # cosine taper: window centres count more than their borders
TRANSFORMS = [(0, 0), (0, 1), (2, 0), (2, 1)][:N_TTA]                                                           # (quarter turns, flip along width)


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
        for i in range(0, len(coords), 8):
            cs = coords[i:i + 8]; b = torch.stack([x[:, y:y + P, xx:xx + P] for (y, xx) in cs])
            if k: b = torch.rot90(b, k, (2, 3))
            if f: b = b.flip(3)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")): lo = model(b)
            p = torch.softmax(lo.float(), 1)
            if f: p = p.flip(3)
            if k: p = torch.rot90(p, -k, (2, 3))
            for j, (y, xx) in enumerate(cs): acc[:, y:y + P, xx:xx + P] += p[j] * WIN2D; wsum[:, y:y + P, xx:xx + P] += WIN2D
    return (acc / wsum.clamp(min=1e-6))[:, :H, :W].cpu()


OUT = "/kaggle/working/predictions"; os.makedirs(OUT, exist_ok=True)
rois = sorted(glob.glob(f"{roi_dir}/*.png")); assert len(rois) == 170, len(rois)
t0 = time.time(); counts = np.zeros(5, np.int64)
for n, p in enumerate(rois):
    img = np.array(Image.open(p).convert("RGB")); pr = predict_probs(img)
    pred = (pr.argmax(0) + 1).numpy().astype(np.uint8); assert pred.shape == img.shape[:2]
    counts += np.bincount(pred.ravel(), minlength=5)
    Image.fromarray(pred, mode="L").save(f"{OUT}/{os.path.basename(p)}")
    if n % 10 == 0: print(f"{n + 1}/170 | {os.path.basename(p)} {img.shape[:2]} | {time.time() - t0:.0f}s | peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
print(f"done in {time.time() - t0:.0f}s | pixel share other/non-inv/invasive/necrosis:", np.round(counts[1:] / counts.sum(), 4).tolist(), flush=True)
bad = []; names = sorted(os.path.basename(p) for p in rois); assert sorted(os.listdir(OUT)) == names
for p in rois:
    a = Image.open(p); b = Image.open(f"{OUT}/{os.path.basename(p)}"); arr = np.array(b)
    if b.mode != "L" or b.size != a.size or not set(np.unique(arr)).issubset({1, 2, 3, 4}): bad.append((os.path.basename(p), b.mode, b.size, a.size))
print("validator: files", len(names), "| problems", len(bad), bad[:3], flush=True); assert not bad
with zipfile.ZipFile(f"/kaggle/working/beetle_own_{TAG}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for nme in names: z.write(f"{OUT}/{nme}", nme)
print("zip written:", os.path.getsize(f"/kaggle/working/beetle_own_{TAG}.zip") / 2**20, "MB"); shutil.rmtree(OUT)
