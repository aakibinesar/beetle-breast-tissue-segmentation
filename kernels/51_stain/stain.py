# CPU notebook: stain / colour statistics of the 170 evaluation ROIs and of the training slides (from the packed patches), to test the
# hypothesis that our model loses on some centres because their stain / scanner look is not covered by the training slides.
# Output: rois_stain.csv (one row per ROI) and train_stain.csv (one row per training slide, mean over up to 12 patches). Tissue pixels only.
import os, io, glob, json, re, time
import numpy as np
from PIL import Image
T0 = time.time()
def stamp(m): print(f"[t+{time.time() - T0:5.0f}s] {m}", flush=True)

# Ruifrok & Johnston stain vectors (H, E, DAB) -> optical-density deconvolution
HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]]); HED = HED / np.linalg.norm(HED, axis=1, keepdims=True); INV = np.linalg.inv(HED)

def stats(img, step=4):
    a = img[::step, ::step].reshape(-1, 3).astype(np.float64)
    gray = a.mean(1); tissue = a[(gray < 215) & (a.max(1) - a.min(1) > 12)]          # drop white background and grey pixels
    if len(tissue) < 200: tissue = a
    od = -np.log10((tissue + 1.0) / 256.0); hed = od @ INV
    mx, mn = tissue.max(1), tissue.min(1); sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    return dict(r=tissue[:, 0].mean(), g=tissue[:, 1].mean(), b=tissue[:, 2].mean(), bright=tissue.mean(), contrast=tissue.mean(1).std(), sat=sat.mean(),
                h=hed[:, 0].mean(), e=hed[:, 1].mean(), h_sd=hed[:, 0].std(), e_sd=hed[:, 1].std(), tissue_frac=len(tissue) / max(1, a.shape[0]))

roi_root = next(r for r, d, f in os.walk("/kaggle/input") if "rois" in d and "model" in d)
rois = sorted(glob.glob(f"{roi_root}/rois/*.png")); stamp(f"{len(rois)} ROIs")
rows = []
for p in rois:
    n = os.path.basename(p); m = re.match(r"patient(\d+)_wsi(\d+)_roi(\d+)", n)
    rows.append(dict(name=n, patient=int(m.group(1)), **stats(np.array(Image.open(p).convert("RGB")))))
import csv
def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
write("/kaggle/working/rois_stain.csv", rows); stamp("ROI stats written")

packed = next(r for r, d, f in os.walk("/kaggle/input") if r.endswith("packed") and "index.json" in f)
index = [e for e in json.load(open(f"{packed}/index.json")) if e["split"] == "train"]
by_slide = {}
for e in index: by_slide.setdefault(re.match(r"(patient\d+_wsi\d+)", e["name"]).group(1), []).append(e)
stamp(f"{len(by_slide)} training slides in the packed set")
rng = np.random.default_rng(0); fh = {}; trows = []
for sid, es in sorted(by_slide.items()):
    pick = [es[i] for i in rng.choice(len(es), min(12, len(es)), replace=False)]; ss = []
    for e in pick:
        f = fh.get(e["shard"]) or fh.setdefault(e["shard"], open(f"{packed}/shard_{e['shard']:03d}.bin", "rb")); f.seek(e["io"]); ib = f.read(e["il"])
        ss.append(stats(np.array(Image.open(io.BytesIO(ib)).convert("RGB")), step=2))
    trows.append(dict(slide=sid, patient=int(re.match(r"patient(\d+)", sid).group(1)), n_patches=len(pick), **{k: float(np.mean([s[k] for s in ss])) for k in ss[0]}))
write("/kaggle/working/train_stain.csv", trows); stamp(f"training-slide stats written ({len(trows)} slides)")
print("DONE", flush=True); os._exit(0)
