# BEETLE patches consistency check (CPU): compare the two extraction runs (beetle-31-prep: 22 small slides, beetle-33-prep2: 63 large slides).
import os, glob, collections, json
import numpy as np, pandas as pd
from PIL import Image

roots = []
for r, dirs, files in os.walk("/kaggle/input"):
    if r.endswith("patches") and "train" in dirs and "val" in dirs: roots.append(r); dirs[:] = []
print("roots:", roots, flush=True)
rng = np.random.default_rng(0); names = ["unannot", "other", "non-inv", "invasive", "necrosis"]
rows = []
for r in roots:
    run = r.split("/")[3]
    man = pd.read_csv(f"{r}/manifest.csv")
    for sp in ("train", "val"):
        m = man[man.split == sp]; files = sorted(os.listdir(f"{r}/{sp}/images"))
        print(f"\n=== {run} / {sp}: manifest rows {len(m)}, image files {len(files)}, slides {m.slide.nunique()}", flush=True)
        if sp == "train": print("   centre-class counts (train):", m.centre_class.value_counts().sort_index().to_dict(), "| spacing_um min/median/max %.3f/%.3f/%.3f" % (m.spacing_um.min(), m.spacing_um.median(), m.spacing_um.max()), flush=True)
        samp = rng.choice(len(files), min(400, len(files)), replace=False)
        px = np.zeros(5, np.int64); rgb = []; present = collections.Counter(); per_slide = collections.defaultdict(lambda: np.zeros(5, np.int64))
        for i in samp:
            f = files[i][:-4]; im = np.asarray(Image.open(f"{r}/{sp}/images/{f}.jpg").convert("RGB")); mk = np.asarray(Image.open(f"{r}/{sp}/masks/{f}.png"))
            b = np.bincount(mk.ravel(), minlength=5)[:5]; px += b; per_slide[f.rsplit("_", 2)[0]] += b
            rgb.append(np.concatenate([im.reshape(-1, 3).mean(0), im.reshape(-1, 3).std(0), [im.mean()]]))
            for c in range(1, 5): present[c] += int(b[c] > 0)
        rgb = np.array(rgb); lab = px[1:].sum()
        print(f"   sampled {len(samp)} patches | labelled share per class:", {names[c]: round(px[c] / lab, 3) for c in range(1, 5)}, "| unannotated share of all px: %.3f" % (px[0] / px.sum()), flush=True)
        print("   patches containing each class:", {names[c]: present[c] for c in range(1, 5)}, flush=True)
        print("   mean RGB %s | within-patch std RGB %s | mean brightness %.1f (p5 %.1f, p95 %.1f)" % (np.round(rgb[:, :3].mean(0), 1), np.round(rgb[:, 3:6].mean(0), 1), rgb[:, 6].mean(), np.percentile(rgb[:, 6], 5), np.percentile(rgb[:, 6], 95)), flush=True)
        n_inv = sum(1 for s, b in per_slide.items() if b[3] > 0); print(f"   slides (in sample) with invasive px: {n_inv}/{len(per_slide)}", flush=True)
