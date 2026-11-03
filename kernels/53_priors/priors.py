# CPU notebook: class pixel counts of EVERY patch mask in the packed set beetle-43-pack2, to measure the class prior the BEETLE models were trained on
# (raw share, and the share after the rarity-weighted sampler of kernels/32_train). Output: patch_counts.csv (split, name, c0..c4 = pixels of mask value 0..4) + summary printed.
import os, io, json, time, re
import numpy as np
from PIL import Image
T0 = time.time()
def stamp(m): print(f"[t+{time.time() - T0:5.0f}s] {m}", flush=True)

packed = next(r for r, d, f in os.walk("/kaggle/input") if r.endswith("packed") and "index.json" in f)
index = json.load(open(f"{packed}/index.json")); stamp(f"{len(index)} patches in {packed}")
fh = {}; rows = []
for k, e in enumerate(index):
    f = fh.get(e["shard"]) or fh.setdefault(e["shard"], open(f"{packed}/shard_{e['shard']:03d}.bin", "rb"))
    f.seek(e["mo"]); m = np.array(Image.open(io.BytesIO(f.read(e["ml"]))))
    c = np.bincount(m.ravel(), minlength=5)[:5]
    rows.append((e["split"], e["name"], e["pres"], *[int(x) for x in c]))
    if (k + 1) % 5000 == 0: stamp(f"{k + 1}/{len(index)}")
import csv
with open("/kaggle/working/patch_counts.csv", "w", newline="") as fo:
    w = csv.writer(fo); w.writerow(["split", "name", "pres", "c0", "c1", "c2", "c3", "c4"]); w.writerows(rows)

C = np.array([r[3:] for r in rows], dtype=np.float64); pres = np.array([r[2] for r in rows]); split = np.array([r[0] for r in rows])
def share(mask, w=None):
    c = (C[mask] * (w[mask][:, None] if w is not None else 1)).sum(0); lab = c[1:]; return (lab / lab.sum()).round(4).tolist(), round(float(c[0] / c.sum()), 4)
for sp in ("train", "val"):
    print(f"{sp}: labelled-pixel share (other, non-inv, invasive, necrosis) {share(split == sp)[0]} | unannotated pixel share {share(split == sp)[1]}", flush=True)
# the rarity-weighted sampler of the trainer: weight of a patch = sum over its classes of 1 / (#train patches containing that class)
tr = split == "train"; P = np.array([[(p >> c) & 1 for c in range(4)] for p in pres])
n_c = P[tr].sum(0).clip(min=1); wgt = np.zeros(len(rows)); wgt[tr] = (P[tr] / n_c).sum(1) + 1e-6
print("train: labelled-pixel share AFTER the sampler weights (the effective training prior):", share(tr, wgt)[0], flush=True)
print("patches containing each class (train):", P[tr].sum(0).tolist(), flush=True)
stamp("DONE"); os._exit(0)
