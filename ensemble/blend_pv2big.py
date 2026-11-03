import os, glob, zipfile, io, time
import numpy as np
from PIL import Image
B = os.environ.get("BEETLE_DIR", ".")   # folder that holds out_probs/, out_blend_pv2/, ... (downloaded Kaggle outputs)
def files(pattern): return {os.path.basename(f): f for f in glob.glob(pattern, recursive=True)}
M = {"new": files(f"{B}/out_pv2bigprobs/probs/*.npz"), "v2": files(f"{B}/out_probs/probs/*.npz"), "pv2": files(f"{B}/out_blend_pv2/probs_dl/**/*.npz")}
names = sorted(M["new"]); assert all(sorted(m) == names for m in M.values()) and len(names) == 170
zorg = zipfile.ZipFile(f"{B}/out_blend_pv2/zips/beetle_org.zip"); znew = zipfile.ZipFile(f"{B}/out_pv2bigprobs/beetle_own_pv2big.zip")
# weights fixed before any score
VARS = {"pv2big_w50org": {"new": 0.5, "org": 0.5}, "ens3b_org": {"new": 1 / 6, "v2": 1 / 6, "pv2": 1 / 6, "org": 0.5}}
print("variants:", {k: {m: round(w, 4) for m, w in v.items()} for k, v in VARS.items()})
preds = {k: {} for k in VARS}; ref_org = {}; ref_new = {}; agree = np.zeros(2); t0 = time.time()
for i, nm in enumerate(names):
    a = {k: np.load(M[k][nm]) for k in M}; png = nm[:-4] + ".png"
    p = {k: a[k]["own"].astype(np.float32) / 255 for k in M}; p["org"] = a["v2"]["org"].astype(np.float32) / 255
    for v, w in VARS.items(): preds[v][png] = (sum(wt * p[m] for m, wt in w.items()).argmax(0) + 1).astype(np.uint8)
    s = np.array(Image.open(znew.open(png))); agree += [((p["new"].argmax(0) + 1) == s).sum(), s.size]; ref_new[png] = s
    ref_org[png] = np.array(Image.open(zorg.open(png)))
    if i % 40 == 0: print(i + 1, "/ 170", f"{time.time() - t0:.0f}s", flush=True)
print("saved probabilities reproduce the new zip on:", round(float(agree[0] / agree[1]), 5))
def dice_vs(a, b):
    inter = np.zeros(4); sa = np.zeros(4); sb = np.zeros(4); ag = tot = 0
    for nm in a:
        x, y = a[nm], b[nm]; ag += int((x == y).sum()); tot += x.size
        for c in range(1, 5): inter[c - 1] += ((x == c) & (y == c)).sum(); sa[c - 1] += (x == c).sum(); sb[c - 1] += (y == c).sum()
    return 2 * inter / np.maximum(sa + sb, 1), ag / tot
dd, ag = dice_vs(ref_new, ref_org); print("new model alone vs organisers: per-class", np.round(dd, 3).tolist(), f"mean {dd.mean():.3f}, pixel {ag:.4f}")
OUT = f"{B}/out_pv2bigcands"; os.makedirs(OUT, exist_ok=True)
for v, d in preds.items():
    names_png = sorted(d); assert names_png == sorted(ref_org)
    with zipfile.ZipFile(f"{OUT}/beetle_{v}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for nm in names_png:
            assert set(np.unique(d[nm][::7, ::7])) <= {1, 2, 3, 4}; buf = io.BytesIO(); Image.fromarray(d[nm], mode="L").save(buf, "PNG"); z.writestr(nm, buf.getvalue())
    zz = zipfile.ZipFile(f"{OUT}/beetle_{v}.zip"); assert sorted(zz.namelist()) == names_png
    bad = [nm for nm in names_png if Image.open(zz.open(nm)).mode != "L" or Image.open(zz.open(nm)).size != Image.open(zorg.open(nm)).size]
    dd, ag = dice_vs(d, ref_org); share = np.bincount(np.concatenate([d[nm].ravel()[::13] for nm in names_png]), minlength=5)[1:]; share = share / share.sum()
    print(f"{v:14s}: {len(names_png)} files, validator problems {len(bad)} | pixel share {np.round(share, 4).tolist()} | agreement with organisers mean {dd.mean():.3f}, pixel {ag:.4f} | {os.path.getsize(f'{OUT}/beetle_{v}.zip') / 2**20:.1f} MB")
