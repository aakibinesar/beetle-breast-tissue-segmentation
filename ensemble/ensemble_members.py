import os, glob, zipfile, io, sys, time, json
import numpy as np
from PIL import Image

B = os.environ.get("BEETLE_DIR", ".")   # folder that holds out_probs/, out_blend_pv2/, ... (downloaded Kaggle outputs)
def files(pattern): return {os.path.basename(f): f for f in glob.glob(pattern, recursive=True)}
MEMBERS = {"v2": files(f"{B}/out_probs/probs/*.npz"), "pv2": files(f"{B}/out_blend_pv2/probs_dl/**/*.npz"), "pv2abl": files(f"{B}/out_blend_pv2abl/probs_dl/**/*.npz")}
if os.path.isdir(f"{B}/out_v3probs"): MEMBERS["v3"] = files(f"{B}/out_v3probs/**/*.npz")
names = sorted(MEMBERS["v2"]); assert all(sorted(m) == names for m in MEMBERS.values()) and len(names) == 170, {k: len(v) for k, v in MEMBERS.items()}
keys = list(MEMBERS); print("members:", keys)
ZIPS = {"v2": f"{B}/out_v2/beetle_own_v2.zip", "pv2": f"{B}/out_blend_pv2/zips/beetle_own.zip", "pv2abl": f"{B}/out_blend_pv2abl/zips/beetle_own.zip", "v3": f"{B}/out_v3inf/beetle_own_v3.zip"}
zorg = zipfile.ZipFile(f"{B}/out_blend_pv2/zips/beetle_org.zip"); zs = {k: zipfile.ZipFile(ZIPS[k]) for k in keys}
n = len(keys); VARS = {f"ens{n}_own": {k: 1 / n for k in keys}, f"ens{n}_org": dict({k: 0.5 / n for k in keys}, org=0.5)}        # weights fixed before any score; in the 'org' variant our models together are half
print("variants:", {k: {m: round(w, 4) for m, w in v.items()} for k, v in VARS.items()})
OUT = f"{B}/out_ens"; os.makedirs(OUT, exist_ok=True); preds = {k: {} for k in VARS}; agree = {k: np.zeros(2) for k in keys}; ref_org = {}; t0 = time.time()
for i, nm in enumerate(names):
    arrs = {k: np.load(MEMBERS[k][nm]) for k in keys}; png = nm[:-4] + ".png"
    p = {k: arrs[k]["own"].astype(np.float32) / 255 for k in keys}; p["org"] = arrs["v2"]["org"].astype(np.float32) / 255
    for v, w in VARS.items(): preds[v][png] = (sum(wt * p[m] for m, wt in w.items()).argmax(0) + 1).astype(np.uint8)
    for k in keys:
        s = np.array(Image.open(zs[k].open(png))); agree[k] += [((p[k].argmax(0) + 1) == s).sum(), s.size]
    ref_org[png] = np.array(Image.open(zorg.open(png)))
    if i % 40 == 0: print(i + 1, "/ 170", f"{time.time() - t0:.0f}s", flush=True)
print("saved probabilities reproduce the submitted zips on:", {k: round(float(a[0] / a[1]), 5) for k, a in agree.items()})
def dice_vs(a, b):
    inter = np.zeros(4); sa = np.zeros(4); sb = np.zeros(4); ag = tot = 0
    for nm in a:
        x, y = a[nm], b[nm]; ag += int((x == y).sum()); tot += x.size
        for c in range(1, 5): inter[c - 1] += ((x == c) & (y == c)).sum(); sa[c - 1] += (x == c).sum(); sb[c - 1] += (y == c).sum()
    return 2 * inter / np.maximum(sa + sb, 1), ag / tot
for v, d in preds.items():
    names_png = sorted(d); assert names_png == sorted(ref_org)
    with zipfile.ZipFile(f"{OUT}/beetle_{v}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for nm in names_png:
            assert set(np.unique(d[nm][::7, ::7])) <= {1, 2, 3, 4}; buf = io.BytesIO(); Image.fromarray(d[nm], mode="L").save(buf, "PNG"); z.writestr(nm, buf.getvalue())
    zz = zipfile.ZipFile(f"{OUT}/beetle_{v}.zip"); assert sorted(zz.namelist()) == names_png
    bad = [nm for nm in names_png if Image.open(zz.open(nm)).mode != "L" or Image.open(zz.open(nm)).size != Image.open(zorg.open(nm)).size]
    dd, ag = dice_vs(d, ref_org); share = np.bincount(np.concatenate([d[nm].ravel()[::13] for nm in names_png]), minlength=5)[1:]; share = share / share.sum()
    print(f"{v:10s}: {len(names_png)} files, validator problems {len(bad)} | pixel share {np.round(share, 4).tolist()} | agreement with organisers per-class {np.round(dd, 3).tolist()} mean {dd.mean():.3f}, pixel {ag:.4f} | {os.path.getsize(f'{OUT}/beetle_{v}.zip') / 2**20:.1f} MB")
