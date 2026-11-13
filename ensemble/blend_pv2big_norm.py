import os, glob, zipfile, io, time
import numpy as np
from PIL import Image

B = os.environ.get("BEETLE_DIR", ".")   # folder that holds the downloaded Kaggle outputs (out_pv2bignorm_probs/, out_probs/, ...)
def files(pattern): return {os.path.basename(f): f for f in glob.glob(pattern, recursive=True)}
NEW = files(f"{B}/out_pv2bignorm_probs/**/*.npz"); NEW = {k: v for k, v in NEW.items() if not k.startswith("check_raw")}
ORG = files(f"{B}/out_probs/probs/*.npz")                       # the saved probabilities of the organisers' nnU-Net sit in key 'org' of these files
names = sorted(NEW); assert names == sorted(ORG) and len(names) == 170, (len(NEW), len(ORG))
zorg = zipfile.ZipFile(f"{B}/out_blend_pv2/zips/beetle_org.zip"); znew = zipfile.ZipFile(f"{B}/out_pv2bignorm_zip/beetle_own_pv2big-norm.zip")
W = {"new": 0.5, "org": 0.5}                                     # fixed in advance: our model carries half of the weight
print("weights:", W)
preds, ref_org, ref_new, agree = {}, {}, {}, np.zeros(2); t0 = time.time()
for i, nm in enumerate(names):
    png = nm[:-4] + ".png"; pn = np.load(NEW[nm])["own"].astype(np.float32) / 255; po = np.load(ORG[nm])["org"].astype(np.float32) / 255
    preds[png] = ((W["new"] * pn + W["org"] * po).argmax(0) + 1).astype(np.uint8)
    s = np.array(Image.open(znew.open(png))); agree += [((pn.argmax(0) + 1) == s).sum(), s.size]; ref_new[png] = s; ref_org[png] = np.array(Image.open(zorg.open(png)))
    if i % 40 == 0: print(i + 1, "/ 170", f"{time.time() - t0:.0f}s", flush=True)
print("saved probabilities reproduce the colour-matched zip on: %.5f" % (agree[0] / agree[1]))
# the organisers' probabilities reproduce their own zip too (argmax)
chk = np.mean([((np.load(ORG[nm])["org"].argmax(0) + 1) == ref_org[nm[:-4] + ".png"]).mean() for nm in names[::17]]); print("organisers' saved probabilities reproduce their zip on (10 ROIs): %.5f" % chk)
OUT = f"{B}/out_pv2bignorm_blend"; os.makedirs(OUT, exist_ok=True); path = f"{OUT}/beetle_pv2big_norm_w50org.zip"
with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
    for nm in sorted(preds):
        assert set(np.unique(preds[nm][::7, ::7])) <= {1, 2, 3, 4}; buf = io.BytesIO(); Image.fromarray(preds[nm], mode="L").save(buf, "PNG"); z.writestr(nm, buf.getvalue())
zz = zipfile.ZipFile(path); assert zz.testzip() is None and sorted(zz.namelist()) == sorted(preds)
bad = [nm for nm in preds if Image.open(zz.open(nm)).mode != "L" or Image.open(zz.open(nm)).size != Image.open(zorg.open(nm)).size]
def dice_vs(a, b):
    inter = np.zeros(4); sa = np.zeros(4); sb = np.zeros(4); ag = tot = 0
    for nm in a:
        x, y = a[nm], b[nm]; ag += int((x == y).sum()); tot += x.size
        for c in range(1, 5): inter[c - 1] += ((x == c) & (y == c)).sum(); sa[c - 1] += (x == c).sum(); sb[c - 1] += (y == c).sum()
    return 2 * inter / np.maximum(sa + sb, 1), ag / tot
dd, ag = dice_vs(preds, ref_org); dn, an = dice_vs(preds, ref_new)
share = np.bincount(np.concatenate([preds[nm].ravel()[::13] for nm in preds]), minlength=5)[1:]; share = share / share.sum()
print(f"blend: {len(preds)} files, validator problems {len(bad)} | pixel share {np.round(share, 4).tolist()} | agreement with organisers mean {dd.mean():.3f}, pixel {ag:.4f} | with the colour-matched model pixel {an:.4f} | {os.path.getsize(path) / 2**20:.1f} MB")
