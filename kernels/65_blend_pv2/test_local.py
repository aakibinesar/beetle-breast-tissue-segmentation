"""Local dry run of the Phikon-v2 blend notebook in its stub mode (BLEND_TEST=1: random probabilities instead of the two models; the model code itself is tested in 62_train_pv2 / 64_infer_pv2): checks that the generated script
parses, writes own / org / blend zips that pass the validator, and saves the probabilities. usage: python test_local.py"""
import os, sys, shutil, subprocess, tempfile, zipfile, ast
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); TMP = tempfile.mkdtemp(prefix="pv2blend_")
os.makedirs(f"{TMP}/kernels/65_blend_pv2"); os.makedirs(f"{TMP}/kernels/50_blend"); os.makedirs(f"{TMP}/src")
shutil.copy(f"{HERE}/build.sh", f"{TMP}/kernels/65_blend_pv2/build.sh"); shutil.copy(f"{HERE}/../50_blend/driver.py", f"{TMP}/kernels/50_blend/driver.py"); shutil.copy(f"{HERE}/../../src/beetle_seg_pv2.py", f"{TMP}/src/beetle_seg_pv2.py")
K = f"{TMP}/kernels/65_blend_pv2"; roi = f"{TMP}/roi_root"; os.makedirs(f"{roi}/rois"); rng = np.random.default_rng(0)
sizes = {"a.png": (300, 420), "b.png": (256, 256)}
for n, (h, w) in sizes.items(): Image.fromarray((rng.random((h, w, 3)) * 255).astype(np.uint8)).save(f"{roi}/rois/{n}")
env = dict(os.environ, BLEND_TEST="1", BEETLE_ROIROOT=roi, BEETLE_OUT=f"{TMP}/out"); os.makedirs(f"{TMP}/out")
subprocess.run(["sh", "build.sh", "t", "beetle-62-train-t", "4", "0.5,0.7"], cwd=K, env=env, check=True, capture_output=True)
ast.parse(open(f"{K}/blend_pv2.py", encoding="utf-8").read())
meta = open(f"{K}/kernel-metadata.json").read(); assert "beetle-60-phikonv2-fetch" in meta and "beetle-62-train-t" in meta and "beetle-20-fetch" in meta
p = subprocess.run([sys.executable, "blend_pv2.py"], cwd=K, env=env, capture_output=True, text=True)
if p.returncode != 0: print(p.stdout[-2500:], p.stderr[-2500:]); raise SystemExit("blend stub run failed")
for v in ("own", "org", "blend_w50", "blend_w70"):
    z = zipfile.ZipFile(f"{TMP}/out/beetle_{v}.zip"); assert sorted(z.namelist()) == sorted(sizes), (v, z.namelist())
    for n, (h, w) in sizes.items(): a = np.array(Image.open(z.open(n))); assert a.shape == (h, w) and set(np.unique(a)) <= {1, 2, 3, 4}
assert len(os.listdir(f"{TMP}/out/probs")) == 2
print("blend notebook dry run OK: zips own/org/blend_w50/blend_w70 valid, probs saved; metadata sources:", [s for s in ("beetle-20-fetch", "beetle-60-phikonv2-fetch", "beetle-62-train-t") if s in meta])
print([l for l in p.stdout.split(chr(10)) if l.startswith("  ")][:4]); shutil.rmtree(TMP, ignore_errors=True)
