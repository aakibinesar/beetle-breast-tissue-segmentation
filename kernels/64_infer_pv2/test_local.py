"""CPU test of the Phikon-v2 inference notebook with the tiny stand-in backbone (env BEETLE_TINY=1): builds the notebook script with build.sh, runs it on 3 fake ROIs of odd sizes with a saved tiny checkpoint,
checks the validator, the zip, that a 512x512 ROI (one window, no TTA) equals the direct argmax of the model, and the SMOKE mode (random decoder, no checkpoint). usage: python test_local.py"""
import os, sys, shutil, subprocess, tempfile, zipfile
import numpy as np, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); TMP = tempfile.mkdtemp(prefix="pv2inf_")
os.makedirs(f"{TMP}/kernels/64"); os.makedirs(f"{TMP}/src")
for f in ("build.sh", "driver.py"): shutil.copy(f"{HERE}/{f}", f"{TMP}/kernels/64/{f}")
shutil.copy(f"{HERE}/../../src/beetle_seg_pv2.py", f"{TMP}/src/beetle_seg_pv2.py")
K = f"{TMP}/kernels/64"; IN = f"{TMP}/in"
os.environ["BEETLE_TINY"] = "1"; sys.path.insert(0, f"{TMP}/src"); import beetle_seg_pv2 as M
os.makedirs(f"{IN}/fetch/rois"); os.makedirs(f"{IN}/fetch/model"); os.makedirs(f"{IN}/train")
rng = np.random.default_rng(0); sizes = {"roi_a.png": (700, 900), "roi_b.png": (512, 512), "roi_c.png": (300, 640)}
for n, (h, w) in sizes.items(): Image.fromarray((rng.random((h, w, 3)) * 255).astype(np.uint8)).save(f"{IN}/fetch/rois/{n}")
torch.manual_seed(0); m = M.ViTSeg(M.build_phikon(None)).eval(); torch.save(m.state_dict(), f"{IN}/train/seg_best_t.pt")
env = dict(os.environ, BEETLE_TINY="1", BEETLE_DEV="cpu", BEETLE_IN=IN, BEETLE_N_ROIS="3")

def run(name, nt, smoke=0):
    out = f"{TMP}/out_{name}"; os.makedirs(out, exist_ok=True); e = dict(env, BEETLE_OUT=out, SMOKE=str(smoke), SMOKE_CPU="1" if smoke else "0")
    subprocess.run(["sh", "build.sh", name, "beetle-62-train-t", str(nt)], cwd=K, env=e, check=True, capture_output=True)
    p = subprocess.run([sys.executable, "infer_seg_pv2.py"], cwd=K, env=e, capture_output=True, text=True)
    if p.returncode != 0: print(p.stdout[-2500:], p.stderr[-2500:]); raise SystemExit(f"run {name} failed")
    return p.stdout, out

o1, out1 = run("t1", 1)
z = zipfile.ZipFile(f"{out1}/beetle_own_t1.zip"); assert sorted(z.namelist()) == sorted(sizes), z.namelist()
for n, (h, w) in sizes.items():
    a = np.array(Image.open(z.open(n))); assert a.shape == (h, w) and set(np.unique(a)) <= {1, 2, 3, 4}, (n, a.shape)
a = np.array(Image.open(z.open("roi_b.png")))
with torch.no_grad(): direct = (m(torch.from_numpy(np.array(Image.open(f"{IN}/fetch/rois/roi_b.png"))).permute(2, 0, 1)[None].float() / 255).argmax(1)[0] + 1).numpy()
assert (a == direct).all(), "stitched 512x512 prediction differs from the direct argmax"
print("1. full mode OK: zip with", len(z.namelist()), "files, shapes", {n: tuple(np.array(Image.open(z.open(n))).shape) for n in sizes}, "| 512x512 ROI == direct argmax")
o4, out4 = run("t4", 4); z4 = zipfile.ZipFile(f"{out4}/beetle_own_t4.zip"); assert sorted(z4.namelist()) == sorted(sizes)
print("2. 4-way TTA OK (", [l for l in o4.split(chr(10)) if l.startswith('done in')][0][:60], ")")
os_, outs = run("smoke", 4, smoke=2)
assert "SMOKE OK" in os_ and not os.path.exists(f"{outs}/beetle_own_smoke.zip")
print("3. smoke mode OK:", [l for l in os_.split(chr(10)) if l.startswith("SMOKE OK")][0][:140]); print([l for l in open(f"{K}/kernel-metadata.json").read().split(chr(10)) if "enable_gpu" in l][0].strip())
print("ALL LOCAL INFERENCE TESTS PASSED"); shutil.rmtree(TMP, ignore_errors=True)
