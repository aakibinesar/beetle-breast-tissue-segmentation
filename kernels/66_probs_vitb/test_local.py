"""CPU test of the ViT-B probability notebook: random ViT-B model + saved checkpoint, 2 fake ROIs; checks the zip, the saved probabilities (uint8, shape [4,H,W], argmax == the zip labels up to rounding). usage: python test_local.py"""
import os, sys, shutil, subprocess, tempfile, zipfile
import numpy as np, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); TMP = tempfile.mkdtemp(prefix="vitbprobs_")
os.makedirs(f"{TMP}/kernels/66"); os.makedirs(f"{TMP}/src")
for f in ("build.sh", "driver.py"): shutil.copy(f"{HERE}/{f}", f"{TMP}/kernels/66/{f}")
shutil.copy(f"{HERE}/../../src/beetle_seg.py", f"{TMP}/src/beetle_seg.py"); K = f"{TMP}/kernels/66"; IN = f"{TMP}/in"
sys.path.insert(0, f"{TMP}/src"); import beetle_seg as M
os.makedirs(f"{IN}/fetch/rois"); os.makedirs(f"{IN}/fetch/model"); os.makedirs(f"{IN}/train")
rng = np.random.default_rng(0); sizes = {"roi_a.png": (600, 700), "roi_b.png": (512, 512)}
for n, (h, w) in sizes.items(): Image.fromarray((rng.random((h, w, 3)) * 255).astype(np.uint8)).save(f"{IN}/fetch/rois/{n}")
torch.manual_seed(0); m = M.ViTSeg(M.build_phikon(None)).eval()
with torch.no_grad(): m.out.weight.mul_(80); m.out.bias.normal_(0, 4)          # a random model gives near-uniform probabilities; sharpen them so that uint8 rounding cannot flip the arg-max (trained models are confident)
torch.save(m.state_dict(), f"{IN}/train/seg_best_t.pt")
out = f"{TMP}/out"; os.makedirs(out); env = dict(os.environ, BEETLE_DEV="cpu", BEETLE_IN=IN, BEETLE_OUT=out, BEETLE_N_ROIS="2")
subprocess.run(["sh", "build.sh", "t", "beetle-32-train-t", "1"], cwd=K, env=env, check=True, capture_output=True)
p = subprocess.run([sys.executable, "probs_vitb.py"], cwd=K, env=env, capture_output=True, text=True)
if p.returncode != 0: print(p.stdout[-2500:], p.stderr[-2500:]); raise SystemExit("run failed")
z = zipfile.ZipFile(f"{out}/beetle_own_t.zip"); assert sorted(z.namelist()) == sorted(sizes)
for n, (h, w) in sizes.items():
    lab = np.array(Image.open(z.open(n))); d = np.load(f"{out}/probs/{n[:-4]}.npz")["own"]
    assert lab.shape == (h, w) and d.shape == (4, h, w) and d.dtype == np.uint8, (n, lab.shape, d.shape, d.dtype)
    agree = ((d.argmax(0) + 1) == lab).mean(); assert agree > 0.995, agree
print("ViT-B probability notebook OK: zip + probs consistent (argmax of the saved probabilities == labels on >99.5% of pixels)")
shutil.rmtree(TMP, ignore_errors=True)
