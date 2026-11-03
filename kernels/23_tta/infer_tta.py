# BEETLE baseline reproduction: the organisers' released nnU-Net 5-fold ensemble (model.zip, Zenodo 16812932) run on the 170 evaluation ROIs.
# Not our model. Follows code/inference.py of github.com/DIAGNijmegen/beetle; differences: (1) patched copy of dataset.json (label 0 named "background") because
# the installed nnU-Net fork rejects "unannotated"; (2) argmax over the four tissue channels only, so no pixel is ever 0 (platform wants 1-4).
import os, sys, io, json, time, glob, shutil, subprocess, importlib, traceback, zipfile
import numpy as np
TTA_MODE = "flips4"     # flips4: {identity, flip-W, flip-H, flip-H+W}; dihedral8 adds the four 90-degree rotations

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True); return r.returncode, (r.stdout + r.stderr)[-600:]

for cmd in ['pip install -q --no-deps "git+https://github.com/DIAGNijmegen/nnUNet-for-pathology.git@92cf757fac30a0e73a7a5f7caa5cd58bd08a4660#egg=nnunetv2"',
            'pip install -q --no-deps "git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@1e4c6ca939c5e372a0b626739c24443f31bff505"',
            'pip install -q acvl-utils batchgenerators dynamic-network-architectures connected-components-3d']:
    print("$", cmd[:100], sh(cmd)[0], flush=True)
for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"): os.environ.setdefault(k, f"/tmp/{k}")
PIPNAME = {"cc3d": "connected-components-3d", "skimage": "scikit-image", "cv2": "opencv-python-headless", "sklearn": "scikit-learn", "yaml": "pyyaml", "PIL": "pillow"}
for attempt in range(15):
    try:
        importlib.invalidate_caches()
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.training.nnUNetTrainer.variants.pathology.nnUNetTrainer_custom_dataloader_test import nnUNetTrainer_custom_dataloader_test
        break
    except ModuleNotFoundError as e:
        n = e.name.split(".")[0]; print("installing missing", n, flush=True); sh(f"pip install -q {PIPNAME.get(n, n.replace('_', '-'))}")
import torch
# The organisers' checkpoints contain numpy scalars; torch>=2.6 defaults to weights_only=True and refuses them. The files come from the official Zenodo record and their md5 was
# verified against it (beetle-20-fetch), the notebook is private and offline-of-secrets, so load them the way the organisers' torch 2.0.1 did.
_orig_load = torch.load
def _load(*a, **k): k["weights_only"] = False; return _orig_load(*a, **k)
torch.load = _load
from pathlib import Path
from PIL import Image
from nnunetv2.utilities.file_path_utilities import load_json
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

root = next(r for r, d, f in os.walk("/kaggle/input") if "rois" in d and "model" in d)
mdir = glob.glob(f"{root}/model/model/*")[0]
mp = Path("/tmp/beetle_model"); mp.mkdir(exist_ok=True)
ds = json.load(open(f"{mdir}/dataset.json")); ds["labels"] = {("background" if v == 0 else k): v for k, v in ds["labels"].items()}
json.dump(ds, open(mp / "dataset.json", "w")); shutil.copy(f"{mdir}/plans.json", mp / "plans.json")
for k in range(5):
    if not (mp / f"fold_{k}").exists(): os.symlink(f"{mdir}/fold_{k}", mp / f"fold_{k}")

trainer = nnUNetTrainer_custom_dataloader_test(load_json(mp / "plans.json"), "2d", 0, load_json(mp / "dataset.json"))
predictor = nnUNetPredictor()
predictor.initialize_from_trained_model_folder(mp, use_folds=(0, 1, 2, 3, 4), checkpoint_name="checkpoint_best.pth")
print("predictor ready", flush=True)

OUT = Path("/kaggle/working/predictions"); OUT.mkdir(exist_ok=True, parents=True)
rois = sorted(glob.glob(f"{root}/rois/*.png")); print(len(rois), "ROIs", flush=True)
t0 = time.time(); counts = np.zeros(5, np.int64)


def run(patch):
    p = torch.tensor(np.ascontiguousarray(patch), dtype=torch.float32)
    logits = predictor.get_logits_list_from_preprocessed_data(p)
    return np.mean([trainer.label_manager.apply_inference_nonlin(l).numpy() for l in logits], 0)


# dihedral transforms: (k quarter-turns, flip along width after the turn). flips4 = k in {0,2} x flip in {0,1}
TRANSFORMS = [(k, f) for k in ((0, 2) if TTA_MODE == "flips4" else (0, 1, 2, 3)) for f in (0, 1)]
print("TTA transforms (k, flip):", TRANSFORMS, flush=True)


def fwd(x, k, f):                                   # x: (C, 1, H, W)
    x = np.rot90(x, k, axes=(2, 3)) if k else x
    return np.flip(x, axis=3) if f else x


def inv(s, k, f):                                   # s: (5, H', W') probabilities in the transformed frame -> original frame
    s = np.flip(s, axis=2) if f else s
    return np.rot90(s, -k, axes=(1, 2)) if k else s


base = {}                                            # baseline predictions (beetle-22 output) for an agreement check
bz = next((p for p in glob.glob("/kaggle/input/**/beetle_predictions.zip", recursive=True)), None)
if bz:
    with zipfile.ZipFile(bz) as z:
        for nme in z.namelist(): base[nme] = np.array(Image.open(io.BytesIO(z.read(nme))))
print("baseline predictions available:", len(base), flush=True)
agree = []; changed = np.zeros((5, 5), np.int64)

for n, path in enumerate(rois):
    img = np.array(Image.open(path).convert("RGB"))
    patch = (np.expand_dims(img, 0).astype(np.float32) / 255.0).transpose(3, 0, 1, 2)         # (C, 1, H, W)
    acc = None
    for (k, f) in TRANSFORMS:
        x = fwd(patch, k, f)
        try:
            sm = run(x)
        except torch.cuda.OutOfMemoryError:
            print("OOM -> retry fp16", flush=True); torch.cuda.empty_cache()
            with torch.autocast("cuda", dtype=torch.float16): sm = run(x)
        sm = inv(np.squeeze(sm).astype(np.float32), k, f)
        acc = sm if acc is None else acc + sm
    sm = acc / len(TRANSFORMS)
    pred = (np.argmax(sm[1:5], 0) + 1).astype(np.uint8); assert pred.shape == img.shape[:2]
    nm = os.path.basename(path)
    if nm in base:
        agree.append(float((pred == base[nm]).mean()))
        for a_ in range(1, 5):
            for b_ in range(1, 5): changed[a_, b_] += int(((base[nm] == a_) & (pred == b_)).sum())
    counts += np.bincount(pred.ravel(), minlength=5)
    Image.fromarray(pred, mode="L").save(OUT / nm)
    if n % 10 == 0: print(f"{n + 1}/{len(rois)} | {nm} | {time.time() - t0:.0f}s | mean agreement with baseline {np.mean(agree) if agree else float('nan'):.4f}", flush=True)
print(f"TTA inference done in {time.time() - t0:.0f}s | pixel share other/non-inv/invasive/necrosis:", np.round(counts[1:] / counts.sum(), 4).tolist(), flush=True)
if agree:
    print("agreement with baseline: mean %.4f min %.4f" % (np.mean(agree), np.min(agree)))
    print("baseline->TTA pixel transition matrix (rows baseline 1-4, cols TTA 1-4):", flush=True); print(changed[1:, 1:], flush=True)

# ---- validator + zip ----
bad = []
names = sorted(os.path.basename(p) for p in rois)
assert sorted(os.listdir(OUT)) == names, "file list differs from the ROI list"
for p in rois:
    a = Image.open(p); b = Image.open(OUT / os.path.basename(p)); arr = np.array(b)
    if b.mode != "L" or b.size != a.size or not set(np.unique(arr)).issubset({1, 2, 3, 4}): bad.append((os.path.basename(p), b.mode, b.size, a.size, np.unique(arr).tolist()))
print("validator: files", len(names), "| problems", len(bad), bad[:3], flush=True); assert not bad
with zipfile.ZipFile("/kaggle/working/beetle_predictions_tta.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for nme in names: z.write(OUT / nme, nme)
print("zip written:", os.path.getsize("/kaggle/working/beetle_predictions_tta.zip") / 2**20, "MB")
shutil.rmtree(OUT)
