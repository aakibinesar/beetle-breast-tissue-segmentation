N_TTA, TAG, WEIGHTS = 4, "v3", (0.3,0.5,)
import os
"""ViT (torchvision layout, e.g. Phikon ViT-B/16) + light CNN skip branch + conv decoder for 4-class tissue segmentation (classes 1-4 -> logits 0-3)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ViTSeg(nn.Module):
    def __init__(s, vit, feat_layers=(3, 6, 9, 12), n_cls=4, dec=128):
        super().__init__()
        s.vit = vit; s.layers = tuple(feat_layers); d = vit.hidden_dim
        s.proj = nn.ModuleList([nn.Conv2d(d, dec, 1) for _ in s.layers])
        s.fuse = conv_bn(dec * len(s.layers), 256)
        s.c1 = nn.Sequential(conv_bn(3, 32), conv_bn(32, 32))                    # 1/1 skip (sharp boundaries; the ViT grid is only 1/16)
        s.c2 = nn.Sequential(conv_bn(32, 64, 3, 2), conv_bn(64, 64))            # 1/2
        s.c4 = nn.Sequential(conv_bn(64, 96, 3, 2), conv_bn(96, 96))            # 1/4
        s.c8 = nn.Sequential(conv_bn(96, 128, 3, 2), conv_bn(128, 128))         # 1/8
        s.d8 = conv_bn(256 + 128, 192); s.d4 = conv_bn(192 + 96, 128); s.d2 = conv_bn(128 + 64, 96); s.d1 = conv_bn(96 + 32, 64)
        s.out = nn.Conv2d(64, n_cls, 1)
        s.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        s.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def backbone_trainable(s):
        return any(p.requires_grad for p in s.vit.parameters())

    def tokens(s, x):                                                            # x: normalised [B,3,H,W], H and W multiples of 16
        v = s.vit; B, _, H, W = x.shape; h, w = H // 16, W // 16
        t = v.conv_proj(x).flatten(2).transpose(1, 2)
        t = torch.cat([v.class_token.expand(B, -1, -1), t], 1)
        pe = v.encoder.pos_embedding; g = int(math.sqrt(pe.shape[1] - 1))
        pg = pe[:, 1:].reshape(1, g, g, -1).permute(0, 3, 1, 2)
        pg = F.interpolate(pg.float(), size=(h, w), mode="bicubic", align_corners=False).permute(0, 2, 3, 1).reshape(1, h * w, -1)
        t = v.encoder.dropout(t + torch.cat([pe[:, :1], pg.to(pe.dtype)], 1))
        feats = []
        for i, blk in enumerate(v.encoder.layers, 1):
            t = blk(t)
            if i in s.layers: feats.append(v.encoder.ln(t)[:, 1:].transpose(1, 2).reshape(B, -1, h, w))
        return feats

    def forward(s, x01):                                                         # x01: float [B,3,H,W] in 0..1
        x = (x01 - s.mean) / s.std
        f1 = s.c1(x); f2 = s.c2(f1); f4 = s.c4(f2); f8 = s.c8(f4)
        if s.backbone_trainable(): feats = s.tokens(x)
        else:
            with torch.no_grad(): feats = s.tokens(x)
        p = s.fuse(torch.cat([pr(f.float() if f.dtype != x.dtype else f) for pr, f in zip(s.proj, feats)], 1))
        up = lambda a, ref: F.interpolate(a, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        y = s.d8(torch.cat([up(p, f8), f8], 1)); y = s.d4(torch.cat([up(y, f4), f4], 1)); y = s.d2(torch.cat([up(y, f2), f2], 1)); y = s.d1(torch.cat([up(y, f1), f1], 1))
        return s.out(y)


def build_phikon(weights_path):
    from torchvision.models import VisionTransformer
    vit = VisionTransformer(image_size=224, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=4 * 768, num_classes=0)
    vit.heads = nn.Identity()
    if weights_path: vit.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return vit


def set_trainable(model, last_blocks):
    for p in model.vit.parameters(): p.requires_grad = False
    if last_blocks > 0:
        for blk in list(model.vit.encoder.layers)[-last_blocks:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in model.vit.encoder.ln.parameters(): p.requires_grad = True

# ---------------- BEETLE blend: the organisers' released nnU-Net 5-fold ensemble + OUR model, softmax-averaged, on the 170 evaluation ROIs ----------------
# Writes one zip per variant: own (ours alone), org (organisers' model alone = the baseline reproduction, sanity check), blend_wXX (XX = percent weight of OUR model).
# Everything the blend needs is also saved as uint8 probabilities (probs/), so other weightings can be made later on a CPU without re-running the GPU.
import os, sys, glob, io, json, time, zipfile, shutil, subprocess, importlib
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
TEST = bool(os.environ.get("BLEND_TEST"))                 # local dry run: stub models, tiny ROIs
dev = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
def stamp(m): print(f"[t+{time.time() - T0:6.0f}s] {m}", flush=True)
def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True); return r.returncode, (r.stdout + r.stderr)[-600:]

roi_root = os.environ.get("BEETLE_ROIROOT") or next(r for r, d, f in os.walk("/kaggle/input") if "rois" in d and "model" in d)
roi_dir = f"{roi_root}/rois"
OUTW = os.environ.get("BEETLE_OUT", "/kaggle/working")

if not TEST:
    ck = next(os.path.join(r, f) for r, d, fs in os.walk("/kaggle/input") for f in fs if f.startswith("seg_best_") and f.endswith(".pt"))
    stamp(f"ROIs: {roi_dir} | own checkpoint: {ck}")
    # ---- our model ----
    model = ViTSeg(build_phikon(None)); model.load_state_dict(torch.load(ck, map_location="cpu"), strict=True); model = model.to(dev).eval()
    # ---- the organisers' model (same setup as the baseline reproduction notebook beetle-22-infer) ----
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
    _orig_load = torch.load
    def _load(*a, **k): k["weights_only"] = False; return _orig_load(*a, **k)          # the organisers' checkpoints hold numpy scalars (official files, md5 verified in beetle-20-fetch)
    torch.load = _load
    from pathlib import Path
    from nnunetv2.utilities.file_path_utilities import load_json
    mdir = glob.glob(f"{roi_root}/model/model/*")[0]
    mp = Path("/tmp/beetle_model"); mp.mkdir(exist_ok=True)
    ds = json.load(open(f"{mdir}/dataset.json")); ds["labels"] = {("background" if v == 0 else k): v for k, v in ds["labels"].items()}
    json.dump(ds, open(mp / "dataset.json", "w")); shutil.copy(f"{mdir}/plans.json", mp / "plans.json")
    for k in range(5):
        if not (mp / f"fold_{k}").exists(): os.symlink(f"{mdir}/fold_{k}", mp / f"fold_{k}")
    trainer = nnUNetTrainer_custom_dataloader_test(load_json(mp / "plans.json"), "2d", 0, load_json(mp / "dataset.json"))
    predictor = nnUNetPredictor(); predictor.initialize_from_trained_model_folder(mp, use_folds=(0, 1, 2, 3, 4), checkpoint_name="checkpoint_best.pth")
    stamp("both models ready")

    P, STRIDE = 512, 256
    win = torch.hann_window(P, periodic=False).clamp(min=1e-3); WIN2D = (win[:, None] * win[None, :]).to(dev)
    TRANSFORMS = [(0, 0), (0, 1), (2, 0), (2, 1)][:N_TTA]

    def own_probs(img):                                   # img uint8 [H,W,3] -> softmax over our 4 classes [4,H,W] float32 (numpy)
        H, W = img.shape[:2]; ph, pw = max(P, H), max(P, W)
        ys = list(range(0, ph - P + 1, STRIDE)); xs = list(range(0, pw - P + 1, STRIDE))
        if ys[-1] != ph - P: ys.append(ph - P)
        if xs[-1] != pw - P: xs.append(pw - P)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255)
        x = F.pad(x[None], (0, pw - W, 0, ph - H), mode="reflect")[0].to(dev) if (ph > H or pw > W) else x.to(dev)
        acc = torch.zeros(4, ph, pw, device=dev); wsum = torch.zeros(1, ph, pw, device=dev)
        coords = [(y, xx) for y in ys for xx in xs]
        for (k, f) in TRANSFORMS:
            for i in range(0, len(coords), 8):
                cs = coords[i:i + 8]; b = torch.stack([x[:, y:y + P, xx:xx + P] for (y, xx) in cs])
                if k: b = torch.rot90(b, k, (2, 3))
                if f: b = b.flip(3)
                with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")): lo = model(b)
                p = torch.softmax(lo.float(), 1)
                if f: p = p.flip(3)
                if k: p = torch.rot90(p, -k, (2, 3))
                for j, (y, xx) in enumerate(cs): acc[:, y:y + P, xx:xx + P] += p[j] * WIN2D; wsum[:, y:y + P, xx:xx + P] += WIN2D
        return (acc / wsum.clamp(min=1e-6))[:, :H, :W].cpu().numpy()

    def _org_run(patch):
        p = torch.tensor(patch, dtype=torch.float32)
        logits = predictor.get_logits_list_from_preprocessed_data(p)
        return np.mean([trainer.label_manager.apply_inference_nonlin(l).numpy() for l in logits], 0)

    def org_probs(img):                                   # the organisers' 5-fold softmax (5 channels: 0 = unannotated) -> channels 1-4 renormalised to sum 1 [4,H,W]
        patch = (np.expand_dims(img, 0).astype(np.float32) / 255.0).transpose(3, 0, 1, 2)
        try: sm = _org_run(patch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            with torch.autocast("cuda", dtype=torch.float16): sm = _org_run(patch)
        sm = np.squeeze(sm)[1:5].astype(np.float32)
        return sm / np.clip(sm.sum(0, keepdims=True), 1e-6, None)
else:
    rng = np.random.default_rng(0)
    def own_probs(img): q = rng.random((4,) + img.shape[:2]).astype(np.float32) ** 3; return q / q.sum(0, keepdims=True)
    def org_probs(img): q = rng.random((4,) + img.shape[:2]).astype(np.float32) ** 3; return q / q.sum(0, keepdims=True)

VARIANTS = ["own", "org"] + [f"blend_w{int(round(w * 100)):02d}" for w in WEIGHTS]
OUT = f"{OUTW}/predictions"; PROBS = f"{OUTW}/probs"; os.makedirs(PROBS, exist_ok=True)
for v in VARIANTS: os.makedirs(f"{OUT}/{v}", exist_ok=True)
rois = sorted(glob.glob(f"{roi_dir}/*.png")); stamp(f"{len(rois)} ROIs, variants {VARIANTS}")
if not TEST: assert len(rois) == 170, len(rois)
counts = {v: np.zeros(5, np.int64) for v in VARIANTS}; agree = {v: [0, 0] for v in VARIANTS}          # pixels agreeing with the organisers' model alone, total pixels
t0 = time.time()
for n, path in enumerate(rois):
    name = os.path.basename(path); img = np.array(Image.open(path).convert("RGB"))
    po = own_probs(img); pg = org_probs(img)
    assert po.shape == pg.shape == (4,) + img.shape[:2], (po.shape, pg.shape, img.shape)
    preds = {}
    for v in VARIANTS:
        if v == "own": p = po
        elif v == "org": p = pg
        else: w = int(v[-2:]) / 100.0; p = w * po + (1 - w) * pg
        preds[v] = (p.argmax(0) + 1).astype(np.uint8)
        Image.fromarray(preds[v], mode="L").save(f"{OUT}/{v}/{name}"); counts[v] += np.bincount(preds[v].ravel(), minlength=5)
    for v in VARIANTS: agree[v][0] += int((preds[v] == preds["org"]).sum()); agree[v][1] += preds[v].size
    np.savez_compressed(f"{PROBS}/{name[:-4]}.npz", own=np.round(po * 255).astype(np.uint8), org=np.round(pg * 255).astype(np.uint8))
    if n % 10 == 0: stamp(f"{n + 1}/{len(rois)} | {name} {img.shape[:2]} | {time.time() - t0:.0f}s" + (f" | peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if dev == "cuda" else ""))
print("pixel share other/non-inv/invasive/necrosis and agreement with the organisers' model alone:", flush=True)
for v in VARIANTS: print(f"  {v:11s} {np.round(counts[v][1:] / counts[v].sum(), 4).tolist()} | agreement with org {agree[v][0] / agree[v][1]:.4f}", flush=True)

names = sorted(os.path.basename(p) for p in rois)
for v in VARIANTS:                                        # validator + zip per variant
    assert sorted(os.listdir(f"{OUT}/{v}")) == names, v; bad = []
    for p in rois:
        a = Image.open(p); b = Image.open(f"{OUT}/{v}/{os.path.basename(p)}"); arr = np.array(b)
        if b.mode != "L" or b.size != a.size or not set(np.unique(arr)).issubset({1, 2, 3, 4}): bad.append(os.path.basename(p))
    print(f"validator {v}: files {len(names)} | problems {len(bad)} {bad[:3]}", flush=True); assert not bad
    with zipfile.ZipFile(f"{OUTW}/beetle_{v}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for nme in names: z.write(f"{OUT}/{v}/{nme}", nme)
shutil.rmtree(OUT)
stamp("DONE: " + ", ".join(f"beetle_{v}.zip {os.path.getsize(f'{OUTW}/beetle_{v}.zip') / 2**20:.1f} MB" for v in VARIANTS) + f" | probs/ {sum(os.path.getsize(f'{PROBS}/{f}') for f in os.listdir(PROBS)) / 2**30:.2f} GB")
sys.stdout.flush(); os._exit(0)
