N_TTA, TAG = 4, "v3"
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

# ---------------- BEETLE: run our own segmentation model on the 170 evaluation ROIs (sliding windows with overlap, dihedral TTA) ----------------
import os, glob, io, json, time, zipfile, shutil
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

dev = "cuda" if torch.cuda.is_available() else "cpu"
roi_dir = next(os.path.join(r, "rois") for r, d, f in os.walk("/kaggle/input") if "rois" in d and "model" in d)
ck = next(os.path.join(r, f) for r, d, fs in os.walk("/kaggle/input") for f in fs if f.startswith("seg_best_") and f.endswith(".pt"))
print("ROIs:", roi_dir, "| checkpoint:", ck, flush=True)
if not os.environ.get("BEETLE_TEST"): model = ViTSeg(build_phikon(None)); model.load_state_dict(torch.load(ck, map_location="cpu"), strict=True); model = model.to(dev).eval()
P, STRIDE = 512, 256
win = torch.hann_window(P, periodic=False).clamp(min=1e-3); WIN2D = (win[:, None] * win[None, :]).to(dev)       # cosine taper: window centres count more than their borders
TRANSFORMS = [(0, 0), (0, 1), (2, 0), (2, 1)][:N_TTA]                                                           # (quarter turns, flip along width)


def predict_probs(img):                                   # img uint8 [H,W,3] -> softmax [4,H,W] (float32, cpu)
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
    return (acc / wsum.clamp(min=1e-6))[:, :H, :W].cpu()


OUT = "/kaggle/working/predictions"; os.makedirs(OUT, exist_ok=True)
rois = sorted(glob.glob(f"{roi_dir}/*.png")); assert len(rois) == 170, len(rois)
t0 = time.time(); counts = np.zeros(5, np.int64)
for n, p in enumerate(rois):
    img = np.array(Image.open(p).convert("RGB")); pr = predict_probs(img)
    pred = (pr.argmax(0) + 1).numpy().astype(np.uint8); assert pred.shape == img.shape[:2]
    counts += np.bincount(pred.ravel(), minlength=5)
    Image.fromarray(pred, mode="L").save(f"{OUT}/{os.path.basename(p)}")
    if n % 10 == 0: print(f"{n + 1}/170 | {os.path.basename(p)} {img.shape[:2]} | {time.time() - t0:.0f}s | peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
print(f"done in {time.time() - t0:.0f}s | pixel share other/non-inv/invasive/necrosis:", np.round(counts[1:] / counts.sum(), 4).tolist(), flush=True)
bad = []; names = sorted(os.path.basename(p) for p in rois); assert sorted(os.listdir(OUT)) == names
for p in rois:
    a = Image.open(p); b = Image.open(f"{OUT}/{os.path.basename(p)}"); arr = np.array(b)
    if b.mode != "L" or b.size != a.size or not set(np.unique(arr)).issubset({1, 2, 3, 4}): bad.append((os.path.basename(p), b.mode, b.size, a.size))
print("validator: files", len(names), "| problems", len(bad), bad[:3], flush=True); assert not bad
with zipfile.ZipFile(f"/kaggle/working/beetle_own_{TAG}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for nme in names: z.write(f"{OUT}/{nme}", nme)
print("zip written:", os.path.getsize(f"/kaggle/working/beetle_own_{TAG}.zip") / 2**20, "MB"); shutil.rmtree(OUT)
