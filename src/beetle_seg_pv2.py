"""Phikon-v2 (DINOv2 ViT-L/16, 303M parameters, Owkin non-commercial licence) + light CNN skip branch + conv decoder for 4-class tissue segmentation (classes 1-4 -> logits 0-3).
Same decoder as beetle_seg.py (which wraps the torchvision Phikon ViT-B); only the backbone differs. API-compatible with beetle_seg.py: ViTSeg, build_phikon(path), set_trainable(model, last_blocks),
so the inference driver (kernels/36_infer_seg) works unchanged with this file concatenated in place of beetle_seg.py.
The backbone is the Hugging Face transformers Dinov2Model (weights: beetle-60-phikonv2-fetch output, phikon-v2/); it is reached only through the small DinoBackbone adapter below
(features / blocks / final_norms), so a tiny stand-in (TinyBackbone, env BEETLE_TINY=1) can replace it in local CPU tests."""
import os, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class DinoBackbone(nn.Module):
    """Adapter around transformers.Dinov2Model. features(x, layers): list of [B, D, h, w] maps = output of block i (i = 1..n) after the final LayerNorm (as DINOv2's get_intermediate_layers(norm=True))."""
    def __init__(s, model):
        super().__init__()
        s.m = model; s.hidden_dim = model.config.hidden_size; s.n_layers = model.config.num_hidden_layers; s.patch = model.config.patch_size

    def features(s, x, layers):
        B, _, H, W = x.shape; h, w = H // s.patch, W // s.patch
        hs = s.m(pixel_values=x, output_hidden_states=True).hidden_states            # hs[0] = embeddings, hs[i] = output of block i (before the final norm)
        return [s.m.layernorm(hs[i])[:, 1:].transpose(1, 2).reshape(B, -1, h, w) for i in layers]       # drop the CLS token

    def blocks(s): return list(s.m.encoder.layer)
    def final_norms(s): return [s.m.layernorm]


class TinyBackbone(nn.Module):
    """A few-parameter stand-in with the same interface, for CPU unit tests of the training / resume logic (env BEETLE_TINY=1). NOT a ViT."""
    def __init__(s, hidden=32, n_layers=4, patch=16):
        super().__init__()
        s.hidden_dim, s.n_layers, s.patch = hidden, n_layers, patch
        s.embed = nn.Conv2d(3, hidden, patch, patch)
        s.layer = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU()) for _ in range(n_layers)])
        s.norm = nn.LayerNorm(hidden)

    def features(s, x, layers):
        B, _, H, W = x.shape; h, w = H // s.patch, W // s.patch
        t = s.embed(x).flatten(2).transpose(1, 2); out = []
        for i, blk in enumerate(s.layer, 1):
            t = t + blk(t)
            if i in layers: out.append(s.norm(t).transpose(1, 2).reshape(B, -1, h, w))
        return out

    def blocks(s): return list(s.layer)
    def final_norms(s): return [s.norm]


class ViTSeg(nn.Module):
    def __init__(s, vit, feat_layers=None, n_cls=4, dec=128):
        super().__init__()
        n = vit.n_layers; s.vit = vit; s.layers = tuple(feat_layers) if feat_layers else (n // 4, n // 2, 3 * n // 4, n)      # 24 blocks -> 6, 12, 18, 24 (ViT-B/12 used 3, 6, 9, 12)
        d = vit.hidden_dim
        s.proj = nn.ModuleList([nn.Conv2d(d, dec, 1) for _ in s.layers])
        s.fuse = conv_bn(dec * len(s.layers), 256)
        s.c1 = nn.Sequential(conv_bn(3, 32), conv_bn(32, 32))                    # 1/1 skip (sharp boundaries; the ViT grid is only 1/16)
        s.c2 = nn.Sequential(conv_bn(32, 64, 3, 2), conv_bn(64, 64))            # 1/2
        s.c4 = nn.Sequential(conv_bn(64, 96, 3, 2), conv_bn(96, 96))            # 1/4
        s.c8 = nn.Sequential(conv_bn(96, 128, 3, 2), conv_bn(128, 128))         # 1/8
        s.d8 = conv_bn(256 + 128, 192); s.d4 = conv_bn(192 + 96, 128); s.d2 = conv_bn(128 + 64, 96); s.d1 = conv_bn(96 + 32, 64)
        s.out = nn.Conv2d(64, n_cls, 1)
        s.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))      # Phikon-v2's preprocessor uses the ImageNet statistics as well
        s.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def backbone_trainable(s):
        return any(p.requires_grad for p in s.vit.parameters())

    def forward(s, x01):                                                         # x01: float [B,3,H,W] in 0..1, H and W multiples of 16
        x = (x01 - s.mean) / s.std
        f1 = s.c1(x); f2 = s.c2(f1); f4 = s.c4(f2); f8 = s.c8(f4)
        if s.backbone_trainable(): feats = s.vit.features(x, s.layers)
        else:
            with torch.no_grad(): feats = s.vit.features(x, s.layers)
        p = s.fuse(torch.cat([pr(f.float() if f.dtype != x.dtype else f) for pr, f in zip(s.proj, feats)], 1))
        up = lambda a, ref: F.interpolate(a, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        y = s.d8(torch.cat([up(p, f8), f8], 1)); y = s.d4(torch.cat([up(y, f4), f4], 1)); y = s.d2(torch.cat([up(y, f2), f2], 1)); y = s.d1(torch.cat([up(y, f1), f1], 1))
        return s.out(y)


def _find_phikon_v2_dir():
    for r, dirs, files in os.walk(os.environ.get("BEETLE_IN", "/kaggle/input")):
        dirs[:] = [d for d in dirs if d != "patches"]
        if "config.json" in files:
            try:
                if json.load(open(os.path.join(r, "config.json"))).get("model_type") == "dinov2": return r
            except Exception: pass
    raise FileNotFoundError("no Phikon-v2 (model_type dinov2) config.json under the input folders")


def build_phikon(weights_path):
    """weights_path: directory with config.json + model.safetensors (training: pretrained backbone) or None (inference: the config alone, random init; the weights come from our saved state dict)."""
    if os.environ.get("BEETLE_TINY"): return TinyBackbone()
    from transformers import Dinov2Config, Dinov2Model
    if weights_path: m = Dinov2Model.from_pretrained(weights_path, attn_implementation="sdpa")
    else:
        cfg = Dinov2Config.from_pretrained(_find_phikon_v2_dir())
        try: m = Dinov2Model._from_config(cfg, attn_implementation="sdpa")
        except Exception: m = Dinov2Model(cfg)
    return DinoBackbone(m)


def set_trainable(model, last_blocks):
    for p in model.vit.parameters(): p.requires_grad = False
    if last_blocks > 0:
        for blk in model.vit.blocks()[-last_blocks:]:
            for p in blk.parameters(): p.requires_grad = True
        for nm in model.vit.final_norms():
            for p in nm.parameters(): p.requires_grad = True
