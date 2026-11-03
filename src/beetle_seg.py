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
