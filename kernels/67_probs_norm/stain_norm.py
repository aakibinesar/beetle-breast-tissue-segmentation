# ---------------- BEETLE: test-time colour matching (Reinhard, CIELAB, tissue pixels only, soft tissue mask) ----------------
# Every evaluation ROI is shifted/scaled in CIELAB so that its tissue colour statistics (mean and standard deviation of L, a, b) equal the average statistics of the
# TRAINING patches of the model (target_from_packed). No labels, no learning, no parameter is tuned on the test set: the tissue definition is the one of kernels/51_stain,
# the scale ratio is clipped to [0.5, 2] and the tissue weight map is a Gaussian blur (sigma 2 px) of the tissue mask, so the white background stays untouched.
# (sigma was 6 px in the first CPU check: thin or fragmented tissue was then only partly shifted, median colour distance after matching 0.99 for ROIs with < 20% tissue; 2 px was chosen from colour distances only, before any submission.)
# Pure torch (device-agnostic) + numpy; no OpenCV / scikit-image needed.
import io, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_M = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float64)     # linear sRGB -> XYZ (D65)
_W = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float64)
_EPS, _KAPPA = 216 / 24389, 24389 / 27


def rgb2lab(x):                      # float [..., 3] in 0..1 (sRGB) -> CIELAB (L 0..100, a/b about -128..127), same dtype / device
    M, W = _M.to(x.dtype).to(x.device), _W.to(x.dtype).to(x.device)
    lin = torch.where(x <= 0.04045, x / 12.92, ((x.clamp(min=0) + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ M.T) / W
    f = torch.where(xyz > _EPS, xyz.clamp(min=1e-12) ** (1 / 3), (_KAPPA * xyz + 16) / 116)
    return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab2rgb(lab):                    # inverse of rgb2lab, result clipped to 0..1
    M, W = _M.to(lab.dtype).to(lab.device), _W.to(lab.dtype).to(lab.device)
    fy = (lab[..., 0] + 16) / 116; fx = fy + lab[..., 1] / 500; fz = fy - lab[..., 2] / 200
    f = torch.stack([fx, fy, fz], -1); f3 = f ** 3
    xyz = torch.where(f3 > _EPS, f3, (116 * f - 16) / _KAPPA) * W
    lin = (xyz @ torch.linalg.inv(M).T).clamp(0, 1)
    return torch.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin.clamp(min=1e-12) ** (1 / 2.4) - 0.055).clamp(0, 1)


def tissue_mask(x):                  # x float [..., 3] in 0..1; same rule as kernels/51_stain: drop white background and grey pixels
    g = x.mean(-1) * 255
    return (g < 215) & ((x.max(-1).values - x.min(-1).values) * 255 > 12)


def lab_stats(rgb, step=4, min_pixels=500, dev="cpu"):      # uint8 [H,W,3] -> (mean[3], std[3]) of the tissue pixels in CIELAB (float64 numpy), or None if too little tissue
    x = torch.from_numpy(np.ascontiguousarray(rgb[::step, ::step])).to(dev).float() / 255
    m = tissue_mask(x)
    if int(m.sum()) < min_pixels: return None
    lab = rgb2lab(x)[m].double()
    return lab.mean(0).cpu().numpy(), lab.std(0).cpu().numpy()


def _blur(m, sigma):                 # separable Gaussian blur of a float [H,W] map (reflect padding)
    r = int(3 * sigma); k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=m.dtype, device=m.device) / sigma) ** 2); k = k / k.sum()
    x = F.pad(m[None, None], (r, r, 0, 0), mode="reflect"); x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, r, r), mode="reflect"); return F.conv2d(x, k.view(1, 1, -1, 1))[0, 0]


def normalise(rgb, target, dev="cpu", ratio_clip=(0.5, 2.0), sigma=2.0, min_pixels=500):
    """uint8 [H,W,3] -> (uint8 [H,W,3] colour-matched image, info dict) ; (image unchanged, None) when the ROI has too little tissue."""
    s = lab_stats(rgb, step=4, min_pixels=min_pixels, dev=dev)
    if s is None: return rgb, None
    mu, sd = s; t_mu, t_sd = np.array(target["mean"]), np.array(target["std"])
    scale = np.clip(t_sd / np.maximum(sd, 1e-6), *ratio_clip)
    x = torch.from_numpy(np.ascontiguousarray(rgb)).to(dev).float() / 255
    m = tissue_mask(x).float(); w = _blur(m, sigma)[..., None]
    lab = rgb2lab(x); lab = (lab - torch.tensor(mu, dtype=lab.dtype, device=dev)) * torch.tensor(scale, dtype=lab.dtype, device=dev) + torch.tensor(t_mu, dtype=lab.dtype, device=dev)
    lab[..., 0] = lab[..., 0].clamp(0, 100); lab[..., 1:] = lab[..., 1:].clamp(-128, 127)
    out = (w * lab2rgb(lab) + (1 - w) * x) * 255
    info = dict(src_mean=mu.tolist(), src_std=sd.tolist(), scale=scale.tolist(), tissue_frac=float(m.mean()))
    return out.round().clamp(0, 255).to(torch.uint8).cpu().numpy(), info


_DINO_RENAMES = [(".attention.attention.query.", ".attention.q_proj."), (".attention.attention.key.", ".attention.k_proj."), (".attention.attention.value.", ".attention.v_proj."), (".attention.output.dense.", ".attention.o_proj.")]


def remap_dino_keys(sd, have):
    """Newer transformers versions renamed the attention layers of Dinov2Model (query/key/value/output.dense -> q_proj/k_proj/v_proj/o_proj); the checkpoints were saved with the old names.
    If the model does not know the checkpoint's keys, rename them (same tensors); the caller loads with strict=True, so a wrong mapping still fails loudly."""
    if set(sd) <= set(have): return sd
    out = {}
    for k, v in sd.items():
        for a, b in _DINO_RENAMES: k = k.replace(a, b)
        out[k] = v
    return out


def colour_distance(mean, target):   # root-mean-square distance of the three channel means to the target, in units of the target's standard deviation
    return float(np.sqrt(np.mean(((np.asarray(mean) - np.array(target["mean"])) / np.array(target["std"])) ** 2)))


def target_from_packed(packed_dirs, exclude_slides=(), per_slide=12, seed=0, split="train", step=2, dev="cpu"):
    """Target statistics = average over the TRAINING slides of the model (packed patch sets; excluded slides left out) of the per-slide mean and within-slide std (up to per_slide random patches per slide)."""
    rng = np.random.default_rng(seed); by_slide = {}; exclude = set(exclude_slides)
    for d in packed_dirs:
        for e in json.load(open(f"{d}/index.json")):
            sid = "_".join(e["name"].split("_")[:2])
            if e["split"] == split and sid not in exclude: by_slide.setdefault(sid, []).append((d, e))
    means, stds, n_patches, fh = [], [], 0, {}
    for sid in sorted(by_slide):
        es = by_slide[sid]; pick = [es[i] for i in rng.choice(len(es), min(per_slide, len(es)), replace=False)]; st = []
        for d, e in pick:
            f = fh.get((d, e["shard"])) or fh.setdefault((d, e["shard"]), open(f"{d}/shard_{e['shard']:03d}.bin", "rb")); f.seek(e["io"])
            s = lab_stats(np.array(Image.open(io.BytesIO(f.read(e["il"]))).convert("RGB")), step=step, min_pixels=200, dev=dev)
            if s is not None: st.append(s)
        if st: means.append(np.mean([s[0] for s in st], 0)); stds.append(np.mean([s[1] for s in st], 0)); n_patches += len(st)
    for f in fh.values(): f.close()
    assert len(means) >= 3, f"only {len(means)} training slides with tissue statistics"
    return dict(mean=np.mean(means, 0).tolist(), std=np.mean(stds, 0).tolist(), n_slides=len(means), n_patches=n_patches, n_excluded=len(exclude))
