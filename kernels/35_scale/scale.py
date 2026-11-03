# BEETLE: is the evaluation ROI pixel size the same as our 0.5 um/px training patches? (CPU) PNG metadata + image-statistics scale matching.
import os, glob
import numpy as np, cv2
from PIL import Image
roi_dir = next(os.path.join(r, "rois") for r, d, f in os.walk("/kaggle/input") if "rois" in d and "model" in d)
rois = sorted(glob.glob(f"{roi_dir}/*.png")); print("ROIs:", len(rois), flush=True)
for p in rois[:3]:
    im = Image.open(p); print(os.path.basename(p), im.size, im.mode, "info:", {k: v for k, v in im.info.items() if k in ("dpi", "pHYs", "spacing", "resolution")}, flush=True)
pdirs = []
for r, dirs, files in os.walk("/kaggle/input"):
    if r.endswith("patches") and "val" in dirs: pdirs.append(r + "/val/images"); dirs[:] = []
rng = np.random.default_rng(0)
def hf_stats(g):                                    # g: float gray image; returns mean gradient magnitude / std and fraction of spectral energy above 1/4 sampling frequency
    g = g - g.mean(); s = g.std() + 1e-6
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3); grad = np.sqrt(gx ** 2 + gy ** 2).mean() / s
    F = np.abs(np.fft.fftshift(np.fft.fft2(g * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))))) ** 2
    h, w = F.shape; yy, xx = np.mgrid[:h, :w]; r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    return grad, F[(r > 0.25) & (r < 1.0)].sum() / F[(r > 0.0) & (r < 1.0)].sum()
def stats_roi(p):
    g = np.asarray(Image.open(p).convert("L"), np.float32); c = rng.integers(0, [g.shape[0] - 512 + 1, g.shape[1] - 512 + 1]); return hf_stats(g[c[0]:c[0] + 512, c[1]:c[1] + 512])
R = np.array([stats_roi(p) for p in rois for _ in range(2)])
print("ROI 512px crops: gradient/std %.4f (sd %.4f) | HF energy fraction %.5f (sd %.5f)" % (R[:, 0].mean(), R[:, 0].std(), R[:, 1].mean(), R[:, 1].std()), flush=True)
files = [f for d in pdirs for f in glob.glob(d + "/*.jpg")]; pick = rng.choice(len(files), 340, replace=False)
for f_ in (0.5, 0.7071, 1.0, 1.4142, 2.0):
    out = []
    for i in pick:
        g = np.asarray(Image.open(files[i]).convert("L"), np.float32)
        if f_ != 1.0: g = cv2.resize(g, None, fx=f_, fy=f_, interpolation=cv2.INTER_AREA if f_ < 1 else cv2.INTER_LINEAR)
        if min(g.shape) < 256: continue
        g = g[:min(512, g.shape[0]), :min(512, g.shape[1])]; out.append(hf_stats(g))
    O = np.array(out); print("train/val patches rescaled by x%.3f (n=%d): gradient/std %.4f | HF energy fraction %.5f" % (f_, len(O), O[:, 0].mean(), O[:, 1].mean()), flush=True)
