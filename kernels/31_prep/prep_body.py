# BEETLE own-model step 1: fetch selected development slides + masks by HTTP range from Zenodo 16812932 (CPU notebook), cut 512x512 patches at 0.5 um/px with masks.
# Train: class-balanced centres (up to PER_CLASS per class per slide). Val: uniformly random centres over labelled pixels (natural class mix). Everything private; data licence CC BY-NC-SA 4.0.
import os, sys, io, json, time, struct, subprocess, urllib.request, collections, threading, shutil, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); return r.returncode, (r.stdout + r.stderr)[-300:]
print(sh("pip install -q tifffile imagecodecs"), flush=True)
import tifffile, cv2
from PIL import Image

SEL = json.loads(SEL_JSON)
if SMOKE:                                                   # 1 val + 2 train slides, the smallest ones, to test the whole path in minutes
    sm = sorted(SEL, key=lambda s: s["mb"]); SEL = [next(s for s in sm if s["split"] == "val")] + [s for s in sm if s["split"] == "train"][:2]
print("slides to process:", len(SEL), flush=True)
BASE = "https://zenodo.org/records/16812932/files"
OUT = "/kaggle/working/patches"; TMP = "/tmp/slides"; os.makedirs(TMP, exist_ok=True)
for sp in ("train", "val"):
    os.makedirs(f"{OUT}/{sp}/images", exist_ok=True); os.makedirs(f"{OUT}/{sp}/masks", exist_ok=True)
PER_CLASS, VAL_PATCHES, P, TARGET_UM = 50, 80, 512, 0.5
MAX_PIXELS = 7e9 // 3                                   # skip levels whose RGB array would exceed ~7 GB
rng = np.random.default_rng(0)


def fetch(zipname, off, size, dst):
    url = f"{BASE}/{zipname}?download=1"
    for attempt in range(4):
        try:
            hdr = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={off}-{off + 29}"}), timeout=120).read()
            n, m = struct.unpack("<HH", hdr[26:30]); start = off + 30 + n + m
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + size - 1}"}), timeout=900) as r, open(dst, "wb") as f:
                while True:
                    b = r.read(1 << 22)
                    if not b: break
                    f.write(b)
            assert os.path.getsize(dst) == size, "short read"; return
        except Exception as e:
            print(f"  fetch retry {attempt} {os.path.basename(dst)}: {type(e).__name__} {str(e)[:80]}", flush=True); time.sleep(5 * (attempt + 1))
    raise RuntimeError("fetch failed " + dst)


def spacing_um(page):
    xr = page.tags["XResolution"].value; unit = page.tags["ResolutionUnit"].value; unit = getattr(unit, "value", unit)
    ppu = xr[0] / xr[1]
    return (1e4 / ppu) if unit == 3 else (25400 / ppu if unit == 2 else float("nan"))


AHEAD = threading.BoundedSemaphore(10)                      # at most 10 slides on disk at once (downloads must not run far ahead of processing)


def download(s):
    sid = s["sid"]; t0 = time.time(); AHEAD.acquire()
    fetch("images.zip", s["wsi"][0], s["wsi"][2], f"{TMP}/{sid}.tif"); fetch("annotations.zip", s["mask"][0], s["mask"][2], f"{TMP}/{sid}_m.tif")
    return s, time.time() - t0


manifest = []; tot = collections.Counter(); skipped = []
lock = threading.Lock()


def process(s, dl_secs):
    sid, sp = s["sid"], s["split"]; t0 = time.time()
    try:
        with tifffile.TiffFile(f"{TMP}/{sid}.tif") as tw, tifffile.TiffFile(f"{TMP}/{sid}_m.tif") as tm:
            sp0 = spacing_um(tw.pages[0]); lv = [(i, spacing_um(tw.pages[i])) for i in range(len(tw.series[0].levels))]
            i, spl = min(lv, key=lambda t: abs(t[1] - TARGET_UM))
            if abs(spl - TARGET_UM) / TARGET_UM > 0.5: raise RuntimeError(f"no level near 0.5um: {lv}")
            H, W = tw.series[0].levels[i].shape[:2]
            if H * W > MAX_PIXELS: raise RuntimeError(f"level too large {H}x{W}")
            img = tw.series[0].levels[i].asarray(); msk = tm.series[0].levels[i].asarray()
            assert msk.shape == img.shape[:2], (msk.shape, img.shape)
        scale = spl / TARGET_UM                                     # >1 means the level is coarser than 0.5 um/px -> upsample
        if abs(scale - 1) > 0.03:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        H, W = msk.shape
        if H < P or W < P: raise RuntimeError(f"slide smaller than a patch {H}x{W}")
        coarse = msk[::8, ::8]; centres = []
        if sp == "train":
            for c in (1, 2, 3, 4):
                ys, xs = np.nonzero(coarse == c)
                if len(ys): k = rng.choice(len(ys), min(PER_CLASS, len(ys)), replace=False); centres += [(int(ys[j]) * 8 + int(rng.integers(8)), int(xs[j]) * 8 + int(rng.integers(8)), c) for j in k]
        else:
            ys, xs = np.nonzero(coarse > 0)
            if len(ys): k = rng.choice(len(ys), min(VAL_PATCHES, len(ys)), replace=False); centres = [(int(ys[j]) * 8, int(xs[j]) * 8, 0) for j in k]
        n = 0; cls = np.zeros(5, np.int64)
        for (cy, cx, c) in centres:
            y0 = int(np.clip(cy - P // 2, 0, H - P)); x0 = int(np.clip(cx - P // 2, 0, W - P))
            pm = msk[y0:y0 + P, x0:x0 + P]; pi = img[y0:y0 + P, x0:x0 + P]
            if sp == "val" and (pm > 0).mean() < 0.3: continue
            name = f"{sid}_{y0}_{x0}"
            Image.fromarray(pi).save(f"{OUT}/{sp}/images/{name}.jpg", quality=95); Image.fromarray(pm).save(f"{OUT}/{sp}/masks/{name}.png")
            cls += np.bincount(pm.ravel(), minlength=5)[:5]; n += 1
            manifest.append(f"{sp},{sid},{y0},{x0},{spl:.4f},{c}")
        with lock: tot.update({f"{sp}_patches": n, f"{sp}_slides": 1}); tot.update({f"{sp}_px{c}": int(cls[c]) for c in range(5)})
        print(f"[{sp}] {sid}: {s['mb']:.0f} MB, spacing L0 {sp0:.3f} -> level {i} ({spl:.3f} um/px, scale {scale:.2f}), {H}x{W}, patches {n}, label px 1-4 {cls[1:].tolist()} | dl {dl_secs:.0f}s proc {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        skipped.append(sid); print(f"[{sp}] SKIP {sid}: {type(e).__name__} {str(e)[:150]}", flush=True)
    finally:
        for f in (f"{TMP}/{sid}.tif", f"{TMP}/{sid}_m.tif"):
            try: os.remove(f)
            except OSError: pass
        AHEAD.release()


t00 = time.time()
with ThreadPoolExecutor(5) as ex:
    futs = [ex.submit(download, s) for s in SEL]                  # downloads run ahead (3 at a time); disk holds at most a handful of slides
    for fu in futs:
        s, dl = fu.result(); process(s, dl)
        used = sum(os.path.getsize(os.path.join(r, f)) for r, d, fs in os.walk(OUT) for f in fs) / 2**30
        if used > 12: print("output size cap reached, stopping", flush=True); [x.cancel() for x in futs]; break
open(f"{OUT}/manifest.csv", "w").write("split,slide,y0,x0,spacing_um,centre_class\n" + "\n".join(manifest))
print("DONE in %.0f s" % (time.time() - t00), dict(tot), "| skipped", skipped, flush=True)
print("output GB: %.2f" % (sum(os.path.getsize(os.path.join(r, f)) for r, d, fs in os.walk(OUT) for f in fs) / 2**30), flush=True)
