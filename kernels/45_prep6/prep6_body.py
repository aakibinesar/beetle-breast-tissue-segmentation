# BEETLE own-model step 1: fetch selected development slides + masks by HTTP range from Zenodo 16812932 (CPU notebook), cut 512x512 patches at 0.5 um/px with masks.
# Train: class-balanced centres (up to PER_CLASS per class per slide). Val: uniformly random centres over labelled pixels (natural class mix). Everything private; data licence CC BY-NC-SA 4.0.
import os, sys, io, json, time, struct, subprocess, urllib.request, collections, threading, shutil, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); return r.returncode, (r.stdout + r.stderr)[-300:]
print(sh("pip install -q -U tifffile imagecodecs zarr"), flush=True)
import tifffile, cv2, zarr
print('tifffile', tifffile.__version__, 'zarr', zarr.__version__, flush=True)
from PIL import Image

SEL = [s for s in json.loads(SEL_JSON) if not s["sid"].startswith("TCGA")]          # TCGA-hosted slides downloaded at ~1 MB/s: left out
SEL = sorted(SEL, key=lambda s: -s["mb"])[PART::NPARTS]                              # this notebook's share
if SMOKE:                                                   # 1 val + 1 train slide, the smallest of each
    sm = sorted(SEL, key=lambda s: s["mb"])
    SEL = [s for s in sm if s["split"] == "train"][:2]                          # smoke: the two smallest train slides (this selection has no val slides)
print("slides to process:", len(SEL), flush=True)
BASE = "https://zenodo.org/records/16812932/files"
OUT = "/kaggle/working/patches"; TMP = "/tmp/slides"; os.makedirs(TMP, exist_ok=True)
for sp in ("train", "val"):
    os.makedirs(f"{OUT}/{sp}/images", exist_ok=True); os.makedirs(f"{OUT}/{sp}/masks", exist_ok=True)
PER_CLASS, VAL_PATCHES, P, TARGET_UM = 60, 80, 512, 0.5
MAX_PIXELS = 7e9 // 3                                   # skip levels whose RGB array would exceed ~7 GB
rng = np.random.default_rng(0)


def fetch(zipname, off, size, dst):
    """Range-fetch one stored member of a Zenodo zip. Resumes after any error (partial file is kept), 8 attempts with growing back-off."""
    url = f"{BASE}/{zipname}?download=1"; start = None
    for attempt in range(8):
        try:
            if start is None:
                hdr = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={off}-{off + 29}"}), timeout=120).read()
                n, m = struct.unpack("<HH", hdr[26:30]); start = off + 30 + n + m
            have = os.path.getsize(dst) if os.path.exists(dst) else 0
            if have >= size: return
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{start + size - 1}"}), timeout=300) as r, open(dst, "ab") as f:
                while True:
                    b = r.read(1 << 22)
                    if not b: break
                    f.write(b)
            if os.path.getsize(dst) == size: return
            raise IOError(f"short read {os.path.getsize(dst)}/{size}")
        except Exception as e:
            print(f"  fetch retry {attempt} {os.path.basename(dst)}: {type(e).__name__} {str(e)[:80]}", flush=True); time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError("fetch failed " + dst)


def spacing_um(page):
    xr = page.tags["XResolution"].value; unit = page.tags["ResolutionUnit"].value; unit = getattr(unit, "value", unit)
    ppu = xr[0] / xr[1]
    return (1e4 / ppu) if unit == 3 else (25400 / ppu if unit == 2 else float("nan"))


AHEAD = threading.BoundedSemaphore(14)                      # at most 14 slides on disk at once (downloads must not run far ahead of processing)


def download(s):
    sid = s["sid"]; t0 = time.time(); AHEAD.acquire()
    files = (f"{TMP}/{sid}.tif", f"{TMP}/{sid}_m.tif")
    try:
        for f in files:
            if os.path.exists(f): os.remove(f)
        fetch("images.zip", s["wsi"][0], s["wsi"][2], files[0]); fetch("annotations.zip", s["mask"][0], s["mask"][2], files[1])
        return s, time.time() - t0, None                    # on success process() releases the semaphore
    except Exception as e:
        for f in files:
            try: os.remove(f)
            except OSError: pass
        AHEAD.release()
        return s, time.time() - t0, f"{type(e).__name__}: {str(e)[:120]}"


manifest = []; tot = collections.Counter(); skipped = []
lock = threading.Lock()


def zlevel(tf, i):
    """Lazy tile-wise reader for pyramid level i. tifffile exposes a pyramid as a zarr group with one array per level ('0', '1', ...)."""
    ser = tf.series[0]
    z = zarr.open(ser.aszarr(), mode="r")
    if hasattr(z, "shape"): return z if i == 0 else zarr.open(ser.levels[i].aszarr(), mode="r")
    keys = sorted(z.keys()) if hasattr(z, "keys") else []
    if not getattr(zlevel, "logged", False): print("zarr group keys:", keys[:12], flush=True); zlevel.logged = True
    if str(i) in keys: return z[str(i)]
    za = zarr.open(ser.levels[i].aszarr(), mode="r")
    return za if hasattr(za, "shape") else za[sorted(za.keys())[0]]


def process(s, dl_secs):
    sid, sp = s["sid"], s["split"]; t0 = time.time()
    try:
        with tifffile.TiffFile(f"{TMP}/{sid}.tif") as tw, tifffile.TiffFile(f"{TMP}/{sid}_m.tif") as tm:
            nl = len(tw.series[0].levels); lv = [(i, spacing_um(tw.pages[i])) for i in range(nl)]; sp0 = lv[0][1]
            i, spl = min(lv, key=lambda t: abs(t[1] - TARGET_UM))
            if abs(spl - TARGET_UM) / TARGET_UM > 0.5: raise RuntimeError(f"no level near 0.5um: {lv}")
            shp = [tuple(tw.series[0].levels[k].shape[:2]) for k in range(nl)]
            H, W = shp[i]
            # coarse mask level for choosing patch centres: the largest level with <= 5e7 pixels
            j = next((k for k in range(nl) if shp[k][0] * shp[k][1] <= 5e7), nl - 1)
            fy, fx = H / shp[j][0], W / shp[j][1]
            coarse = tm.series[0].levels[j].asarray()
            zi, zm = zlevel(tw, i), zlevel(tm, i)                         # lazy, tile-wise readers: only the windows we ask for are decoded
            assert zm.shape[:2] == zi.shape[:2] == (H, W), (zm.shape, zi.shape, H, W)
            scale = spl / TARGET_UM; w = int(round(P / scale))             # window size at this level that spans P pixels at 0.5 um/px
            if H < w or W < w: raise RuntimeError(f"slide smaller than a patch {H}x{W}")
            centres = []
            if sp == "train":
                for c in (1, 2, 3, 4):
                    ys, xs = np.nonzero(coarse == c)
                    if len(ys): k = rng.choice(len(ys), min(PER_CLASS, len(ys)), replace=False); centres += [(int((ys[q] + rng.random()) * fy), int((xs[q] + rng.random()) * fx), c) for q in k]
            else:
                ys, xs = np.nonzero(coarse > 0)
                if len(ys): k = rng.choice(len(ys), min(VAL_PATCHES * 2, len(ys)), replace=False); centres = [(int((ys[q] + rng.random()) * fy), int((xs[q] + rng.random()) * fx), 0) for q in k]
            n = 0; cls = np.zeros(5, np.int64)
            for (cy, cx, c) in centres:
                if sp == "val" and n >= VAL_PATCHES: break
                y0 = int(np.clip(cy - w // 2, 0, H - w)); x0 = int(np.clip(cx - w // 2, 0, W - w))
                pm = np.asarray(zm[y0:y0 + w, x0:x0 + w]); pi = np.asarray(zi[y0:y0 + w, x0:x0 + w])
                if w != P:
                    pi = cv2.resize(pi, (P, P), interpolation=cv2.INTER_AREA if w > P else cv2.INTER_LINEAR); pm = cv2.resize(pm, (P, P), interpolation=cv2.INTER_NEAREST)
                if sp == "val" and (pm > 0).mean() < 0.3: continue
                if pi.shape != (P, P, 3) or pm.shape != (P, P): continue
                name = f"{sid}_{y0}_{x0}"
                Image.fromarray(pi).save(f"{OUT}/{sp}/images/{name}.jpg", quality=95); Image.fromarray(pm).save(f"{OUT}/{sp}/masks/{name}.png")
                cls += np.bincount(pm.ravel(), minlength=5)[:5]; n += 1
                manifest.append(f"{sp},{sid},{y0},{x0},{spl:.4f},{c}")
        with lock: tot.update({f"{sp}_patches": n, f"{sp}_slides": 1}); tot.update({f"{sp}_px{c}": int(cls[c]) for c in range(5)})
        print(f"[{sp}] {sid}: {s['mb']:.0f} MB, spacing L0 {sp0:.3f} -> level {i} ({spl:.3f} um/px, scale {scale:.2f}), {H}x{W}, coarse level {j} ({shp[j]}), patches {n}, label px 1-4 {cls[1:].tolist()} | dl {dl_secs:.0f}s proc {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        skipped.append(sid); print(f"[{sp}] SKIP {sid}: {type(e).__name__} {str(e)[:200]}", flush=True)
        traceback.print_exc(limit=3)
    finally:
        for f in (f"{TMP}/{sid}.tif", f"{TMP}/{sid}_m.tif"):
            try: os.remove(f)
            except OSError: pass
        AHEAD.release()


STOP = threading.Event()


def orchestrate(items, dl_fn, proc_fn, deadline, threads=3):
    """Downloads run in a small pool, processing in this thread. A failed download is recorded and skipped (it never blocks the others);
    at the deadline everything stops cleanly so that the output written so far is kept."""
    from concurrent.futures import TimeoutError as FTimeout
    ex = ThreadPoolExecutor(threads); done, failed = [], []
    futs = [ex.submit(dl_fn, s) for s in items]
    try:
        for s, fu in zip(items, futs):
            left = deadline - time.time()
            if STOP.is_set() or left <= 0: print("deadline/size cap reached - stopping", flush=True); break
            try: s2, dl, err = fu.result(timeout=left)
            except FTimeout: print("deadline reached while waiting for", s["sid"], flush=True); break
            except Exception as e: failed.append((s["sid"], repr(e)[:100])); print(f"[dl-error] {s['sid']}: {e!r}"[:200], flush=True); continue
            if err: failed.append((s["sid"], err)); print(f"[dl-fail] {s['sid']}: {err}", flush=True); continue
            proc_fn(s2, dl); done.append(s["sid"])
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return done, failed


def write_manifest():
    open(f"{OUT}/manifest.csv", "w").write("split,slide,y0,x0,spacing_um,centre_class\n" + "\n".join(manifest))


def process_and_save(s, dl):
    process(s, dl); write_manifest()
    used = sum(os.path.getsize(os.path.join(r, f)) for r, d, fs in os.walk(OUT) for f in fs) / 2**30
    if used > 12: print("output size cap reached, stopping", flush=True); STOP.set()


if __name__ == "__main__":
    t00 = time.time()
    end = t00 + DEADLINE_MIN * 60
    done, failed = orchestrate(SEL, download, process_and_save, end)
    ids = {f[0] for f in failed}; again = [s for s in SEL if s["sid"] in ids]
    if again and time.time() < end and not STOP.is_set():                       # one more try for the slides whose download failed
        print("retrying", len(again), "failed downloads", flush=True)
        done2, failed = orchestrate(again, download, process_and_save, end); done += done2
    write_manifest()
    print("DONE in %.0f s" % (time.time() - t00), dict(tot), "| processed", len(done), "of", len(SEL), "| download failures", failed, "| skipped in processing", skipped, flush=True)
    print("output GB: %.2f" % (sum(os.path.getsize(os.path.join(r, f)) for r, d, fs in os.walk(OUT) for f in fs) / 2**30), flush=True)
    sys.stdout.flush(); os._exit(0)                         # background download threads must not keep the notebook alive
