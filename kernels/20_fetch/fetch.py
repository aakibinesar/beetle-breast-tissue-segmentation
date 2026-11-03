# BEETLE: fetch the 170 evaluation ROI PNGs (range requests into the 147 GB images.zip) and the organisers' model.zip from Zenodo record 16812932.
# Data licence CC BY-NC-SA 4.0: used here for non-commercial benchmarking; not redistributed (kernel is private).
import io, json, os, sys, time, hashlib, zipfile, urllib.request

REC = "16812932"
BASE = f"https://zenodo.org/records/{REC}/files"
OUT = "/kaggle/working/beetle"; os.makedirs(f"{OUT}/rois", exist_ok=True); os.makedirs(f"{OUT}/model", exist_ok=True)


def get(url, headers=None, tries=5):
    for t in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=180)
        except Exception as e:
            print(f"retry {t}: {type(e).__name__} {str(e)[:80]}", flush=True); time.sleep(5 * (t + 1))
    raise RuntimeError("download failed: " + url)


meta = json.load(get(f"https://zenodo.org/api/records/{REC}"))
files = {f["key"]: f for f in meta["files"]}
for k, f in files.items(): print(k, f["size"], f["checksum"], flush=True)
SIZE = files["images.zip"]["size"]


class RangeFile(io.RawIOBase):
    def __init__(s): s.pos = 0; s.fetched = 0
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s, off, whence=0):
        s.pos = off if whence == 0 else (s.pos + off if whence == 1 else SIZE + off); return s.pos
    def read(s, n=-1):
        if n < 0 or s.pos + n > SIZE: n = SIZE - s.pos
        if n <= 0: return b""
        d = get(f"{BASE}/images.zip?download=1", {"Range": f"bytes={s.pos}-{s.pos + n - 1}"}).read()
        s.pos += len(d); s.fetched += len(d); return d
    def readinto(s, b):
        d = s.read(len(b)); b[:len(d)] = d; return len(d)


t0 = time.time(); rf = RangeFile(); z = zipfile.ZipFile(rf)
roi = [i for i in z.infolist() if "evaluation/rois/" in i.filename and i.filename.endswith(".png")]
print(f"ROI entries: {len(roi)}", flush=True); assert len(roi) == 170
for n, i in enumerate(roi):
    dst = f"{OUT}/rois/{os.path.basename(i.filename)}"
    with z.open(i) as src, open(dst, "wb") as f: f.write(src.read())            # zipfile verifies the CRC32 while reading
    if n % 20 == 0: print(f"{n + 1}/170 | {rf.fetched / 2**20:.0f} MB fetched | {time.time() - t0:.0f}s", flush=True)
print(f"ROIs done: {len(os.listdir(OUT + '/rois'))} files, {sum(os.path.getsize(OUT + '/rois/' + f) for f in os.listdir(OUT + '/rois')) / 2**20:.0f} MB, CRC-verified | {time.time() - t0:.0f}s", flush=True)

# model.zip: stream to disk, verify md5 against the Zenodo record, then extract
mz = f"{OUT}/model.zip"; md5 = hashlib.md5()
with get(f"{BASE}/model.zip?download=1") as r, open(mz, "wb") as f:
    while True:
        b = r.read(1 << 22)
        if not b: break
        f.write(b); md5.update(b)
want = files["model.zip"]["checksum"].split(":")[-1]
print("model.zip md5", md5.hexdigest(), "expected", want, "OK" if md5.hexdigest() == want else "MISMATCH", flush=True)
assert md5.hexdigest() == want
with zipfile.ZipFile(mz) as zz:
    zz.extractall(f"{OUT}/model")
os.remove(mz)
for r, d, fs in os.walk(f"{OUT}/model"):
    for f in fs: print(os.path.join(r, f).replace(OUT, ""), os.path.getsize(os.path.join(r, f)), flush=True)
print("done", f"{time.time() - t0:.0f}s")
