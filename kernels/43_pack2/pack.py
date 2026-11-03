# CPU notebook: pack all BEETLE patch folders (old runs + the five prep4 parts) into a few large shard files + one index, so a GPU training session
# mounts ONE source with ~15 files instead of 7 sources with ~56,000 small files.
# shard_XXX.bin = raw jpg bytes and png bytes written back to back (no re-encoding); index.json lists, per patch, where its image and mask bytes are.
import os, io, json, time, random
import numpy as np
from PIL import Image

INROOT = os.environ.get("BEETLE_IN", "/kaggle/input"); OUT = os.environ.get("BEETLE_OUT", "/kaggle/working") + "/packed"
SHARD_BYTES = 400 * 2 ** 20
os.makedirs(OUT, exist_ok=True); T0 = time.time()
def stamp(m): print(f"[t+{time.time() - T0:6.0f}s] {m}", flush=True)

roots = []
for r, dirs, files in os.walk(INROOT):
    if r.endswith("patches") and "train" in dirs: roots.append(r); dirs[:] = []
roots.sort(); stamp(f"{len(roots)} patch roots: {roots}")

index, seen = [], set(); shard_id, shard_f, shard_size = -1, None, 0
def new_shard():
    global shard_id, shard_f, shard_size
    if shard_f: shard_f.close()
    shard_id += 1; shard_f = open(f"{OUT}/shard_{shard_id:03d}.bin", "wb"); shard_size = 0
new_shard()
for r in roots:
    for sp in ("train", "val"):
        if not os.path.isdir(f"{r}/{sp}/images"): stamp(f"{r.split('/')[-2]}/{sp}: none (empty folders are not kept in Kaggle outputs)"); continue
        names = sorted(f[:-4] for f in os.listdir(f"{r}/{sp}/images"))
        stamp(f"{r.split('/')[-2]}/{sp}: {len(names)} patches")
        for n in names:
            key = (sp, n)
            assert key not in seen, f"duplicate patch name {key}"; seen.add(key)
            ib = open(f"{r}/{sp}/images/{n}.jpg", "rb").read(); mb = open(f"{r}/{sp}/masks/{n}.png", "rb").read()
            u = np.unique(np.array(Image.open(io.BytesIO(mb))))
            pres = sum(1 << c for c in range(4) if (c + 1) in u)                  # bit c set = class c+1 (1 other, 2 non-inv, 3 invasive, 4 necrosis) occurs in the mask
            if shard_size > SHARD_BYTES: new_shard()
            io_off = shard_f.tell(); shard_f.write(ib); mo_off = shard_f.tell(); shard_f.write(mb); shard_size += len(ib) + len(mb)
            index.append(dict(split=sp, name=n, shard=shard_id, io=io_off, il=len(ib), mo=mo_off, ml=len(mb), pres=pres))
shard_f.close()
json.dump(index, open(f"{OUT}/index.json", "w"))
stamp(f"packed {len(index)} patches into {shard_id + 1} shards, {sum(os.path.getsize(f'{OUT}/{f}') for f in os.listdir(OUT)) / 2 ** 30:.2f} GB")

# verification: re-read 400 random entries from the shards and compare the bytes with the original files
random.seed(0); bad = 0
root_of = {}
for r in roots:
    for sp in ("train", "val"):
        if not os.path.isdir(f"{r}/{sp}/images"): continue
        for f in os.listdir(f"{r}/{sp}/images"): root_of[(sp, f[:-4])] = r
for e in random.sample(index, min(400, len(index))):
    with open(f"{OUT}/shard_{e['shard']:03d}.bin", "rb") as f:
        f.seek(e["io"]); ib = f.read(e["il"]); f.seek(e["mo"]); mb = f.read(e["ml"])
    r = root_of[(e["split"], e["name"])]
    if ib != open(f"{r}/{e['split']}/images/{e['name']}.jpg", "rb").read() or mb != open(f"{r}/{e['split']}/masks/{e['name']}.png", "rb").read(): bad += 1
stamp(f"verification of 400 random patches: {bad} mismatches")
cnt = {sp: [sum(1 for e in index if e['split'] == sp and (e['pres'] >> c) & 1) for c in range(4)] for sp in ("train", "val")}
print("patches containing each class (other, non-inv, invasive, necrosis):", cnt, "| train", sum(1 for e in index if e['split'] == 'train'), "val", sum(1 for e in index if e['split'] == 'val'), flush=True)
assert bad == 0
stamp("DONE")
