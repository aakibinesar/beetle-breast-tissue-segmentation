"""CPU tests of the Phikon-v2 training notebook logic with a tiny stand-in backbone (env BEETLE_TINY=1; the real Dinov2 adapter needs transformers and the 1.1 GB weights, so it is checked separately on Kaggle).
Runs the GENERATED notebook script (build.sh + src/beetle_seg_pv2.py + driver.py, copied to a temp folder) on a fake packed patch set:
  1. model: shapes at 512 px, only the last N blocks + final norm trainable
  2. resume: 4 epochs uninterrupted == 2 epochs + RESUME for 2 epochs (history rows and final weights identical)
  3. stop rule on the stain-shift score, TIME budget and TIMING mode behave as documented.
usage: python test_local.py"""
import os, sys, io, json, shutil, subprocess, tempfile
import numpy as np, torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, "..", "..", "src")
TMP = tempfile.mkdtemp(prefix="pv2test_"); print("temp folder:", TMP)
os.makedirs(f"{TMP}/kernels/62_train_pv2"); os.makedirs(f"{TMP}/src")
for f in ("build.sh", "driver.py"): shutil.copy(f"{HERE}/{f}", f"{TMP}/kernels/62_train_pv2/{f}")
shutil.copy(f"{SRC}/beetle_seg_pv2.py", f"{TMP}/src/beetle_seg_pv2.py")
K = f"{TMP}/kernels/62_train_pv2"

# ---------- 1. model shapes / trainable parts (module imported directly) ----------
os.environ["BEETLE_TINY"] = "1"
sys.path.insert(0, f"{TMP}/src"); import beetle_seg_pv2 as M
m = M.ViTSeg(M.build_phikon(None)); x = torch.rand(2, 3, 512, 512)
assert m(x).shape == (2, 4, 512, 512), m(x).shape
assert m.layers == (1, 2, 3, 4), m.layers
M.set_trainable(m, 2)
n_bb = sum(p.numel() for p in m.vit.parameters() if p.requires_grad); blocks = m.vit.blocks()
assert all(p.requires_grad for b in blocks[-2:] for p in b.parameters()) and not any(p.requires_grad for b in blocks[:-2] for p in b.parameters())
assert all(p.requires_grad for n in m.vit.final_norms() for p in n.parameters()) and not any(p.requires_grad for p in m.vit.embed.parameters())
print("1. model OK: output", tuple(m(x).shape), "| feature layers", m.layers, "| trainable backbone params", n_bb)
M.set_trainable(m, 0); assert not m.backbone_trainable(); assert m(x).shape == (2, 4, 512, 512)

# ---------- fake packed patch set: patients 1-3 (+ shift patients 25, 26), 64x64 patches ----------
def make_packed(root):
    d = f"{root}/data/packed"; os.makedirs(d); rng = np.random.default_rng(0); index, blob = [], bytearray()
    def add(split, patient, k):
        img = (rng.random((64, 64, 3)) * 255).astype(np.uint8); mask = np.zeros((64, 64), np.uint8)
        for c in range(1, 5):
            if rng.random() < 0.7: y0, x0 = rng.integers(0, 40, 2); mask[y0:y0 + 24, x0:x0 + 24] = c
        bi, bm = io.BytesIO(), io.BytesIO(); Image.fromarray(img).save(bi, "JPEG"); Image.fromarray(mask).save(bm, "PNG")
        e = dict(name=f"patient{patient:03d}_{k}", split=split, shard=0, io=len(blob), il=len(bi.getvalue())); blob.extend(bi.getvalue())
        e.update(mo=len(blob), ml=len(bm.getvalue())); blob.extend(bm.getvalue()); e["pres"] = sum(1 << (c - 1) for c in range(1, 5) if (mask == c).any()); index.append(e)
    for p in (1, 2, 3, 25, 26):
        for k in range(8 if p in (1, 2, 3) else 4): add("train", p, k)
    for k in range(8): add("val", 4, k)
    open(f"{d}/shard_000.bin", "wb").write(bytes(blob)); json.dump(index, open(f"{d}/index.json", "w"))
IN = f"{TMP}/in"; os.makedirs(IN); make_packed(IN); json.dump({"patients": [25, 26]}, open(f"{TMP}/shift.json", "w"))


def run(tag, epochs, out, extra=None, sub=False):
    env = dict(os.environ, BEETLE_TINY="1", BEETLE_DEV="cpu", BEETLE_AMP="0", BEETLE_WORKERS="0", BEETLE_IN=IN, BEETLE_OUT=out, SHIFT_JSON=f"{TMP}/shift.json", STAIN="0.1", BB_DELAY="0.25")
    env.update(extra or {}); os.makedirs(out, exist_ok=True)
    r = subprocess.run(["sh", "build.sh", tag, str(epochs), "4", "5e-4", "3e-5", "2", "0"], cwd=K, env=env, capture_output=True, text=True); assert r.returncode == 0, r.stderr
    p = subprocess.run([sys.executable, "train_seg_pv2.py"], cwd=K, env=env, capture_output=True, text=True)
    if p.returncode != 0: print(p.stdout[-3000:], p.stderr[-3000:]); raise SystemExit(f"run {tag} failed")
    return p.stdout, json.load(open(f"{out}/metrics_{tag}.json"))


# ---------- 2. resume equivalence ----------
outU, mU = run("U", 4, f"{TMP}/outU", dict(SHIFT_STOP="0", TIME_BUDGET_H="0"))
os.makedirs(f"{IN}/prevA")
outA, mA = run("A", 4, f"{IN}/prevA", dict(SHIFT_STOP="0", TIME_BUDGET_H="0", STOP_AFTER="2"))
assert mA["epochs_done"] == 2 and "session stop" in mA["stopped"], mA["stopped"]
assert os.path.exists(f"{IN}/prevA/resume_A.pt")
outB, mB = run("B", 4, f"{TMP}/outB", dict(SHIFT_STOP="0", TIME_BUDGET_H="0", RESUME_TAG="A"))
assert "RESUMED from A: 2 epochs done" in outB and mB["epochs_done"] == 4 and mB["stopped"] == "all epochs done"
key = lambda h: (h["epoch"], h["train_loss"], h["overall"], h["mean_class"], tuple(h["per_class"]), h["shift"]["mean_class"], tuple(h["shift"]["per_class"]))
hu, hb = [key(h) for h in mU["history"]], [key(h) for h in mB["history"]]
assert hu == hb, f"history differs:\n{hu}\n{hb}"
su, sb = torch.load(f"{TMP}/outU/seg_last_U.pt"), torch.load(f"{TMP}/outB/seg_last_B.pt")
assert su.keys() == sb.keys() and all(torch.equal(su[k], sb[k]) for k in su), "final weights differ"
assert os.path.exists(f"{TMP}/outB/seg_best_B.pt") and abs(mU["best_score"] - mB["best_score"]) < 1e-12
print("2. resume OK: 4 epochs uninterrupted == 2 + resume 2 (history rows and all", len(su), "weight tensors identical); best score", round(mB["best_score"], 4))
print("   per-epoch train loss:", [h["train_loss"] for h in mB["history"]])
bb_lr = [l for l in outU.split("\n") if l.startswith("ep0 0/")]
print("   first lines:", bb_lr[:1])

# ---------- 3. stop rules and timing ----------
outS, mS = run("S", 8, f"{TMP}/outS", dict(SHIFT_STOP="1", TIME_BUDGET_H="0"))
sh = [h["shift"]["mean_class"] for h in mS["history"]]
first_drop = next((i for i in range(1, len(sh)) if sh[i] < sh[i - 1]), None)
if first_drop is None: assert mS["epochs_done"] == 8 and mS["stopped"] == "all epochs done"
else: assert mS["epochs_done"] == first_drop + 1 and "stain-shift score fell 1 epochs" in mS["stopped"], (mS["epochs_done"], first_drop, mS["stopped"])
print("3a. shift-score stop rule OK: scores", [round(v, 4) for v in sh], "->", mS["stopped"])
outT, mT = run("T", 6, f"{TMP}/outT", dict(SHIFT_STOP="0", TIME_BUDGET_H="0.00000001"))            # a budget of ~0 s: must stop after the first epoch (resume file kept)
assert mT["epochs_done"] == 1 and "time budget" in mT["stopped"] and os.path.exists(f"{TMP}/outT/resume_T.pt"), mT["stopped"]
print("3b. time-budget stop OK:", mT["stopped"])
env = dict(os.environ, BEETLE_TINY="1", BEETLE_DEV="cpu", BEETLE_AMP="0", BEETLE_WORKERS="0", BEETLE_IN=IN, BEETLE_OUT=f"{TMP}/outM", SHIFT_JSON=f"{TMP}/shift.json", TIMING="3")
subprocess.run(["sh", "build.sh", "M", "1", "4", "5e-4", "3e-5", "2", "0"], cwd=K, env=env, check=True, capture_output=True)
p = subprocess.run([sys.executable, "train_seg_pv2.py"], cwd=K, env=env, capture_output=True, text=True); assert p.returncode == 0 and "PROJECTION:" in p.stdout, p.stdout[-1500:] + p.stderr[-1500:]
print("3c. timing mode OK:"); print("   " + "\n   ".join(l for l in p.stdout.split("\n") if l.startswith(("TIMING", "PROJECTION"))))
print("sections 1-3 passed")

# ---------- 4. slide-level exclusion (ablation back to the original slides) ----------
import json as _json
_json.dump([{"sid": "patient001_0"}, {"sid": "patient001_1"}], open(f"{os.path.dirname(os.path.abspath(__file__))}/_excl_test.json", "w"))
if True:
    TMP2 = tempfile.mkdtemp(prefix="pv2ex_"); os.makedirs(f"{TMP2}/kernels/62_train_pv2"); os.makedirs(f"{TMP2}/src")
    for f in ("build.sh", "driver.py"): shutil.copy(f"{HERE}/{f}", f"{TMP2}/kernels/62_train_pv2/{f}")
    shutil.copy(f"{SRC}/beetle_seg_pv2.py", f"{TMP2}/src/beetle_seg_pv2.py"); K2 = f"{TMP2}/kernels/62_train_pv2"
    envx = dict(os.environ, BEETLE_TINY="1", BEETLE_DEV="cpu", BEETLE_AMP="0", BEETLE_WORKERS="0", BEETLE_IN=IN, BEETLE_OUT=f"{TMP2}/out", SHIFT_JSON=f"{TMP}/shift.json", EXCLUDE_JSON=f"{HERE}/_excl_test.json", TIMING="1"); os.makedirs(f"{TMP2}/out")
    subprocess.run(["sh", "build.sh", "X", "1", "4", "5e-4", "3e-5", "2", "0"], cwd=K2, env=envx, check=True, capture_output=True)
    px = subprocess.run([sys.executable, "train_seg_pv2.py"], cwd=K2, env=envx, capture_output=True, text=True); assert px.returncode == 0, px.stdout[-1500:] + px.stderr[-1500:]
    line = [l for l in px.stdout.split("\n") if l.startswith("train patches")][0]; assert "(full 22)" in line and "excluded 2 slides" in px.stdout, line
    print("4. slide exclusion OK:", line); shutil.rmtree(TMP2, ignore_errors=True); os.remove(f"{HERE}/_excl_test.json")


# ---------- 5. two packed patch sets (old pack + new pack) read together ----------
NL = chr(10)
def make_packed_set(root, sub, patients, n_val):
    d = f"{root}/{sub}/packed"; os.makedirs(d); rng2 = np.random.default_rng(7 + len(patients)); index, blob = [], bytearray()
    def add(split, patient, k):
        img = (rng2.random((64, 64, 3)) * 255).astype(np.uint8); mask = np.zeros((64, 64), np.uint8)
        for c in range(1, 5):
            if rng2.random() < 0.7: y0, x0 = rng2.integers(0, 40, 2); mask[y0:y0 + 24, x0:x0 + 24] = c
        bi, bm = io.BytesIO(), io.BytesIO(); Image.fromarray(img).save(bi, "JPEG"); Image.fromarray(mask).save(bm, "PNG")
        e = dict(name=f"patient{patient:03d}_wsi1_{k}", split=split, shard=0, io=len(blob), il=len(bi.getvalue())); blob.extend(bi.getvalue())
        e.update(mo=len(blob), ml=len(bm.getvalue())); blob.extend(bm.getvalue()); e["pres"] = sum(1 << (c - 1) for c in range(1, 5) if (mask == c).any()); index.append(e)
    for p in patients:
        for k in range(8): add("train", p, k)
    for k in range(n_val): add("val", 90, k)
    open(f"{d}/shard_000.bin", "wb").write(bytes(blob)); json.dump(index, open(f"{d}/index.json", "w"))
TMP5 = tempfile.mkdtemp(prefix="pv2two_"); IN5 = f"{TMP5}/in"; os.makedirs(IN5)
make_packed_set(IN5, "setA", [1, 2, 25], 8); make_packed_set(IN5, "setB", [3, 4, 26], 0)          # shard files with the SAME name in two folders: must not be mixed up
json.dump({"patients": [25, 26]}, open(f"{TMP5}/shift.json", "w")); os.makedirs(f"{TMP5}/kernels/62_train_pv2"); os.makedirs(f"{TMP5}/src")
for f in ("build.sh", "driver.py"): shutil.copy(f"{HERE}/{f}", f"{TMP5}/kernels/62_train_pv2/{f}")
shutil.copy(f"{SRC}/beetle_seg_pv2.py", f"{TMP5}/src/beetle_seg_pv2.py"); K5 = f"{TMP5}/kernels/62_train_pv2"; os.makedirs(f"{TMP5}/out")
env5 = dict(os.environ, BEETLE_TINY="1", BEETLE_DEV="cpu", BEETLE_AMP="0", BEETLE_WORKERS="0", BEETLE_IN=IN5, BEETLE_OUT=f"{TMP5}/out", SHIFT_JSON=f"{TMP5}/shift.json")
subprocess.run(["sh", "build.sh", "Y", "2", "4", "5e-4", "3e-5", "2", "0"], cwd=K5, env=env5, check=True, capture_output=True)
p5 = subprocess.run([sys.executable, "train_seg_pv2.py"], cwd=K5, env=env5, capture_output=True, text=True); assert p5.returncode == 0, p5.stdout[-1500:] + p5.stderr[-1500:]
lines5 = p5.stdout.split(NL); l5 = [l for l in lines5 if l.startswith("train patches")][0]
assert "2 packed set(s), 56 patches in total" in p5.stdout and "(full 32)" in l5 and "val patches 8 (full 8)" in l5, (l5, [l for l in lines5 if "packed" in l])
print("5. two packed sets OK:", [l for l in lines5 if "packed set(s)" in l][0].split("] ")[-1], "|", l5); shutil.rmtree(TMP5, ignore_errors=True)

print("ALL LOCAL TESTS PASSED")
shutil.rmtree(TMP, ignore_errors=True)
