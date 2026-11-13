"""CPU tests of the colour-matching inference notebook (kernels/67_probs_norm). usage: python test_local.py
Part A (stain_norm.py): colour conversion against reference values and round trip, identity when the target equals the image's own statistics, matching moves a differently stained image onto the target,
white background far from tissue stays untouched, images without tissue are returned unchanged, the scale ratio is clipped, target statistics from fake packed training sets (excluded slides and the val split are ignored, deterministic).
Part B (build.sh + driver.py with the tiny stand-in backbone, env BEETLE_TINY=1): full mode (zip, saved probabilities, norm_stats.csv, montages, and the 512x512 ROI equals the direct argmax of the model on the COLOUR-MATCHED image),
NORM_ONLY mode (no model, no zip, CPU notebook without checkpoint source), SMOKE mode."""
import io, json, os, shutil, subprocess, sys, tempfile, zipfile
import numpy as np, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import stain_norm as S


def fake_tissue(h, w, base, nuc, seed):                      # white background (240) with one big tissue region: a "base" colour with darker "nucleus" spots
    rng = np.random.default_rng(seed); img = np.full((h, w, 3), 240, np.float32)
    yy, xx = np.mgrid[:h, :w]; inside = (((yy - h / 2) / (h * 0.38)) ** 2 + ((xx - w / 2) / (w * 0.38)) ** 2) < 1
    spots = rng.random((h, w)) < 0.25
    img[inside] = np.array(base, np.float32); img[inside & spots] = np.array(nuc, np.float32)
    img += rng.normal(0, 4, img.shape).astype(np.float32); return np.clip(img, 0, 255).astype(np.uint8), inside


A = dict(base=(225, 150, 195), nuc=(110, 70, 150))            # "training" stain
B = dict(base=(190, 110, 160), nuc=(70, 40, 110))             # darker, more saturated "test" stain

# ---------------- Part A ----------------
red = S.rgb2lab(torch.tensor([[[1.0, 0.0, 0.0]]])).flatten().numpy(); white = S.rgb2lab(torch.tensor([[[1.0, 1.0, 1.0]]])).flatten().numpy()
assert np.allclose(red, [53.24, 80.09, 67.20], atol=0.1), red
assert abs(white[0] - 100) < 0.05 and abs(white[1]) < 0.05 and abs(white[2]) < 0.05, white
rnd = np.random.default_rng(1).integers(0, 256, (300, 300, 3), dtype=np.uint8)
back = (S.lab2rgb(S.rgb2lab(torch.from_numpy(rnd).float() / 255)) * 255).round().to(torch.uint8).numpy()
assert np.abs(back.astype(int) - rnd.astype(int)).max() <= 1, np.abs(back.astype(int) - rnd.astype(int)).max()
print("A1. CIELAB conversion OK (red / white reference values, uint8 round trip max error <= 1)")

imgA, _ = fake_tissue(600, 700, seed=0, **A); imgB, insideB = fake_tissue(600, 700, seed=1, **B)
mA, sA = S.lab_stats(imgA); target = dict(mean=mA.tolist(), std=sA.tolist())
out, info = S.normalise(imgA, target); assert out.dtype == np.uint8 and out.shape == imgA.shape and info is not None
assert np.abs(out.astype(int) - imgA.astype(int)).mean() < 1.0, np.abs(out.astype(int) - imgA.astype(int)).mean()
assert np.allclose(info["scale"], 1.0, atol=1e-6)
print("A2. identity OK: an image whose statistics equal the target is left unchanged (mean abs diff %.3f)" % np.abs(out.astype(int) - imgA.astype(int)).mean())

outB, infoB = S.normalise(imgB, target); mB, _ = S.lab_stats(imgB); mo, so = S.lab_stats(outB)
d0, d1 = S.colour_distance(mB, target), S.colour_distance(mo, target)
assert d1 < 0.3 * d0 and np.abs(mo - mA).max() < 2.5, (d0, d1, mo, mA)
print("A3. matching OK: distance to the target %.2f -> %.2f (mean L/a/b %s -> %s, target %s)" % (d0, d1, np.round(mB, 1).tolist(), np.round(mo, 1).tolist(), np.round(mA, 1).tolist()))
far = np.zeros((600, 700), bool); far[:60, :60] = True; far[-60:, -60:] = True               # corners are > 40 px away from the tissue ellipse
assert (outB[far] == imgB[far]).all(), "background far from tissue changed"
assert np.abs(outB[insideB].astype(int) - imgB[insideB].astype(int)).mean() > 5      # tissue really changed
print("A4. background far from tissue untouched, tissue changed")
white_img = np.full((400, 400, 3), 245, np.uint8); o, i = S.normalise(white_img, target); assert i is None and o is white_img
print("A5. image without tissue returned unchanged")
_, ib = S.normalise(imgB, dict(mean=mA.tolist(), std=(sA * 10).tolist())); assert max(ib["scale"]) <= 2.0 + 1e-9, ib["scale"]
_, ic = S.normalise(imgB, dict(mean=mA.tolist(), std=(sA * 0.01).tolist())); assert min(ic["scale"]) >= 0.5 - 1e-9, ic["scale"]
print("A6. scale ratio clipped to [0.5, 2]")

def make_packed(root, slides):                                # slides: list of (name, split, palette, seed); writes root/packed/{index.json, shard_000.bin}
    os.makedirs(f"{root}/packed"); idx = []; f = open(f"{root}/packed/shard_000.bin", "wb")
    for sname, split, pal, seed in slides:
        for k in range(3):
            im, _ = fake_tissue(256, 256, seed=seed * 10 + k, **pal); b = io.BytesIO(); Image.fromarray(im).save(b, "JPEG", quality=95); ib = b.getvalue()
            mb = io.BytesIO(); Image.fromarray(np.ones((256, 256), np.uint8)).save(mb, "PNG"); mb = mb.getvalue(); io_off = f.tell(); f.write(ib); mo_off = f.tell(); f.write(mb)
            idx.append(dict(split=split, name=f"{sname}_{k}_x", shard=0, io=io_off, il=len(ib), mo=mo_off, ml=len(mb), pres=0))
    f.close(); json.dump(idx, open(f"{root}/packed/index.json", "w"))
TMPA = tempfile.mkdtemp(prefix="pnorm_a_")
make_packed(f"{TMPA}/p1", [("patient1_wsi1", "train", A, 1), ("patient2_wsi1", "train", A, 2), ("patient5_wsi1", "val", B, 5)])
make_packed(f"{TMPA}/p2", [("patient3_wsi1", "train", A, 3), ("patient4_wsi1", "train", B, 4)])
dirs = [f"{TMPA}/p1/packed", f"{TMPA}/p2/packed"]
t_ex = S.target_from_packed(dirs, exclude_slides={"patient4_wsi1"}); t_all = S.target_from_packed(dirs); t_ex2 = S.target_from_packed(dirs, exclude_slides={"patient4_wsi1"})
assert t_ex["n_slides"] == 3 and t_all["n_slides"] == 4 and t_ex["n_excluded"] == 1, (t_ex, t_all)            # the val slide never counts, the excluded slide is left out
assert t_ex == t_ex2, "target statistics are not deterministic"
assert np.abs(np.array(t_ex["mean"]) - mA).max() < 2.0 and np.abs(np.array(t_all["mean"]) - np.array(t_ex["mean"])).max() > 2.0, (t_ex["mean"], mA, t_all["mean"])
print("A7. target statistics OK: 3 train slides (1 excluded, val ignored), deterministic, close to the stain of the used slides (%s vs %s)" % (np.round(t_ex["mean"], 1).tolist(), np.round(mA, 1).tolist()))
shutil.rmtree(TMPA, ignore_errors=True)

old_keys = {f"vit.m.encoder.layer.{i}.attention.{p}": torch.zeros(1) for i in (0, 5) for p in ("attention.query.weight", "attention.query.bias", "attention.key.weight", "attention.value.bias", "output.dense.weight", "output.dense.bias")}
old_keys["vit.m.encoder.layer.0.mlp.fc1.weight"] = torch.zeros(1)
new_names = {k.replace(".attention.attention.query.", ".attention.q_proj.").replace(".attention.attention.key.", ".attention.k_proj.").replace(".attention.attention.value.", ".attention.v_proj.").replace(".attention.output.dense.", ".attention.o_proj."): 1 for k in old_keys}
re_ = S.remap_dino_keys(old_keys, set(new_names)); assert set(re_) == set(new_names) and all(re_[k] is old_keys[k2] for k, k2 in zip(re_, old_keys)), sorted(re_)[:4]
assert S.remap_dino_keys(old_keys, set(old_keys)) is old_keys, "keys the model already knows must be left alone"
assert "vit.m.encoder.layer.0.attention.q_proj.weight" in re_ and "vit.m.encoder.layer.5.attention.o_proj.bias" in re_ and "vit.m.encoder.layer.0.mlp.fc1.weight" in re_
print("A8. Dinov2 key renaming OK (query/key/value/output.dense -> q/k/v/o_proj, other keys untouched, no-op when the model knows the keys)")

# ---------------- Part B ----------------
TMP = tempfile.mkdtemp(prefix="pnorm_b_"); os.makedirs(f"{TMP}/kernels/67"); os.makedirs(f"{TMP}/src"); K = f"{TMP}/kernels/67"; IN = f"{TMP}/in"
for f in ("build.sh", "driver.py", "stain_norm.py"): shutil.copy(f"{HERE}/{f}", f"{K}/{f}")
shutil.copy(f"{HERE}/../../src/beetle_seg_pv2.py", f"{TMP}/src/beetle_seg_pv2.py")
os.environ["BEETLE_TINY"] = "1"; sys.path.insert(0, f"{TMP}/src"); import beetle_seg_pv2 as M
os.makedirs(f"{IN}/fetch/rois"); os.makedirs(f"{IN}/fetch/model"); os.makedirs(f"{IN}/train")
sizes = {"roi_a.png": (700, 900), "roi_b.png": (512, 512), "roi_c.png": (300, 640)}
for i, (n, (h, w)) in enumerate(sizes.items()): Image.fromarray(fake_tissue(h, w, seed=20 + i, **B)[0]).save(f"{IN}/fetch/rois/{n}")
make_packed(f"{IN}/pack_a", [("patient1_wsi1", "train", A, 1), ("patient2_wsi1", "train", A, 2)]); make_packed(f"{IN}/pack_b", [("patient3_wsi1", "train", A, 3), ("patient4_wsi1", "train", B, 4)])
json.dump([{"sid": "patient4_wsi1"}], open(f"{TMP}/exclude.json", "w"))
torch.manual_seed(0); m = M.ViTSeg(M.build_phikon(None)).eval(); torch.save(m.state_dict(), f"{IN}/train/seg_best_t.pt")
env = dict(os.environ, BEETLE_TINY="1", BEETLE_DEV="cpu", BEETLE_IN=IN, BEETLE_N_ROIS="3", EXCLUDE_JSON=f"{TMP}/exclude.json")

def run(name, nt, **kw):
    out = f"{TMP}/out_{name}"; os.makedirs(out, exist_ok=True); e = dict(env, BEETLE_OUT=out, **kw)
    subprocess.run(["sh", "build.sh", name, "beetle-62-train-t", str(nt)], cwd=K, env=e, check=True, capture_output=True)
    meta = json.load(open(f"{K}/kernel-metadata.json"))
    p = subprocess.run([sys.executable, "probs_norm.py"], cwd=K, env=e, capture_output=True, text=True)
    if p.returncode != 0: print(p.stdout[-3000:], p.stderr[-3000:]); raise SystemExit(f"run {name} failed")
    return p.stdout, out, meta

o1, out1, meta1 = run("t1", 1)
z = zipfile.ZipFile(f"{out1}/beetle_own_t1.zip"); assert sorted(z.namelist()) == sorted(sizes), z.namelist()
tgt = json.load(open(f"{out1}/target_stats.json")); assert tgt["n_slides"] == 3 and tgt["n_excluded"] == 1, tgt            # patient4 (stain B) excluded from the target
for n, (h, w) in sizes.items():
    lab = np.array(Image.open(z.open(n))); d = np.load(f"{out1}/probs/{n[:-4]}.npz")["own"]
    assert lab.shape == (h, w) and d.shape == (4, h, w) and d.dtype == np.uint8 and set(np.unique(lab)) <= {1, 2, 3, 4}, (n, lab.shape, d.shape)
img_b = np.array(Image.open(f"{IN}/fetch/rois/roi_b.png").convert("RGB")); norm_b, _ = S.normalise(img_b, tgt, dev="cpu"); assert np.abs(norm_b.astype(int) - img_b.astype(int)).mean() > 3
with torch.no_grad(): direct = (m(torch.from_numpy(norm_b).permute(2, 0, 1)[None].float() / 255).argmax(1)[0] + 1).numpy(); raw = (m(torch.from_numpy(img_b).permute(2, 0, 1)[None].float() / 255).argmax(1)[0] + 1).numpy()
got = np.array(Image.open(z.open("roi_b.png"))); assert (got == direct).all(), "prediction is not the argmax on the colour-matched image"; assert (got != raw).any(), "colour matching changed nothing in the prediction (test is not sensitive)"
import csv
rows = list(csv.DictReader(open(f"{out1}/norm_stats.csv"))); assert len(rows) == 3 and all(float(r["dist_after"]) < 0.5 * float(r["dist_before"]) for r in rows), rows
assert len(os.listdir(f"{out1}/montage")) == 3 and os.path.exists(f"{out1}/target_stats.json")
assert os.path.exists(f"{out1}/check_raw_roi_a.npz") and "forward self-check OK" in o1 and o1.index("forward self-check OK") < o1.index("target colour statistics"), "model self-check must come before the (slow) target statistics"
chk = np.load(f"{out1}/check_raw_roi_a.npz")["own"]; assert chk.shape == (4, 700, 900) and chk.dtype == np.uint8
ks = meta1["kernel_sources"]; assert all(any(s.endswith(x) for s in ks) for x in ("beetle-20-fetch", "beetle-43-pack2", "beetle-46-pack3", "beetle-60-phikonv2-fetch", "beetle-62-train-t")) and meta1["enable_gpu"] is True, meta1
print("B1. full mode OK: zip + probabilities + norm_stats.csv + montages; the 512x512 prediction is the argmax on the COLOUR-MATCHED image (differs from the raw-image argmax); distance to the target halves or better on every ROI")

o2, out2, meta2 = run("t2", 1, NORM_ONLY="1")
assert "NORM_ONLY OK" in o2 and not os.path.exists(f"{out2}/beetle_own_t2.zip") and not os.path.exists(f"{out2}/probs") and os.path.exists(f"{out2}/norm_stats.csv") and len(os.listdir(f"{out2}/montage")) == 3
ks2 = meta2["kernel_sources"]; assert meta2["enable_gpu"] is False and not any(s.endswith(("beetle-60-phikonv2-fetch", "beetle-62-train-t")) for s in ks2), meta2
print("B2. NORM_ONLY mode OK: no model / zip / probabilities, CPU notebook without checkpoint source;", [l for l in o2.split(chr(10)) if "colour matching:" in l][0][:150])

o3, out3, meta3 = run("smoke", 4, SMOKE="2", SMOKE_CPU="1")
assert "SMOKE OK" in o3 and not os.path.exists(f"{out3}/beetle_own_smoke.zip") and meta3["enable_gpu"] is False
print("B3. smoke mode OK:", [l for l in o3.split(chr(10)) if l.startswith("SMOKE OK")][0][:100])
print("ALL LOCAL TESTS PASSED"); shutil.rmtree(TMP, ignore_errors=True)
