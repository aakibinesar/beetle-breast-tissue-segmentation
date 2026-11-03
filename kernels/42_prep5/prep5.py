SMOKE, PART, NPARTS, DEADLINE_MIN = 0, 3, 4, 170
SEL_JSON = r'''[{"sid": "patient152_wsi1", "patient": 152, "wsi": [29974136863, 12040400, 12040400], "mask": [190832340, 102088, 102088], "mb": 11.5, "split": "train"}, {"sid": "patient196_wsi1", "patient": 196, "wsi": [35639485599, 19116664, 19116664], "mask": [224434376, 217862, 217862], "mb": 18.2, "split": "train"}, {"sid": "patient175_wsi1", "patient": 175, "wsi": [32589623026, 20377528, 20377528], "mask": [209798097, 384102, 384102], "mb": 19.4, "split": "train"}, {"sid": "patient155_wsi1", "patient": 155, "wsi": [30086669059, 22231134, 22231134], "mask": [192008582, 386206, 386206], "mb": 21.2, "split": "train"}, {"sid": "patient174_wsi1", "patient": 174, "wsi": [32566958413, 22664540, 22664540], "mask": [209657174, 140856, 140856], "mb": 21.6, "split": "train"}, {"sid": "patient217_wsi1", "patient": 217, "wsi": [38380741068, 24906740, 24906740], "mask": [235077059, 347380, 347380], "mb": 23.8, "split": "train"}, {"sid": "patient202_wsi1", "patient": 202, "wsi": [37411789777, 28064016, 28064016], "mask": [229401184, 312136, 312136], "mb": 26.8, "split": "train"}, {"sid": "patient205_wsi1", "patient": 205, "wsi": [37507138342, 30073260, 30073260], "mask": [230117363, 277378, 277378], "mb": 28.7, "split": "train"}, {"sid": "patient160_wsi1", "patient": 160, "wsi": [31157113561, 33046386, 33046386], "mask": [200190854, 319368, 319368], "mb": 31.5, "split": "train"}, {"sid": "patient193_wsi1", "patient": 193, "wsi": [35423077146, 33614104, 33614104], "mask": [223119881, 370332, 370332], "mb": 32.1, "split": "train"}, {"sid": "patient200_wsi1", "patient": 200, "wsi": [37209008348, 37206096, 37206096], "mask": [228058605, 373786, 373786], "mb": 35.5, "split": "train"}, {"sid": "patient195_wsi1", "patient": 195, "wsi": [35601709744, 37775782, 37775782], "mask": [224065259, 369050, 369050], "mb": 36.0, "split": "train"}, {"sid": "patient216_wsi1", "patient": 216, "wsi": [38337145191, 43595804, 43595804], "mask": [234652356, 424636, 424636], "mb": 41.6, "split": "train"}, {"sid": "patient167_wsi2", "patient": 167, "wsi": [31534776883, 52282716, 52282716], "mask": [204032194, 639962, 639962], "mb": 49.9, "split": "train"}, {"sid": "patient251_wsi1", "patient": 251, "wsi": [44185938071, 53614394, 53614394], "mask": [249890126, 204372, 204372], "mb": 51.1, "split": "train"}, {"sid": "patient199_wsi1", "patient": 199, "wsi": [35724856910, 56369132, 56369132], "mask": [225474989, 463230, 463230], "mb": 53.8, "split": "train"}, {"sid": "patient162_wsi1", "patient": 162, "wsi": [31277446267, 59944830, 59944830], "mask": [201305776, 505894, 505894], "mb": 57.2, "split": "train"}, {"sid": "patient173_wsi2", "patient": 173, "wsi": [32506648762, 60309578, 60309578], "mask": [209375711, 281396, 281396], "mb": 57.5, "split": "train"}, {"sid": "patient233_wsi1", "patient": 233, "wsi": [41160058277, 62128918, 62128918], "mask": [244204558, 632486, 632486], "mb": 59.3, "split": "train"}, {"sid": "patient190_wsi1", "patient": 190, "wsi": [35206465860, 66496604, 66496604], "mask": [221269887, 575590, 575590], "mb": 63.4, "split": "train"}, {"sid": "patient178_wsi1", "patient": 178, "wsi": [32778273917, 68188160, 68188160], "mask": [211616514, 583488, 583488], "mb": 65.0, "split": "train"}, {"sid": "patient181_wsi1", "patient": 181, "wsi": [33943187148, 83220428, 83220428], "mask": [214121753, 875502, 875502], "mb": 79.4, "split": "train"}, {"sid": "patient223_wsi1", "patient": 223, "wsi": [39740815674, 83421532, 83421532], "mask": [238078807, 639768, 639768], "mb": 79.6, "split": "train"}, {"sid": "patient219_wsi1", "patient": 219, "wsi": [38497862030, 85554270, 85554270], "mask": [235702409, 367748, 367748], "mb": 81.6, "split": "train"}, {"sid": "patient185_wsi1", "patient": 185, "wsi": [34245308824, 88742886, 88742886], "mask": [216240615, 600040, 600040], "mb": 84.6, "split": "train"}, {"sid": "patient227_wsi1", "patient": 227, "wsi": [40303660823, 90646660, 90646660], "mask": [241339834, 311366, 311366], "mb": 86.4, "split": "train"}, {"sid": "patient173_wsi1", "patient": 173, "wsi": [32408198307, 98450382, 98450382], "mask": [208937746, 437898, 437898], "mb": 93.9, "split": "train"}, {"sid": "patient224_wsi1", "patient": 224, "wsi": [39824237279, 107527212, 107527212], "mask": [238718642, 857412, 857412], "mb": 102.5, "split": "train"}, {"sid": "patient225_wsi1", "patient": 225, "wsi": [40150179557, 109008992, 109008992], "mask": [240333666, 608150, 608150], "mb": 104.0, "split": "train"}, {"sid": "patient221_wsi1", "patient": 221, "wsi": [39538651848, 118969910, 118969910], "mask": [237097663, 444218, 444218], "mb": 113.5, "split": "train"}, {"sid": "patient230_wsi1", "patient": 230, "wsi": [40830645996, 144916630, 144916630], "mask": [243018171, 270592, 270592], "mb": 138.2, "split": "train"}, {"sid": "patient194_wsi1", "patient": 194, "wsi": [35456691323, 145018348, 145018348], "mask": [223490280, 574912, 574912], "mb": 138.3, "split": "train"}, {"sid": "patient93_wsi1", "patient": 93, "wsi": [122667058205, 165811834, 165811834], "mask": [490144852, 205894, 205894], "mb": 158.1, "split": "train"}, {"sid": "patient297_wsi1", "patient": 297, "wsi": [67375211115, 171254282, 171254282], "mask": [344422618, 922412, 922412], "mb": 163.3, "split": "train"}, {"sid": "patient13_wsi1", "patient": 13, "wsi": [21778756816, 199312170, 199312170], "mask": [177212133, 446422, 446422], "mb": 190.1, "split": "train"}, {"sid": "patient96_wsi1", "patient": 96, "wsi": [123336254115, 204273772, 204273772], "mask": [490696722, 436676, 436676], "mb": 194.8, "split": "train"}, {"sid": "patient315_wsi1", "patient": 315, "wsi": [77506306664, 204885048, 204885048], "mask": [381799977, 256084, 256084], "mb": 195.4, "split": "train"}, {"sid": "patient10_wsi1", "patient": 10, "wsi": [4599381363, 209906740, 209906740], "mask": [105927072, 596368, 596368], "mb": 200.2, "split": "train"}, {"sid": "patient224_wsi2", "patient": 224, "wsi": [39931764564, 218414920, 218414920], "mask": [239576121, 757478, 757478], "mb": 208.3, "split": "train"}, {"sid": "patient257_wsi1", "patient": 257, "wsi": [46598368507, 245406978, 245406978], "mask": [259194176, 1170628, 1170628], "mb": 234.0, "split": "train"}, {"sid": "patient101_wsi1", "patient": 101, "wsi": [401573081, 248515300, 248515300], "mask": [91474900, 294006, 294006], "mb": 237.0, "split": "train"}, {"sid": "patient58_wsi1", "patient": 58, "wsi": [101514912371, 249901186, 249901186], "mask": [429828526, 767792, 767792], "mb": 238.3, "split": "train"}, {"sid": "patient97_wsi1", "patient": 97, "wsi": [123540527959, 266068160, 266068160], "mask": [491133464, 195052, 195052], "mb": 253.7, "split": "train"}, {"sid": "patient296_wsi1", "patient": 296, "wsi": [67096594542, 278616500, 278616500], "mask": [341999737, 2422814, 2422814], "mb": 265.7, "split": "train"}, {"sid": "patient51_wsi1", "patient": 51, "wsi": [98510591195, 281651578, 281651578], "mask": [425313424, 496120, 496120], "mb": 268.6, "split": "train"}, {"sid": "patient95_wsi1", "patient": 95, "wsi": [123053301245, 282952798, 282952798], "mask": [490483940, 212716, 212716], "mb": 269.8, "split": "train"}, {"sid": "patient87_wsi1", "patient": 87, "wsi": [119363331192, 288789882, 288789882], "mask": [481764079, 1015900, 1015900], "mb": 275.4, "split": "train"}, {"sid": "patient279_wsi1", "patient": 279, "wsi": [59083165421, 299301022, 299301022], "mask": [310918252, 1482704, 1482704], "mb": 285.4, "split": "train"}, {"sid": "patient99_wsi1", "patient": 99, "wsi": [124208436389, 300797060, 300797060], "mask": [491846708, 623388, 623388], "mb": 286.9, "split": "train"}, {"sid": "patient58_wsi3", "patient": 58, "wsi": [102095106785, 303860430, 303860430], "mask": [430994438, 406856, 406856], "mb": 289.8, "split": "train"}, {"sid": "patient121_wsi1", "patient": 121, "wsi": [9904717424, 305598694, 305598694], "mask": [147976117, 4237700, 4237700], "mb": 291.4, "split": "train"}, {"sid": "patient63_wsi1", "patient": 63, "wsi": [106495137026, 323403424, 323403424], "mask": [438930049, 1294900, 1294900], "mb": 308.4, "split": "train"}, {"sid": "patient57_wsi1", "patient": 57, "wsi": [100723207963, 324747050, 324747050], "mask": [429051502, 363230, 363230], "mb": 309.7, "split": "train"}, {"sid": "patient58_wsi2", "patient": 58, "wsi": [101764813629, 330293084, 330293084], "mask": [430596384, 397988, 397988], "mb": 315.0, "split": "train"}, {"sid": "patient110_wsi1", "patient": 110, "wsi": [4809288175, 337484826, 337484826], "mask": [106523506, 1688816, 1688816], "mb": 321.9, "split": "train"}, {"sid": "patient291_wsi1", "patient": 291, "wsi": [65640328481, 340798956, 340798956], "mask": [337210274, 562542, 562542], "mb": 325.0, "split": "train"}, {"sid": "patient80_wsi1", "patient": 80, "wsi": [116960351432, 346752692, 346752692], "mask": [473484555, 728840, 728840], "mb": 330.7, "split": "train"}, {"sid": "patient101_wsi2", "patient": 101, "wsi": [650088454, 364030616, 364030616], "mask": [91768973, 227302, 227302], "mb": 347.2, "split": "train"}, {"sid": "patient107_wsi1", "patient": 107, "wsi": [2829234961, 364263010, 364263010], "mask": [94976628, 398252, 398252], "mb": 347.4, "split": "train"}, {"sid": "patient119_wsi1", "patient": 119, "wsi": [8338678390, 366671778, 366671778], "mask": [139943837, 3008712, 3008712], "mb": 349.7, "split": "train"}, {"sid": "patient85_wsi1", "patient": 85, "wsi": [118769564510, 368910506, 368910506], "mask": [479982713, 721838, 721838], "mb": 351.8, "split": "train"}, {"sid": "patient306_wsi1", "patient": 306, "wsi": [71337627393, 394302644, 394302644], "mask": [364975236, 2284712, 2284712], "mb": 376.0, "split": "train"}, {"sid": "patient312_wsi1", "patient": 312, "wsi": [76400972451, 394941536, 394941536], "mask": [378840822, 1568630, 1568630], "mb": 376.6, "split": "train"}, {"sid": "patient104_wsi1", "patient": 104, "wsi": [1468435463, 396056316, 396056316], "mask": [93459470, 577388, 577388], "mb": 377.7, "split": "train"}, {"sid": "patient100_wsi1", "patient": 100, "wsi": [140, 401572868, 401572868], "mask": [91041553, 433280, 433280], "mb": 383.0, "split": "train"}, {"sid": "patient75_wsi1", "patient": 75, "wsi": [113813036031, 401636810, 401636810], "mask": [466771000, 1549898, 1549898], "mb": 383.0, "split": "train"}, {"sid": "patient77_wsi2", "patient": 77, "wsi": [115051534501, 403165496, 403165496], "mask": [469477866, 1157176, 1157176], "mb": 384.5, "split": "train"}, {"sid": "patient303_wsi1", "patient": 303, "wsi": [70165596128, 406113082, 406113082], "mask": [356982275, 189194, 189194], "mb": 387.3, "split": "train"}, {"sid": "patient62_wsi1", "patient": 62, "wsi": [105583198626, 408726932, 408726932], "mask": [436582657, 841834, 841834], "mb": 389.8, "split": "train"}, {"sid": "patient39_wsi3", "patient": 39, "wsi": [88026008015, 411017456, 411017456], "mask": [394920642, 878238, 878238], "mb": 392.0, "split": "train"}, {"sid": "patient317_wsi1", "patient": 317, "wsi": [78203975958, 414856992, 414856992], "mask": [382214213, 524196, 524196], "mb": 395.6, "split": "train"}, {"sid": "patient28_wsi2", "patient": 28, "wsi": [64701811986, 415933870, 415933870], "mask": [334918035, 423546, 423546], "mb": 396.7, "split": "train"}, {"sid": "patient61_wsi1", "patient": 61, "wsi": [104726642532, 421806968, 421806968], "mask": [435453441, 599244, 599244], "mb": 402.3, "split": "train"}, {"sid": "patient255_wsi1", "patient": 255, "wsi": [45532138723, 423677954, 423677954], "mask": [255363896, 1554076, 1554076], "mb": 404.1, "split": "train"}, {"sid": "patient285_wsi1", "patient": 285, "wsi": [61979383989, 427557770, 427557770], "mask": [323265610, 239718, 239718], "mb": 407.8, "split": "train"}, {"sid": "patient61_wsi2", "patient": 61, "wsi": [105148449572, 434748982, 434748982], "mask": [436052751, 529840, 529840], "mb": 414.6, "split": "train"}, {"sid": "patient318_wsi1", "patient": 318, "wsi": [78618833023, 442963506, 442963506], "mask": [382738476, 586950, 586950], "mb": 422.4, "split": "train"}, {"sid": "patient25_wsi3", "patient": 25, "wsi": [48777790956, 448325562, 448325562], "mask": [266801245, 735644, 735644], "mb": 427.6, "split": "train"}, {"sid": "patient273_wsi1", "patient": 273, "wsi": [56600392323, 448854366, 448854366], "mask": [296670606, 2300144, 2300144], "mb": 428.1, "split": "train"}, {"sid": "patient118_wsi1", "patient": 118, "wsi": [7888066483, 450611834, 450611834], "mask": [136710404, 3233366, 3233366], "mb": 429.7, "split": "train"}, {"sid": "patient57_wsi2", "patient": 57, "wsi": [101047955085, 466957214, 466957214], "mask": [429414798, 413662, 413662], "mb": 445.3, "split": "train"}, {"sid": "patient264_wsi1", "patient": 264, "wsi": [51954337672, 476754512, 476754512], "mask": [279415867, 2579860, 2579860], "mb": 454.7, "split": "train"}, {"sid": "patient23_wsi2", "patient": 23, "wsi": [42106921778, 481469832, 481469832], "mask": [246598983, 363550, 363550], "mb": 459.2, "split": "train"}, {"sid": "patient33_wsi1", "patient": 33, "wsi": [81710645047, 495114850, 495114850], "mask": [385929582, 916022, 916022], "mb": 472.2, "split": "train"}, {"sid": "patient253_wsi1", "patient": 253, "wsi": [44697250801, 496694820, 496694820], "mask": [252081242, 1155082, 1155082], "mb": 473.7, "split": "train"}]'''
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
    SEL = [next(s for s in sm if s["split"] == "val"), next(s for s in sm if s["split"] == "train")]
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
