SMOKE = 0
SEL_JSON = r'''[{"sid": "patient109_wsi1", "patient": 109, "wsi": [4071169228, 528212062, 528212062], "mask": [100975851, 4951154, 4951154], "mb": 503.74227714538574, "split": "val"}, {"sid": "patient126_wsi1", "patient": 126, "wsi": [12168715549, 600287446, 600287446], "mask": [165582724, 831516, 831516], "mb": 572.4787197113037, "split": "val"}, {"sid": "patient171_wsi1", "patient": 171, "wsi": [32331958075, 58164928, 58164928], "mask": [208469530, 277480, 277480], "mb": 55.47039794921875, "split": "val"}, {"sid": "patient218_wsi1", "patient": 218, "wsi": [38405647881, 92214076, 92214076], "mask": [235424506, 277836, 277836], "mb": 87.94219589233398, "split": "val"}, {"sid": "patient226_wsi1", "patient": 226, "wsi": [40259188622, 44472128, 44472128], "mask": [240941883, 397884, 397884], "mb": 42.41192626953125, "split": "val"}, {"sid": "patient236_wsi1", "patient": 236, "wsi": [41421473842, 30078994, 30078994], "mask": [245358659, 302732, 302732], "mb": 28.685564041137695, "split": "val"}, {"sid": "patient247_wsi1", "patient": 247, "wsi": [43286092463, 75011174, 75011174], "mask": [248308240, 86516, 86516], "mb": 71.53623008728027, "split": "val"}, {"sid": "patient270_wsi1", "patient": 270, "wsi": [55395993782, 349852428, 349852428], "mask": [291530541, 1817740, 1817740], "mb": 333.6452751159668, "split": "val"}, {"sid": "patient293_wsi1", "patient": 293, "wsi": [66220239177, 285442196, 285442196], "mask": [337942688, 448072, 448072], "mb": 272.2188911437988, "split": "val"}, {"sid": "patient294_wsi1", "patient": 294, "wsi": [66505681446, 285349378, 285349378], "mask": [338390827, 349280, 349280], "mb": 272.13037300109863, "split": "val"}, {"sid": "patient319_wsi1", "patient": 319, "wsi": [79061796602, 552186288, 552186288], "mask": [383325493, 255542, 255542], "mb": 526.6058807373047, "split": "val"}, {"sid": "patient90_wsi1", "patient": 90, "wsi": [120935517473, 406825072, 406825072], "mask": [484740170, 1364406, 1364406], "mb": 387.97862243652344, "split": "val"}, {"sid": "patient90_wsi2", "patient": 90, "wsi": [121342342617, 412282164, 412282164], "mask": [486104642, 685608, 685608], "mb": 393.18291091918945, "split": "val"}, {"sid": "patient165_wsi1", "patient": 165, "wsi": [31420553878, 26186560, 26186560], "mask": [202775663, 302348, 302348], "mb": 24.97344970703125, "split": "train"}, {"sid": "patient4_wsi1", "patient": 4, "wsi": [97764395064, 338758138, 338758138], "mask": [423309351, 1313930, 1313930], "mb": 323.0649356842041, "split": "train"}, {"sid": "patient148_wsi1", "patient": 148, "wsi": [27204012887, 656552672, 656552672], "mask": [183339612, 582398, 582398], "mb": 626.1374206542969, "split": "train"}, {"sid": "patient276_wsi1", "patient": 276, "wsi": [57839757642, 334954010, 334954010], "mask": [305956561, 305166, 305166], "mb": 319.4370365142822, "split": "train"}, {"sid": "patient203_wsi1", "patient": 203, "wsi": [37439853866, 34888140, 34888140], "mask": [229713387, 224988, 224988], "mb": 33.27192306518555, "split": "train"}, {"sid": "patient20_wsi1", "patient": 20, "wsi": [37777298463, 249680034, 249680034], "mask": [232052062, 274350, 274350], "mb": 238.11343574523926, "split": "train"}, {"sid": "patient89_wsi1", "patient": 89, "wsi": [119799770090, 197551334, 197551334], "mask": [483317415, 321206, 321206], "mb": 188.39963340759277, "split": "train"}, {"sid": "patient89_wsi2", "patient": 89, "wsi": [119997321496, 182901916, 182901916], "mask": [483638687, 474514, 474514], "mb": 174.42885971069336, "split": "train"}, {"sid": "patient161_wsi1", "patient": 161, "wsi": [31190160020, 87286174, 87286174], "mask": [200510289, 795420, 795420], "mb": 83.24258232116699, "split": "train"}, {"sid": "patient42_wsi1", "patient": 42, "wsi": [90525063642, 548045346, 548045346], "mask": [402056843, 2343154, 2343154], "mb": 522.6567707061768, "split": "train"}, {"sid": "patient106_wsi1", "patient": 106, "wsi": [2219195733, 260460242, 260460242], "mask": [94431902, 273756, 273756], "mb": 248.39424324035645, "split": "train"}, {"sid": "patient106_wsi2", "patient": 106, "wsi": [2479656048, 349578840, 349578840], "mask": [94705725, 270836, 270836], "mb": 333.38436126708984, "split": "train"}, {"sid": "patient177_wsi1", "patient": 177, "wsi": [32640399648, 29728888, 29728888], "mask": [210606555, 277618, 277618], "mb": 28.35167694091797, "split": "train"}, {"sid": "patient177_wsi2", "patient": 177, "wsi": [32670128609, 74346680, 74346680], "mask": [210884240, 428040, 428040], "mb": 70.90251922607422, "split": "train"}, {"sid": "patient177_wsi3", "patient": 177, "wsi": [32744475362, 33798482, 33798482], "mask": [211312347, 304100, 304100], "mb": 32.232744216918945, "split": "train"}, {"sid": "patient278_wsi1", "patient": 278, "wsi": [58497656714, 585508634, 585508634], "mask": [308098697, 2819488, 2819488], "mb": 558.3845462799072, "split": "train"}, {"sid": "patient21_wsi1", "patient": 21, "wsi": [38583416373, 431039166, 431039166], "mask": [236070224, 213134, 213134], "mb": 411.0709819793701, "split": "train"}, {"sid": "patient21_wsi2", "patient": 21, "wsi": [39014455611, 446445872, 446445872], "mask": [236283424, 486744, 486744], "mb": 425.7639617919922, "split": "train"}, {"sid": "patient267_wsi1", "patient": 267, "wsi": [53470433847, 340783948, 340783948], "mask": [287174072, 680514, 680514], "mb": 324.99689865112305, "split": "train"}, {"sid": "patient112_wsi1", "patient": 112, "wsi": [5404389563, 388147294, 388147294], "mask": [109842456, 3002290, 3002290], "mb": 370.16610527038574, "split": "train"}, {"sid": "patient295_wsi1", "patient": 295, "wsi": [66791030897, 305563572, 305563572], "mask": [338740174, 3259496, 3259496], "mb": 291.40813064575195, "split": "train"}, {"sid": "patient98_wsi1", "patient": 98, "wsi": [123806596191, 197851800, 197851800], "mask": [491328582, 360404, 360404], "mb": 188.6861801147461, "split": "train"}, {"sid": "patient98_wsi2", "patient": 98, "wsi": [124004448063, 203988254, 203988254], "mask": [491689052, 157590, 157590], "mb": 194.5383586883545, "split": "train"}, {"sid": "patient301_wsi1", "patient": 301, "wsi": [69620323028, 234205208, 234205208], "mask": [353086477, 1006340, 1006340], "mb": 223.3554916381836, "split": "train"}, {"sid": "patient172_wsi1", "patient": 172, "wsi": [32390123076, 18075158, 18075158], "mask": [208747077, 190602, 190602], "mb": 17.23781394958496, "split": "train"}, {"sid": "patient213_wsi1", "patient": 213, "wsi": [38233558758, 21692178, 21692178], "mask": [233908805, 243966, 243966], "mb": 20.687273025512695, "split": "train"}, {"sid": "patient73_wsi1", "patient": 73, "wsi": [113004070925, 487245620, 487245620], "mask": [462899536, 270968, 270968], "mb": 464.67363357543945, "split": "train"}, {"sid": "patient280_wsi1", "patient": 280, "wsi": [60158404582, 323299416, 323299416], "mask": [313477305, 2025502, 2025502], "mb": 308.32234954833984, "split": "train"}, {"sid": "patient246_wsi1", "patient": 246, "wsi": [43151548160, 134544230, 134544230], "mask": [248122175, 185998, 185998], "mb": 128.31137657165527, "split": "train"}, {"sid": "patient252_wsi1", "patient": 252, "wsi": [44239552538, 457698190, 457698190], "mask": [250094565, 1986610, 1986610], "mb": 436.49500846862793, "split": "train"}, {"sid": "patient88_wsi1", "patient": 88, "wsi": [119652121146, 147648872, 147648872], "mask": [482780045, 537304, 537304], "mb": 140.8089370727539, "split": "train"}, {"sid": "patient210_wsi1", "patient": 210, "wsi": [38026978569, 62153448, 62153448], "mask": [232326478, 386312, 386312], "mb": 59.274147033691406, "split": "train"}, {"sid": "patient135_wsi1", "patient": 135, "wsi": [18681482693, 620329438, 620329438], "mask": [173921352, 1137962, 1137962], "mb": 591.5922527313232, "split": "train"}, {"sid": "patient214_wsi1", "patient": 214, "wsi": [38255251009, 59842736, 59842736], "mask": [234152838, 185958, 185958], "mb": 57.07048034667969, "split": "train"}, {"sid": "patient299_wsi1", "patient": 299, "wsi": [67899010009, 390817942, 390817942], "mask": [348276462, 3094100, 3094100], "mb": 372.71303367614746, "split": "train"}, {"sid": "patient120_wsi1", "patient": 120, "wsi": [9513390379, 391326972, 391326972], "mask": [144429194, 3546856, 3546856], "mb": 373.19848251342773, "split": "train"}, {"sid": "patient313_wsi1", "patient": 313, "wsi": [76795914060, 297038338, 297038338], "mask": [380409519, 330008, 330008], "mb": 283.27783393859863, "split": "train"}, {"sid": "patient69_wsi1", "patient": 69, "wsi": [109991101682, 563783178, 563783178], "mask": [449567649, 421290, 421290], "mb": 537.6655368804932, "split": "train"}, {"sid": "patient115_wsi1", "patient": 115, "wsi": [6548074914, 692261444, 692261444], "mask": [117421529, 15938716, 15938716], "mb": 660.1919593811035, "split": "train"}, {"sid": "patient188_wsi1", "patient": 188, "wsi": [34419779488, 22166456, 22166456], "mask": [218639141, 253248, 253248], "mb": 21.13957977294922, "split": "train"}, {"sid": "patient67_wsi1", "patient": 67, "wsi": [108823792246, 522264988, 522264988], "mask": [445328773, 3371470, 3371470], "mb": 498.07070541381836, "split": "train"}, {"sid": "patient74_wsi1", "patient": 74, "wsi": [113491316617, 321719342, 321719342], "mask": [463170570, 3600364, 3600364], "mb": 306.81547355651855, "split": "train"}, {"sid": "patient78_wsi1", "patient": 78, "wsi": [115454700069, 301448320, 301448320], "mask": [470635108, 981156, 981156], "mb": 287.4835205078125, "split": "train"}, {"sid": "patient64_wsi1", "patient": 64, "wsi": [107326418314, 247640150, 247640150], "mask": [440541421, 1423898, 1423898], "mb": 236.1680507659912, "split": "train"}, {"sid": "patient54_wsi1", "patient": 54, "wsi": [99817995821, 219384756, 219384756], "mask": [427678258, 938262, 938262], "mb": 209.22160720825195, "split": "train"}, {"sid": "patient207_wsi1", "patient": 207, "wsi": [37587515084, 13207994, 13207994], "mask": [230699397, 191086, 191086], "mb": 12.596124649047852, "split": "train"}, {"sid": "patient3_wsi1", "patient": 3, "wsi": [88437025543, 214885632, 214885632], "mask": [395798946, 1368972, 1368972], "mb": 204.930908203125, "split": "train"}, {"sid": "patient249_wsi1", "patient": 249, "wsi": [43397324029, 144451972, 144451972], "mask": [248509202, 244024, 244024], "mb": 137.76013565063477, "split": "train"}, {"sid": "patient250_wsi1", "patient": 250, "wsi": [44156683972, 29254026, 29254026], "mask": [249711417, 178642, 178642], "mb": 27.898813247680664, "split": "train"}, {"sid": "patient116_wsi1", "patient": 116, "wsi": [7240336431, 341174878, 341174878], "mask": [133360312, 1715758, 1715758], "mb": 325.36971855163574, "split": "train"}, {"sid": "patient111_wsi1", "patient": 111, "wsi": [5146773074, 257616416, 257616416], "mask": [108212389, 1630000, 1630000], "mb": 245.68215942382812, "split": "train"}, {"sid": "patient228_wsi1", "patient": 228, "wsi": [40394307556, 53411942, 53411942], "mask": [241651267, 316926, 316926], "mb": 50.93759727478027, "split": "train"}, {"sid": "patient228_wsi2", "patient": 228, "wsi": [40447719571, 41394128, 41394128], "mask": [241968260, 335658, 335658], "mb": 39.47651672363281, "split": "train"}, {"sid": "patient300_wsi1", "patient": 300, "wsi": [69325941877, 294381078, 294381078], "mask": [352583004, 503406, 503406], "mb": 280.74367332458496, "split": "train"}, {"sid": "patient36_wsi1", "patient": 36, "wsi": [83533719949, 384194954, 384194954], "mask": [388344544, 1621032, 1621032], "mb": 366.39686012268066, "split": "train"}, {"sid": "patient50_wsi1", "patient": 50, "wsi": [98103153273, 407437850, 407437850], "mask": [424623346, 690012, 690012], "mb": 388.5630130767822, "split": "train"}, {"sid": "patient22_wsi1", "patient": 22, "wsi": [40535249474, 295396450, 295396450], "mask": [242660505, 357600, 357600], "mb": 281.712007522583, "split": "train"}, {"sid": "patient113_wsi1", "patient": 113, "wsi": [5792536930, 345369802, 345369802], "mask": [112844813, 2581186, 2581186], "mb": 329.3703098297119, "split": "train"}, {"sid": "patient130_wsi1", "patient": 130, "wsi": [15833050626, 559078000, 559078000], "mask": [170334449, 808314, 808314], "mb": 533.1783294677734, "split": "train"}, {"sid": "patient244_wsi1", "patient": 244, "wsi": [42992923922, 46218660, 46218660], "mask": [247756621, 116800, 116800], "mb": 44.07754898071289, "split": "train"}, {"sid": "patient32_wsi1", "patient": 32, "wsi": [80252111855, 490991146, 490991146], "mask": [384112202, 655998, 655998], "mb": 468.2456455230713, "split": "train"}, {"sid": "patient32_wsi2", "patient": 32, "wsi": [80743103073, 492588518, 492588518], "mask": [384768266, 434766, 434766], "mb": 469.7690181732178, "split": "train"}, {"sid": "patient32_wsi3", "patient": 32, "wsi": [81235691663, 474953312, 474953312], "mask": [385203098, 726418, 726418], "mb": 452.9507751464844, "split": "train"}, {"sid": "patient282_wsi1", "patient": 282, "wsi": [60763454696, 375236720, 375236720], "mask": [315677953, 1604346, 1604346], "mb": 357.85362243652344, "split": "train"}, {"sid": "patient92_wsi1", "patient": 92, "wsi": [122161448703, 233476756, 233476756], "mask": [489319792, 490066, 490066], "mb": 222.66078567504883, "split": "train"}, {"sid": "patient92_wsi2", "patient": 92, "wsi": [122394925531, 272132602, 272132602], "mask": [489809924, 334862, 334862], "mb": 259.5258731842041, "split": "train"}, {"sid": "patient179_wsi1", "patient": 179, "wsi": [32846462150, 25532620, 25532620], "mask": [212200069, 306970, 306970], "mb": 24.349803924560547, "split": "train"}, {"sid": "patient141_wsi1", "patient": 141, "wsi": [22598354005, 577835246, 577835246], "mask": [178127908, 580194, 580194], "mb": 551.0666332244873, "split": "train"}, {"sid": "patient102_wsi1", "patient": 102, "wsi": [1014119143, 141320522, 141320522], "mask": [91996342, 131584, 131584], "mb": 134.7737522125244, "split": "train"}, {"sid": "patient24_wsi1", "patient": 24, "wsi": [43541776074, 288717844, 288717844], "mask": [248753293, 497428, 497428], "mb": 275.3427925109863, "split": "train"}, {"sid": "patient24_wsi2", "patient": 24, "wsi": [43830493990, 326189910, 326189910], "mask": [249250787, 460564, 460564], "mb": 311.0789394378662, "split": "train"}, {"sid": "patient272_wsi1", "patient": 272, "wsi": [56154250830, 446141420, 446141420], "mask": [295155949, 1514590, 1514590], "mb": 425.4736137390137, "split": "train"}]'''
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
