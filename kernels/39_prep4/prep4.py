SMOKE, PART, NPARTS, DEADLINE_MIN = 0, 4, 5, 170
SEL_JSON = r'''[{"sid": "patient169_wsi1", "patient": 169, "wsi": [31608548633, 23405156, 23405156], "mask": [205367824, 388510, 388510], "mb": 22.3, "split": "val"}, {"sid": "patient304_wsi1", "patient": 304, "wsi": [70571709283, 462750084, 462750084], "mask": [357171536, 4599150, 4599150], "mb": 441.3, "split": "val"}, {"sid": "patient209_wsi1", "patient": 209, "wsi": [37686269664, 91028726, 91028726], "mask": [231637411, 414584, 414584], "mb": 86.8, "split": "val"}, {"sid": "patient166_wsi1", "patient": 166, "wsi": [31446740511, 56435598, 56435598], "mask": [203078078, 593002, 593002], "mb": 53.8, "split": "val"}, {"sid": "patient308_wsi1", "patient": 308, "wsi": [71981064869, 280716754, 280716754], "mask": [370148072, 3661588, 3661588], "mb": 267.7, "split": "val"}, {"sid": "patient153_wsi2", "patient": 153, "wsi": [30027504109, 36680854, 36680854], "mask": [191243136, 366826, 366826], "mb": 35.0, "split": "val"}, {"sid": "patient302_wsi1", "patient": 302, "wsi": [69854528309, 311067746, 311067746], "mask": [354092884, 2889324, 2889324], "mb": 296.7, "split": "val"}, {"sid": "patient232_wsi1", "patient": 232, "wsi": [41045060858, 114997346, 114997346], "mask": [243591077, 613414, 613414], "mb": 109.7, "split": "val"}, {"sid": "patient290_wsi1", "patient": 290, "wsi": [65117745928, 522582480, 522582480], "mask": [335341647, 1868560, 1868560], "mb": 498.4, "split": "val"}, {"sid": "patient168_wsi1", "patient": 168, "wsi": [31587059672, 21488888, 21488888], "mask": [204672223, 695534, 695534], "mb": 20.5, "split": "val"}, {"sid": "patient305_wsi1", "patient": 305, "wsi": [71034459440, 303167880, 303167880], "mask": [361770753, 3204416, 3204416], "mb": 289.1, "split": "train"}, {"sid": "patient286_wsi1", "patient": 286, "wsi": [62406941832, 579739742, 579739742], "mask": [323505395, 4140212, 4140212], "mb": 552.9, "split": "train"}, {"sid": "patient259_wsi1", "patient": 259, "wsi": [47362597957, 678312136, 678312136], "mask": [262679198, 3615318, 3615318], "mb": 646.9, "split": "train"}, {"sid": "patient266_wsi1", "patient": 266, "wsi": [52902028052, 568405722, 568405722], "mask": [284241039, 2932966, 2932966], "mb": 542.1, "split": "train"}, {"sid": "patient49_wsi1", "patient": 49, "wsi": [97220499022, 543895970, 543895970], "mask": [419076505, 4232780, 4232780], "mb": 518.7, "split": "train"}, {"sid": "patient275_wsi1", "patient": 275, "wsi": [57413700167, 426057402, 426057402], "mask": [302130958, 3825536, 3825536], "mb": 406.3, "split": "train"}, {"sid": "patient289_wsi1", "patient": 289, "wsi": [63746392747, 582119848, 582119848], "mask": [329734492, 4835670, 4835670], "mb": 555.2, "split": "train"}, {"sid": "patient47_wsi1", "patient": 47, "wsi": [95205704888, 626859354, 626859354], "mask": [410010763, 6997986, 6997986], "mb": 597.8, "split": "train"}, {"sid": "patient256_wsi1", "patient": 256, "wsi": [45955816750, 642551684, 642551684], "mask": [256918039, 2276070, 2276070], "mb": 612.8, "split": "train"}, {"sid": "TCGA-B6-A0IM-01Z-00-DX1.E2CBD42F-D933-4446-BE7C-C2F8B5AFB976", "patient": null, "wsi": [127031088990, 679647964, 679647964], "mask": [515425127, 4542312, 4542312], "mb": 648.2, "split": "train"}, {"sid": "TCGA-EW-A2FW-01Z-00-DX1.4F948681-81F3-46C5-A4E8-FDE8A4116F7A", "patient": null, "wsi": [129049567832, 469543842, 469543842], "mask": [541436507, 3840920, 3840920], "mb": 447.8, "split": "train"}, {"sid": "TCGA-B6-A0IA-01Z-00-DX1.8A868F0A-6CE2-46A7-AF0C-C5FC5D657D86", "patient": null, "wsi": [126572614534, 458474338, 458474338], "mask": [510052677, 5372338, 5372338], "mb": 437.2, "split": "train"}, {"sid": "TCGA-E2-A15J-01Z-00-DX1.BF7901D1-30B1-4A76-B0A5-E9B8B36EF4C9", "patient": null, "wsi": [128629687004, 419880710, 419880710], "mask": [535230797, 2818704, 2818704], "mb": 400.4, "split": "train"}, {"sid": "patient211_wsi1", "patient": 211, "wsi": [38089132090, 98279406, 98279406], "mask": [232712857, 417922, 417922], "mb": 93.7, "split": "train"}, {"sid": "patient84_wsi1", "patient": 84, "wsi": [118326390744, 443173694, 443173694], "mask": [476003453, 3979194, 3979194], "mb": 422.6, "split": "train"}, {"sid": "patient46_wsi2", "patient": 46, "wsi": [93842200478, 650300564, 650300564], "mask": [407226515, 1395030, 1395030], "mb": 620.2, "split": "train"}, {"sid": "TCGA-C8-A26Z-01Z-00-DX1.1A15D951-F3BD-4024-91DD-E7633C09A837", "patient": null, "wsi": [128143657342, 330448576, 330448576], "mask": [527598573, 2609780, 2609780], "mb": 315.1, "split": "train"}, {"sid": "patient284_wsi1", "patient": 284, "wsi": [61532112834, 447271082, 447271082], "mask": [320635515, 2630028, 2630028], "mb": 426.6, "split": "train"}, {"sid": "patient123_wsi1", "patient": 123, "wsi": [10766219846, 407371920, 407371920], "mask": [155401241, 3454964, 3454964], "mb": 388.5, "split": "train"}, {"sid": "patient180_wsi1", "patient": 180, "wsi": [33866490295, 76696780, 76696780], "mask": [213253656, 868030, 868030], "mb": 73.1, "split": "train"}, {"sid": "patient271_wsi1", "patient": 271, "wsi": [55745846283, 408404474, 408404474], "mask": [293348348, 1807534, 1807534], "mb": 389.5, "split": "train"}, {"sid": "patient52_wsi1", "patient": 52, "wsi": [99014305253, 423314988, 423314988], "mask": [425987984, 1577462, 1577462], "mb": 403.7, "split": "train"}, {"sid": "patient208_wsi1", "patient": 208, "wsi": [37600723151, 85546440, 85546440], "mask": [230890550, 746794, 746794], "mb": 81.6, "split": "train"}, {"sid": "patient254_wsi1", "patient": 254, "wsi": [45193945694, 338192956, 338192956], "mask": [253236391, 2127438, 2127438], "mb": 322.5, "split": "train"}, {"sid": "patient82_wsi1", "patient": 82, "wsi": [117525978774, 502126510, 502126510], "mask": [474344205, 1143100, 1143100], "mb": 478.9, "split": "train"}, {"sid": "patient125_wsi1", "patient": 125, "wsi": [11672393620, 496321856, 496321856], "mask": [161651477, 3931180, 3931180], "mb": 473.3, "split": "train"}, {"sid": "patient309_wsi1", "patient": 309, "wsi": [72261781696, 440690274, 440690274], "mask": [373809727, 648256, 648256], "mb": 420.3, "split": "train"}, {"sid": "patient268_wsi1", "patient": 268, "wsi": [53811217868, 479800992, 479800992], "mask": [287854653, 1835282, 1835282], "mb": 457.6, "split": "train"}, {"sid": "patient307_wsi1", "patient": 307, "wsi": [71731930110, 249134686, 249134686], "mask": [367260015, 2887990, 2887990], "mb": 237.6, "split": "train"}, {"sid": "patient6_wsi1", "patient": 6, "wsi": [110554884932, 358056868, 358056868], "mask": [449989005, 1544100, 1544100], "mb": 341.5, "split": "train"}, {"sid": "patient159_wsi1", "patient": 159, "wsi": [30189856472, 76005662, 76005662], "mask": [193574091, 570368, 570368], "mb": 72.5, "split": "train"}, {"sid": "patient283_wsi1", "patient": 283, "wsi": [61138691489, 393421272, 393421272], "mask": [317282366, 3353082, 3353082], "mb": 375.2, "split": "train"}, {"sid": "patient14_wsi1", "patient": 14, "wsi": [28715093631, 687811714, 687811714], "mask": [186787578, 3219614, 3219614], "mb": 655.9, "split": "train"}, {"sid": "patient108_wsi2", "patient": 108, "wsi": [3620871255, 450297900, 450297900], "mask": [98336872, 2638912, 2638912], "mb": 429.4, "split": "train"}, {"sid": "patient40_wsi1", "patient": 40, "wsi": [88651911246, 528062482, 528062482], "mask": [397167983, 3521272, 3521272], "mb": 503.6, "split": "train"}, {"sid": "patient66_wsi2", "patient": 66, "wsi": [108300917180, 522874994, 522874994], "mask": [442694879, 2633828, 2633828], "mb": 498.7, "split": "train"}, {"sid": "patient189_wsi1", "patient": 189, "wsi": [34441946017, 124474954, 124474954], "mask": [218892456, 830300, 830300], "mb": 118.7, "split": "train"}, {"sid": "patient298_wsi1", "patient": 298, "wsi": [67546465470, 352544466, 352544466], "mask": [345345097, 2931298, 2931298], "mb": 336.2, "split": "train"}, {"sid": "patient274_wsi1", "patient": 274, "wsi": [57049246762, 364453332, 364453332], "mask": [298970817, 3160074, 3160074], "mb": 347.6, "split": "train"}, {"sid": "patient16_wsi1", "patient": 16, "wsi": [31631953862, 666859910, 666859910], "mask": [205756401, 2327312, 2327312], "mb": 636.0, "split": "train"}, {"sid": "patient192_wsi1", "patient": 192, "wsi": [35293063716, 95113796, 95113796], "mask": [222008669, 644948, 644948], "mb": 90.7, "split": "train"}, {"sid": "patient206_wsi1", "patient": 206, "wsi": [37537211675, 50303336, 50303336], "mask": [230394808, 304522, 304522], "mb": 48.0, "split": "train"}, {"sid": "patient48_wsi2", "patient": 48, "wsi": [96504693288, 715805662, 715805662], "mask": [418075769, 1000670, 1000670], "mb": 682.6, "split": "train"}, {"sid": "patient35_wsi1", "patient": 35, "wsi": [82814776349, 718943528, 718943528], "mask": [387378174, 966304, 966304], "mb": 685.6, "split": "train"}, {"sid": "patient222_wsi1", "patient": 222, "wsi": [39657621831, 83193770, 83193770], "mask": [237541948, 536792, 536792], "mb": 79.3, "split": "train"}, {"sid": "patient108_wsi1", "patient": 108, "wsi": [3193498044, 427373138, 427373138], "mask": [95374947, 2961858, 2961858], "mb": 407.6, "split": "train"}, {"sid": "patient46_wsi3", "patient": 46, "wsi": [94492501114, 713203702, 713203702], "mask": [408621611, 1389086, 1389086], "mb": 680.2, "split": "train"}, {"sid": "patient310_wsi1", "patient": 310, "wsi": [75584166227, 469741818, 469741818], "mask": [376409652, 2052184, 2052184], "mb": 448.0, "split": "train"}, {"sid": "patient198_wsi1", "patient": 198, "wsi": [35686793731, 38063106, 38063106], "mask": [225019422, 455500, 455500], "mb": 36.3, "split": "train"}, {"sid": "patient215_wsi1", "patient": 215, "wsi": [38315093818, 22051300, 22051300], "mask": [234338863, 313426, 313426], "mb": 21.0, "split": "train"}, {"sid": "patient19_wsi1", "patient": 19, "wsi": [35781226115, 703518914, 703518914], "mask": [225938286, 1697034, 1697034], "mb": 670.9, "split": "train"}, {"sid": "patient192_wsi2", "patient": 192, "wsi": [35388177585, 34899488, 34899488], "mask": [222653684, 466130, 466130], "mb": 33.3, "split": "train"}, {"sid": "patient48_wsi1", "patient": 48, "wsi": [95832564314, 672128902, 672128902], "mask": [417008815, 1066888, 1066888], "mb": 641.0, "split": "train"}, {"sid": "patient187_wsi1", "patient": 187, "wsi": [34395342727, 24436688, 24436688], "mask": [217869168, 769906, 769906], "mb": 23.3, "split": "train"}, {"sid": "patient212_wsi1", "patient": 212, "wsi": [38187411569, 46147116, 46147116], "mask": [233130846, 777892, 777892], "mb": 44.0, "split": "train"}, {"sid": "patient277_wsi1", "patient": 277, "wsi": [58174711725, 322944916, 322944916], "mask": [306261794, 1836836, 1836836], "mb": 308.0, "split": "train"}, {"sid": "patient134_wsi1", "patient": 134, "wsi": [18082066470, 599416150, 599416150], "mask": [172906303, 1014982, 1014982], "mb": 571.6, "split": "train"}, {"sid": "patient156_wsi1", "patient": 156, "wsi": [30123955435, 21386826, 21386826], "mask": [192543342, 544590, 544590], "mb": 20.4, "split": "train"}, {"sid": "patient157_wsi1", "patient": 157, "wsi": [30145342334, 30378636, 30378636], "mask": [193087999, 290396, 290396], "mb": 29.0, "split": "train"}, {"sid": "patient144_wsi1", "patient": 144, "wsi": [24369674728, 619655380, 619655380], "mask": [180096299, 1064052, 1064052], "mb": 590.9, "split": "train"}, {"sid": "patient314_wsi1", "patient": 314, "wsi": [77092952471, 413354120, 413354120], "mask": [380739594, 1060316, 1060316], "mb": 394.2, "split": "train"}, {"sid": "patient83_wsi1", "patient": 83, "wsi": [118028105356, 298285316, 298285316], "mask": [475487371, 516016, 516016], "mb": 284.5, "split": "train"}, {"sid": "patient91_wsi1", "patient": 91, "wsi": [121754624853, 406823778, 406823778], "mask": [486790316, 2529410, 2529410], "mb": 388.0, "split": "train"}, {"sid": "patient258_wsi1", "patient": 258, "wsi": [46843775558, 518822326, 518822326], "mask": [260364871, 2314260, 2314260], "mb": 494.8, "split": "train"}, {"sid": "patient62_wsi2", "patient": 62, "wsi": [105991925630, 503211324, 503211324], "mask": [437424557, 1505426, 1505426], "mb": 479.9, "split": "train"}, {"sid": "patient27_wsi2", "patient": 27, "wsi": [59682481294, 261285202, 261285202], "mask": [312668481, 490118, 490118], "mb": 249.2, "split": "train"}, {"sid": "patient70_wsi1", "patient": 70, "wsi": [110912941871, 552080490, 552080490], "mask": [451533170, 5771578, 5771578], "mb": 526.5, "split": "train"}, {"sid": "patient269_wsi1", "patient": 269, "wsi": [54291018933, 453358560, 453358560], "mask": [289690002, 1412280, 1412280], "mb": 432.4, "split": "train"}, {"sid": "patient39_wsi2", "patient": 39, "wsi": [87713564441, 312443502, 312443502], "mask": [394103208, 817368, 817368], "mb": 298.0, "split": "train"}, {"sid": "patient79_wsi2", "patient": 79, "wsi": [116027488887, 278123118, 278123118], "mask": [472092496, 539254, 539254], "mb": 265.2, "split": "train"}, {"sid": "patient18_wsi1", "patient": 18, "wsi": [34566421044, 640044744, 640044744], "mask": [219722823, 1546998, 1546998], "mb": 610.4, "split": "train"}, {"sid": "patient103_wsi1", "patient": 103, "wsi": [1155439738, 312995652, 312995652], "mask": [92127993, 1331410, 1331410], "mb": 298.5, "split": "train"}, {"sid": "patient184_wsi1", "patient": 184, "wsi": [34165944735, 79364016, 79364016], "mask": [215542778, 697770, 697770], "mb": 75.7, "split": "train"}, {"sid": "patient86_wsi1", "patient": 86, "wsi": [119138475088, 224856032, 224856032], "mask": [480704617, 1059396, 1059396], "mb": 214.4, "split": "train"}, {"sid": "patient197_wsi1", "patient": 197, "wsi": [35658602336, 28191322, 28191322], "mask": [224652305, 367050, 367050], "mb": 26.9, "split": "train"}, {"sid": "patient265_wsi1", "patient": 265, "wsi": [52431092257, 470935722, 470935722], "mask": [281995794, 2245178, 2245178], "mb": 449.1, "split": "train"}, {"sid": "patient38_wsi2", "patient": 38, "wsi": [85192749263, 729378954, 729378954], "mask": [391588388, 738184, 738184], "mb": 695.6, "split": "train"}, {"sid": "patient201_wsi1", "patient": 201, "wsi": [37285428966, 126360738, 126360738], "mask": [228814091, 587026, 587026], "mb": 120.5, "split": "train"}, {"sid": "patient311_wsi1", "patient": 311, "wsi": [76053908118, 347064260, 347064260], "mask": [378461903, 378852, 378852], "mb": 331.0, "split": "train"}, {"sid": "patient220_wsi1", "patient": 220, "wsi": [39460901555, 77750220, 77750220], "mask": [236770234, 327362, 327362], "mb": 74.1, "split": "train"}, {"sid": "patient154_wsi1", "patient": 154, "wsi": [30064185036, 22483950, 22483950], "mask": [191610029, 398486, 398486], "mb": 21.4, "split": "train"}, {"sid": "patient76_wsi1", "patient": 76, "wsi": [114214672913, 509610014, 509610014], "mask": [468320964, 607622, 607622], "mb": 486.0, "split": "train"}, {"sid": "patient72_wsi1", "patient": 72, "wsi": [112584347629, 419723224, 419723224], "mask": [460790398, 2109072, 2109072], "mb": 400.3, "split": "train"}, {"sid": "patient288_wsi1", "patient": 288, "wsi": [63441627138, 304765536, 304765536], "mask": [327918819, 1815606, 1815606], "mb": 290.6, "split": "train"}, {"sid": "patient163_wsi1", "patient": 163, "wsi": [31337391170, 19486234, 19486234], "mask": [201811737, 453490, 453490], "mb": 18.6, "split": "train"}, {"sid": "patient164_wsi1", "patient": 164, "wsi": [31356877477, 63676328, 63676328], "mask": [202265294, 510302, 510302], "mb": 60.7, "split": "train"}, {"sid": "patient170_wsi1", "patient": 170, "wsi": [32298813844, 33144158, 33144158], "mask": [208083779, 385684, 385684], "mb": 31.6, "split": "train"}]'''
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
