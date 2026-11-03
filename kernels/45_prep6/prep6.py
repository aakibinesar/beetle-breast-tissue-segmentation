SMOKE, PART, NPARTS, DEADLINE_MIN = 0, 4, 5, 170
SEL_JSON = r'''[{"sid": "patient105_wsi1", "patient": 105, "wsi": [1864491852, 354703808, 354703808], "mask": [94036925, 394910, 394910], "mb": 338.3, "split": "train"}, {"sid": "patient114_wsi1", "patient": 114, "wsi": [6137906805, 410168036, 410168036], "mask": [115426066, 1995396, 1995396], "mb": 391.2, "split": "train"}, {"sid": "patient117_wsi1", "patient": 117, "wsi": [7581511382, 306555028, 306555028], "mask": [135076137, 1634200, 1634200], "mb": 292.4, "split": "train"}, {"sid": "patient11_wsi1", "patient": 11, "wsi": [8705350241, 808040066, 808040066], "mask": [142952616, 1476512, 1476512], "mb": 770.6, "split": "train"}, {"sid": "patient122_wsi1", "patient": 122, "wsi": [10210316191, 555903582, 555903582], "mask": [152213884, 3187290, 3187290], "mb": 530.2, "split": "train"}, {"sid": "patient124_wsi1", "patient": 124, "wsi": [11173591839, 498801708, 498801708], "mask": [158856272, 2795138, 2795138], "mb": 475.7, "split": "train"}, {"sid": "patient127_wsi1", "patient": 127, "wsi": [12769003068, 552377196, 552377196], "mask": [166414307, 547764, 547764], "mb": 526.8, "split": "train"}, {"sid": "patient127_wsi2", "patient": 127, "wsi": [13321380337, 541117506, 541117506], "mask": [166962138, 489688, 489688], "mb": 516.0, "split": "train"}, {"sid": "patient128_wsi1", "patient": 128, "wsi": [13862497916, 623077952, 623077952], "mask": [167451893, 835326, 835326], "mb": 594.2, "split": "train"}, {"sid": "patient129_wsi1", "patient": 129, "wsi": [14485575941, 585288248, 585288248], "mask": [168287286, 973280, 973280], "mb": 558.2, "split": "train"}, {"sid": "patient12_wsi1", "patient": 12, "wsi": [15070864262, 762186292, 762186292], "mask": [169260633, 1073750, 1073750], "mb": 726.9, "split": "train"}, {"sid": "patient131_wsi1", "patient": 131, "wsi": [16392128699, 562936058, 562936058], "mask": [171142830, 353496, 353496], "mb": 536.9, "split": "train"}, {"sid": "patient132_wsi1", "patient": 132, "wsi": [16955064830, 561517502, 561517502], "mask": [171496393, 638398, 638398], "mb": 535.5, "split": "train"}, {"sid": "patient133_wsi1", "patient": 133, "wsi": [17516582405, 565483992, 565483992], "mask": [172134858, 771378, 771378], "mb": 539.3, "split": "train"}, {"sid": "patient136_wsi1", "patient": 136, "wsi": [19301812204, 555892038, 555892038], "mask": [175059381, 558646, 558646], "mb": 530.1, "split": "train"}, {"sid": "patient137_wsi1", "patient": 137, "wsi": [19857704315, 554527582, 554527582], "mask": [175618094, 546962, 546962], "mb": 528.8, "split": "train"}, {"sid": "patient138_wsi1", "patient": 138, "wsi": [20412231970, 589791460, 589791460], "mask": [176165123, 542030, 542030], "mb": 562.5, "split": "train"}, {"sid": "patient139_wsi1", "patient": 139, "wsi": [21002023503, 776733240, 776733240], "mask": [176707220, 504846, 504846], "mb": 740.8, "split": "train"}, {"sid": "patient140_wsi1", "patient": 140, "wsi": [21978069058, 620284874, 620284874], "mask": [177658621, 469220, 469220], "mb": 591.5, "split": "train"}, {"sid": "patient142_wsi1", "patient": 142, "wsi": [23176189324, 650570256, 650570256], "mask": [178708169, 857824, 857824], "mb": 620.4, "split": "train"}, {"sid": "patient143_wsi1", "patient": 143, "wsi": [23826759653, 542915002, 542915002], "mask": [179566060, 530172, 530172], "mb": 517.8, "split": "train"}, {"sid": "patient145_wsi1", "patient": 145, "wsi": [24989330181, 557311776, 557311776], "mask": [181160418, 478500, 478500], "mb": 531.5, "split": "train"}, {"sid": "patient146_wsi1", "patient": 146, "wsi": [25546642030, 553700634, 553700634], "mask": [181638985, 465068, 465068], "mb": 528.1, "split": "train"}, {"sid": "patient146_wsi2", "patient": 146, "wsi": [26100342737, 556189046, 556189046], "mask": [182104120, 475818, 475818], "mb": 530.4, "split": "train"}, {"sid": "patient147_wsi1", "patient": 147, "wsi": [26656531856, 547480958, 547480958], "mask": [182580005, 759540, 759540], "mb": 522.1, "split": "train"}, {"sid": "patient149_wsi1", "patient": 149, "wsi": [27860565632, 854527926, 854527926], "mask": [183922077, 2865434, 2865434], "mb": 814.9, "split": "train"}, {"sid": "patient150_wsi1", "patient": 150, "wsi": [29402905417, 548672274, 548672274], "mask": [190007258, 551482, 551482], "mb": 523.3, "split": "train"}, {"sid": "patient151_wsi1", "patient": 151, "wsi": [29951577764, 22559026, 22559026], "mask": [190558807, 273466, 273466], "mb": 21.5, "split": "train"}, {"sid": "patient155_wsi2", "patient": 155, "wsi": [30108900266, 15055096, 15055096], "mask": [192394855, 148420, 148420], "mb": 14.4, "split": "train"}, {"sid": "patient158_wsi1", "patient": 158, "wsi": [30175721043, 14135356, 14135356], "mask": [193378462, 195562, 195562], "mb": 13.5, "split": "train"}, {"sid": "patient15_wsi1", "patient": 15, "wsi": [30265862207, 891251282, 891251282], "mask": [194144526, 6046262, 6046262], "mb": 850.0, "split": "train"}, {"sid": "patient167_wsi1", "patient": 167, "wsi": [31503176182, 31600628, 31600628], "mask": [203671147, 360980, 360980], "mb": 30.1, "split": "train"}, {"sid": "patient176_wsi1", "patient": 176, "wsi": [32610000627, 30398948, 30398948], "mask": [210182266, 424222, 424222], "mb": 29.0, "split": "train"}, {"sid": "patient17_wsi1", "patient": 17, "wsi": [32871994843, 994495380, 994495380], "mask": [212507106, 746484, 746484], "mb": 948.4, "split": "train"}, {"sid": "patient182_wsi1", "patient": 182, "wsi": [34026407649, 76600468, 76600468], "mask": [214997322, 170744, 170744], "mb": 73.1, "split": "train"}, {"sid": "patient183_wsi1", "patient": 183, "wsi": [34103008190, 62936472, 62936472], "mask": [215168133, 374578, 374578], "mb": 60.0, "split": "train"}, {"sid": "patient186_wsi1", "patient": 186, "wsi": [34334051783, 44535710, 44535710], "mask": [216840722, 599506, 599506], "mb": 42.5, "split": "train"}, {"sid": "patient186_wsi2", "patient": 186, "wsi": [34378587566, 16755088, 16755088], "mask": [217440295, 428806, 428806], "mb": 16.0, "split": "train"}, {"sid": "patient191_wsi1", "patient": 191, "wsi": [35272962537, 20101106, 20101106], "mask": [221845544, 163058, 163058], "mb": 19.2, "split": "train"}, {"sid": "patient1_wsi1", "patient": 1, "wsi": [36484745101, 724263176, 724263176], "mask": [227635386, 423154, 423154], "mb": 690.7, "split": "train"}, {"sid": "patient200_wsi2", "patient": 200, "wsi": [37246214517, 39214376, 39214376], "mask": [228432458, 381566, 381566], "mb": 37.4, "split": "train"}, {"sid": "patient204_wsi1", "patient": 204, "wsi": [37474742079, 32396190, 32396190], "mask": [229938442, 178854, 178854], "mb": 30.9, "split": "train"}, {"sid": "patient229_wsi1", "patient": 229, "wsi": [40489113772, 11897036, 11897036], "mask": [242303985, 180294, 180294], "mb": 11.3, "split": "train"}, {"sid": "patient229_wsi2", "patient": 229, "wsi": [40501010881, 34238520, 34238520], "mask": [242484346, 176092, 176092], "mb": 32.7, "split": "train"}, {"sid": "patient231_wsi1", "patient": 231, "wsi": [40975562699, 69498086, 69498086], "mask": [243288830, 302180, 302180], "mb": 66.3, "split": "train"}, {"sid": "patient234_wsi1", "patient": 234, "wsi": [41222187268, 164579912, 164579912], "mask": [244837111, 238560, 238560], "mb": 157.0, "split": "train"}, {"sid": "patient235_wsi1", "patient": 235, "wsi": [41386767253, 34706516, 34706516], "mask": [245075738, 282854, 282854], "mb": 33.1, "split": "train"}, {"sid": "patient237_wsi1", "patient": 237, "wsi": [41451552909, 56221966, 56221966], "mask": [245661458, 163162, 163162], "mb": 53.6, "split": "train"}, {"sid": "patient238_wsi1", "patient": 238, "wsi": [41507774948, 16630276, 16630276], "mask": [245824687, 217688, 217688], "mb": 15.9, "split": "train"}, {"sid": "patient239_wsi1", "patient": 239, "wsi": [41524405297, 125350570, 125350570], "mask": [246042442, 317584, 317584], "mb": 119.5, "split": "train"}, {"sid": "patient23_wsi1", "patient": 23, "wsi": [41649755940, 457165766, 457165766], "mask": [246360093, 238824, 238824], "mb": 436.0, "split": "train"}, {"sid": "patient240_wsi1", "patient": 240, "wsi": [42588391682, 35540826, 35540826], "mask": [246962599, 184170, 184170], "mb": 33.9, "split": "train"}, {"sid": "patient241_wsi1", "patient": 241, "wsi": [42623932581, 192189716, 192189716], "mask": [247146836, 228406, 228406], "mb": 183.3, "split": "train"}, {"sid": "patient242_wsi1", "patient": 242, "wsi": [42816122370, 153788258, 153788258], "mask": [247375309, 297164, 297164], "mb": 146.7, "split": "train"}, {"sid": "patient243_wsi1", "patient": 243, "wsi": [42969910701, 23013148, 23013148], "mask": [247672540, 84014, 84014], "mb": 21.9, "split": "train"}, {"sid": "patient245_wsi1", "patient": 245, "wsi": [43039142655, 112405432, 112405432], "mask": [247873488, 248620, 248620], "mb": 107.2, "split": "train"}, {"sid": "patient247_wsi1", "patient": 247, "wsi": [43286092463, 75011174, 75011174], "mask": [248308240, 86516, 86516], "mb": 71.5, "split": "train"}, {"sid": "patient248_wsi1", "patient": 248, "wsi": [43361103710, 36220246, 36220246], "mask": [248394823, 114312, 114312], "mb": 34.5, "split": "train"}, {"sid": "patient260_wsi1", "patient": 260, "wsi": [49556886518, 734293406, 734293406], "mask": [267852967, 4817900, 4817900], "mb": 700.3, "split": "train"}, {"sid": "patient261_wsi1", "patient": 261, "wsi": [50291179997, 490434524, 490434524], "mask": [272670934, 2446762, 2446762], "mb": 467.7, "split": "train"}, {"sid": "patient262_wsi1", "patient": 262, "wsi": [50781614594, 645179448, 645179448], "mask": [275117763, 2399764, 2399764], "mb": 615.3, "split": "train"}, {"sid": "patient263_wsi1", "patient": 263, "wsi": [51426794115, 527543484, 527543484], "mask": [277517594, 1898206, 1898206], "mb": 503.1, "split": "train"}, {"sid": "patient26_wsi1", "patient": 26, "wsi": [54744377566, 300500034, 300500034], "mask": [291102349, 195932, 195932], "mb": 286.6, "split": "train"}, {"sid": "patient26_wsi2", "patient": 26, "wsi": [55044877672, 351116038, 351116038], "mask": [291298347, 232128, 232128], "mb": 334.9, "split": "train"}, {"sid": "patient27_wsi1", "patient": 27, "wsi": [59382466516, 300014706, 300014706], "mask": [312401023, 267392, 267392], "mb": 286.1, "split": "train"}, {"sid": "patient27_wsi3", "patient": 27, "wsi": [59943766568, 214637942, 214637942], "mask": [313158665, 318574, 318574], "mb": 204.7, "split": "train"}, {"sid": "patient281_wsi1", "patient": 281, "wsi": [60481704071, 281750552, 281750552], "mask": [315502874, 175012, 175012], "mb": 268.7, "split": "train"}, {"sid": "patient287_wsi1", "patient": 287, "wsi": [62986681647, 454945418, 454945418], "mask": [327645674, 273078, 273078], "mb": 433.9, "split": "train"}, {"sid": "patient28_wsi1", "patient": 28, "wsi": [64328512668, 373299246, 373299246], "mask": [334570229, 347740, 347740], "mb": 356.0, "split": "train"}, {"sid": "patient292_wsi1", "patient": 292, "wsi": [65981127510, 239111594, 239111594], "mask": [337772883, 169738, 169738], "mb": 228.0, "split": "train"}, {"sid": "patient29_wsi1", "patient": 29, "wsi": [68289828024, 312967082, 312967082], "mask": [351370629, 503292, 503292], "mb": 298.5, "split": "train"}, {"sid": "patient2_wsi1", "patient": 2, "wsi": [68602795178, 723146628, 723146628], "mask": [351873987, 708952, 708952], "mb": 689.6, "split": "train"}, {"sid": "patient30_wsi1", "patient": 30, "wsi": [72702472043, 516189208, 516189208], "mask": [374458050, 478222, 478222], "mb": 492.3, "split": "train"}, {"sid": "patient30_wsi2", "patient": 30, "wsi": [73218661323, 555680866, 555680866], "mask": [374936338, 295210, 295210], "mb": 529.9, "split": "train"}, {"sid": "patient30_wsi3", "patient": 30, "wsi": [73774342261, 524596522, 524596522], "mask": [375231614, 467662, 467662], "mb": 500.3, "split": "train"}, {"sid": "patient30_wsi4", "patient": 30, "wsi": [74298938855, 555432994, 555432994], "mask": [375699342, 295350, 295350], "mb": 529.7, "split": "train"}, {"sid": "patient30_wsi5", "patient": 30, "wsi": [74854371921, 729794234, 729794234], "mask": [375994758, 414828, 414828], "mb": 696.0, "split": "train"}, {"sid": "patient316_wsi1", "patient": 316, "wsi": [77711191785, 492784100, 492784100], "mask": [382056128, 158018, 158018], "mb": 470.0, "split": "train"}, {"sid": "patient31_wsi1", "patient": 31, "wsi": [79613982963, 638128820, 638128820], "mask": [383581102, 531034, 531034], "mb": 608.6, "split": "train"}, {"sid": "patient34_wsi1", "patient": 34, "wsi": [82205759969, 609016308, 609016308], "mask": [386845670, 532438, 532438], "mb": 580.8, "split": "train"}, {"sid": "patient37_wsi1", "patient": 37, "wsi": [83917914975, 684507350, 684507350], "mask": [389965642, 896882, 896882], "mb": 652.8, "split": "train"}, {"sid": "patient38_wsi1", "patient": 38, "wsi": [84602422397, 590326794, 590326794], "mask": [390862590, 725732, 725732], "mb": 563.0, "split": "train"}, {"sid": "patient38_wsi3", "patient": 38, "wsi": [85922128289, 595979182, 595979182], "mask": [392326638, 327138, 327138], "mb": 568.4, "split": "train"}, {"sid": "patient38_wsi4", "patient": 38, "wsi": [86518107543, 911292372, 911292372], "mask": [392653842, 477736, 477736], "mb": 869.1, "split": "train"}, {"sid": "patient39_wsi1", "patient": 39, "wsi": [87429399987, 284164382, 284164382], "mask": [393131644, 971498, 971498], "mb": 271.0, "split": "train"}, {"sid": "patient40_wsi2", "patient": 40, "wsi": [89179973800, 654151258, 654151258], "mask": [400689321, 1030546, 1030546], "mb": 623.8, "split": "train"}, {"sid": "patient41_wsi1", "patient": 41, "wsi": [89834125130, 690938440, 690938440], "mask": [401719933, 336844, 336844], "mb": 658.9, "split": "train"}, {"sid": "patient43_wsi1", "patient": 43, "wsi": [91073109060, 802132408, 802132408], "mask": [404400063, 356290, 356290], "mb": 765.0, "split": "train"}, {"sid": "patient44_wsi1", "patient": 44, "wsi": [91875241540, 416308026, 416308026], "mask": [404756419, 341122, 341122], "mb": 397.0, "split": "train"}, {"sid": "patient45_wsi1", "patient": 45, "wsi": [92291549638, 766873800, 766873800], "mask": [405097607, 674846, 674846], "mb": 731.3, "split": "train"}, {"sid": "patient46_wsi1", "patient": 46, "wsi": [93058423510, 783776896, 783776896], "mask": [405772519, 1453930, 1453930], "mb": 747.5, "split": "train"}, {"sid": "patient53_wsi1", "patient": 53, "wsi": [99437620313, 380375436, 380375436], "mask": [427565512, 112680, 112680], "mb": 362.8, "split": "train"}, {"sid": "patient55_wsi1", "patient": 55, "wsi": [100037380649, 304088934, 304088934], "mask": [428616586, 170254, 170254], "mb": 290.0, "split": "train"}, {"sid": "patient56_wsi1", "patient": 56, "wsi": [100341469655, 381738236, 381738236], "mask": [428786906, 264530, 264530], "mb": 364.1, "split": "train"}, {"sid": "patient59_wsi1", "patient": 59, "wsi": [102398967287, 416660200, 416660200], "mask": [431401360, 234124, 234124], "mb": 397.4, "split": "train"}, {"sid": "patient59_wsi2", "patient": 59, "wsi": [102815627559, 368222122, 368222122], "mask": [431635550, 250154, 250154], "mb": 351.2, "split": "train"}, {"sid": "patient5_wsi1", "patient": 5, "wsi": [103183849753, 774572350, 774572350], "mask": [431885770, 2676250, 2676250], "mb": 738.7, "split": "train"}, {"sid": "patient60_wsi1", "patient": 60, "wsi": [103958422174, 371026456, 371026456], "mask": [434562085, 316978, 316978], "mb": 353.8, "split": "train"}, {"sid": "patient60_wsi2", "patient": 60, "wsi": [104329448702, 397193758, 397193758], "mask": [434879129, 574246, 574246], "mb": 378.8, "split": "train"}, {"sid": "patient63_wsi2", "patient": 63, "wsi": [106818540522, 507877720, 507877720], "mask": [440225015, 316340, 316340], "mb": 484.3, "split": "train"}, {"sid": "patient65_wsi1", "patient": 65, "wsi": [107574058536, 223269804, 223269804], "mask": [441965385, 318798, 318798], "mb": 212.9, "split": "train"}, {"sid": "patient66_wsi1", "patient": 66, "wsi": [107797328412, 503588696, 503588696], "mask": [442284249, 410564, 410564], "mb": 480.3, "split": "train"}, {"sid": "patient68_wsi1", "patient": 68, "wsi": [109346057306, 378889066, 378889066], "mask": [448700309, 578098, 578098], "mb": 361.3, "split": "train"}, {"sid": "patient68_wsi2", "patient": 68, "wsi": [109724946444, 266155166, 266155166], "mask": [449278473, 289110, 289110], "mb": 253.8, "split": "train"}, {"sid": "patient70_wsi2", "patient": 70, "wsi": [111465022433, 506887684, 506887684], "mask": [457304814, 1246770, 1246770], "mb": 483.4, "split": "train"}, {"sid": "patient71_wsi1", "patient": 71, "wsi": [111971910189, 612437368, 612437368], "mask": [458551650, 2238682, 2238682], "mb": 584.1, "split": "train"}, {"sid": "patient77_wsi1", "patient": 77, "wsi": [114724282999, 327251430, 327251430], "mask": [468928652, 549148, 549148], "mb": 312.1, "split": "train"}, {"sid": "patient79_wsi1", "patient": 79, "wsi": [115756148461, 271340354, 271340354], "mask": [471616330, 476100, 476100], "mb": 258.8, "split": "train"}, {"sid": "patient7_wsi1", "patient": 7, "wsi": [116305612077, 654739284, 654739284], "mask": [472631816, 852674, 852674], "mb": 624.4, "split": "train"}, {"sid": "patient81_wsi1", "patient": 81, "wsi": [117307104196, 218874506, 218874506], "mask": [474213461, 130678, 130678], "mb": 208.7, "split": "train"}, {"sid": "patient8_wsi1", "patient": 8, "wsi": [120180223484, 755293918, 755293918], "mask": [484113267, 626838, 626838], "mb": 720.3, "split": "train"}, {"sid": "patient94_wsi1", "patient": 94, "wsi": [122832870111, 220431062, 220431062], "mask": [490350812, 133062, 133062], "mb": 210.2, "split": "train"}, {"sid": "patient9_wsi1", "patient": 9, "wsi": [124509233521, 748718014, 748718014], "mask": [492470162, 3847238, 3847238], "mb": 714.0, "split": "train"}]'''
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
