"""Choose the stain-shift validation patients for the BEETLE retrain.

Inputs (local): out_stain2/slides_stain.csv (stain stats of ALL slides of beetle-43-pack2, from the CPU notebook beetle-52-stain2),
out_stain/rois_stain.csv (170 evaluation ROIs), out_stain/train_stain.csv (the 154 slides the v2 model was trained on), out_pack2/packed/index.json.
Rule: the held-out patients must be NEW (no slide of theirs was seen by the warm-start model v2 and none is in the normal validation set), and
their stain should be about as far from the remaining training slides as the evaluation ROIs are (median ROI distance), so the validation mimics the real shift.
Output: shift.json {"patients": [...]} + a short report.   Usage: python select_shift.py [max_patch_share=0.25] [min_patients=8]
"""
import json, re, sys, collections
import numpy as np, pandas as pd

MAX_SHARE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
MIN_PAT = int(sys.argv[2]) if len(sys.argv) > 2 else 8
FEATS = ["r", "g", "b", "h", "e", "sat", "contrast"]                       # same colour / stain features as the earlier analysis (beetle/roi_stain_vs_agreement.csv)

sl = pd.read_csv("out_stain2/slides_stain.csv"); rois = pd.read_csv("out_stain/rois_stain.csv"); old = pd.read_csv("out_stain/train_stain.csv")
index = json.load(open("out_pack2/packed/index.json"))
n_patch = collections.Counter(); split_of = {}
for e in index:
    s = re.match(r"(patient\d+_wsi\d+)", e["name"]).group(1); n_patch[(s, e["split"])] += 1; split_of[s] = e["split"]
sl["n_patch"] = [n_patch[(s, split_of[s])] for s in sl.slide]; sl["split"] = [split_of[s] for s in sl.slide]
old_slides = set(old.slide); sl["seen_by_v2"] = sl.slide.isin(old_slides)
val_pat = set(sl[sl.split == "val"].patient); seen_pat = set(sl[sl.seen_by_v2].patient)
sl["candidate"] = (sl.split == "train") & (~sl.seen_by_v2) & (~sl.patient.isin(val_pat | seen_pat))                          # patient-level: no slide of the patient seen by v2 or in val
pat_ok = sl.groupby("patient").candidate.all(); cand_pat = sorted(pat_ok[pat_ok].index)
print(f"slides {len(sl)} (train {int((sl.split=='train').sum())}, val {int((sl.split=='val').sum())}) | seen by v2: {int(sl.seen_by_v2.sum())} | new train slides: {int(((sl.split=='train') & ~sl.seen_by_v2).sum())} | candidate patients (all slides new): {len(cand_pat)}")

mu, sd = sl[FEATS].mean(), sl[FEATS].std().replace(0, 1)
z = lambda d: ((d[FEATS] - mu) / sd).to_numpy()
Zs, Zr = z(sl), z(rois)
def nn_dist(a, ref): return np.sqrt(((a[:, None, :] - ref[None, :, :]) ** 2).sum(-1)).min(1)
roi_med = float(np.median(nn_dist(Zr, Zs[(sl.split == "train").to_numpy()]))); print(f"evaluation ROIs: median distance to the nearest training slide {roi_med:.2f}")

rows = []
for p in cand_pat:
    m = (sl.patient == p).to_numpy(); others = (sl.split == "train").to_numpy() & ~m
    d = float(nn_dist(Zs[m], Zs[others]).mean()); rows.append(dict(patient=p, slides=int(m.sum()), patches=int(sl.n_patch[m].sum()), dist=d, h=float(sl.h[m].mean())))
c = pd.DataFrame(rows).sort_values("dist", ascending=False).reset_index(drop=True)
budget = MAX_SHARE * float(sl[(sl.split == "train") & (~sl.seen_by_v2)].n_patch.sum())
far = c[c.dist >= roi_med]; chosen = []; tot = 0
for _, r in (far.sort_values("dist") if len(far) >= MIN_PAT else c).iterrows():          # nearest to the ROI-like distance first (typical shift, not the extreme outliers)
    if tot + r.patches > budget and len(chosen) >= MIN_PAT: continue
    chosen.append(int(r.patient)); tot += int(r.patches)
sel = c[c.patient.isin(chosen)]
print(f"candidates at least as far as the ROI median ({roi_med:.2f}): {len(far)} of {len(c)} | chosen {len(chosen)} patients, {int(sel.slides.sum())} slides, {tot} patches "
      f"({tot / float(sl[(sl.split == 'train') & (~sl.seen_by_v2)].n_patch.sum()):.0%} of the new patches; budget {MAX_SHARE:.0%})")
print(f"chosen: distance mean {sel.dist.mean():.2f} (candidates overall {c.dist.mean():.2f}); haematoxylin mean {sel.h.mean():.3f} vs training slides {sl[sl.split=='train'].h.mean():.3f} vs ROIs {rois.h.mean():.3f}")
json.dump(dict(patients=sorted(chosen), n_slides=int(sel.slides.sum()), n_patches=int(tot), rule=f"new patients, NN stain distance >= ROI median {roi_med:.2f}, <= {MAX_SHARE:.0%} of new patches"), open("shift.json", "w"), indent=1)
c.to_csv("shift_candidates.csv", index=False); print("wrote shift.json, shift_candidates.csv")
