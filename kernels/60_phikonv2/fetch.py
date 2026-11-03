# CPU notebook: download owkin/phikon-v2 (Hugging Face, ViT-L/16 DINOv2-style pathology backbone) once and keep the files as this notebook's OUTPUT,
# so that later (GPU) notebooks mount them offline. No token is used: if the repo turns out to be gated the download stops with a clear message.
# Licence: Owkin non-commercial licence (see the saved README.md); the weights stay private and are not redistributed.
import os, json, time, hashlib, sys
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
T0 = time.time()
def stamp(m): print(f"[t+{time.time() - T0:5.0f}s] {m}", flush=True)

from huggingface_hub import snapshot_download, list_repo_files
REPO, DEST = "owkin/phikon-v2", "/kaggle/working/phikon-v2"
try:
    files = list_repo_files(REPO); stamp(f"{REPO} files: {files}")
    path = snapshot_download(REPO, local_dir=DEST)
except Exception as e:
    print("DOWNLOAD FAILED:", type(e).__name__, str(e)[:400], "\n-> if this says 401 / gated: the repo needs accepted terms + a Hugging Face token; the user has to download the files and upload them as a Kaggle Model.", flush=True); os._exit(2)
stamp(f"downloaded to {path}")

tot = 0
for r, d, fs in os.walk(DEST):
    d[:] = [x for x in d if x != ".cache"]
    for f in sorted(fs):
        p = os.path.join(r, f); n = os.path.getsize(p); tot += n
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] if n < 3 * 2**30 else "(skipped)"
        print(f"  {os.path.relpath(p, DEST):45s} {n / 2**20:9.1f} MB  sha256[:16] {h}", flush=True)
stamp(f"total {tot / 2**30:.2f} GB")
for name in ("config.json", "preprocessor_config.json"):
    p = os.path.join(DEST, name)
    if os.path.exists(p): print(f"--- {name}\n" + open(p, encoding="utf-8").read()[:1800], flush=True)
rd = os.path.join(DEST, "README.md")
if os.path.exists(rd): print("--- README.md (first 2500 chars: licence / usage)\n" + open(rd, encoding="utf-8").read()[:2500], flush=True)

# sanity load on CPU: architecture, parameter count, output shape on a random 224 px input (no weights are modified)
try:
    import torch
    from transformers import AutoModel
    m = AutoModel.from_pretrained(DEST).eval()
    n = sum(p.numel() for p in m.parameters()); x = torch.randn(1, 3, 224, 224)
    with torch.no_grad(): o = m(pixel_values=x, output_hidden_states=True)
    cfg = m.config
    print(f"model class {type(m).__name__} | parameters {n / 1e6:.1f}M | hidden {cfg.hidden_size} | layers {cfg.num_hidden_layers} | heads {cfg.num_attention_heads} | patch {getattr(cfg, 'patch_size', '?')} | "
          f"image {getattr(cfg, 'image_size', '?')} | last_hidden_state {tuple(o.last_hidden_state.shape)} | hidden_states {len(o.hidden_states)} x {tuple(o.hidden_states[-1].shape)}", flush=True)
    x2 = torch.randn(1, 3, 512, 512)
    with torch.no_grad(): o2 = m(pixel_values=x2, output_hidden_states=True)
    print(f"512 px input works: last_hidden_state {tuple(o2.last_hidden_state.shape)} (tokens incl. CLS/register tokens)", flush=True)
except Exception as e:
    import traceback; traceback.print_exc(); print("sanity load failed (files are still saved):", repr(e)[:300], flush=True)
print("DONE", flush=True); os._exit(0)
