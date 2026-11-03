# ---------------- CPU check of the Phikon-v2 backbone adapter against the real weights (no GPU) ----------------
import os, sys, time, json
import torch, transformers
print("torch", torch.__version__, "transformers", transformers.__version__, "| cpu threads", torch.get_num_threads(), flush=True)
wdir = next(r for r, d, f in os.walk("/kaggle/input") if "model.safetensors" in f and "config.json" in f); print("weights folder:", wdir, flush=True)
t0 = time.time(); bb = build_phikon(wdir); print(f"backbone built in {time.time() - t0:.0f}s | attention implementation: {bb.m.config._attn_implementation} | hidden {bb.hidden_dim} | blocks {bb.n_layers} | patch {bb.patch} | {sum(p.numel() for p in bb.parameters()) / 1e6:.1f}M params", flush=True)
bb.eval(); ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, extra, flush=True)
with torch.no_grad():
    x = torch.rand(2, 3, 512, 512); x = (x - 0.45) / 0.225
    out = bb.m(pixel_values=x, output_hidden_states=True); hs = out.hidden_states
    check("hidden_states has embeddings + 24 blocks", len(hs) == 25, f"(got {len(hs)})")
    check("hidden state shape [B, 1+32*32, 1024]", tuple(hs[24].shape) == (2, 1 + 32 * 32, 1024), str(tuple(hs[24].shape)))
    f = bb.features(x, (6, 12, 18, 24))
    check("4 feature maps [B,1024,32,32]", len(f) == 4 and all(tuple(t.shape) == (2, 1024, 32, 32) for t in f), str([tuple(t.shape) for t in f]))
    last = out.last_hidden_state[:, 1:].transpose(1, 2).reshape(2, -1, 32, 32)
    check("layer-24 feature == last_hidden_state (final norm applied the same way)", torch.allclose(f[3], last, atol=1e-5), f"max diff {(f[3] - last).abs().max():.2e}")
    check("features finite", all(torch.isfinite(t).all() for t in f))
    xn = torch.rand(1, 3, 512, 768); fn = bb.features(xn, (24,))
    check("non-square 512x768 input -> [1,1024,32,48]", tuple(fn[0].shape) == (1, 1024, 32, 48), str(tuple(fn[0].shape)))
    f1 = bb.features(x[:1], (24,))[0]; check("batch of 1 equals batch of 2 (first item)", torch.allclose(f1, f[3][:1], atol=1e-4), f"max diff {(f1 - f[3][:1]).abs().max():.2e}")
# the whole segmentation model: forward, trainable parts, one backward step on a small input
model = ViTSeg(bb); print("ViTSeg feature layers:", model.layers, "| total params %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
set_trainable(model, 6); tr = lambda m_: sum(p.numel() for p in m_.parameters() if p.requires_grad)
print(f"trainable: backbone {tr(model.vit) / 1e6:.1f}M, decoder {(tr(model) - tr(model.vit)) / 1e6:.1f}M", flush=True)
blocks = model.vit.blocks(); check("only the last 6 blocks + final norm trainable", all(p.requires_grad for b in blocks[-6:] for p in b.parameters()) and not any(p.requires_grad for b in blocks[:-6] for p in b.parameters()) and all(p.requires_grad for p in model.vit.m.layernorm.parameters()) and not any(p.requires_grad for p in model.vit.m.embeddings.parameters()))
model.train(); xs = torch.rand(2, 3, 256, 256); t0 = time.time(); y = model(xs); check("forward [2,4,256,256]", tuple(y.shape) == (2, 4, 256, 256), f"({time.time() - t0:.1f}s)")
loss = torch.nn.functional.cross_entropy(y, torch.randint(0, 4, (2, 256, 256))); loss.backward()
g_last = blocks[-1].attention.attention.query.weight.grad; g_first = blocks[0].attention.attention.query.weight.grad
check("gradient reaches the last trainable block", g_last is not None and g_last.abs().sum() > 0)
check("no gradient in the frozen first block", g_first is None)
# inference path: config-only backbone + state dict round trip
sd = model.state_dict(); m2 = ViTSeg(build_phikon(None)); print(m2.load_state_dict(sd, strict=True), flush=True)
m2.eval(); model.eval()
with torch.no_grad(): d = (m2(xs) - model(xs)).abs().max()
check("inference build (config only) + state dict reproduces the outputs", d < 1e-5, f"max diff {d:.2e}")
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED", flush=True)
