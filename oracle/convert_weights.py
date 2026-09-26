#!/usr/bin/env python
"""convert_weights.py — per-component MLX weights for mlx-vosr-swift from the upstream checkpoints.

    python convert_weights.py --src <vosr checkout> --out <dir>

Writes transformer_{fp32,fp16,bf16}.safetensors, vae_{fp32,fp16}.safetensors, dinov2_{fp32,fp16}.safetensors + config.json.
Key contract (the Swift modules load these keys with 0 missing / 0 unused):
  • conv weights (4-D) transposed PyTorch (O,I,kh,kw) → MLX (O,kh,kw,I);
  • numeric nn.Sequential children remapped to MLX-native `layers.N` paths (t_embedder.mlp.{0,2} → t_embedder.mlp.layers.{0,2},
    t_block.1 → t_block.layers.1, final_layer.adaLN_modulation.1 → …adaLN_modulation.layers.1, VAE resample.{1} → resample.layers.1);
  • VAE RMSNorm2D gamma (C,1,1) squeezed to (C) (channels-last layout);
  • DINOv2: blocks 0–17 only (the DiT consumes layer_17), mask_token/norm dropped, pos_embed replaced by the model's own
    interpolate_pos_encoding at the fixed 448² input (37² → 32² grid, bicubic, interpolate_offset 0.1) — [1, 1025, 1024];
  • the DiT's feat_rope.freqs_cos/sin buffers are kept (exact 32×32 tables).
Every array is materialised through numpy before saving (no lazy tensors).
"""
import argparse, json, os, sys, re, numpy as np
p = argparse.ArgumentParser(); p.add_argument("--src", required=True); p.add_argument("--out", required=True); a = p.parse_args()
os.chdir(os.path.abspath(a.src)); sys.path.insert(0, os.getcwd())
import torch
from safetensors.torch import load_file
from safetensors.numpy import save_file
torch.hub.set_dir("preset/ckpts/torch_cache")
os.makedirs(a.out, exist_ok=True)

def seq_remap(k):
    k = re.sub(r"^t_embedder\.mlp\.(\d+)\.", r"t_embedder.mlp.layers.\1.", k)
    k = re.sub(r"^t_block\.(\d+)\.", r"t_block.layers.\1.", k)
    k = re.sub(r"^final_layer\.adaLN_modulation\.(\d+)\.", r"final_layer.adaLN_modulation.layers.\1.", k)
    k = re.sub(r"\.resample\.(\d+)\.", r".resample.layers.\1.", k)
    return k
def to_np(t): return np.ascontiguousarray(t.detach().cpu().float().numpy())
def save(name, d, dtype_out):
    out = {}
    for k, v in d.items():
        if dtype_out == "bf16": arr = torch.from_numpy(v).to(torch.bfloat16).view(torch.int16).numpy()   # raw bf16 bits
        elif dtype_out == "fp16": arr = v.astype(np.float16)
        else: arr = v
        out[k] = np.ascontiguousarray(arr)
    path = os.path.join(a.out, name)
    if dtype_out == "bf16":
        # safetensors.numpy has no bf16 dtype; write via torch with real bf16 tensors
        from safetensors.torch import save_file as tsave
        tsave({k: torch.from_numpy(v).to(torch.bfloat16).contiguous() for k, v in d.items()}, path)
    else:
        save_file(out, path)
    print(f"  wrote {name}: {len(out)} tensors, {os.path.getsize(path)/1e9:.2f} GB")

# --- transformer ---
sd = load_file("preset/ckpts/VOSR2/checkpoints/ema_model.safetensors")
tr, n_conv = {}, 0
for k, v in sd.items():
    if v.ndim == 4: v = v.permute(0, 2, 3, 1); n_conv += 1
    tr[seq_remap(k)] = to_np(v)
assert n_conv == 1, n_conv
print(f"transformer: {len(tr)} tensors ({n_conv} conv transposed)")
save("transformer_fp32.safetensors", tr, "fp32"); save("transformer_bf16.safetensors", tr, "bf16"); save("transformer_fp16.safetensors", tr, "fp16")

# --- VAE ---
sd = load_file("preset/ckpts/Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors")
va, n_conv, n_gamma = {}, 0, 0
for k, v in sd.items():
    if v.ndim == 4: v = v.permute(0, 2, 3, 1); n_conv += 1
    elif k.endswith(".gamma") and v.ndim == 3: v = v.reshape(-1); n_gamma += 1
    va[seq_remap(k)] = to_np(v)
print(f"vae: {len(va)} tensors ({n_conv} convs transposed, {n_gamma} gammas squeezed)")
save("vae_fp32.safetensors", va, "fp32"); save("vae_fp16.safetensors", va, "fp16")
vae_cfg = json.load(open("preset/ckpts/Qwen-Image-vae-2d/config.json"))

# --- DINOv2 (blocks 0..17, pos_embed pre-interpolated at 448²) ---
sd = torch.load("preset/ckpts/torch_cache/checkpoints/dinov2_vitl14_pretrain.pth", map_location="cpu", weights_only=True)
venc = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")   # for interpolate_pos_encoding (exact upstream op)
venc.eval()
with torch.no_grad():
    dummy = torch.zeros(1, 1 + (448 // 14) ** 2, 1024)
    pos448 = venc.interpolate_pos_encoding(dummy, 448, 448)          # [1, 1025, 1024]
assert pos448.shape == (1, 1025, 1024), pos448.shape
KEEP_BLOCKS = 18
di = {}
for k, v in sd.items():
    m = re.match(r"^blocks\.(\d+)\.", k)
    if m and int(m.group(1)) >= KEEP_BLOCKS: continue
    if k in ("mask_token", "norm.weight", "norm.bias", "pos_embed"): continue
    if v.ndim == 4: v = v.permute(0, 2, 3, 1)
    di[k] = to_np(v)
di["pos_embed"] = to_np(pos448)
print(f"dinov2: {len(di)} tensors (blocks 0..{KEEP_BLOCKS-1}, pos_embed [1,1025,1024] at 448²)")
save("dinov2_fp32.safetensors", di, "fp32"); save("dinov2_fp16.safetensors", di, "fp16")

cfg = json.load(open("preset/ckpts/VOSR2/args.json"))
json.dump({"transformer": cfg, "vae": vae_cfg,
           "dinov2": {"arch": "vit_large", "patch_size": 14, "embed_dim": 1024, "depth_kept": KEEP_BLOCKS, "num_heads": 16,
                      "mlp_ratio": 4, "layerscale": True, "qkv_bias": True, "ln_eps": 1e-6, "input_size": 448,
                      "feature_layer": cfg["layer_dinov2b_list"][0], "pos_embed": "interpolated to 448² (offset 0.1, bicubic)"},
           "key_contract": "conv (O,kh,kw,I); Sequential children → layers.N; VAE gamma (C); DINOv2 blocks 0..17 only"},
          open(os.path.join(a.out, "config.json"), "w"), indent=1)
print("config.json written")
