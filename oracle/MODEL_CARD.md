---
library_name: mlx
license: apache-2.0
license_link: https://github.com/cswry/VOSR/blob/main/LICENSE
pipeline_tag: image-to-image
base_model: CSWRY/VOSR
tags:
  - mlx
  - super-resolution
  - image-restoration
  - vosr
---

# {{REPO}}

**VOSR2** ([CSWRY/VOSR](https://huggingface.co/CSWRY/VOSR), "vision-only" one-step generative super-resolution,
CVPR 2026) converted to MLX in **{{LANE}}** for the Swift/MLX port
[`xocialize/mlx-vosr-swift`](https://github.com/xocialize/mlx-vosr-swift) (MLXEngine `imageUpscale`, the
generative stills tier). Three components, converted by that repo's `oracle/convert_weights.py` from the upstream
checkpoints, per-tensor exact (no re-quantisation on upload):

| file | component | notes |
|---|---|---|
| `transformer_{{LANE}}.safetensors` | LightningDiT 1.394 B (`VOSR2/checkpoints/ema_model.safetensors`) | patch conv (O,kH,kW,I); `nn.Sequential` children under `layers.N`; RoPE tables kept |
| `vae_{{LANE}}.safetensors` | Qwen-Image 2-D VAE (`Qwen-Image-vae-2d`) | convs (O,kH,kW,I); RMSNorm2D gamma squeezed to (C) |
| `dinov2_{{LANE}}.safetensors` | DINOv2 ViT-L/14, blocks 0–17 (`dinov2_vitl14_pretrain.pth`) | only the block the DiT consumes and below; `pos_embed` pre-interpolated for the 448² input |
| `config.json` | the upstream `args.json` + VAE config + the DINOv2 truncation | key contract in the converter's docstring |

Lanes: `mlx-community/VOSR2-fp16` is the shipping lane (the port derives its int8 tier from it at load: group-64
weight-only quantisation of the DiT Linears); `mlx-community/VOSR2-fp32` is the parity / reference lane.
bf16 is not published — measured 37.6 dB against the fp32 reference (the DiT residual stream reaches |126|).

## Parity

Against the upstream PyTorch implementation on CPU fp32 (same input, same noise), decoded 512² output:
fp32 lane **115.5 dB** PSNR (tiled 1024²: 121.9 dB); fp16 lane **49.3 dB**; fp16 + int8 DiT **46.7 dB**.
Details and the per-sub-op table: `PORTING-SPEC.md` in the port repo.

## Use with mlx-vosr-swift

```swift
import MLXServeCore, MLXVOSR
let engine = MLXServeEngine()
let id = try await engine.register(VOSRUpscalePackage.registration, configuration: VOSRConfiguration(quant: .fp16))
let out = try await engine.run(ImageUpscaleRequest(image: image, scale: 2), package: id)   // ×1…×4
```

The engine materializes this repo into its model store on first use (`WeightSourcing`).

## Licences and provenance

Apache-2.0 throughout: VOSR2 (CSWRY/VOSR), the Qwen-Image VAE (Wan 2.1 / Qwen-Image), DINOv2 (Meta). The training
corpus of VOSR2 is not disclosed by its authors; this re-host takes the declared weight licence as governing.
⚠️ Generative: it rewrites letters in already-legible small text — route text-heavy inputs through a legibility guard.
