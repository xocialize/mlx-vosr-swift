# mlx-vosr-swift

Swift/MLX port of **VOSR2** ([CSWRY/VOSR](https://huggingface.co/CSWRY/VOSR), CVPR 2026, Apache-2.0) — a
vision-only, one-step generative super-resolution model (1.394 B LightningDiT conditioned on DINOv2-L features, in
the Qwen-Image 2-D VAE latent space) — for MLXEngine's `imageUpscale` capability as the **generative stills tier**
beside Real-ESRGAN / RealPLKSR / SeedVR2.

Two products, the fleet shape:

- `VOSRMLX` — engine-agnostic core: `AutoencoderKLQwenImage2D`, `DINOv2ViT` (blocks 0–17), `LightningDiT`,
  `VOSR2Pipeline` (the one-step tiled driver), the two exact bicubic resamplers, the exact-coverage loader.
- `MLXVOSR` — the MLXEngine `ModelPackage` over the core (S7 in `PORTING-SPEC.md`; not yet built).
- `vosr-smoke` — the CLI gate lane (`keys` / `gate` / `run`).

## Status (2026-09-26)

Port gates S0–S6 **passed** on the CPU fp32 lane against the PyTorch oracle (`PORTING-SPEC.md`): key contract
849 / 186 / 256 tensors, every sub-op ≤ 5e-5 relative, end-to-end 512² **115.5 dB**, tiled 1024² **121.9 dB**.
GPU tiers vs that oracle: fp16 **49.3 dB** (the default), fp16 + int8 DiT **46.7 dB** (the budget fallback),
fp32 = the parity lane (TF32 on M5 unless the process sets `MLX_ENABLE_TF32=0` — AB-L-0175); bf16 rejected
(37.6 dB). Timings on an M5 Max: 512² ×1 0.3 s, ×2 1.6 s, ×4 (→2048²) ~6 s.

S7 (`MLXVOSR`): `VOSRUpscalePackage` + `VOSRConfiguration` built with the offline C0–C14 / MAT-1..5 / CAN-1..3
suites green and the engine path exercised by `vosr-smoke engine`. Still to do before it can be consumed:
publish this repo + tag, upload the converted weights to `mlx-community/VOSR2-fp16` / `-fp32` (`oracle/upload_weights.py`),
add the registry row in `mlx-engine-swift/docs/model-registry.md`, register the tier in ForgeCore behind
the text-legibility guard, and run `GuardedUpscaleLiveTests` with VOSR2 as the generative tier.

## Build / gates

```bash
xcrun swift build -c release --build-system swiftbuild --target VOSRSmoke   # → .build/out/Products/Release/vosr-smoke
xcrun swift test -c release --build-system swiftbuild                       # offline suites (XCTest for anything that touches MLX)
vosr-smoke keys  <weightsDir>                                               # S0
vosr-smoke gate  <weightsDir> <goldens.safetensors> [--only …] [--gpu]      # S1/S2/S5/S6
vosr-smoke run   <in.png> <out.png> <weightsDir> [--scale N] [--fp16|--bf16] [--int8]
vosr-smoke engine <in.png> <out.png> <weightsDir> [--scale N] [--int8|--fp32] [--raw]   # the real MLXServeEngine path
```

## Weights

Converted by `oracle/convert_weights.py` from the upstream checkpoints (transformer fp32 5.6 GB / bf16 2.8 GB; VAE
fp32 178 MB / fp16 89 MB; DINOv2 blocks 0–17 fp32 914 MB / fp16 457 MB). Re-hosted on mlx-community (`<UpstreamRepoName>-<quant>` grammar); the S7 wrapper's
`WeightSourcing` points at `mlx-community/VOSR2-fp16` (shipping lane; int8 derived at load) and `mlx-community/VOSR2-fp32` (parity lane).

## Licences

Weights Apache-2.0 (VOSR2 DiT; Qwen-Image VAE from Wan2.1/Qwen-Image; DINOv2). Port code Apache-2.0 — an isomorphic
derivative of the upstream architecture files (`models/lightningdit.py`, `models/qwenimage_vae2d.py`, dinov2
`layers/`) plus EVA-02's MIT `pos_embed.py`. See `LICENSE`.
