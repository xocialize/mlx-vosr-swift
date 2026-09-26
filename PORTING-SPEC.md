# mlx-vosr-swift — porting spec (VOSR2 → Swift/MLX)

Reference: `cswry/VOSR` (Apache-2.0), checkpoint `CSWRY/VOSR` `VOSR2/` (1.394 B fp32) + `Qwen-Image-vae-2d/` +
`torch_cache/checkpoints/dinov2_vitl14_pretrain.pth`. Evaluation, gates and the decision to port: `mlxengine-forge/Docs/VOSR-EVAL.md`
(§6 port plan, §10 gate results, §11 text guard). Path B (novel architecture), PyTorch → Swift directly with per-sub-op
goldens from the PyTorch oracle (`oracle/dump_goldens.py`, CPU fp32); no Python-MLX rung (every component has a Swift
donor in the fleet: DiT ports, `QwenVAE.swift`, `DINOv3.swift`).

## Components (isomorphic to upstream `models/`)

| Swift | upstream | notes |
|---|---|---|
| `QwenImageVAE2D` | `models/qwenimage_vae2d.py` | 2D conversion of the Wan/Qwen 3D-causal VAE: RMSNorm2D, ResidualBlock2D, one mid `AttentionBlock2D`, Resample2D; 16-ch latent, `latents_mean/std` normalisation |
| `DINOv2` | torch-hub `dinov2_vitl14` (`vision_transformer.py`, `layers/`) | ViT-L/14, learned pos_embed interpolated (bicubic, `interpolate_offset` quirk) to the 448² grid; LayerScale; **truncated after block 17** — only `layer_17` patch tokens feed the DiT |
| `LightningDiT` | `models/lightningdit.py` + `pos_embed.py` (EVA 2D RoPE) + `swiglu_ffn.py` + `rmsnorm.py` | adaLN-single (`scale_shift_table` + shared `t_block`), RMSNorm, QK-norm, 2D RoPE (rotate_half), SwiGLU (w12/w3), per-block cross-attn to DINOv2 tokens via `layer_norm` + `mlp_ca`; `forward_flexible` (dynamic RoPE for non-training tiles) |
| `VOSRSampler` / `TiledLatentDriver` | `vosr.py` `sample_onestep`, `inference_vosr_onestep.py` `tiled_latent_inference`, `tiled_vae.py` | one Euler step; 512²/32 latent tiles with Gaussian-weighted velocity blending, per-tile DINOv2 features; VAE full-image first, Gaussian-blended tiles as fallback; **colour fix OFF by default** (measured −0.3…−3 SSIMULACRA2) |

## Gate tooling

`vosr-smoke` (Sources/Smoke, built with `xcrun swift build -c release --build-system swiftbuild --target VOSRSmoke`;
binary at `.build/out/Products/Release/vosr-smoke`, AB-L-0171) is the CLI lane:

```
vosr-smoke keys <weightsDir>                                         # S0
vosr-smoke gate <weightsDir> <goldens.safetensors> [--only vae,dino,dit,e2e,tiled] [--gpu]   # S1/S2/S5
vosr-smoke run <in.png> <out.png> <weightsDir> [--scale N] [--bf16] [--seed N] [--cpu] [--vae-tile N]   # S2b
```

Weights = `oracle/convert_weights.py` output (`transformer_{fp32,bf16}` / `vae_{fp32,fp16}` / `dinov2_{fp32,fp16}` +
`config.json`; key contract in its docstring). Goldens = `oracle/dump_goldens.py` output at
`training-resources/Corpus/derived/vosr2-fixtures/goldens/` (42 arrays: VAE encode/decode taps, DINOv2 patch_embed /
block0 / block17 / z, DiT x_embedder / t_embedder / t_block / layer_norm / mlp_ca / block0 internals / blocks 1·17·35 /
final_layer / output, the injected noise, the 512² one-step output, and the 1024² 3×3-tiled output). Gate metric =
max|Δ| relative to the golden's |max| (`rel`) plus cosine; the CPU fp32 lane is the strict one, `--gpu` reports the
Metal gap (TF32 GEMMs on M5, Winograd conv window — the VAE routes its in-window 3×3 convs through conv3d, AB-L-0163).
`Tests/VOSRMLXTests` carries the weight-free structure tests and the env-gated S0 (`VOSR_WEIGHTS=<weightsDir>`) and
PIL-bicubic parity (`VOSR_GOLDENS_DIR`) tests.

## Phase gates (stamp AFTER the run — a row is `pending` until a tool result says otherwise)

| phase | gate | status |
|---|---|---|
| S0 key contract | `vosr-smoke keys`: 0 missing / 0 unused for transformer (849) / vae (186) / dinov2 (256) | **PASSED 2026-09-26** — 849 / 186 / 256 tensors, 1.394 B / 0.045 B / 0.228 B params (`oracle/reports/s0-keys.log`) |
| S1a VAE | `gate --only vae`: moments · latent_raw · latent_norm rel ≤ 2e-3, decode rel ≤ 5e-3 (CPU fp32) | **PASSED 2026-09-26** — quant_conv rel 2.5e-6 · latent_norm 4.1e-6 · decode 5.2e-5 (PSNR 130.6 dB), cos ≥ 0.9999999 (`s1-vae-dino.log`) |
| S1b DINOv2 | `gate --only dino`: input448 rel ≤ 1e-4, patch_embed/block0 ≤ 1e-3, block17/z ≤ 2e-3 | **PASSED 2026-09-26** — input448 rel 4.4e-5 · patch_embed 1.2e-6 · block0 2.8e-6 · block17/z 2.4e-5 (max|Δ| 2.4e-3 on |ref|max 100) (`s1-vae-dino.log`) |
| S1c DiT sub-ops | `gate --only dit`: x_embedder · t_embedder · t_block · layer_norm · mlp_ca · block0.norm1 · block0 · block1 ≤ 1e-3; chained block17/35 ≤ 5e-3; final_layer ≤ 2e-3; output ≤ 5e-3 | **PASSED 2026-09-26** — 12/12, worst rel 8.1e-6 (chained block17, max|Δ| 1.0e-3 on |ref|max 126); output rel 9.0e-7, cos 1.0000000; 8.8 s CPU (`s1c-dit.log`) |
| S2 e2e golden | `gate --only e2e`: torch noise injected; velocity/latent_out ≤ 5e-3, decoded 512² rel ≤ 1e-2 (report PSNR) | **PASSED 2026-09-26** — velocity rel 2.2e-6 · latent_out 4.6e-6 · decoded rel 2.2e-4, **PSNR 115.5 dB** vs the oracle; 9.4 s CPU (`s2-e2e-tiled.log`) |
| S2b GPU eyeball | `run` on GPU per shipping tier, ×1/×2, decoded output beside the PyTorch fixture (`vosr2-fixtures/`) | **PASSED 2026-09-26** (M5 Max) — fp32 ×1 512² 0.4 s / peak 8.44 GB, ×2 →1024² 2.0 s / 10.11 GB; **fp16** ×1 0.3 s / 4.67 GB, ×2 1.6 s / 5.98 GB; bf16 same speed. Eyeball: clean restoration, hoodie lettering rewritten exactly like the PyTorch run (the AB-L-0172 caveat). Fidelity of the half tiers vs the CPU-fp32 oracle (torch noise): **fp16 49.3 dB**, bf16 37.6 dB → bf16 is NOT the shipping tier (8-bit mantissa on a residual stream reaching \|126\|); the reference's own fp16 measures 56–59 dB at ×4 (`oracle/reports/s2b/`, `s6-quant.log`) |
| S5 tiling | `gate --only tiled`: the 1024² 3×3 latent-tile path vs the reference's own tiled output, rel ≤ 1e-2 | **PASSED 2026-09-26** — latent rel 4.2e-6, output rel 1.4e-4, **PSNR 121.8 dB** vs the reference's tiled output (same grid 0/56/64, Gaussian blend, per-tile DINOv2); 61.6 s CPU fp32 for 9 tiles (`s2-e2e-tiled.log`) |
| S6 int8 DiT | `gate --gpu --only quant`: velocity cosine and decoded PSNR vs the fp32 oracle at the golden noise, per tier | **PASSED 2026-09-26** — fp16: velocity cos 0.9999962, **49.3 dB**; **fp16 + int8 DiT** (group 64, every block/cross-attn/mlp_ca/final Linear; the three Sequential-held Linears stay fp16): velocity cos 0.9999927, **46.7 dB**, peak MLX 3.32 GB at 512² / 4.70 GB at 1024² ×2 (vs fp16 4.67 / 5.98), same speed (0.3 s / 1.7 s); bf16: 37.6 dB — rejected. Reference calibration: PyTorch fp16 56–59 dB and int8 50–53 dB vs its fp32 at ×4 (`s6-quant.log`, `oracle/reports/s2b/`) |
| S7 engine wrap | `VOSRUpscalePackage` C0–C14 + MAT-1..5 + CAN-1..3; split footprint measured; registry row | **BUILT + OFFLINE-GREEN 2026-09-26** — `MLXVOSR` (`VOSRConfiguration`: fp16 default / int8 BudgetAware fallback below 7 GB / fp32 parity lane; `WeightSourcing` one role per lane on `mlx-community/VOSR2-fp16` / `-fp32`, explicit `weightsDirectory` honored first; `ModelStorable`, `WeightPrewarming`, `QuantConfigured`; `VOSRUpscalePackage`: Apache/Apache, surface `vosr2-upscale`, ×1–×4 via the Pillow-exact pre-upsample + reflect pad, CAN entry + per-tile + pre-decode checkpoints, `RunProgress` upsample/chunk, `unload → clearCache`). Suites: wrapper 14 Swift Testing + 5 XCTest (MAT-1..5 per lane incl. the flat store layout, CAN-1..3), core 11 XCTest + 3. Real engine (`vosr-smoke engine`, explicit dir): 512²→1024² fp16 2.2 s / peak 5.98 GB, int8 raw-BGRA 2.3 s / 4.70 GB; 1024²→4096² with 1024-px VAE tiles fp16 38.8–69 s / 6.64–7.04 GB, int8 69 s / 5.35 GB (timings vary 2× across the session — sustained-load variance, not re-measured cold); seam check on the tiled 4096² output: blend-band gradient 0.95× / 1.27× interior (no seam). Registry row added to `mlx-engine-swift/docs/model-registry.md` (+ README tracked 79→80), UNCOMMITTED there. **Not done:** repo not pushed/tagged, weights uploading to mlx-community (`oracle/upload_weights.py`), ForgeCore registration + `GuardedUpscaleLiveTests` (AB-T-0185), in-app phys_footprint re-baseline (the manifest's footprints are MLX-peak PROVISIONAL) |

## GPU lane finding — the fp32 Metal gap is TF32, all of it (2026-09-26)

`gate --gpu` (M5 Max, mlx-swift 0.31.6, fp32 weights) with mlx's default `MLX_ENABLE_TF32=1`: 16/25 pass — DINOv2
block17 rel 7.4e-3 (max|Δ| 0.74 on |ref|max 100), DiT chained block17 rel 6.3e-2, decoded e2e **PSNR 58.0 dB**
(`oracle/reports/s2b-gpu-fp32.log`). Same binary with `MLX_ENABLE_TF32=0`: **21/21 pass, e2e PSNR 115.0 dB** —
identical to the CPU lane — at the same speed (DiT lane 0.6 s vs 0.5 s) (`s2b-gpu-fp32-notf32.log`). So an "fp32"
run on M5 is TF32-class unless the PROCESS disables TF32 before its first GEMM; a package cannot do that for its
host. Consequences: the fp32 tier is the parity/reference lane, not a quality promise over bf16 on M5; the shipping
tier is bf16 (S2b/S6 measure it against the CPU-fp32 oracle, which is the honest reference). The VAE's in-window
3×3 convs were already routed through conv3d (AB-L-0163) — with TF32 on they still pass (decode PSNR 99 dB).

## Hazards carried in

- **mlx#3797 NAX split-K GEMM** (mlx-swift ≤ 0.31.6): half-precision GEMMs with a large K corrupt for rows in a
  window (~1024…4522 at K ≈ 12–13.5k in sibling ports). The DiT's largest K is 4608 (`mlp_ca.fc2`) / 4096 (`w3`) at
  1024 rows — probe on the bf16 GPU lane before trusting it; row-chunk ≤ 896 if it bites.
- **RNG**: MLX normal ≠ torch randn; every parity gate injects the torch noise. Production draws `MLXRandom.normal`.
- **Non-square inputs**: `forward_flexible` asserts square DiT inputs; the tile driver always uses square latent tiles
  but a latent smaller than a tile in one dim only (e.g. 400×800 px) would reach the DiT non-square — the wrapper
  must pad to a multiple of 16 px and never hand the core a latent with min side < 64 unless square.
- **CLI tools that drive `MLXServeEngine` must not block the main thread** (`DispatchSemaphore` around a `Task`): the
  engine hops through main-actor monitors and the process deadlocks at 0 % CPU. Use top-level `await` (SE-0343) and an
  explicit `exit(0)` at the end — the engine's actors / Metal threads otherwise keep a finished process alive.
- **`RunProgress.$sink.withValue` around `engine.run` sees no reports** — the engine runs the package outside the caller's
  task-local scope; the package reports (per tile) but a CLI observer must go through the engine's `RunMonitor`.
- **The 2048² full-image VAE peak (13.9 GB) is conv-map-bound, not attention-bound** — chunking the scores did not move it;
  only tiling (or a leaner decoder) does. `vaeWholeFramePixels` = 2048² is the measured line.
- **PIL vs torch bicubic**: the LR→HR upsample is Pillow (a = −0.5, 8-bit fixed point, two quantised passes);
  the DINOv2 448 resize is torch (a = −0.75, float). Both reproduced exactly in `Resample.swift`.
