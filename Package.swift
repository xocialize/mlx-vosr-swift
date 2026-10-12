// swift-tools-version: 6.2
import PackageDescription

// mlx-vosr-swift — VOSR2 (CSWRY/VOSR, CVPR 2026, Apache-2.0): a from-scratch 1.4B vision-only generative
// super-resolution DiT, ported PyTorch → Swift/MLX for MLXEngine as the Forge generative STILLS upscale tier
// (AB-T-0184; evaluation + gates in mlxengine-forge/Docs/VOSR-EVAL.md). ONE repo, TWO products, the
// mlx-realplksr-swift shape:
//   • VOSRMLX — engine-agnostic Swift/MLX core, isomorphic to the upstream repo's models/: the 2D Qwen-Image VAE
//     (models/qwenimage_vae2d.py), DINOv2-L truncated at the consumed block (torch-hub dinov2), LightningDiT with
//     per-block cross-attention (models/lightningdit.py), the one-step flow sampler and the tiled latent driver
//     (inference_vosr_onestep.py). No MLXToolKit dependency.
//   • MLXVOSR — the MLXEngine `imageUpscale` ModelPackage over that core (VOSRUpscalePackage).
// Weights: converted per component (transformer bf16 / vae fp16 / dinov2 fp16) by oracle/convert_weights.py and
// re-hosted under a namespace we control (weight-durability rule). Parity: oracle/dump_goldens.py dumps
// per-sub-op goldens from the PyTorch reference on the CPU; gates run on the CPU stream (S0–S7 in PORTING-SPEC.md).
// Licences: weights Apache-2.0 (VOSR; VAE = Wan2.1 + Qwen-Image Apache-2.0; DINOv2 Apache-2.0), port code Apache-2.0 (an isomorphic derivative of the upstream Apache-2.0 arch files + EVA's MIT pos_embed).
let package = Package(
    name: "mlx-vosr-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "VOSRMLX", targets: ["VOSRMLX"]),
        .library(name: "MLXVOSR", targets: ["MLXVOSR"]),
        .executable(name: "vosr-smoke", targets: ["VOSRSmoke"]),   // CLI gate modes + drive the package
    ],
    dependencies: [
        // ≥ 0.67.0: contract 1.52.0 — `.rawRGBA16` and `ToolDescriptor.rawImageFormats` (16 bits out, AB-T-0233).
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.67.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
        // ≥0.1.1: 0.1.0's exact conv reached an mlx channel-pad helper that frees its zero fill value
        // mid-flight; under a capped MLX cache that killed the host with a Metal "Invalid Resource"
        // fault (AB-T-0214).
        .package(url: "https://github.com/xocialize/mlx-exact-conv-swift", from: "0.1.1"),
    ],
    targets: [
        .target(
            name: "VOSRMLX",
            dependencies: [
                .product(name: "MLXExactConv", package: "mlx-exact-conv-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "MLXVOSR",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                "VOSRMLX",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "VOSRMLXTests",
            dependencies: [
                "VOSRMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXVOSRTests",
            dependencies: [
                "MLXVOSR",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ]
        ),
        .executableTarget(
            name: "VOSRSmoke",
            dependencies: [
                "VOSRMLX",
                "MLXVOSR",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            path: "Sources/Smoke",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
