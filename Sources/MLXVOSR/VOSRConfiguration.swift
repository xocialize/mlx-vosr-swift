import Foundation
import MLXToolKit
import VOSRMLX

/// Init-time configuration for `VOSRUpscalePackage` (C9). Stable for the session.
///
/// VOSR2 ships ONE weight set per precision lane (`mlx-community/VOSR2-fp16`, `-fp32`); the int8 tier is produced at
/// load from the fp16 transformer (group-64 weight-only quantization of every block / cross-attention /
/// `mlp_ca` Linear — measured 46.7 dB vs the fp32 oracle, fp16 49.3 dB; PORTING-SPEC S6), so `quant`
/// selects the files (fp16 or fp32) and whether the DiT is quantized after loading:
///
///   quant   files            resident (weights)   measured MLX peak, 1024² out
///   .fp16   *_fp16 (3.35 GB) ≈ 3.4 GB              5.98 GB
///   .int8   *_fp16 + int8    ≈ 2.0 GB              4.70 GB   ← the BudgetAware fallback
///   .fp32   *_fp32 (6.7 GB)  ≈ 6.7 GB             10.11 GB   (the parity lane; TF32 on M5 — AB-L-0175)
///
/// bf16 is deliberately NOT offered (37.6 dB: an 8-bit mantissa on a residual stream that reaches |126|).
public struct VOSRConfiguration: PackageConfiguration, ModelStorable, QuantConfigured, BudgetAware {
    /// `.fp16` (default), `.int8` or `.fp32`; anything else is treated as `.fp16`.
    public var quant: Quant
    /// Deterministic per seed (MLX RNG; the reference's torch noise is only reproducible in the gates).
    public var seed: UInt64
    /// Integer scale when a request omits one (1…4). VOSR2 restores at ×1 and upsamples ×2/×4 through a
    /// Pillow-exact bicubic pre-upsample — the reference pipeline's own input path.
    public var defaultScale: Int
    /// DiT tile geometry in output pixels (the reference's 512 / 32; the latent tile is 64 / 8).
    public var tileSize: Int
    public var tileOverlap: Int
    /// Above this many OUTPUT pixels the VAE encodes/decodes in Gaussian-blended tiles of `vaeTileSize`
    /// (the reference's `--vae_tile_size` lever). The VAE's mid-block attention is single-head over EVERY
    /// latent position, O(N²): a 2048² output (256² latent) is the largest full-image size measured
    /// (13.9 GB MLX peak before query-chunking), and a 4096² one asks Metal for a 137 GB score matrix.
    /// Default 2048² keeps every measured envelope on the seam-free path and tiles above it.
    public var vaeWholeFramePixels: Int
    public var vaeTileSize: Int

    /// Absolute path to the converted weights (`oracle/convert_weights.py` layout). **Honored OVER the
    /// engine-stamped store root** — dev-mode escape hatch, never touches the network.
    public var weightsDirectory: URL?
    /// Where the engine materializes weights — stamped from its `ModelStore.root` (`ModelStorable`).
    public var modelsRootDirectory: URL?
    /// Real headroom at load time, stamped by the governor (`BudgetAware`): below `fp16MinBudgetBytes`
    /// an fp16 configuration loads the int8 tier instead.
    public var availableBudgetBytes: UInt64?

    /// The Hub repos a fresh machine fetches from — mlx-community's `<UpstreamRepoName>-<quant>` grammar, one repo per
    /// precision lane (the int8 tier is derived from the fp16 files at load, so it rides `VOSR2-fp16`).
    public static let fp16Repo = "mlx-community/VOSR2-fp16"
    public static let fp32Repo = "mlx-community/VOSR2-fp32"
    /// The repo serving a lane.
    public static func repo(for quant: Quant) -> String { quant == .fp32 ? fp32Repo : fp16Repo }
    /// fp16 needs ~3.4 GB resident + ~2.6 GB transient at a 1024² output; below this the int8 tier is loaded.
    public static let fp16MinBudgetBytes: UInt64 = 7_000_000_000
    public static let maxScale = 4

    public init(quant: Quant = .fp16,
                seed: UInt64 = 0,
                defaultScale: Int = 2,
                tileSize: Int = 512,
                tileOverlap: Int = 32,
                vaeWholeFramePixels: Int = 2048 * 2048,
                vaeTileSize: Int = 1024,
                weightsDirectory: URL? = nil,
                modelsRootDirectory: URL? = nil,
                availableBudgetBytes: UInt64? = nil) {
        self.quant = quant
        self.seed = seed
        self.defaultScale = defaultScale
        self.tileSize = tileSize
        self.tileOverlap = tileOverlap
        self.vaeWholeFramePixels = vaeWholeFramePixels
        self.vaeTileSize = vaeTileSize
        self.weightsDirectory = weightsDirectory
        self.modelsRootDirectory = modelsRootDirectory
        self.availableBudgetBytes = availableBudgetBytes
    }

    /// The precision lane actually loaded: `.fp32` and `.int8` as declared, everything else `.fp16`.
    public var effectiveQuant: Quant {
        switch quant {
        case .fp32, .int8: return quant
        default: return .fp16
        }
    }

    /// The files a lane needs (the converter's names). int8 rides the fp16 files.
    public static func files(for quant: Quant) -> [String] {
        let half = ["transformer_fp16.safetensors", "vae_fp16.safetensors", "dinov2_fp16.safetensors", "config.json"]
        let full = ["transformer_fp32.safetensors", "vae_fp32.safetensors", "dinov2_fp32.safetensors", "config.json"]
        return quant == .fp32 ? full : half
    }

    /// The repo this configuration's lane materializes from.
    public var repo: String { Self.repo(for: effectiveQuant) }

    /// The directory `load()` reads: the explicit `weightsDirectory`, else the store's flat repo
    /// directory (`<root>/models--mlx-community--VOSR2-fp16/`, where the engine's materializer lands files).
    public func resolvedWeightsDirectory(storeRoot: URL?) -> URL? {
        weightsDirectory ?? ModelStore(root: storeRoot).directory(for: repo)
    }

    // Persist only the portable knobs; stamped roots, budgets and absolute paths are per-session.
    private enum CodingKeys: String, CodingKey {
        case quant, seed, defaultScale, tileSize, tileOverlap, vaeWholeFramePixels, vaeTileSize
    }
}

/// Fresh-machine sources (contract 1.24: the ENGINE downloads them into the store before `load()`).
/// One role per precision lane so an fp16/int8 configuration never pulls the 6.7 GB fp32 set.
extension VOSRConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        let lane = effectiveQuant == .fp32 ? "fp32" : "fp16"
        return [WeightSource(role: "vosr2-\(lane)", repo: repo, revision: "main",
                             matching: Self.files(for: effectiveQuant))]
    }

    /// Explicit `weightsDirectory` first (complete → nothing missing, incomplete → the lane is missing),
    /// then the MS-2 default probe over the store.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let dir = weightsDirectory {
            let complete = Self.files(for: effectiveQuant).allSatisfy {
                FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path)
            }
            return complete ? [] : weightSources
        }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }
}

/// Cold-start page-in of the lane's files from wherever they resolve.
extension VOSRConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        guard let dir = resolvedWeightsDirectory(storeRoot: modelsRootDirectory) else { return [] }
        return Self.files(for: effectiveQuant).map { dir.appendingPathComponent($0) }
    }
}
