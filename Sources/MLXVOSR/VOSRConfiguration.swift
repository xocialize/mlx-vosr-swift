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
///   quant   files            resident (weights)   in-app `phys_footprint` peak, 512² → 2048² (×4, VAE tiled — AB-D-0120)
///   .fp16   *_fp16 (3.35 GB) ≈ 3.4 GB              9.6 GB   (MLX pool peak 6.87 GB; 28.9 s on an M5 Max)
///   .int8   *_fp16 + int8    ≈ 2.0 GB              8.4 GB   (5.59 GB; 28.3 s)   ← the BudgetAware fallback
///   .fp32   *_fp32 (6.7 GB)  ≈ 6.7 GB             not measured in-app — the parity lane (TF32 on M5, AB-L-0175)
///
/// The declared footprints (`fp16Footprint` …) are those peaks split as resident + activation, with ~6 % margin;
/// `vosr-smoke engine` on the real MLXServeEngine (pool cap applied) read within 2 % of the ForgeOptimizer app on the
/// whole-frame path (20.09 vs 19.66 GB), so the engine-path smoke is the pre-release proxy and the app the basis.
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
    /// latent position, O(N²): a 2048² output (256² latent) is the largest full-image size that runs at all
    /// (16.9 GB MLX peak with query-chunking), and a 4096² one asks Metal for a 137 GB score matrix.
    ///
    /// Default **1024²** (AB-D-0120, 2026-10-10): a 1024² output decodes whole-frame, anything larger tiles. Measured
    /// on a 2048² output, the tiled decode is indistinguishable from the whole-frame one — PSNR 48.1 dB, the
    /// blend-band seam probe at the whole-frame image's own ratio (0.946× vs 0.949×) — and the ×4 MLX peak falls
    /// 16.86 → 6.87 GB at the same speed, which is what lets the fp16 tier fit a 16 GB Mac. The old 2048² default is
    /// still a lever for parity work on the RAW path only: under the engine's 2 GiB MLX cache cap the whole-frame 2048²
    /// decode comes out WRONG (8.7 dB vs the oracle, deterministic; `vosr-smoke run --cache-limit 2147483648`
    /// reproduces it, the tiled decode under the same cap is right) — so an engine host must not raise this past 1024².
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
    /// In-app `phys_footprint` basis (ForgeOptimizer `UpscaleMemoryBench` + `vosr-smoke engine`, M5 Max, 2026-10-10,
    /// 512² → 2048² at ×4 on the tiled-VAE default; AB-T-0185 step 6). resident = the governor's charge for the
    /// loaded weights; activation = measured phys peak − resident, rounded up ~6 %. The in-app whole-frame numbers
    /// they replace were 19.66 GB (fp16) and 14.34 GB (int8) — provisional MLX-peak declarations had said 9.4 / 8.0.
    public static let fp16Footprint = QuantFootprint(quant: .fp16, residentBytes: 3_400_000_000, peakActivationBytes: 6_600_000_000)
    public static let int8Footprint = QuantFootprint(quant: .int8, residentBytes: 2_000_000_000, peakActivationBytes: 6_800_000_000)
    /// The parity lane is not run in the app; declared as a DERIVED upper bound (weights 6.7 GB; activation 2× fp16's),
    /// so the governor refuses it where it might not fit rather than admit on a guess.
    public static let fp32Footprint = QuantFootprint(quant: .fp32, residentBytes: 6_700_000_000, peakActivationBytes: 13_200_000_000)
    /// Below this much governor headroom an fp16 configuration loads the int8 tier instead: fp16's whole declared
    /// working set (resident + activation = 10.0 GB), so the fallback engages exactly where fp16 would be refused.
    public static var fp16MinBudgetBytes: UInt64 { fp16Footprint.residentBytes + fp16Footprint.peakActivationBytes }
    public static let maxScale = 4

    public init(quant: Quant = .fp16,
                seed: UInt64 = 0,
                defaultScale: Int = 2,
                tileSize: Int = 512,
                tileOverlap: Int = 32,
                vaeWholeFramePixels: Int = 1024 * 1024,   // AB-D-0120
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
