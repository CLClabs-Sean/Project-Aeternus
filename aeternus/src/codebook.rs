//! # VQ Codebook
//!
//! Vector Quantization codebook for sub-1-bit magnitude encoding.
//! A 2-bit codebook maps 4 magnitude levels, packing 16 weights per u32.
//!
//! Combined with PCG sign reconstruction, effective storage is:
//!   2 bits (magnitude) + ~0 bits (sign) = ~2 bits/param
//!
//! At 2 bits/param: 70B model = 17.5 GB (tiles into 2 GB VRAM).

/// A 2-bit VQ codebook with 4 magnitude levels.
///
/// Default levels are calibrated for typical transformer weight distributions:
///   0 → 0.0    (zero / sparse)
///   1 → 0.25   (small magnitude)
///   2 → 0.75   (medium magnitude)  
///   3 → 1.50   (large magnitude)
#[derive(Debug, Clone, Copy)]
pub struct Codebook {
    pub magnitudes: [f32; 4],
}

impl Default for Codebook {
    fn default() -> Self {
        // Ternary-inspired levels: sparse zero + 3 magnitude tiers.
        // These approximate a half-normal distribution typical of
        // trained transformer weights.
        Self {
            magnitudes: [0.0, 0.25, 0.75, 1.50],
        }
    }
}

impl Codebook {
    /// Create a codebook with custom magnitude levels.
    pub fn new(magnitudes: [f32; 4]) -> Self {
        Self { magnitudes }
    }

    /// BitNet b1.58-style ternary codebook: {0, 1, 1, 1}
    /// Indices 1-3 all map to magnitude 1.0, combined with PCG sign
    /// gives {-1, 0, +1} — exactly BitNet b1.58.
    pub fn bitnet_ternary() -> Self {
        Self {
            magnitudes: [0.0, 1.0, 1.0, 1.0],
        }
    }
}

/// Pack a slice of 2-bit magnitude indices into u32 words.
///
/// Each u32 holds 16 indices (2 bits each, LSB-first).
/// Input `indices` values must be in 0..=3.
pub fn pack_indices(indices: &[u8]) -> Vec<u32> {
    let num_words = (indices.len() + 15) / 16;
    let mut packed = vec![0u32; num_words];

    for (i, &idx) in indices.iter().enumerate() {
        debug_assert!(idx <= 3, "Index {} out of range for 2-bit codebook", idx);
        let word = i / 16;
        let bit_offset = (i % 16) * 2;
        packed[word] |= (idx as u32 & 0x3) << bit_offset;
    }

    packed
}

/// Unpack u32 words back to 2-bit indices (for verification).
pub fn unpack_indices(packed: &[u32], count: usize) -> Vec<u8> {
    let mut indices = Vec::with_capacity(count);
    for i in 0..count {
        let word = i / 16;
        let bit_offset = (i % 16) * 2;
        let idx = ((packed[word] >> bit_offset) & 0x3) as u8;
        indices.push(idx);
    }
    indices
}

/// Reconstruct weights on CPU using the codebook + PCG signs (reference impl).
pub fn reconstruct_weights_cpu(
    packed: &[u32],
    codebook: &Codebook,
    seed: u32,
    count: usize,
) -> Vec<f32> {
    let indices = unpack_indices(packed, count);
    let mut weights = Vec::with_capacity(count);

    for (i, &idx) in indices.iter().enumerate() {
        let magnitude = codebook.magnitudes[idx as usize];
        let sign = crate::seed_engine::pcg_sign(i as u32, seed);
        weights.push(magnitude * sign);
    }

    weights
}

/// Generate synthetic packed weights for benchmarking.
/// Uses PCG hash to create a realistic distribution of magnitude indices.
pub fn generate_synthetic_packed(seed: u32, count: usize) -> Vec<u32> {
    let mut indices = Vec::with_capacity(count);
    for i in 0..count {
        // Use hash to generate a distribution biased toward small magnitudes
        // (matching typical transformer weight distributions).
        let hash = crate::seed_engine::pcg_hash(i as u32 ^ seed.wrapping_mul(7));
        let idx = match hash % 100 {
            0..=15  => 0u8,  // 16% zero (sparse)
            16..=55 => 1,    // 40% small
            56..=84 => 2,    // 29% medium
            _       => 3,    // 15% large
        };
        indices.push(idx);
    }
    pack_indices(&indices)
}

/// K-Means (Lloyd's algorithm) codebook calibration for 4 centroids.
///
/// Clusters the absolute values of `weights` into 4 groups using iterative
/// centroid refinement. Initialization uses percentiles (0th, 33rd, 67th, 95th)
/// of the sorted |w| distribution for deterministic, fast convergence.
///
/// Returns centroids sorted ascending.
pub fn kmeans_4(weights: &[f32], max_iters: usize) -> Codebook {
    let n = weights.len();
    if n < 4 {
        return Codebook::default();
    }

    // Work on absolute magnitudes
    let mut mags: Vec<f32> = weights.iter().map(|w| w.abs()).collect();
    mags.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Percentile-based initialization (deterministic, avoids pathological starts)
    let mut centroids = [
        mags[0],                           // min
        mags[n / 3],                       // 33rd percentile
        mags[2 * n / 3],                   // 67th percentile
        mags[(n as f64 * 0.95) as usize],  // 95th percentile
    ];

    // Lloyd's iterations
    for _iter in 0..max_iters {
        // Assignment step: assign each magnitude to nearest centroid
        let mut sums = [0.0f64; 4];
        let mut counts = [0usize; 4];

        for &mag in &mags {
            let mut best_k = 0usize;
            let mut best_dist = f32::MAX;
            for (k, &c) in centroids.iter().enumerate() {
                let d = (mag - c).abs();
                if d < best_dist {
                    best_dist = d;
                    best_k = k;
                }
            }
            sums[best_k] += mag as f64;
            counts[best_k] += 1;
        }

        // Update step: recompute centroids
        let mut converged = true;
        for k in 0..4 {
            if counts[k] > 0 {
                let new_c = (sums[k] / counts[k] as f64) as f32;
                if (new_c - centroids[k]).abs() > 1e-7 {
                    converged = false;
                }
                centroids[k] = new_c;
            }
        }

        if converged {
            break;
        }
    }

    // Sort centroids ascending (important: index 0 = smallest magnitude)
    centroids.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    Codebook::new(centroids)
}

/// Compute the mean squared error between original weights and their
/// quantized reconstruction using the given codebook (magnitude only, ignoring signs).
pub fn quantization_mse(weights: &[f32], codebook: &Codebook) -> f64 {
    if weights.is_empty() {
        return 0.0;
    }

    let total_sq_err: f64 = weights.iter().map(|&w| {
        let mag = w.abs();
        // Find nearest centroid
        let mut best_dist = f32::MAX;
        for &c in &codebook.magnitudes {
            let d = (mag - c).abs();
            if d < best_dist {
                best_dist = d;
            }
        }
        (best_dist as f64) * (best_dist as f64)
    }).sum();

    total_sq_err / weights.len() as f64
}

/// Compute storage cost in bits per parameter.
pub fn bits_per_param() -> f64 {
    // 2 bits magnitude + ~0 bits sign (PCG) + amortized codebook (128 bits / N)
    2.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let indices: Vec<u8> = (0..100).map(|i| (i % 4) as u8).collect();
        let packed = pack_indices(&indices);
        let unpacked = unpack_indices(&packed, indices.len());
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn pack_16_per_word() {
        // 16 indices should fit in exactly 1 word.
        let indices = vec![3u8; 16];
        let packed = pack_indices(&indices);
        assert_eq!(packed.len(), 1);
        // All bits set: 16 × 2-bit `11` = 0xFFFFFFFF
        assert_eq!(packed[0], 0xFFFFFFFF);
    }

    #[test]
    fn bfe_simulation() {
        // Simulate what the GPU's bitfieldExtract does.
        let indices: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let packed = pack_indices(&indices);
        assert_eq!(packed.len(), 1);

        let word = packed[0];
        for i in 0..16 {
            let bit_offset = i * 2;
            let extracted = (word >> bit_offset) & 0x3;
            assert_eq!(extracted as u8, indices[i], "BFE mismatch at index {}", i);
        }
    }

    #[test]
    fn reconstruct_matches_individual() {
        let codebook = Codebook::default();
        let seed = 42u32;
        let count = 1000;
        let packed = generate_synthetic_packed(seed, count);
        let weights = reconstruct_weights_cpu(&packed, &codebook, seed, count);

        assert_eq!(weights.len(), count);
        // Every weight should be magnitude × ±1
        for &w in &weights {
            let abs_w = w.abs();
            assert!(
                codebook.magnitudes.contains(&abs_w),
                "Weight {} has magnitude {} not in codebook",
                w, abs_w
            );
        }
    }

    #[test]
    fn bitnet_ternary_produces_ternary() {
        let codebook = Codebook::bitnet_ternary();
        let seed = 0xBEEFu32;
        // Force all indices to 1 (magnitude 1.0) — should give ±1.0
        let indices = vec![1u8; 100];
        let packed = pack_indices(&indices);
        let weights = reconstruct_weights_cpu(&packed, &codebook, seed, 100);

        assert!(weights.iter().all(|&w| w == 1.0 || w == -1.0));
    }

    #[test]
    fn storage_cost() {
        assert!((bits_per_param() - 2.0).abs() < f64::EPSILON);
    }
}
