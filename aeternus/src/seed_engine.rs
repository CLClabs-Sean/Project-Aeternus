//! # Seed Engine
//!
//! CPU-side reference implementation of the 32-bit PCG hash for procedural
//! sign reconstruction. Mirrors the GPU shader exactly so we can validate
//! GPU output against CPU reference.

/// PCG hash — identical to the GLSL `pcg_hash` in pim_sign_regen.comp.
///
/// Single-pass 32-bit hash. No 64-bit arithmetic needed.
#[inline]
pub fn pcg_hash(input: u32) -> u32 {
    let state = input.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

/// Reconstruct the sign (+1.0 or -1.0) for a single weight.
///
/// Deterministic: `pcg_sign(idx, seed)` always returns the same value.
#[inline]
pub fn pcg_sign(weight_index: u32, seed: u32) -> f32 {
    let hash = pcg_hash(weight_index ^ seed);
    if hash % 2 == 0 { 1.0 } else { -1.0 }
}

/// Apply sign reconstruction to a weight buffer in-place (CPU reference).
/// Mirrors exactly what the GPU shader does.
pub fn apply_signs_cpu(weights: &mut [f32], seed: u32) {
    for (i, w) in weights.iter_mut().enumerate() {
        *w *= pcg_sign(i as u32, seed);
    }
}

/// Compute the effective storage cost in bits for the sign information
/// when using PWR.
///
/// With procedural reconstruction, the answer is always the seed size
/// (32 bits) amortized over all parameters.
pub fn effective_sign_bits_per_param(param_count: u64) -> f64 {
    32.0 / param_count as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg_sign_determinism() {
        let seed = 0xDEAD_BEEFu32;
        let signs_a: Vec<f32> = (0..10_000).map(|i| pcg_sign(i, seed)).collect();
        let signs_b: Vec<f32> = (0..10_000).map(|i| pcg_sign(i, seed)).collect();
        assert_eq!(signs_a, signs_b, "PCG sign reconstruction must be deterministic");
    }

    #[test]
    fn pcg_sign_distribution() {
        let seed = 0x1234_5678u32;
        let count = 100_000u32;
        let positive: usize = (0..count)
            .filter(|&i| pcg_sign(i, seed) == 1.0)
            .count();
        let ratio = positive as f64 / count as f64;
        assert!(
            (ratio - 0.5).abs() < 0.01,
            "Sign distribution should be ~50/50, got {:.4}",
            ratio
        );
    }

    #[test]
    fn pcg_sign_in_place() {
        let seed = 42u32;
        let mut weights = vec![1.0f32; 1000];
        apply_signs_cpu(&mut weights, seed);

        // Verify each weight is either +1.0 or -1.0.
        assert!(weights.iter().all(|&w| w == 1.0 || w == -1.0));

        // Apply signs again — should restore original (all 1.0).
        apply_signs_cpu(&mut weights, seed);
        assert!(
            weights.iter().all(|&w| (w - 1.0).abs() < f32::EPSILON),
            "Double sign application should restore original weights"
        );
    }

    #[test]
    fn effective_bits_sub_one() {
        let bits = effective_sign_bits_per_param(1_000_000_000);
        assert!(
            bits < 1e-6,
            "Sign bits per param for 1B model should be negligible, got {}",
            bits
        );
    }

    #[test]
    fn different_seeds_different_signs() {
        let seed_a = 0x1111u32;
        let seed_b = 0x2222u32;
        let signs_a: Vec<f32> = (0..1000).map(|i| pcg_sign(i, seed_a)).collect();
        let signs_b: Vec<f32> = (0..1000).map(|i| pcg_sign(i, seed_b)).collect();
        let matches: usize = signs_a.iter().zip(signs_b.iter())
            .filter(|(a, b)| a == b).count();
        let match_ratio = matches as f64 / 1000.0;
        assert!(
            (match_ratio - 0.5).abs() < 0.05,
            "Different seeds should produce ~50% overlap (random), got {:.4}",
            match_ratio
        );
    }
}
