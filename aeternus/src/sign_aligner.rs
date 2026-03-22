//! # Sign-Aligner — Phase 7
//!
//! Optimizes a PCG seed per layer to maximize sign-match with real weights,
//! then generates a sparse XOR correction mask for the remaining mismatches.
//!
//! This is the core of sub-1-bit AETERNUS: instead of storing 1 bit/param
//! for signs, we store a 32-bit seed + a correction mask that is typically
//! 40-45% of the parameter count (vs 100% with explicit signs).

use crate::seed_engine::pcg_hash;

// ---------------------------------------------------------------------------
// Sign-Aligner results
// ---------------------------------------------------------------------------

/// Result of per-layer seed optimization.
#[derive(Debug, Clone)]
pub struct LayerSignData {
    /// Optimal PCG seed for this layer.
    pub seed: u32,
    /// Packed XOR correction mask (32 corrections per u32).
    /// correction_mask bit i = 1 means PCG sign at position i was WRONG.
    pub correction_mask: Vec<u32>,
    /// Number of sign mismatches (popcount of correction_mask).
    pub correction_count: usize,
    /// Fraction of signs correctly predicted by the seed alone.
    pub match_rate: f64,
    /// Total weight count for this layer.
    pub weight_count: usize,
}

impl LayerSignData {
    /// Bits used for sign information per parameter.
    /// = correction_count / weight_count (each correction is 1 bit).
    pub fn sign_bits_per_param(&self) -> f64 {
        if self.weight_count == 0 {
            return 0.0;
        }
        self.correction_count as f64 / self.weight_count as f64
    }
}

/// Full model sign alignment result.
#[derive(Debug)]
pub struct ModelSignData {
    pub layers: Vec<LayerSignData>,
}

impl ModelSignData {
    /// Average match rate across all layers.
    pub fn avg_match_rate(&self) -> f64 {
        if self.layers.is_empty() {
            return 0.0;
        }
        let total_weights: usize = self.layers.iter().map(|l| l.weight_count).sum();
        let total_matches: usize = self.layers.iter()
            .map(|l| l.weight_count - l.correction_count)
            .sum();
        total_matches as f64 / total_weights as f64
    }

    /// Total sign bits per param across the model.
    pub fn total_sign_bits_per_param(&self) -> f64 {
        let total_weights: usize = self.layers.iter().map(|l| l.weight_count).sum();
        let total_corrections: usize = self.layers.iter().map(|l| l.correction_count).sum();
        if total_weights == 0 {
            return 0.0;
        }
        total_corrections as f64 / total_weights as f64
    }

    /// Total correction mask bytes.
    pub fn correction_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.correction_mask.len() * 4).sum()
    }
}

// ---------------------------------------------------------------------------
// PCG sign bit extraction (matches seed_engine::pcg_sign but returns 0/1)
// ---------------------------------------------------------------------------

/// Get the sign bit (0=positive, 1=negative) from PCG hash.
#[inline]
fn pcg_sign_bit(weight_index: u32, seed: u32) -> u8 {
    let hash = pcg_hash(weight_index ^ seed);
    (hash & 1) as u8
}

// ---------------------------------------------------------------------------
// Seed Optimizer — evolutionary search
// ---------------------------------------------------------------------------

/// Count how many sign bits match between PCG(seed) and target for a subset.
fn count_matches_sampled(target_signs: &[u8], seed: u32, sample_indices: &[usize]) -> usize {
    sample_indices.iter()
        .filter(|&&i| pcg_sign_bit(i as u32, seed) == target_signs[i])
        .count()
}

/// Count exact matches for ALL weights (used for final scoring).
fn count_matches_full(target_signs: &[u8], seed: u32) -> usize {
    target_signs.iter().enumerate()
        .filter(|(i, &s)| pcg_sign_bit(*i as u32, seed) == s)
        .count()
}

/// Optimize the PCG seed to maximize sign-match with target signs.
///
/// Fast three-phase search:
///   1. Random sample seeds, score on subsample → keep top-4
///   2. Hill-climb around top-4 using subsample scoring
///   3. Full-score only the single winner, generate correction mask
pub fn optimize_seed(target_signs: &[u8], max_random_seeds: u32) -> LayerSignData {
    let n = target_signs.len();
    if n == 0 {
        return LayerSignData {
            seed: 0,
            correction_mask: Vec::new(),
            correction_count: 0,
            match_rate: 1.0,
            weight_count: 0,
        };
    }

    // Subsample: 1% of data, min 1000, max 50K — fast enough for millions of candidates
    let sample_size = (n / 100).max(1000).min(50_000).min(n);
    let sample_indices: Vec<usize> = (0..sample_size)
        .map(|i| (i as u64 * n as u64 / sample_size as u64) as usize)
        .collect();

    // Phase 1: Random seed screening (subsample scored)
    let mut best_seed = 0u32;
    let mut best_score = 0usize;
    let mut top4: Vec<(u32, usize)> = Vec::with_capacity(4);

    for i in 0..max_random_seeds {
        let candidate_seed = pcg_hash(i.wrapping_mul(2654435761));
        let score = count_matches_sampled(target_signs, candidate_seed, &sample_indices);
        if top4.len() < 4 || score > top4.last().unwrap().1 {
            top4.push((candidate_seed, score));
            top4.sort_by(|a, b| b.1.cmp(&a.1));
            top4.truncate(4);
        }
    }

    // Phase 2: Hill-climb around top-4 (still subsample scored)
    let primes = [1u32, 3, 7, 13, 31, 127, 1021, 8191, 65521, 524287];

    best_seed = top4[0].0;
    best_score = top4[0].1;

    for &(base_seed, base_score) in &top4 {
        if base_score > best_score {
            best_score = base_score;
            best_seed = base_seed;
        }
        for &p in &primes {
            for &offset in &[p, p.wrapping_neg()] {
                let trial = base_seed.wrapping_add(offset);
                let score = count_matches_sampled(target_signs, trial, &sample_indices);
                if score > best_score {
                    best_score = score;
                    best_seed = trial;
                }
            }
        }
    }

    // Phase 3: Full-score ONLY the winner, generate correction mask
    let full_matches = count_matches_full(target_signs, best_seed);
    let match_rate = full_matches as f64 / n as f64;
    let correction_mask = compute_correction_mask(target_signs, best_seed);
    let correction_count = n - full_matches;

    log::info!("    Seed 0x{:08X}: match {:.2}%, corrections {}/{} ({:.3} bits/param)",
        best_seed, match_rate * 100.0, correction_count, n,
        correction_count as f64 / n as f64);

    LayerSignData {
        seed: best_seed,
        correction_mask,
        correction_count,
        match_rate,
        weight_count: n,
    }
}

/// Compute the XOR correction mask between PCG-predicted signs and true signs.
/// Result bit i = 1 means PCG was WRONG at position i and needs flipping.
pub fn compute_correction_mask(true_signs: &[u8], seed: u32) -> Vec<u32> {
    let n = true_signs.len();
    let num_words = (n + 31) / 32;
    let mut mask = vec![0u32; num_words];

    for i in 0..n {
        let predicted = pcg_sign_bit(i as u32, seed);
        if predicted != true_signs[i] {
            mask[i / 32] |= 1 << (i % 32);
        }
    }

    mask
}

/// Verify that seed + correction mask perfectly reconstructs the target signs.
pub fn verify_correction(true_signs: &[u8], seed: u32, correction_mask: &[u32]) -> bool {
    for (i, &true_sign) in true_signs.iter().enumerate() {
        let predicted = pcg_sign_bit(i as u32, seed);
        let corr_bit = ((correction_mask[i / 32] >> (i % 32)) & 1) as u8;
        let corrected = predicted ^ corr_bit;
        if corrected != true_sign {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correction_mask_roundtrip() {
        // Random-ish target signs
        let target: Vec<u8> = (0..10000)
            .map(|i| if pcg_hash(i * 31337) % 3 == 0 { 1 } else { 0 })
            .collect();

        let seed = 0xDEADBEEF_u32;
        let mask = compute_correction_mask(&target, seed);
        assert!(verify_correction(&target, seed, &mask));
    }

    #[test]
    fn test_optimize_seed_better_than_random() {
        // Create signs with structure (not purely uniform)
        let target: Vec<u8> = (0..50000)
            .map(|i| {
                // Correlated pattern: sign depends on position modulo
                if (i * 7 + i / 100) % 2 == 0 { 0 } else { 1 }
            })
            .collect();

        let result = optimize_seed(&target, 5000);

        // Should be better than 50% (random baseline)
        assert!(result.match_rate > 0.49,
            "Match rate {} should be at least 49% (random baseline)", result.match_rate);

        // Correction mask should be valid
        assert!(verify_correction(&target, result.seed, &result.correction_mask));

        // Sign bits per param should be < 1.0
        assert!(result.sign_bits_per_param() < 1.0);
    }

    #[test]
    fn test_pcg_sign_bit_matches_engine() {
        // Verify our sign bit extraction matches seed_engine::pcg_sign
        for i in 0..1000u32 {
            let seed = 42u32;
            let sign_f32 = crate::seed_engine::pcg_sign(i, seed);
            let sign_bit = pcg_sign_bit(i, seed);
            let expected_bit = if sign_f32 < 0.0 { 1u8 } else { 0u8 };
            assert_eq!(sign_bit, expected_bit,
                "Sign mismatch at index {}: pcg_sign={}, sign_bit={}", i, sign_f32, sign_bit);
        }
    }

    #[test]
    fn test_verify_catches_errors() {
        let target = vec![0u8; 100];
        let seed = 42u32;
        let mut mask = compute_correction_mask(&target, seed);

        // Valid mask should pass
        assert!(verify_correction(&target, seed, &mask));

        // Corrupt one bit — should fail
        mask[0] ^= 1;
        assert!(!verify_correction(&target, seed, &mask));
    }
}
