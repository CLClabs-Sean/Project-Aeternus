//! # Low-Rank Binary Factorization via ADMM
//!
//! Approximate W ≈ A · diag(s) · B^T where:
//!   A ∈ {-1,+1}^{m×r}  (packed binary, 32 per u32)
//!   B ∈ {-1,+1}^{n×r}  (packed binary, 32 per u32)
//!   s ∈ R^r             (f32 scale per rank slice)
//!
//! Two-phase solver:
//!   Phase 1: Greedy rank-1 deflation (initialization)
//!   Phase 2: Coordinate descent refinement (re-optimize each slice against W)
//!
//! Storage: r(m + n) bits + r×32 bits ≈ 2r/n bits/param for square matrices.

/// Packed binary factors for a single weight matrix.
#[derive(Debug)]
pub struct BinaryFactors {
    pub a_packed: Vec<u32>,
    pub b_packed: Vec<u32>,
    pub scales: Vec<f32>,
    pub rank: usize,
    pub m: usize,
    pub n: usize,
}

impl BinaryFactors {
    pub fn storage_bits(&self) -> usize {
        self.m * self.rank + self.n * self.rank + self.rank * 32
    }

    pub fn bits_per_param(&self) -> f64 {
        self.storage_bits() as f64 / (self.m * self.n) as f64
    }

    pub fn storage_bytes(&self) -> usize {
        (self.storage_bits() + 7) / 8
    }
}

// ---------------------------------------------------------------------------
// Bit packing helpers
// ---------------------------------------------------------------------------

#[inline]
fn get_binary(packed: &[u32], idx: usize) -> f32 {
    let word = packed[idx / 32];
    let bit = (word >> (idx % 32)) & 1;
    if bit == 1 { 1.0 } else { -1.0 }
}

#[inline]
fn set_binary(packed: &mut [u32], idx: usize, value: f32) {
    let word_idx = idx / 32;
    let bit_pos = idx % 32;
    if value >= 0.0 {
        packed[word_idx] |= 1u32 << bit_pos;
    } else {
        packed[word_idx] &= !(1u32 << bit_pos);
    }
}

fn words_for(n: usize) -> usize {
    (n + 31) / 32
}

// ---------------------------------------------------------------------------
// ADMM Solver
// ---------------------------------------------------------------------------

/// Factorize W[m×n] into binary factors A, B and scales s.
///
/// Phase 1: Greedy rank-1 deflation for initialization.
/// Phase 2: Coordinate descent refinement on original W.
pub fn admm_factorize(w: &[f32], m: usize, n: usize, rank: usize, max_iters: usize) -> BinaryFactors {
    assert_eq!(w.len(), m * n, "Weight matrix size mismatch");
    assert!(rank > 0 && rank <= m.min(n), "Rank must be in [1, min(m,n)]");

    let a_words = words_for(m);
    let b_words = words_for(n);

    let mut a_packed = vec![0u32; a_words * rank];
    let mut b_packed = vec![0u32; b_words * rank];
    let mut scales = vec![0.0f32; rank];

    let mut a_col = vec![0.0f32; m];
    let mut b_col = vec![0.0f32; n];

    // --- Phase 1: Greedy rank-1 deflation ---
    let mut residual = w.to_vec();
    for k in 0..rank {
        let b_off = k * b_words;
        let a_off = k * a_words;

        // Init B[:,k] randomly
        for j in 0..n {
            let hash = crate::seed_engine::pcg_hash((k * n + j) as u32);
            b_col[j] = if hash & 1 == 0 { 1.0f32 } else { -1.0 };
            set_binary(&mut b_packed[b_off..b_off + b_words], j, b_col[j]);
        }

        // Alternating optimization on residual
        for _iter in 0..max_iters {
            for i in 0..m {
                let mut dot = 0.0f32;
                let row = i * n;
                for j in 0..n { dot += residual[row + j] * b_col[j]; }
                a_col[i] = if dot >= 0.0 { 1.0 } else { -1.0 };
            }
            for j in 0..n {
                let mut dot = 0.0f32;
                for i in 0..m { dot += residual[i * n + j] * a_col[i]; }
                b_col[j] = if dot >= 0.0 { 1.0 } else { -1.0 };
            }
        }

        // Compute scale: s[k] = (a^T R b) / (m * n)
        let mut a_r_b = 0.0f64;
        for i in 0..m {
            let row = i * n;
            let mut rd = 0.0f64;
            for j in 0..n { rd += residual[row + j] as f64 * b_col[j] as f64; }
            a_r_b += a_col[i] as f64 * rd;
        }
        scales[k] = (a_r_b / (m as f64 * n as f64)) as f32;

        // Pack
        for i in 0..m { set_binary(&mut a_packed[a_off..a_off + a_words], i, a_col[i]); }
        for j in 0..n { set_binary(&mut b_packed[b_off..b_off + b_words], j, b_col[j]); }

        // Deflate
        let s_k = scales[k];
        for i in 0..m {
            let row = i * n;
            for j in 0..n { residual[row + j] -= s_k * a_col[i] * b_col[j]; }
        }
    }

    // --- Phase 2: Coordinate descent refinement on original W ---
    // Re-compute residual from scratch using original W
    // For each slice k: add back this slice, re-optimize against true residual, deflate
    let refinement_sweeps = 5; // more sweeps for better convergence
    for _sweep in 0..refinement_sweeps {
        // Recompute full residual from scratch each sweep
        residual.copy_from_slice(w);
        for k2 in 0..rank {
            let s2 = scales[k2];
            let a2_off = k2 * a_words;
            let b2_off = k2 * b_words;
            for i in 0..m {
                let a_val = get_binary(&a_packed[a2_off..a2_off + a_words], i);
                let row = i * n;
                for j in 0..n {
                    let b_val = get_binary(&b_packed[b2_off..b2_off + b_words], j);
                    residual[row + j] -= s2 * a_val * b_val;
                }
            }
        }

        for k in 0..rank {
            let b_off = k * b_words;
            let a_off = k * a_words;
            let old_s = scales[k];

            // Add back this slice's contribution
            for i in 0..m {
                let a_val = get_binary(&a_packed[a_off..a_off + a_words], i);
                let row = i * n;
                for j in 0..n {
                    let b_val = get_binary(&b_packed[b_off..b_off + b_words], j);
                    residual[row + j] += old_s * a_val * b_val;
                }
            }

            // Extract current B[:,k]
            for j in 0..n {
                b_col[j] = get_binary(&b_packed[b_off..b_off + b_words], j);
            }

            // Re-optimize A and B against the residual (which now excludes only this slice)
            for _iter in 0..max_iters {
                for i in 0..m {
                    let mut dot = 0.0f32;
                    let row = i * n;
                    for j in 0..n { dot += residual[row + j] * b_col[j]; }
                    a_col[i] = if dot >= 0.0 { 1.0 } else { -1.0 };
                }
                for j in 0..n {
                    let mut dot = 0.0f32;
                    for i in 0..m { dot += residual[i * n + j] * a_col[i]; }
                    b_col[j] = if dot >= 0.0 { 1.0 } else { -1.0 };
                }
            }

            // Re-compute scale
            let mut a_r_b = 0.0f64;
            for i in 0..m {
                let row = i * n;
                let mut rd = 0.0f64;
                for j in 0..n { rd += residual[row + j] as f64 * b_col[j] as f64; }
                a_r_b += a_col[i] as f64 * rd;
            }
            let new_s = (a_r_b / (m as f64 * n as f64)) as f32;
            scales[k] = new_s;

            // Pack updated vectors
            for i in 0..m { set_binary(&mut a_packed[a_off..a_off + a_words], i, a_col[i]); }
            for j in 0..n { set_binary(&mut b_packed[b_off..b_off + b_words], j, b_col[j]); }

            // Deflate with new values
            for i in 0..m {
                let row = i * n;
                for j in 0..n { residual[row + j] -= new_s * a_col[i] * b_col[j]; }
            }
        }
    }

    BinaryFactors { a_packed, b_packed, scales, rank, m, n }
}

/// Reconstruct the full weight matrix from binary factors.
pub fn reconstruct_cpu(factors: &BinaryFactors) -> Vec<f32> {
    let m = factors.m;
    let n = factors.n;
    let a_words = words_for(m);
    let b_words = words_for(n);
    let mut w = vec![0.0f32; m * n];

    for k in 0..factors.rank {
        let s_k = factors.scales[k];
        let a_off = k * a_words;
        let b_off = k * b_words;

        for i in 0..m {
            let a_val = get_binary(&factors.a_packed[a_off..a_off + a_words], i);
            for j in 0..n {
                let b_val = get_binary(&factors.b_packed[b_off..b_off + b_words], j);
                w[i * n + j] += s_k * a_val * b_val;
            }
        }
    }

    w
}

/// Compute MSE between original weights and binary factor reconstruction.
pub fn factorization_mse(w: &[f32], factors: &BinaryFactors) -> f64 {
    let recon = reconstruct_cpu(factors);
    let n = w.len() as f64;
    let sq_err: f64 = w.iter().zip(recon.iter())
        .map(|(&a, &b)| {
            let d = (a - b) as f64;
            d * d
        })
        .sum();
    sq_err / n
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_pack_roundtrip() {
        let mut packed = vec![0u32; 2];
        for i in 0..64 {
            let val = if i % 3 == 0 { 1.0 } else { -1.0 };
            set_binary(&mut packed, i, val);
        }
        for i in 0..64 {
            let expected = if i % 3 == 0 { 1.0 } else { -1.0 };
            assert_eq!(get_binary(&packed, i), expected, "Mismatch at {}", i);
        }
    }

    #[test]
    fn rank1_identity_approx() {
        let a = [1.0f32, 1.0, -1.0, -1.0];
        let b = [1.0f32, -1.0, 1.0, -1.0];
        let mut w = vec![0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                w[i * 4 + j] = a[i] * b[j] * 0.5;
            }
        }

        let factors = admm_factorize(&w, 4, 4, 1, 10);
        let mse = factorization_mse(&w, &factors);
        assert!(mse < 1e-6, "Rank-1 matrix should be perfectly recovered, MSE={}", mse);
    }

    #[test]
    fn storage_calc() {
        let factors = BinaryFactors {
            a_packed: vec![0; 64],
            b_packed: vec![0; 64],
            scales: vec![0.0; 1],
            rank: 1,
            m: 2048,
            n: 2048,
        };
        let bpp = factors.bits_per_param();
        assert!(bpp < 0.001);
    }
}
