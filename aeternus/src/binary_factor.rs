//! # Low-Rank Binary Factorization via ADMM
//!
//! Approximate W ≈ A · diag(s) · B^T where:
//!   A ∈ {-1,+1}^{m×r}  (packed binary, 32 per u32)
//!   B ∈ {-1,+1}^{n×r}  (packed binary, 32 per u32)
//!   s ∈ R^r             (f32 scale per rank slice)
//!
//! Supports Hessian-weighted optimization:
//!   minimize Σ_ij h[j] · (W[i,j] - [A diag(s) B^T][i,j])^2
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
// Importance Weighting
// ---------------------------------------------------------------------------

/// Compute per-column squared L2 norm as importance proxy.
/// h[j] = Σ_i W[i,j]^2
///
/// This is a zero-data Hessian proxy: columns with large weights are
/// more important to reconstruct accurately.
pub fn column_importance(w: &[f32], m: usize, n: usize) -> Vec<f64> {
    let mut h = vec![0.0f64; n];
    for i in 0..m {
        let row = i * n;
        for j in 0..n {
            let v = w[row + j] as f64;
            h[j] += v * v;
        }
    }
    // Normalize so mean(h) = 1.0 (prevents scale issues in ADMM)
    let mean = h.iter().sum::<f64>() / n as f64;
    if mean > 0.0 {
        for v in h.iter_mut() { *v /= mean; }
    }
    h
}

/// Statistics about importance distribution.
pub fn importance_stats(h: &[f64]) -> (f64, f64, f64) {
    let min = h.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = h.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ratio = if min > 0.0 { max / min } else { f64::INFINITY };
    (min, max, ratio)
}

// ---------------------------------------------------------------------------
// ADMM Solver with Importance Weighting
// ---------------------------------------------------------------------------

/// Factorize W[m×n] into binary factors A, B and scales s.
///
/// When `importance` is Some(h) with h.len() == n:
///   A-step: a[i] = sign(Σ_j h[j] · R[i,j] · b[j])
///   B-step: b[j] = sign(h[j] · Σ_i R[i,j] · a[i])
///   S-step: s = (Σ_ij h[j] · a[i] · R[i,j] · b[j]) / (Σ_ij h[j])
///
/// When None: uniform weighting (original Frobenius norm).
pub fn admm_factorize(
    w: &[f32], m: usize, n: usize, rank: usize, max_iters: usize,
    importance: Option<&[f64]>,
) -> BinaryFactors {
    assert_eq!(w.len(), m * n, "Weight matrix size mismatch");
    assert!(rank > 0 && rank <= m.min(n), "Rank must be in [1, min(m,n)]");
    if let Some(h) = importance {
        assert_eq!(h.len(), n, "Importance vector length must match n");
    }

    let a_words = words_for(m);
    let b_words = words_for(n);

    let mut a_packed = vec![0u32; a_words * rank];
    let mut b_packed = vec![0u32; b_words * rank];
    let mut scales = vec![0.0f32; rank];

    let mut a_col = vec![0.0f32; m];
    let mut b_col = vec![0.0f32; n];

    // Uniform weights fallback
    let uniform = vec![1.0f64; n];
    let h: &[f64] = importance.unwrap_or(&uniform);

    // Precompute sum of importance for S-step denominator
    let h_sum: f64 = h.iter().sum();

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

        // Weighted alternating optimization on residual
        for _iter in 0..max_iters {
            // A-step: a[i] = sign(Σ_j h[j] · R[i,j] · b[j])
            for i in 0..m {
                let mut dot = 0.0f64;
                let row = i * n;
                for j in 0..n {
                    dot += h[j] * residual[row + j] as f64 * b_col[j] as f64;
                }
                a_col[i] = if dot >= 0.0 { 1.0 } else { -1.0 };
            }
            // B-step: b[j] = sign(h[j] · Σ_i R[i,j] · a[i])
            for j in 0..n {
                let mut dot = 0.0f64;
                for i in 0..m {
                    dot += residual[i * n + j] as f64 * a_col[i] as f64;
                }
                // Weight by h[j]: if h[j] is large, the sign is more determined
                b_col[j] = if h[j] * dot >= 0.0 { 1.0 } else { -1.0 };
            }
        }

        // Weighted scale: s = (Σ_ij h[j] · a[i] · R[i,j] · b[j]) / (Σ_ij h[j])
        // Since h[j] ≥ 0 and (a[i]·b[j])^2 = 1:
        // denominator = m · Σ_j h[j] (each h[j] counted m times across rows)
        let mut num = 0.0f64;
        for i in 0..m {
            let row = i * n;
            for j in 0..n {
                num += h[j] * a_col[i] as f64 * residual[row + j] as f64 * b_col[j] as f64;
            }
        }
        scales[k] = (num / (m as f64 * h_sum)) as f32;

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

    // --- Phase 2: Weighted coordinate descent refinement ---
    let refinement_sweeps = 5;
    for _sweep in 0..refinement_sweeps {
        // Recompute full residual from scratch
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

            // Add back this slice
            for i in 0..m {
                let a_val = get_binary(&a_packed[a_off..a_off + a_words], i);
                let row = i * n;
                for j in 0..n {
                    let b_val = get_binary(&b_packed[b_off..b_off + b_words], j);
                    residual[row + j] += old_s * a_val * b_val;
                }
            }

            // Extract B[:,k]
            for j in 0..n {
                b_col[j] = get_binary(&b_packed[b_off..b_off + b_words], j);
            }

            // Weighted re-optimization
            for _iter in 0..max_iters {
                for i in 0..m {
                    let mut dot = 0.0f64;
                    let row = i * n;
                    for j in 0..n {
                        dot += h[j] * residual[row + j] as f64 * b_col[j] as f64;
                    }
                    a_col[i] = if dot >= 0.0 { 1.0 } else { -1.0 };
                }
                for j in 0..n {
                    let mut dot = 0.0f64;
                    for i in 0..m {
                        dot += residual[i * n + j] as f64 * a_col[i] as f64;
                    }
                    b_col[j] = if h[j] * dot >= 0.0 { 1.0 } else { -1.0 };
                }
            }

            // Weighted scale
            let mut num = 0.0f64;
            for i in 0..m {
                let row = i * n;
                for j in 0..n {
                    num += h[j] * a_col[i] as f64 * residual[row + j] as f64 * b_col[j] as f64;
                }
            }
            scales[k] = (num / (m as f64 * h_sum)) as f32;

            // Pack
            for i in 0..m { set_binary(&mut a_packed[a_off..a_off + a_words], i, a_col[i]); }
            for j in 0..n { set_binary(&mut b_packed[b_off..b_off + b_words], j, b_col[j]); }

            // Deflate
            let new_s = scales[k];
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

/// Compute weighted MSE: Σ h[j] · (W[i,j] - W'[i,j])^2 / (m · Σ h[j])
pub fn weighted_mse(w: &[f32], factors: &BinaryFactors, importance: &[f64]) -> f64 {
    let recon = reconstruct_cpu(factors);
    let m = factors.m;
    let n = factors.n;
    let h_sum: f64 = importance.iter().sum();
    let mut wmse = 0.0f64;

    for i in 0..m {
        let row = i * n;
        for j in 0..n {
            let d = (w[row + j] - recon[row + j]) as f64;
            wmse += importance[j] * d * d;
        }
    }

    wmse / (m as f64 * h_sum)
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

        let factors = admm_factorize(&w, 4, 4, 1, 10, None);
        let mse = factorization_mse(&w, &factors);
        assert!(mse < 1e-6, "Rank-1 should be perfect, MSE={}", mse);
    }

    #[test]
    fn column_importance_basic() {
        // 2×3 matrix: col 0 has large values, col 2 has small
        let w = vec![10.0, 1.0, 0.1,
                     10.0, 1.0, 0.1];
        let h = column_importance(&w, 2, 3);
        assert_eq!(h.len(), 3);
        // h[0] should be much larger than h[2]
        assert!(h[0] > h[2] * 50.0, "col0={} col2={}", h[0], h[2]);
    }

    #[test]
    fn weighted_factorization_focuses_on_important_cols() {
        // Matrix where col 0 has large values, col 1 has small
        let mut w = vec![0.0f32; 64]; // 8×8
        for i in 0..8 {
            w[i * 8] = 5.0;     // col 0: large
            w[i * 8 + 1] = 0.01; // col 1: tiny
        }

        let h = column_importance(&w, 8, 8);
        let factors_w = admm_factorize(&w, 8, 8, 1, 10, Some(&h));
        let factors_u = admm_factorize(&w, 8, 8, 1, 10, None);

        let wmse_w = weighted_mse(&w, &factors_w, &h);
        let wmse_u = weighted_mse(&w, &factors_u, &h);
        // Weighted solver should achieve lower weighted MSE
        assert!(wmse_w <= wmse_u * 1.01, "Weighted={} Uniform={}", wmse_w, wmse_u);
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
        assert!(factors.bits_per_param() < 0.001);
    }
}
