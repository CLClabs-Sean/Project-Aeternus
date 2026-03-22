//! # Low-Rank Binary Factorization via ADMM
//!
//! Approximate W ≈ A · diag(s) · B^T where:
//!   A ∈ {-1,+1}^{m×r}  (packed binary, 32 per u32)
//!   B ∈ {-1,+1}^{n×r}  (packed binary, 32 per u32)
//!   s ∈ R^r             (f32 scale per rank slice)
//!
//! ADMM alternates:
//!   A-step: A = sign(W · B · diag(s))
//!   B-step: B = sign(W^T · A · diag(s))
//!   S-step: s_k = (A[:,k]^T · W · B[:,k]) / (m · n)
//!
//! Storage: r(m + n) bits + r×32 bits ≈ 2r/n bits/param for square matrices.

/// Packed binary factors for a single weight matrix.
#[derive(Debug)]
pub struct BinaryFactors {
    /// Packed A matrix: {-1,+1}^{m×r}, stored as bits (1=+1, 0=-1).
    /// Layout: r columns, each column is ceil(m/32) u32 words.
    pub a_packed: Vec<u32>,
    /// Packed B matrix: {-1,+1}^{n×r}, same layout.
    pub b_packed: Vec<u32>,
    /// Scale vector s ∈ R^r.
    pub scales: Vec<f32>,
    pub rank: usize,
    pub m: usize, // rows of original W
    pub n: usize, // cols of original W
}

impl BinaryFactors {
    /// Total storage in bits.
    pub fn storage_bits(&self) -> usize {
        let a_bits = self.m * self.rank;
        let b_bits = self.n * self.rank;
        let s_bits = self.rank * 32;
        a_bits + b_bits + s_bits
    }

    /// Bits per parameter.
    pub fn bits_per_param(&self) -> f64 {
        self.storage_bits() as f64 / (self.m * self.n) as f64
    }

    /// Total storage in bytes.
    pub fn storage_bytes(&self) -> usize {
        (self.storage_bits() + 7) / 8
    }
}

// ---------------------------------------------------------------------------
// Bit packing helpers
// ---------------------------------------------------------------------------

/// Get binary value at position `idx` from packed array. Returns +1.0 or -1.0.
#[inline]
fn get_binary(packed: &[u32], idx: usize) -> f32 {
    let word = packed[idx / 32];
    let bit = (word >> (idx % 32)) & 1;
    if bit == 1 { 1.0 } else { -1.0 }
}

/// Set binary value at position `idx`. value >= 0 → 1 (represents +1), else 0 (-1).
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

/// Words needed to pack `n` binary values.
fn words_for(n: usize) -> usize {
    (n + 31) / 32
}

// ---------------------------------------------------------------------------
// ADMM Solver
// ---------------------------------------------------------------------------

/// Factorize W[m×n] into binary factors A, B and scales s using ADMM.
///
/// Frobenius-norm minimization (no calibration data needed):
///   minimize ||W - A · diag(s) · B^T||_F
///
/// `rank` controls the approximation quality and storage cost.
/// `max_iters` is the number of ADMM alternating steps.
pub fn admm_factorize(w: &[f32], m: usize, n: usize, rank: usize, max_iters: usize) -> BinaryFactors {
    assert_eq!(w.len(), m * n, "Weight matrix size mismatch");
    assert!(rank > 0 && rank <= m.min(n), "Rank must be in [1, min(m,n)]");

    let a_words_per_col = words_for(m);
    let b_words_per_col = words_for(n);

    let mut a_packed = vec![0u32; a_words_per_col * rank];
    let mut b_packed = vec![0u32; b_words_per_col * rank];
    let mut scales = vec![0.0f32; rank];

    // --- Initialization ---
    // Initialize B with random binary values (deterministic via PCG hash)
    for k in 0..rank {
        let col_offset = k * b_words_per_col;
        for j in 0..n {
            let hash = crate::seed_engine::pcg_hash((k * n + j) as u32);
            let val = if hash & 1 == 0 { 1.0f32 } else { -1.0 };
            set_binary(&mut b_packed[col_offset..col_offset + b_words_per_col], j, val);
        }
    }

    // Temporary dense columns for matrix operations
    let mut a_col = vec![0.0f32; m];
    let mut b_col = vec![0.0f32; n];

    // Residual: R = W (we'll update it as we go for greedy rank-1 updates)
    let mut residual = w.to_vec();

    // --- Greedy rank-1 sweeps ---
    // For each rank slice, find best binary rank-1 approximation of the residual
    for k in 0..rank {
        let b_col_offset = k * b_words_per_col;
        let a_col_offset = k * a_words_per_col;

        // Extract B[:,k] as dense
        for j in 0..n {
            b_col[j] = get_binary(&b_packed[b_col_offset..b_col_offset + b_words_per_col], j);
        }

        // Run alternating optimization for this rank slice on the residual
        for _iter in 0..max_iters {
            // A-step: a[:,k] = sign(R · b[:,k])
            for i in 0..m {
                let mut dot = 0.0f32;
                for j in 0..n {
                    dot += residual[i * n + j] * b_col[j];
                }
                a_col[i] = if dot >= 0.0 { 1.0 } else { -1.0 };
            }

            // B-step: b[:,k] = sign(R^T · a[:,k])
            for j in 0..n {
                let mut dot = 0.0f32;
                for i in 0..m {
                    dot += residual[i * n + j] * a_col[i];
                }
                b_col[j] = if dot >= 0.0 { 1.0 } else { -1.0 };
            }
        }

        // S-step: s[k] = (a[:,k]^T · R · b[:,k]) / (m * n)
        // Actually: s[k] = (a^T R b) / (||a||^2 · ||b||^2) = (a^T R b) / (m · n)
        // since ||a||^2 = m, ||b||^2 = n for binary ±1 vectors
        let mut aRb = 0.0f64;
        for i in 0..m {
            let mut row_dot = 0.0f64;
            for j in 0..n {
                row_dot += residual[i * n + j] as f64 * b_col[j] as f64;
            }
            aRb += a_col[i] as f64 * row_dot;
        }
        let s_k = (aRb / (m as f64 * n as f64)) as f32;
        scales[k] = s_k;

        // Pack A[:,k] and B[:,k]
        for i in 0..m {
            set_binary(&mut a_packed[a_col_offset..a_col_offset + a_words_per_col], i, a_col[i]);
        }
        for j in 0..n {
            set_binary(&mut b_packed[b_col_offset..b_col_offset + b_words_per_col], j, b_col[j]);
        }

        // Update residual: R -= s[k] * a[:,k] * b[:,k]^T
        for i in 0..m {
            for j in 0..n {
                residual[i * n + j] -= s_k * a_col[i] * b_col[j];
            }
        }
    }

    BinaryFactors {
        a_packed,
        b_packed,
        scales,
        rank,
        m,
        n,
    }
}

/// Reconstruct the full weight matrix from binary factors (CPU, for verification).
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
        let mut packed = vec![0u32; 2]; // 64 bits
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
        // Rank-1 matrix: outer product of [1,1,-1,-1] and [1,-1,1,-1]
        let a = [1.0f32, 1.0, -1.0, -1.0];
        let b = [1.0f32, -1.0, 1.0, -1.0];
        let mut w = vec![0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                w[i * 4 + j] = a[i] * b[j] * 0.5; // scale = 0.5
            }
        }

        let factors = admm_factorize(&w, 4, 4, 1, 10);
        let mse = factorization_mse(&w, &factors);
        assert!(mse < 1e-6, "Rank-1 matrix should be perfectly recovered, MSE={}", mse);
    }

    #[test]
    fn storage_calc() {
        let factors = BinaryFactors {
            a_packed: vec![0; 64],  // 2048 bits = 2048 elements
            b_packed: vec![0; 64],
            scales: vec![0.0; 1],
            rank: 1,
            m: 2048,
            n: 2048,
        };
        let bpp = factors.bits_per_param();
        // rank=1: (2048 + 2048 + 32) / (2048*2048) ≈ 0.00098
        assert!(bpp < 0.001);
    }
}
