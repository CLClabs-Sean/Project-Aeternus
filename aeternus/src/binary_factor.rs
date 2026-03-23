//! # Low-Rank Binary Factorization via ADMM
//!
//! Approximate W ≈ A · diag(s) · B^T where:
//!   A ∈ {-1,+1}^{m×r}  (packed binary, 32 per u32)
//!   B ∈ {-1,+1}^{n×r}  (packed binary, 32 per u32)
//!   s ∈ R^r             (f32 scale per rank slice)
//!
//! Uses Gram-matrix ADMM with joint updates:
//!   A-step: A = sign(W · B · (B^T B)^{-1})
//!   B-step: B = sign(W^T · A · (A^T A)^{-1})
//!   S-step: s[k] = (A[:,k]^T · W · B[:,k]) / (m · n)  (global least-squares)
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
// Dense r×r matrix operations (for Gram matrix)
// ---------------------------------------------------------------------------

/// Compute Gram matrix G = X^T X where X is stored as packed binary columns.
/// X ∈ {-1,+1}^{dim × rank}, packed as `rank` columns of `words_per_col` u32s.
/// Result: G[i][j] stored row-major in Vec<f64>, size rank×rank.
fn gram_matrix(packed: &[u32], dim: usize, rank: usize, words_per_col: usize) -> Vec<f64> {
    let mut g = vec![0.0f64; rank * rank];

    for i in 0..rank {
        // Diagonal: G[i][i] = dim (all ±1 squared = 1)
        g[i * rank + i] = dim as f64;

        for j in (i + 1)..rank {
            // G[i][j] = X[:,i]^T · X[:,j] = sum of x_i * x_j
            // For binary: each matching pair contributes +1, each mismatch -1
            // So G[i][j] = 2 * popcount(xnor(a,b)) - dim
            let col_i = i * words_per_col;
            let col_j = j * words_per_col;
            let mut dot = 0i64;

            for w in 0..words_per_col {
                let xnor = !(packed[col_i + w] ^ packed[col_j + w]);
                let matching = xnor.count_ones() as i64;
                // Each word has 32 bits; matching bits contribute +1, mismatches -1
                dot += 2 * matching - 32;
            }
            // Correct for padding bits in the last word
            let padding = words_per_col * 32 - dim;
            if padding > 0 {
                // The padding bits are all 0 in both columns, so xnor gives 1s
                // We need to subtract the overcounting: padding bits counted as matching
                dot -= padding as i64; // remove the +1s from padding
                dot += padding as i64; // add back as -1s... wait
                // Actually: padding bits are 0 in both → xnor = 1 → counted as matching
                // But they shouldn't exist in the vector. We overcounted by `padding` matches.
                // Correct: dot -= 2 * padding (each padding bit was +1, should be 0, so remove +1 and there's no -1)
                // Simpler: just recompute correctly
            }

            // Actually, let's just do the simple dense computation for correctness
            let mut dot_val = 0.0f64;
            for d in 0..dim {
                let a = get_binary(&packed[col_i..col_i + words_per_col], d);
                let b = get_binary(&packed[col_j..col_j + words_per_col], d);
                dot_val += a as f64 * b as f64;
            }

            g[i * rank + j] = dot_val;
            g[j * rank + i] = dot_val; // symmetric
        }
    }

    g
}

/// Solve G x = b via Cholesky (G is symmetric positive definite).
/// G is rank×rank row-major. b is length rank. Returns x.
fn cholesky_solve(g: &[f64], b: &[f64], rank: usize) -> Vec<f64> {
    // Cholesky: G = L L^T
    let mut l = vec![0.0f64; rank * rank];

    for i in 0..rank {
        for j in 0..=i {
            let mut sum = 0.0f64;
            for k in 0..j {
                sum += l[i * rank + k] * l[j * rank + k];
            }
            if i == j {
                let diag = g[i * rank + i] - sum;
                l[i * rank + j] = if diag > 0.0 { diag.sqrt() } else { 1e-10 };
            } else {
                l[i * rank + j] = (g[i * rank + j] - sum) / l[j * rank + j];
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0f64; rank];
    for i in 0..rank {
        let mut sum = 0.0f64;
        for j in 0..i {
            sum += l[i * rank + j] * y[j];
        }
        y[i] = (b[i] - sum) / l[i * rank + i];
    }

    // Back substitution: L^T x = y
    let mut x = vec![0.0f64; rank];
    for i in (0..rank).rev() {
        let mut sum = 0.0f64;
        for j in (i + 1)..rank {
            sum += l[j * rank + i] * x[j]; // L^T[i][j] = L[j][i]
        }
        x[i] = (y[i] - sum) / l[i * rank + i];
    }

    x
}

// ---------------------------------------------------------------------------
// ADMM Solver with Gram-Matrix Joint Updates
// ---------------------------------------------------------------------------

/// Factorize W[m×n] into binary factors A, B and scales s.
///
/// Phase 1: Greedy rank-1 deflation for initialization.
/// Phase 2: Gram-matrix ADMM joint refinement.
pub fn admm_factorize(w: &[f32], m: usize, n: usize, rank: usize, max_iters: usize) -> BinaryFactors {
    assert_eq!(w.len(), m * n, "Weight matrix size mismatch");
    assert!(rank > 0 && rank <= m.min(n), "Rank must be in [1, min(m,n)]");

    let a_words = words_for(m);
    let b_words = words_for(n);

    let mut a_packed = vec![0u32; a_words * rank];
    let mut b_packed = vec![0u32; b_words * rank];
    let mut scales = vec![0.0f32; rank];

    // Temp dense vectors
    let mut a_col = vec![0.0f32; m];
    let mut b_col = vec![0.0f32; n];

    // --- Phase 1: Greedy rank-1 deflation (initialization) ---
    let mut residual = w.to_vec();
    for k in 0..rank {
        let b_off = k * b_words;
        let a_off = k * a_words;

        // Init B[:,k] randomly
        for j in 0..n {
            let hash = crate::seed_engine::pcg_hash((k * n + j) as u32);
            let val = if hash & 1 == 0 { 1.0f32 } else { -1.0 };
            set_binary(&mut b_packed[b_off..b_off + b_words], j, val);
            b_col[j] = val;
        }

        // Alternating A/B optimization on residual
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

        // Scale
        let mut a_r_b = 0.0f64;
        for i in 0..m {
            let mut row_dot = 0.0f64;
            let row = i * n;
            for j in 0..n { row_dot += residual[row + j] as f64 * b_col[j] as f64; }
            a_r_b += a_col[i] as f64 * row_dot;
        }
        let s_k = (a_r_b / (m as f64 * n as f64)) as f32;
        scales[k] = s_k;

        // Pack
        for i in 0..m { set_binary(&mut a_packed[a_off..a_off + a_words], i, a_col[i]); }
        for j in 0..n { set_binary(&mut b_packed[b_off..b_off + b_words], j, b_col[j]); }

        // Deflate
        for i in 0..m {
            let row = i * n;
            for j in 0..n { residual[row + j] -= s_k * a_col[i] * b_col[j]; }
        }
    }

    // --- Phase 2: Gram-matrix joint ADMM refinement ---
    // Instead of per-slice coordinate descent, do joint updates:
    //   A = sign(W · B · (B^T B)^{-1} · diag(s))
    //   B = sign(W^T · A · (A^T A)^{-1} · diag(s))
    //   s[k] = (A[:,k]^T · W · B[:,k]) / (m · n)

    let joint_sweeps = 3;
    for _sweep in 0..joint_sweeps {
        // --- Joint A-update ---
        // Compute B^T B (r×r Gram matrix)
        let g_b = gram_matrix(&b_packed, n, rank, b_words);

        // For each row i of A: A[i,:] = sign( Σ_j W[i,j] * B[j,:] * (B^T B)^{-1} * s )
        // = sign( (W[i,:] · B) · (B^T B)^{-1} · diag(s) )
        // First compute W[i,:] · B → vector of length r
        for i in 0..m {
            let row = i * n;
            // Compute rhs = W[i,:] · B (dense, length r)
            let mut rhs = vec![0.0f64; rank];
            for k in 0..rank {
                let b_off = k * b_words;
                let mut dot = 0.0f64;
                for j in 0..n {
                    dot += w[row + j] as f64 * get_binary(&b_packed[b_off..b_off + b_words], j) as f64;
                }
                rhs[k] = dot;
            }

            // Solve (B^T B) z = rhs → z = (B^T B)^{-1} rhs
            let z = cholesky_solve(&g_b, &rhs, rank);

            // A[i,k] = sign(z[k] * s[k])
            for k in 0..rank {
                let val = z[k] * scales[k] as f64;
                let a_off = k * a_words;
                set_binary(&mut a_packed[a_off..a_off + a_words], i, if val >= 0.0 { 1.0 } else { -1.0 });
            }
        }

        // --- Joint B-update ---
        let g_a = gram_matrix(&a_packed, m, rank, a_words);

        for j in 0..n {
            // Compute rhs = W[:,j]^T · A (length r)
            let mut rhs = vec![0.0f64; rank];
            for k in 0..rank {
                let a_off = k * a_words;
                let mut dot = 0.0f64;
                for i in 0..m {
                    dot += w[i * n + j] as f64 * get_binary(&a_packed[a_off..a_off + a_words], i) as f64;
                }
                rhs[k] = dot;
            }

            let z = cholesky_solve(&g_a, &rhs, rank);

            for k in 0..rank {
                let val = z[k] * scales[k] as f64;
                let b_off = k * b_words;
                set_binary(&mut b_packed[b_off..b_off + b_words], j, if val >= 0.0 { 1.0 } else { -1.0 });
            }
        }

        // --- Joint S-update (global least-squares) ---
        for k in 0..rank {
            let a_off = k * a_words;
            let b_off = k * b_words;
            let mut a_w_b = 0.0f64;
            for i in 0..m {
                let a_val = get_binary(&a_packed[a_off..a_off + a_words], i) as f64;
                let row = i * n;
                let mut row_dot = 0.0f64;
                for j in 0..n {
                    row_dot += w[row + j] as f64 * get_binary(&b_packed[b_off..b_off + b_words], j) as f64;
                }
                a_w_b += a_val * row_dot;
            }
            scales[k] = (a_w_b / (m as f64 * n as f64)) as f32;
        }
    }

    BinaryFactors { a_packed, b_packed, scales, rank, m, n }
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
    fn cholesky_solve_identity() {
        // 2×2 identity: solve I x = [3, 7] → x = [3, 7]
        let g = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 7.0];
        let x = cholesky_solve(&g, &b, 2);
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 7.0).abs() < 1e-10);
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
