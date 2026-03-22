//! # Walsh-Hadamard Transform (WHT)
//!
//! In-place fast Walsh-Hadamard transform for weight rotation.
//! Used to de-randomize weight signs before quantization (Phase 7.6).
//!
//! Mathematical basis: y = Wx = (WH^T)(Hx) = W'x'
//! - W' = WH^T (rotated weights, stored)
//! - x' = Hx   (rotated input, computed at inference)

/// In-place normalized Walsh-Hadamard Transform.
///
/// Input length MUST be a power of 2. The transform is its own inverse
/// (up to normalization), so we apply 1/sqrt(n) scaling.
pub fn wht_inplace(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two(), "WHT requires power-of-2 length, got {}", n);

    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let a = data[j];
                let b = data[j + h];
                data[j] = a + b;
                data[j + h] = a - b;
            }
        }
        h *= 2;
    }

    // Normalize by 1/sqrt(n)
    let scale = 1.0 / (n as f32).sqrt();
    for v in data.iter_mut() {
        *v *= scale;
    }
}

/// Pad a slice to the next power of 2, apply WHT, return only the original length.
fn wht_padded(data: &[f32]) -> Vec<f32> {
    let n = data.len();
    if n.is_power_of_two() {
        let mut buf = data.to_vec();
        wht_inplace(&mut buf);
        buf
    } else {
        let padded_len = n.next_power_of_two();
        let mut buf = vec![0.0f32; padded_len];
        buf[..n].copy_from_slice(data);
        wht_inplace(&mut buf);
        // Return full padded result — caller must track original dimension
        buf
    }
}

/// Rotate weight matrix rows using WHT.
///
/// For a weight matrix W[rows × cols], applies WHT to each row independently.
/// If cols is not a power of 2, pads each row to the next power of 2.
///
/// Returns (rotated_weights, padded_cols) where padded_cols is the actual
/// column dimension after padding.
pub fn rotate_weight_rows(weights: &[f32], rows: usize, cols: usize) -> (Vec<f32>, usize) {
    let padded_cols = if cols.is_power_of_two() { cols } else { cols.next_power_of_two() };
    let mut rotated = vec![0.0f32; rows * padded_cols];

    for r in 0..rows {
        let src_start = r * cols;
        let dst_start = r * padded_cols;
        // Copy row data (rest is already zero = padding)
        rotated[dst_start..dst_start + cols].copy_from_slice(&weights[src_start..src_start + cols]);
        // WHT in-place on the padded row
        wht_inplace(&mut rotated[dst_start..dst_start + padded_cols]);
    }

    (rotated, padded_cols)
}

/// Rotate input vector using WHT (applied before GEMV dispatch).
///
/// If x.len() is not a power of 2, pads to padded_cols and rotates.
/// Returns the rotated vector with length = padded_cols.
pub fn rotate_input(x: &[f32], padded_cols: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; padded_cols];
    let copy_len = x.len().min(padded_cols);
    buf[..copy_len].copy_from_slice(&x[..copy_len]);
    wht_inplace(&mut buf);
    buf
}

/// Measure sign entropy of a weight vector.
///
/// Returns (positive_fraction, average_run_length).
/// Perfectly random signs: positive_fraction ≈ 0.5, avg_run ≈ 2.0.
/// Clustered signs: positive_fraction varies, avg_run >> 2.0.
pub fn sign_stats(weights: &[f32]) -> (f64, f64) {
    if weights.is_empty() {
        return (0.5, 1.0);
    }

    let n = weights.len();
    let positive_count = weights.iter().filter(|&&w| w >= 0.0).count();
    let positive_frac = positive_count as f64 / n as f64;

    // Count sign runs
    let mut run_count = 1usize;
    for i in 1..n {
        if (weights[i] >= 0.0) != (weights[i - 1] >= 0.0) {
            run_count += 1;
        }
    }
    let avg_run = n as f64 / run_count as f64;

    (positive_frac, avg_run)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wht_identity_roundtrip() {
        // WHT applied twice (with normalization) should return the original.
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut data = original.clone();
        wht_inplace(&mut data);
        wht_inplace(&mut data);
        for (a, b) in original.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-5, "Roundtrip failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn wht_known_values() {
        // WHT of [1, 1, 1, 1] = [2, 0, 0, 0] (normalized by 1/sqrt(4) = 0.5)
        let mut data = vec![1.0, 1.0, 1.0, 1.0];
        wht_inplace(&mut data);
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 0.0).abs() < 1e-5);
        assert!((data[2] - 0.0).abs() < 1e-5);
        assert!((data[3] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn rotate_rows_preserves_dimensions() {
        let weights = vec![1.0; 4 * 8]; // 4 rows × 8 cols
        let (rotated, padded_cols) = rotate_weight_rows(&weights, 4, 8);
        assert_eq!(padded_cols, 8); // 8 is already power of 2
        assert_eq!(rotated.len(), 4 * 8);
    }

    #[test]
    fn rotate_rows_pads_non_power_of_2() {
        let weights = vec![1.0; 3 * 5]; // 3 rows × 5 cols
        let (rotated, padded_cols) = rotate_weight_rows(&weights, 3, 5);
        assert_eq!(padded_cols, 8); // next power of 2 after 5
        assert_eq!(rotated.len(), 3 * 8);
    }

    #[test]
    fn sign_stats_random() {
        // Random-ish signs should have avg_run ≈ 2.0
        let data: Vec<f32> = (0..1000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let (pos_frac, avg_run) = sign_stats(&data);
        assert!((pos_frac - 0.5).abs() < 0.01);
        assert!((avg_run - 1.0).abs() < 0.01); // alternating = run length 1
    }
}
