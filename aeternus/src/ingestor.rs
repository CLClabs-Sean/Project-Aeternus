//! # Safetensors Ingestor
//!
//! Zero-copy weight loading from HuggingFace `.safetensors` files.
//! Memory-maps files via `memmap2`, parses the JSON header, then
//! quantizes FP16/BF16 weight tensors to 2-bit VQ magnitude indices.
//!
//! Pipeline: `.safetensors` → mmap → FP16 → quantize → pack → PackedLayer

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use half::f16;
use memmap2::Mmap;

use crate::codebook::{self, Codebook};
use crate::micro_model::{Activation, MicroModel, PackedLayer};

// ---------------------------------------------------------------------------
// Safetensors format: [8-byte header_len][JSON header][data block]
// ---------------------------------------------------------------------------

/// Metadata for a single tensor within a safetensors file.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize), // (begin, end) relative to data block start
}

/// A memory-mapped safetensors file with parsed metadata.
pub struct SafetensorsFile {
    _mmap: Mmap,
    data_start: usize,
    tensors: HashMap<String, TensorMeta>,
}

impl SafetensorsFile {
    /// Open and memory-map a safetensors file.
    pub fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err("File too small for safetensors header".into());
        }

        // First 8 bytes: little-endian u64 = header size
        let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let data_start = 8 + header_len;

        if mmap.len() < data_start {
            return Err(format!(
                "File too small: header says {} bytes, file is {} bytes",
                data_start, mmap.len()
            ).into());
        }

        // Parse JSON header
        let header_json = std::str::from_utf8(&mmap[8..data_start])?;
        let header: serde_json::Value = serde_json::from_str(header_json)?;

        let mut tensors = HashMap::new();
        if let serde_json::Value::Object(map) = &header {
            for (name, info) in map {
                // Skip __metadata__ key
                if name == "__metadata__" {
                    continue;
                }

                let dtype = info["dtype"]
                    .as_str()
                    .ok_or_else(|| format!("Missing dtype for tensor '{}'", name))?
                    .to_string();

                let shape: Vec<usize> = info["shape"]
                    .as_array()
                    .ok_or_else(|| format!("Missing shape for tensor '{}'", name))?
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect();

                let offsets = info["data_offsets"]
                    .as_array()
                    .ok_or_else(|| format!("Missing data_offsets for tensor '{}'", name))?;
                let begin = offsets[0].as_u64().unwrap() as usize;
                let end = offsets[1].as_u64().unwrap() as usize;

                tensors.insert(name.clone(), TensorMeta {
                    name: name.clone(),
                    dtype,
                    shape,
                    data_offsets: (begin, end),
                });
            }
        }

        log::info!("Loaded safetensors: {} tensors, data starts at byte {}",
            tensors.len(), data_start);

        Ok(Self {
            _mmap: mmap,
            data_start,
            tensors,
        })
    }

    /// List all tensor names in the file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Get metadata for a named tensor.
    pub fn tensor_meta(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.get(name)
    }

    /// Get raw bytes for a named tensor (zero-copy from mmap).
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let meta = self.tensors.get(name)?;
        let start = self.data_start + meta.data_offsets.0;
        let end = self.data_start + meta.data_offsets.1;
        Some(&self._mmap[start..end])
    }

    /// Read a tensor as f32 values, converting from FP16/BF16/F32 as needed.
    pub fn tensor_as_f32(&self, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let meta = self.tensors.get(name)
            .ok_or_else(|| format!("Tensor '{}' not found", name))?;
        let data = self.tensor_data(name).unwrap();

        match meta.dtype.as_str() {
            "F16" => {
                let f16_slice: &[f16] = bytemuck::cast_slice(data);
                Ok(f16_slice.iter().map(|v| v.to_f32()).collect())
            }
            "BF16" => {
                let bf16_slice: &[half::bf16] = bytemuck::cast_slice(data);
                Ok(bf16_slice.iter().map(|v| v.to_f32()).collect())
            }
            "F32" => {
                let f32_slice: &[f32] = bytemuck::cast_slice(data);
                Ok(f32_slice.to_vec())
            }
            other => Err(format!("Unsupported dtype: {}", other).into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-shard loader
// ---------------------------------------------------------------------------

/// Load all safetensors shards from a directory.
pub fn load_shards(dir: &Path) -> Result<Vec<SafetensorsFile>, Box<dyn std::error::Error>> {
    let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().map(|e| e == "safetensors").unwrap_or(false) {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    shard_paths.sort(); // Ensure deterministic order
    log::info!("Found {} safetensors shards in {:?}", shard_paths.len(), dir);

    let mut shards = Vec::new();
    for path in &shard_paths {
        log::info!("  Loading shard: {:?}", path.file_name().unwrap());
        shards.push(SafetensorsFile::open(path)?);
    }
    Ok(shards)
}

/// Find a tensor by name across all shards.
fn find_tensor<'a>(shards: &'a [SafetensorsFile], name: &str) -> Option<&'a SafetensorsFile> {
    shards.iter().find(|s| s.tensor_meta(name).is_some())
}

// ---------------------------------------------------------------------------
// VQ Quantization (CPU, per-layer calibrated)
// ---------------------------------------------------------------------------

/// Calibrate a 4-level codebook from weight statistics.
///
/// Uses half-normal quartile boundaries:
///   level 0 = 0.0       (zero/sparse)
///   level 1 = 0.38σ     (small)
///   level 2 = 0.98σ     (medium)
///   level 3 = 2.05σ     (large)
pub fn calibrate_codebook(weights: &[f32]) -> Codebook {
    let n = weights.len() as f64;
    if n == 0.0 {
        return Codebook::default();
    }

    // Compute std of absolute values (half-normal)
    let abs_mean: f64 = weights.iter().map(|w| w.abs() as f64).sum::<f64>() / n;
    let abs_var: f64 = weights.iter()
        .map(|w| {
            let d = w.abs() as f64 - abs_mean;
            d * d
        })
        .sum::<f64>() / n;
    let sigma = abs_var.sqrt() as f32;

    if sigma < 1e-10 {
        return Codebook::default();
    }

    // Half-normal quartile centroids (not boundaries — centroids within each quartile)
    Codebook::new([
        0.0,
        0.38 * sigma,
        0.98 * sigma,
        2.05 * sigma,
    ])
}

/// Quantize f32 weights to 2-bit magnitude indices using nearest-centroid.
/// Returns (magnitude_indices, sign_bits) where sign_bits[i] = 1 if weight[i] < 0.
pub fn quantize_layer(weights: &[f32], codebook: &Codebook) -> (Vec<u8>, Vec<u8>) {
    let mut indices = Vec::with_capacity(weights.len());
    let mut signs = Vec::with_capacity(weights.len());

    for &w in weights {
        // Separate sign and magnitude
        signs.push(if w < 0.0 { 1u8 } else { 0u8 });
        let mag = w.abs();

        // Nearest-centroid search
        let mut best_idx = 0u8;
        let mut best_dist = f32::MAX;
        for (k, &centroid) in codebook.magnitudes.iter().enumerate() {
            let dist = (mag - centroid).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = k as u8;
            }
        }
        indices.push(best_idx);
    }

    (indices, signs)
}

/// Pack sign bits into u32 words (32 signs per word).
pub fn pack_signs(signs: &[u8]) -> Vec<u32> {
    let num_words = (signs.len() + 31) / 32;
    let mut packed = vec![0u32; num_words];
    for (i, &s) in signs.iter().enumerate() {
        if s != 0 {
            packed[i / 32] |= 1 << (i % 32);
        }
    }
    packed
}

// ---------------------------------------------------------------------------
// Llama 3 Ingestor
// ---------------------------------------------------------------------------

/// Llama 3 architecture config.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
}

impl LlamaConfig {
    /// Llama 3 8B default config.
    pub fn llama3_8b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 14336,
            num_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
        }
    }

    /// TinyLlama 1.1B config (for testing).
    pub fn tiny_llama() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 5632,
            num_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: 4,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }
}

/// Layer tensor names for a given Llama layer index.
fn llama_layer_tensors(layer_idx: usize) -> Vec<(String, Activation)> {
    let prefix = format!("model.layers.{}", layer_idx);
    vec![
        // Self-attention projections
        (format!("{}.self_attn.q_proj.weight", prefix), Activation::None),
        (format!("{}.self_attn.k_proj.weight", prefix), Activation::None),
        (format!("{}.self_attn.v_proj.weight", prefix), Activation::None),
        (format!("{}.self_attn.o_proj.weight", prefix), Activation::None),
        // MLP (SwiGLU)
        (format!("{}.mlp.gate_proj.weight", prefix), Activation::SiLU),
        (format!("{}.mlp.up_proj.weight", prefix), Activation::None),
        (format!("{}.mlp.down_proj.weight", prefix), Activation::None),
    ]
}

/// Ingest a Llama model from safetensors shards.
///
/// Returns a MicroModel with real quantized weights and optimized sign seeds.
pub fn ingest_llama(
    weights_dir: &Path,
    config: &LlamaConfig,
) -> Result<MicroModel, Box<dyn std::error::Error>> {
    log::info!("Ingesting Llama model from {:?}", weights_dir);
    log::info!("  Config: {:?}", config);

    let shards = load_shards(weights_dir)?;

    let mut layers = Vec::new();
    let mut layer_seeds: Vec<u32> = Vec::new();
    let mut total_corrections = 0usize;
    let mut total_weights = 0usize;
    let mut total_mse = 0.0f64;

    for layer_idx in 0..config.num_layers {
        let tensor_specs = llama_layer_tensors(layer_idx);

        for (tensor_name, activation) in &tensor_specs {
            let shard = find_tensor(&shards, tensor_name)
                .ok_or_else(|| format!("Tensor '{}' not found in any shard", tensor_name))?;

            let meta = shard.tensor_meta(tensor_name).unwrap();
            log::info!("  Layer {}: {} — shape {:?}, dtype {}",
                layer_idx, tensor_name, meta.shape, meta.dtype);

            // Read tensor as f32
            let f32_data = shard.tensor_as_f32(tensor_name)?;

            // Determine dimensions (safetensors stores as [rows, cols])
            let rows = meta.shape[0] as u32;
            let cols = if meta.shape.len() > 1 { meta.shape[1] as u32 } else { 1 };

            // Phase 6: K-Means codebook calibration
            let codebook = codebook::kmeans_4(&f32_data, 10);
            let old_codebook = calibrate_codebook(&f32_data);
            let kmeans_mse = codebook::quantization_mse(&f32_data, &codebook);
            let quartile_mse = codebook::quantization_mse(&f32_data, &old_codebook);
            let improvement = if quartile_mse > 0.0 { (1.0 - kmeans_mse / quartile_mse) * 100.0 } else { 0.0 };
            log::info!("    codebook: [{:.4}, {:.4}, {:.4}, {:.4}]  MSE: {:.6} (vs quartile {:.6}, {:.1}% better)",
                codebook.magnitudes[0], codebook.magnitudes[1],
                codebook.magnitudes[2], codebook.magnitudes[3],
                kmeans_mse, quartile_mse, improvement);
            total_mse += kmeans_mse * f32_data.len() as f64;

            // Quantize to 2-bit magnitude + 1-bit sign
            let (magnitude_indices, sign_bits) = quantize_layer(&f32_data, &codebook);

            // Phase 7: Optimize PCG seed for sign alignment
            let sign_data = crate::sign_aligner::optimize_seed(&sign_bits, 10_000);

            total_corrections += sign_data.correction_count;
            total_weights += sign_data.weight_count;
            layer_seeds.push(sign_data.seed);

            // Pack magnitudes (16 per u32)
            let packed_magnitudes = codebook::pack_indices(&magnitude_indices);

            layers.push(PackedLayer {
                packed_weights: packed_magnitudes,
                rows,
                cols,
                activation: *activation,
                correction_mask: Some(sign_data.correction_mask),
                codebook,
            });
        }

        if (layer_idx + 1) % 4 == 0 || layer_idx == config.num_layers - 1 {
            let running_rate = 1.0 - (total_corrections as f64 / total_weights as f64);
            log::info!("  Progress: {}/{} layers | cumulative match {:.2}% | {:.3} sign bits/param",
                layer_idx + 1, config.num_layers,
                running_rate * 100.0,
                total_corrections as f64 / total_weights as f64);
        }
    }

    // Use per-layer seeds (store the first one as global for compatibility)
    let global_seed = layer_seeds.first().copied().unwrap_or(0);
    let global_codebook = Codebook::default();

    let model = MicroModel::from_layers(
        "llama3",
        layers,
        global_seed,
        global_codebook,
        false, // Phase 7.6 reverted: Hadamard rotation disabled (negative result)
    );

    let sign_bits_per_param = total_corrections as f64 / total_weights as f64;
    let mag_bits = 2.0;
    let total_bits = mag_bits + sign_bits_per_param;
    let avg_mse = if total_weights > 0 { total_mse / total_weights as f64 } else { 0.0 };

    log::info!("Ingestion complete:");
    log::info!("  Total params: {}", model.total_params());
    log::info!("  Packed bytes: {} ({:.2} bits/param)",
        model.packed_bytes(), total_bits);
    log::info!("  Magnitude: {:.1} bits/param (k-means MSE: {:.6})", mag_bits, avg_mse);
    log::info!("  Sign corrections: {} / {} ({:.3} bits/param)",
        total_corrections, total_weights, sign_bits_per_param);

    Ok(model)
}

// ---------------------------------------------------------------------------
// Phase 8a: Binary Factorization Quality Benchmark
// ---------------------------------------------------------------------------

/// Run ADMM binary factorization on weight tensors and compare MSE vs k-means VQ.
pub fn ingest_binary_quality(
    weights_dir: &Path,
    config: &LlamaConfig,
    rank: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let shards = load_shards(weights_dir)?;
    let max_tensors = 4;
    let admm_iters = 10;
    let mut tensor_count = 0;

    log::info!("Binary factorization benchmark: rank={}, max_tensors={}", rank, max_tensors);

    for layer_idx in 0..config.num_layers {
        if tensor_count >= max_tensors { break; }
        let tensor_specs = llama_layer_tensors(layer_idx);

        for (tensor_name, _activation) in &tensor_specs {
            if tensor_count >= max_tensors { break; }

            let shard = match find_tensor(&shards, tensor_name) {
                Some(s) => s,
                None => continue,
            };

            let meta = shard.tensor_meta(tensor_name).unwrap();
            let f32_data = shard.tensor_as_f32(tensor_name)?;
            let rows = meta.shape[0];
            let cols = if meta.shape.len() > 1 { meta.shape[1] } else { 1 };

            log::info!("Tensor: {} [{} x {}] ({} params)", tensor_name, rows, cols, f32_data.len());

            // K-Means VQ baseline
            let vq_start = std::time::Instant::now();
            let codebook = codebook::kmeans_4(&f32_data, 10);
            let vq_mse = codebook::quantization_mse(&f32_data, &codebook);
            let vq_time = vq_start.elapsed();

            let effective_rank = rank.min(rows).min(cols);

            // Compute column importance (Hessian proxy)
            let importance = crate::binary_factor::column_importance(&f32_data, rows, cols);
            let (h_min, h_max, h_ratio) = crate::binary_factor::importance_stats(&importance);
            log::info!("  Importance: min={:.4} max={:.4} dynamic_range={:.1}x", h_min, h_max, h_ratio);

            // ADMM binary factorization with importance weighting
            let admm_start = std::time::Instant::now();
            let factors = crate::binary_factor::admm_factorize(
                &f32_data, rows, cols, effective_rank, admm_iters,
                Some(&importance),
            );
            let binary_mse = crate::binary_factor::factorization_mse(&f32_data, &factors);
            let weighted_mse = crate::binary_factor::weighted_mse(&f32_data, &factors, &importance);
            let admm_time = admm_start.elapsed();

            let mse_ratio = if vq_mse > 0.0 { binary_mse / vq_mse } else { f64::INFINITY };
            let bpp = factors.bits_per_param();

            log::info!("  VQ (k-means):  MSE={:.8}  bpp=2.50  time={:.2}s", vq_mse, vq_time.as_secs_f64());
            log::info!("  Binary (r={}): MSE={:.8}  wMSE={:.8}  bpp={:.3}  time={:.2}s  ratio={:.2}x",
                effective_rank, binary_mse, weighted_mse, bpp, admm_time.as_secs_f64(), mse_ratio);

            if binary_mse < vq_mse {
                log::info!("  >> BINARY WINS ({:.1}% lower MSE at {:.2} bpp)",
                    (1.0 - binary_mse / vq_mse) * 100.0, bpp);
            } else {
                log::info!("  >> VQ WINS ({:.1}% lower MSE at 2.50 bpp)",
                    (1.0 - vq_mse / binary_mse) * 100.0);
            }

            tensor_count += 1;
        }
    }

    log::info!("Binary factorization benchmark complete ({} tensors)", tensor_count);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibrate_codebook() {
        // Normal-ish distribution
        let weights: Vec<f32> = (0..10000)
            .map(|i| {
                let x = (i as f32 - 5000.0) / 1000.0;
                x * 0.1 // small weights
            })
            .collect();
        let cb = calibrate_codebook(&weights);
        assert!(cb.magnitudes[0] == 0.0);
        assert!(cb.magnitudes[1] > 0.0);
        assert!(cb.magnitudes[2] > cb.magnitudes[1]);
        assert!(cb.magnitudes[3] > cb.magnitudes[2]);
    }

    #[test]
    fn test_quantize_roundtrip() {
        let codebook = Codebook::new([0.0, 0.25, 0.75, 1.50]);
        let weights = vec![0.1, -0.3, 0.8, -1.2, 0.0, 1.6];
        let (indices, signs) = quantize_layer(&weights, &codebook);

        // Check indices are reasonable
        assert_eq!(indices.len(), 6);
        assert!(indices.iter().all(|&i| i <= 3));

        // Check signs
        assert_eq!(signs, vec![0, 1, 0, 1, 0, 0]);
    }

    #[test]
    fn test_pack_signs() {
        let signs = vec![1u8, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let packed = pack_signs(&signs);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0x80000005); // bits 0, 2, 31 set
    }

    #[test]
    fn test_quantization_mse() {
        // Generate weights with known distribution
        let codebook = Codebook::new([0.0, 0.25, 0.75, 1.50]);
        let weights: Vec<f32> = (0..1000)
            .map(|i| ((i as f32 * 0.01).sin() * 1.5))
            .collect();

        let (indices, signs) = quantize_layer(&weights, &codebook);

        // Reconstruct and measure MSE
        let mut mse = 0.0f64;
        for (i, &w) in weights.iter().enumerate() {
            let reconstructed = codebook.magnitudes[indices[i] as usize]
                * if signs[i] == 1 { -1.0 } else { 1.0 };
            let err = (w as f64 - reconstructed as f64);
            mse += err * err;
        }
        mse /= weights.len() as f64;

        // MSE should be reasonable for 2-bit quantization
        assert!(mse < 0.5, "MSE too high: {}", mse);
    }
}
