//! # Micro Model Harness
//!
//! Tiny neural networks for end-to-end validation of the fused GEMV pipeline.
//! Each model is a sequence of packed weight layers with activations.
//! GPU forward pass is validated against CPU reference.

use ash::vk;
use crate::codebook::{self, Codebook};
use crate::seed_engine;
use crate::vulkan_fabric::VulkanContext;
use crate::vulkan_fabric::buffer::AllocatedBuffer;
use crate::vulkan_fabric::gemv_pipeline::{GemvPipeline, GemvPushConstants};
use crate::vulkan_fabric::activation_pipeline::{ActivationPipeline, ActivationPushConstants};

// ---------------------------------------------------------------------------
// Model definition
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    None,
    ReLU,
    SiLU,
}

impl Activation {
    fn mode(&self) -> u32 {
        match self {
            Activation::None => u32::MAX, // sentinel — skip dispatch
            Activation::ReLU => 0,
            Activation::SiLU => 1,
        }
    }

    fn apply_cpu(&self, x: f32) -> f32 {
        match self {
            Activation::None => x,
            Activation::ReLU => x.max(0.0),
            Activation::SiLU => x / (1.0 + (-x).exp()),
        }
    }
}

#[derive(Debug)]
pub struct PackedLayer {
    pub packed_weights: Vec<u32>,
    pub rows: u32,  // output dim
    pub cols: u32,  // input dim
    pub activation: Activation,
    /// Phase 7: XOR correction mask for sign alignment.
    /// None = use PCG seed directly (synthetic), Some = apply XOR corrections (real weights).
    pub correction_mask: Option<Vec<u32>>,
}

pub struct MicroModel {
    pub name: String,
    pub layers: Vec<PackedLayer>,
    pub seed: u32,
    pub codebook: Codebook,
}

impl MicroModel {
    /// Create a model from a list of (output_dim, input_dim, activation) tuples.
    pub fn new(name: &str, architecture: &[(u32, u32, Activation)], seed: u32) -> Self {
        let codebook = Codebook::default();
        let layers: Vec<PackedLayer> = architecture.iter().enumerate().map(|(i, &(rows, cols, act))| {
            let layer_seed = seed.wrapping_add(i as u32 * 12345);
            let total_weights = (rows as usize) * (cols as usize);
            let packed = codebook::generate_synthetic_packed(layer_seed, total_weights);
            PackedLayer {
                packed_weights: packed,
                rows,
                cols,
                activation: act,
                correction_mask: None,
            }
        }).collect();

        Self {
            name: name.to_string(),
            layers,
            seed,
            codebook,
        }
    }

    /// Create a model from pre-built layers (for ingested real weights).
    pub fn from_layers(name: &str, layers: Vec<PackedLayer>, seed: u32, codebook: Codebook) -> Self {
        Self {
            name: name.to_string(),
            layers,
            seed,
            codebook,
        }
    }

    /// Total parameters across all layers.
    pub fn total_params(&self) -> u64 {
        self.layers.iter().map(|l| l.rows as u64 * l.cols as u64).sum()
    }

    /// Packed size in bytes (weights + correction masks).
    pub fn packed_bytes(&self) -> usize {
        self.layers.iter().map(|l| {
            let weight_bytes = l.packed_weights.len() * 4;
            let mask_bytes = l.correction_mask.as_ref().map_or(0, |m| m.len() * 4);
            weight_bytes + mask_bytes
        }).sum()
    }
}

// ---------------------------------------------------------------------------
// Preset architectures
// ---------------------------------------------------------------------------

pub fn nano() -> MicroModel {
    MicroModel::new("nano", &[
        (128, 64, Activation::ReLU),
        (32, 128, Activation::None),
    ], 0xDEAD_BEEF)
}

pub fn micro() -> MicroModel {
    MicroModel::new("micro", &[
        (512, 256, Activation::ReLU),
        (256, 512, Activation::ReLU),
        (128, 256, Activation::None),
    ], 0xCAFE_BABE)
}

pub fn mini() -> MicroModel {
    MicroModel::new("mini", &[
        (2048, 1024, Activation::ReLU),
        (1024, 2048, Activation::ReLU),
        (512, 1024, Activation::None),
    ], 0xBAAD_F00D)
}

pub fn small() -> MicroModel {
    MicroModel::new("small", &[
        (4096, 4096, Activation::ReLU),
        (4096, 4096, Activation::ReLU),
        (4096, 4096, Activation::None),
    ], 0xFACE_FEED)
}

/// ~1.3B params — 24-layer transformer (H=2048, FFN=8192).
pub fn medium() -> MicroModel {
    let h: u32 = 2048;
    let ffn: u32 = 8192;
    let n_layers: usize = 24;
    let mut arch = Vec::new();
    for _ in 0..n_layers {
        arch.push((h, h, Activation::None));     // attn QKV
        arch.push((h, h, Activation::None));     // attn output
        arch.push((ffn, h, Activation::SiLU));   // FFN up
        arch.push((h, ffn, Activation::None));   // FFN down
    }
    MicroModel::new("medium", &arch, 0x7070_1B00)
}

/// ~6.7B params — 32-layer transformer (H=4096, FFN=11008).
pub fn large() -> MicroModel {
    let h: u32 = 4096;
    let ffn: u32 = 11008;
    let n_layers: usize = 32;
    let mut arch = Vec::new();
    for _ in 0..n_layers {
        arch.push((h, h, Activation::None));
        arch.push((h, h, Activation::None));
        arch.push((ffn, h, Activation::SiLU));
        arch.push((h, ffn, Activation::None));
    }
    MicroModel::new("large", &arch, 0x7070_7B00)
}

/// ~65B params — 80-layer transformer (H=8192, FFN=28672).
pub fn xl() -> MicroModel {
    let h: u32 = 8192;
    let ffn: u32 = 28672;
    let n_layers: usize = 80;
    let mut arch = Vec::new();
    for _ in 0..n_layers {
        arch.push((h, h, Activation::None));
        arch.push((h, h, Activation::None));
        arch.push((ffn, h, Activation::SiLU));
        arch.push((h, ffn, Activation::None));
    }
    MicroModel::new("xl", &arch, 0x7070_70B0)
}

pub fn get_model(name: &str) -> Option<MicroModel> {
    match name {
        "nano" => Some(nano()),
        "micro" => Some(micro()),
        "mini" => Some(mini()),
        "small" => Some(small()),
        "medium" => Some(medium()),
        "large" => Some(large()),
        "xl" => Some(xl()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// CPU reference forward pass
// ---------------------------------------------------------------------------

fn cpu_gemv(
    packed_w: &[u32], codebook: &Codebook,
    x: &[f32], rows: usize, cols: usize, seed: u32,
) -> Vec<f32> {
    let words_per_row = (cols + 15) / 16;
    let mut y = vec![0.0f32; rows];

    for row in 0..rows {
        let mut acc = 0.0f64;
        for col in 0..cols {
            let word_idx = row * words_per_row + col / 16;
            let bit_offset = (col % 16) * 2;
            let mag_idx = ((packed_w[word_idx] >> bit_offset) & 3) as usize;
            let magnitude = codebook.magnitudes[mag_idx];
            let sign = seed_engine::pcg_sign((row * cols + col) as u32, seed);
            acc += (magnitude as f64) * (sign as f64) * (x[col] as f64);
        }
        y[row] = acc as f32;
    }
    y
}

pub fn forward_cpu(model: &MicroModel, input: &[f32]) -> Vec<f32> {
    let mut x = input.to_vec();

    for (i, layer) in model.layers.iter().enumerate() {
        let layer_seed = model.seed.wrapping_add(i as u32 * 12345);
        let mut y = cpu_gemv(
            &layer.packed_weights, &model.codebook,
            &x, layer.rows as usize, layer.cols as usize, layer_seed,
        );

        // Apply activation.
        for v in y.iter_mut() {
            *v = layer.activation.apply_cpu(*v);
        }

        x = y;
    }
    x
}

// ---------------------------------------------------------------------------
// GPU forward pass
// ---------------------------------------------------------------------------

pub fn forward_gpu(model: &MicroModel, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // For models with many layers, use the VRAM-resident path.
    if model.layers.len() > 16 {
        return forward_gpu_vram_resident(model, input);
    }

    let ctx = VulkanContext::new()?;
    let gemv_pipeline = GemvPipeline::new(&ctx.device)?;
    let act_pipeline = ActivationPipeline::new(&ctx.device)?;

    // Upload codebook (shared across all layers).
    let mut codebook_buf = AllocatedBuffer::new_staging_with_data(
        &ctx.device, &ctx.allocator, &model.codebook.magnitudes, "codebook",
    )?;

    // Command infrastructure.
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(ctx.compute_queue_family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let cmd_pool = unsafe { ctx.device.create_command_pool(&pool_info, None)? };

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd_bufs = unsafe { ctx.device.allocate_command_buffers(&alloc_info)? };
    let cmd = cmd_bufs[0];

    let fence = unsafe { ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)? };

    // Current activation vector = input.
    let mut current_data = input.to_vec();

    for (i, layer) in model.layers.iter().enumerate() {
        let layer_seed = model.seed.wrapping_add(i as u32 * 12345);
        let rows = layer.rows as usize;
        let cols = layer.cols as usize;

        // Upload packed weights for this layer.
        let mut packed_buf = AllocatedBuffer::new_staging_with_data(
            &ctx.device, &ctx.allocator, &layer.packed_weights, "packed_w",
        )?;

        // Upload current input vector.
        let mut x_buf = AllocatedBuffer::new_staging_with_data(
            &ctx.device, &ctx.allocator, &current_data, "input_x",
        )?;

        // Output buffer (host-visible for readback).
        let mut y_buf = AllocatedBuffer::new_storage(
            &ctx.device, &ctx.allocator, (rows as u64) * 4,
            gpu_allocator::MemoryLocation::CpuToGpu, "output_y",
        )?;

        // Bind GEMV descriptors.
        let gemv_set = gemv_pipeline.bind_buffers(
            &ctx.device,
            packed_buf.buffer, packed_buf.size,
            codebook_buf.buffer, codebook_buf.size,
            x_buf.buffer, x_buf.size,
            y_buf.buffer, y_buf.size,
        )?;

        let push = GemvPushConstants {
            seed: layer_seed,
            m: layer.rows,
            k: layer.cols,
        };

        // Record GEMV dispatch.
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            ctx.device.begin_command_buffer(cmd, &begin_info)?;
            ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, gemv_pipeline.pipeline);
            ctx.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE,
                gemv_pipeline.pipeline_layout, 0, &[gemv_set], &[],
            );
            ctx.device.cmd_push_constants(
                cmd, gemv_pipeline.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push),
            );
            ctx.device.cmd_dispatch(cmd, layer.rows, 1, 1);

            // Memory barrier between GEMV write and activation read.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier], &[], &[],
            );

            // Activation dispatch (if not None).
            if layer.activation != Activation::None {
                let act_set = act_pipeline.bind_buffer(
                    &ctx.device, y_buf.buffer, y_buf.size,
                )?;
                let act_push = ActivationPushConstants {
                    count: layer.rows,
                    mode: layer.activation.mode(),
                };
                ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, act_pipeline.pipeline);
                ctx.device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::COMPUTE,
                    act_pipeline.pipeline_layout, 0, &[act_set], &[],
                );
                ctx.device.cmd_push_constants(
                    cmd, act_pipeline.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&act_push),
                );
                let groups = (layer.rows + 255) / 256;
                ctx.device.cmd_dispatch(cmd, groups, 1, 1);
            }

            ctx.device.end_command_buffer(cmd)?;
        }

        // Submit and wait.
        let cmd_arr = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmd_arr);
        unsafe {
            ctx.device.reset_fences(&[fence])?;
            ctx.device.queue_submit(ctx.compute_queue, &[submit], fence)?;
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        }

        // Read back output.
        current_data = y_buf.read_back::<f32>(&ctx.device, &ctx.allocator, rows)?;

        // Cleanup layer buffers.
        y_buf.destroy(&ctx.device, &ctx.allocator);
        x_buf.destroy(&ctx.device, &ctx.allocator);
        packed_buf.destroy(&ctx.device, &ctx.allocator);
    }

    // Cleanup shared resources.
    codebook_buf.destroy(&ctx.device, &ctx.allocator);
    act_pipeline.destroy(&ctx.device);
    gemv_pipeline.destroy(&ctx.device);
    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(cmd_pool, None);
    }

    Ok(current_data)
}

// ---------------------------------------------------------------------------
// VRAM-Resident GPU forward pass (all weights pre-uploaded, single dispatch)
// ---------------------------------------------------------------------------

/// VRAM-resident forward pass: uploads ALL packed weights to GPU memory at
/// startup, records a SINGLE command buffer with all layer dispatches and
/// barriers, submits once, waits once, reads back only the final output.
/// This eliminates the CPU↔GPU transfer overhead that dominated the streaming
/// path (~54× slower than peak).
///
/// Uses ping-pong activation buffers on GPU to avoid per-layer readback.
pub fn forward_gpu_vram_resident(
    model: &MicroModel,
    input: &[f32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let total_layers = model.layers.len();
    let start = std::time::Instant::now();

    log::info!(
        "VRAM-resident forward: '{}' — {} layers, {} params, {:.2} MB packed",
        model.name, total_layers, model.total_params(),
        model.packed_bytes() as f64 / 1_048_576.0
    );

    let ctx = VulkanContext::new()?;

    // --- Determine descriptor pool sizes ---
    // Each layer needs 1 GEMV descriptor set (4 bindings).
    // Layers with activations need 1 additional activation set (1 binding).
    let n_act_layers = model.layers.iter()
        .filter(|l| l.activation != Activation::None)
        .count();
    let total_gemv_sets = total_layers;
    let total_act_sets = n_act_layers;
    let total_sets = total_gemv_sets + total_act_sets;
    let total_storage_descriptors = total_gemv_sets * 4 + total_act_sets;

    // Create pipelines with appropriately-sized descriptor pools.
    let gemv_pipeline = GemvPipeline::new_batch(&ctx.device, total_gemv_sets as u32, (total_gemv_sets * 4) as u32)?;
    let act_pipeline = ActivationPipeline::new_batch(&ctx.device, total_act_sets.max(1) as u32, total_act_sets.max(1) as u32)?;

    log::info!(
        "  Descriptor pool: {} sets ({} GEMV + {} act), {} storage descriptors",
        total_sets, total_gemv_sets, total_act_sets, total_storage_descriptors
    );

    // --- Phase 1: Upload ALL weights to VRAM ---
    let upload_start = std::time::Instant::now();

    let mut codebook_buf = AllocatedBuffer::new_staging_with_data(
        &ctx.device, &ctx.allocator, &model.codebook.magnitudes, "codebook",
    )?;

    // Upload all packed weight buffers (use raw bytes to avoid alignment issues).
    let mut weight_bufs: Vec<AllocatedBuffer> = Vec::with_capacity(total_layers);
    for (i, layer) in model.layers.iter().enumerate() {
        let bytes: &[u8] = bytemuck::cast_slice(&layer.packed_weights);
        let buf = AllocatedBuffer::new_staging_with_bytes(
            &ctx.device, &ctx.allocator, bytes,
            &format!("w_{}", i),
        )?;
        weight_bufs.push(buf);
    }

    // Upload input vector.
    let mut input_buf = AllocatedBuffer::new_staging_with_data(
        &ctx.device, &ctx.allocator, input, "input",
    )?;

    // Find max activation dimensions for ping-pong buffers.
    let max_dim: u32 = model.layers.iter()
        .flat_map(|l| [l.rows, l.cols])
        .max()
        .unwrap_or(1);

    // Ping-pong activation buffers — both sized for max dimension.
    let mut ping_buf = AllocatedBuffer::new_storage(
        &ctx.device, &ctx.allocator, (max_dim as u64) * 4,
        gpu_allocator::MemoryLocation::CpuToGpu, "ping",
    )?;
    let mut pong_buf = AllocatedBuffer::new_storage(
        &ctx.device, &ctx.allocator, (max_dim as u64) * 4,
        gpu_allocator::MemoryLocation::CpuToGpu, "pong",
    )?;

    let upload_time = upload_start.elapsed();
    log::info!(
        "  Weights uploaded to VRAM: {:.2} MB in {:.3}s ({:.1} GB/s)",
        model.packed_bytes() as f64 / 1_048_576.0,
        upload_time.as_secs_f64(),
        model.packed_bytes() as f64 / upload_time.as_secs_f64() / 1e9
    );

    // --- Phase 2: Bind ALL descriptor sets upfront ---
    let mut gemv_sets: Vec<vk::DescriptorSet> = Vec::with_capacity(total_layers);
    let mut act_sets: Vec<Option<vk::DescriptorSet>> = Vec::with_capacity(total_layers);

    for (i, layer) in model.layers.iter().enumerate() {
        // Determine which buffers this layer reads/writes.
        // Layer 0 reads from input_buf; subsequent layers alternate ping/pong.
        let (x_buffer, x_size) = if i == 0 {
            (input_buf.buffer, input_buf.size)
        } else if i % 2 == 1 {
            (ping_buf.buffer, (layer.cols as u64) * 4)
        } else {
            (pong_buf.buffer, (layer.cols as u64) * 4)
        };

        // Output alternates: odd layers → ping, even layers → pong.
        // (Except layer 0 which writes to ping.)
        let (y_buffer, y_size) = if i % 2 == 0 {
            (ping_buf.buffer, (layer.rows as u64) * 4)
        } else {
            (pong_buf.buffer, (layer.rows as u64) * 4)
        };

        let gemv_set = gemv_pipeline.bind_buffers(
            &ctx.device,
            weight_bufs[i].buffer, weight_bufs[i].size,
            codebook_buf.buffer, codebook_buf.size,
            x_buffer, x_size,
            y_buffer, y_size,
        )?;
        gemv_sets.push(gemv_set);

        if layer.activation != Activation::None {
            let act_set = act_pipeline.bind_buffer(
                &ctx.device, y_buffer, y_size,
            )?;
            act_sets.push(Some(act_set));
        } else {
            act_sets.push(None);
        }
    }

    // --- Phase 3: Record ONE command buffer for ALL layers ---
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(ctx.compute_queue_family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let cmd_pool = unsafe { ctx.device.create_command_pool(&pool_info, None)? };

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd_bufs = unsafe { ctx.device.allocate_command_buffers(&alloc_info)? };
    let cmd = cmd_bufs[0];

    let fence = unsafe { ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)? };

    let record_start = std::time::Instant::now();

    let begin_info = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        ctx.device.begin_command_buffer(cmd, &begin_info)?;

        for (i, layer) in model.layers.iter().enumerate() {
            let layer_seed = model.seed.wrapping_add(i as u32 * 12345);
            let push = GemvPushConstants {
                seed: layer_seed,
                m: layer.rows,
                k: layer.cols,
            };

            // GEMV dispatch.
            ctx.device.cmd_bind_pipeline(
                cmd, vk::PipelineBindPoint::COMPUTE, gemv_pipeline.pipeline,
            );
            ctx.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE,
                gemv_pipeline.pipeline_layout, 0, &[gemv_sets[i]], &[],
            );
            ctx.device.cmd_push_constants(
                cmd, gemv_pipeline.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push),
            );
            ctx.device.cmd_dispatch(cmd, layer.rows, 1, 1);

            // Barrier between GEMV write and next read.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier], &[], &[],
            );

            // Activation dispatch.
            if let Some(act_set) = act_sets[i] {
                let act_push = ActivationPushConstants {
                    count: layer.rows,
                    mode: layer.activation.mode(),
                };
                ctx.device.cmd_bind_pipeline(
                    cmd, vk::PipelineBindPoint::COMPUTE, act_pipeline.pipeline,
                );
                ctx.device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::COMPUTE,
                    act_pipeline.pipeline_layout, 0, &[act_set], &[],
                );
                ctx.device.cmd_push_constants(
                    cmd, act_pipeline.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&act_push),
                );
                let groups = (layer.rows + 255) / 256;
                ctx.device.cmd_dispatch(cmd, groups, 1, 1);

                // Barrier after activation.
                ctx.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barrier], &[], &[],
                );
            }
        }

        ctx.device.end_command_buffer(cmd)?;
    }

    let record_time = record_start.elapsed();
    log::info!("  Command buffer recorded: {} dispatches in {:.3}ms",
        total_layers + n_act_layers, record_time.as_secs_f64() * 1000.0);

    // --- Phase 4: Single submit, single wait ---
    let dispatch_start = std::time::Instant::now();
    let cmd_arr = [cmd];
    let submit = vk::SubmitInfo::default().command_buffers(&cmd_arr);
    unsafe {
        ctx.device.queue_submit(ctx.compute_queue, &[submit], fence)?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
    }
    let dispatch_time = dispatch_start.elapsed();

    let flops: u64 = model.layers.iter()
        .map(|l| l.rows as u64 * l.cols as u64 * 2)
        .sum();
    let gflops = flops as f64 / dispatch_time.as_secs_f64() / 1e9;
    log::info!(
        "  GPU dispatch: {:.3}s, {:.2} GFLOP/s (compute only, no transfers)",
        dispatch_time.as_secs_f64(), gflops
    );

    // --- Phase 5: Read back final output only ---
    let final_rows = model.layers.last().unwrap().rows as usize;
    // Final output is in ping or pong depending on layer count parity.
    let final_buf = if (total_layers - 1) % 2 == 0 { &mut ping_buf } else { &mut pong_buf };
    let output = final_buf.read_back::<f32>(&ctx.device, &ctx.allocator, final_rows)?;

    let total_time = start.elapsed();
    let total_gflops = flops as f64 / total_time.as_secs_f64() / 1e9;
    log::info!(
        "  Total (upload+record+dispatch+readback): {:.3}s, {:.2} GFLOP/s",
        total_time.as_secs_f64(), total_gflops
    );

    // --- Cleanup ---
    ping_buf.destroy(&ctx.device, &ctx.allocator);
    pong_buf.destroy(&ctx.device, &ctx.allocator);
    input_buf.destroy(&ctx.device, &ctx.allocator);
    for mut wb in weight_bufs {
        wb.destroy(&ctx.device, &ctx.allocator);
    }
    codebook_buf.destroy(&ctx.device, &ctx.allocator);
    act_pipeline.destroy(&ctx.device);
    gemv_pipeline.destroy(&ctx.device);
    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(cmd_pool, None);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

pub struct ValidationResult {
    pub model_name: String,
    pub total_params: u64,
    pub packed_bytes: usize,
    pub num_layers: usize,
    pub max_abs_error: f32,
    pub max_rel_error: f32,
    pub passed: bool,
}

impl std::fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "")?;
        writeln!(f, "================================================================")?;
        writeln!(f, "  AETERNUS — Micro Model Validation: {}", self.model_name)?;
        writeln!(f, "================================================================")?;
        writeln!(f, "  Layers:             {:>38}", self.num_layers)?;
        writeln!(f, "  Total Params:       {:>38}", self.total_params)?;
        writeln!(f, "  Packed Size:         {:>34} bytes", self.packed_bytes)?;
        writeln!(f, "  Max Abs Error:      {:>38.6}", self.max_abs_error)?;
        writeln!(f, "  Max Rel Error:      {:>38.6}", self.max_rel_error)?;
        writeln!(f, "----------------------------------------------------------------")?;
        if self.passed {
            writeln!(f, "  >> VALIDATION PASSED (rel error < 1e-3)")?;
        } else {
            writeln!(f, "  >> VALIDATION FAILED")?;
        }
        writeln!(f, "================================================================")?;
        Ok(())
    }
}

pub fn validate(model: &MicroModel) -> Result<ValidationResult, Box<dyn std::error::Error>> {
    let input_dim = model.layers[0].cols as usize;

    // Generate deterministic input.
    let input: Vec<f32> = (0..input_dim)
        .map(|i| ((seed_engine::pcg_hash(i as u32) % 2000) as f32 / 2000.0) - 0.5)
        .collect();

    log::info!("Running CPU reference for '{}'...", model.name);
    let cpu_out = forward_cpu(model, &input);

    log::info!("Running GPU forward pass for '{}'...", model.name);
    let gpu_out = forward_gpu(model, &input)?;

    assert_eq!(cpu_out.len(), gpu_out.len(), "output dimension mismatch");

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let max_cpu = cpu_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    for (c, g) in cpu_out.iter().zip(gpu_out.iter()) {
        let abs_err = (c - g).abs();
        if abs_err > max_abs { max_abs = abs_err; }
        if max_cpu > 0.0 {
            let rel = abs_err / max_cpu;
            if rel > max_rel { max_rel = rel; }
        }
    }

    let passed = max_rel < 1e-3;

    Ok(ValidationResult {
        model_name: model.name.clone(),
        total_params: model.total_params(),
        packed_bytes: model.packed_bytes(),
        num_layers: model.layers.len(),
        max_abs_error: max_abs,
        max_rel_error: max_rel,
        passed,
    })
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

pub fn bench(model: &MicroModel, iterations: u32) -> Result<(), Box<dyn std::error::Error>> {
    let input_dim = model.layers[0].cols as usize;
    let input: Vec<f32> = (0..input_dim)
        .map(|i| ((seed_engine::pcg_hash(i as u32) % 2000) as f32 / 2000.0) - 0.5)
        .collect();

    println!("\n  Benchmarking '{}' ({} params, {} layers) x {} iterations...",
             model.name, model.total_params(), model.layers.len(), iterations);

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = forward_gpu(model, &input)?;
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let flops_per_pass: u64 = model.layers.iter()
        .map(|l| l.rows as u64 * l.cols as u64 * 2)
        .sum();
    let gflops = (flops_per_pass as f64 / avg_ms) * 1000.0 / 1e9;

    println!("  Avg time/pass:  {:.3} ms", avg_ms);
    println!("  GFLOP/s:        {:.2}", gflops);
    println!("  Total params:   {}", model.total_params());
    println!("  Packed size:    {} bytes\n", model.packed_bytes());

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nano_model_cpu_forward() {
        let model = nano();
        let input = vec![0.5f32; 64];
        let output = forward_cpu(&model, &input);
        assert_eq!(output.len(), 32);
        // Check output is not all zeros (model should produce non-trivial output).
        assert!(output.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn nano_model_gpu_matches_cpu() {
        let model = nano();
        let result = validate(&model).expect("validation should succeed");
        assert!(result.passed, "max rel error: {}", result.max_rel_error);
    }
}
