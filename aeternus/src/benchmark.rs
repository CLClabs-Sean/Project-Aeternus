//! # Benchmark Harness
//!
//! Dispatches the PIM sign reconstruction kernel across tiled passes and
//! measures throughput against PCIe 3.0 x16 bandwidth.
//!
//! **2 GB VRAM constraint**: default tile is 4M weights (16 MB at f32).
//! Total model size is swept via repeated dispatches.

use ash::vk;
use crate::seed_engine;
use crate::vulkan_fabric::VulkanContext;
use crate::vulkan_fabric::buffer::AllocatedBuffer;
use crate::vulkan_fabric::pipeline::{SignReconstructPipeline, PushConstants};

/// PCIe 3.0 x16 theoretical bandwidth in GB/s.
const PCIE3_X16_GBS: f64 = 15.754;

/// Benchmark configuration.
pub struct BenchConfig {
    /// Total number of parameters to reconstruct signs for.
    pub total_params: u64,
    /// Weights per GPU dispatch (tile size). Must fit in VRAM as f32.
    /// Default: 4M (= 16 MB at 4 bytes/weight).
    pub tile_size: u32,
    /// Global seed for sign reconstruction.
    pub seed: u32,
}

impl BenchConfig {
    pub fn new(total_params: u64, tile_size: u32) -> Self {
        Self {
            total_params,
            tile_size,
            seed: 0xCAFE_BABE,
        }
    }
}

/// Benchmark results.
#[derive(Debug)]
pub struct BenchResult {
    pub gpu_name: String,
    pub total_params: u64,
    pub tile_size: u32,
    pub num_tiles: u64,
    pub total_gpu_time_ms: f64,
    pub throughput_gbs: f64,
    pub pcie_ratio: f64,
    pub effective_sign_bits_per_param: f64,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "")?;
        writeln!(f, "================================================================")?;
        writeln!(f, "  AETERNUS  —  PWR Benchmark Results")?;
        writeln!(f, "================================================================")?;
        writeln!(f, "  GPU:                {:>38}", self.gpu_name)?;
        writeln!(f, "  Total Parameters:   {:>35}B",
                 self.total_params / 1_000_000_000)?;
        writeln!(f, "  Tile Size:          {:>32} weights", self.tile_size)?;
        writeln!(f, "  Tiles Dispatched:   {:>38}", self.num_tiles)?;
        writeln!(f, "----------------------------------------------------------------")?;
        writeln!(f, "  GPU Wall Time:      {:>34.2} ms", self.total_gpu_time_ms)?;
        writeln!(f, "  Throughput:         {:>32.3} GB/s", self.throughput_gbs)?;
        writeln!(f, "  PCIe 3.0 x16:       {:>32.3} GB/s", PCIE3_X16_GBS)?;
        writeln!(f, "  PWR / PCIe Ratio:   {:>35.2}x", self.pcie_ratio)?;
        writeln!(f, "----------------------------------------------------------------")?;
        writeln!(f, "  Sign bits/param:    {:>34.2e}",
                 self.effective_sign_bits_per_param)?;
        if self.pcie_ratio > 1.0 {
            writeln!(f, "  >> PWR VALIDATED: reconstruction faster than PCIe transfer")?;
        } else {
            writeln!(f, "  >> PWR below PCIe — kernel optimization needed")?;
        }
        writeln!(f, "================================================================")?;
        Ok(())
    }
}

/// Run the full benchmark.
pub fn run(config: &BenchConfig) -> Result<BenchResult, Box<dyn std::error::Error>> {
    log::info!("Initializing Vulkan...");
    let ctx = VulkanContext::new()?;

    log::info!("Building compute pipeline...");
    let pipeline = SignReconstructPipeline::new(&ctx.device)?;

    // Create the weight buffer (reused across tiles).
    // Fill with 1.0f32 — after sign reconstruction, each element will be +1.0 or -1.0.
    let tile_bytes = (config.tile_size as u64) * 4;
    log::info!(
        "Allocating weight buffer: {} weights = {} MB",
        config.tile_size,
        tile_bytes / (1024 * 1024)
    );

    let ones: Vec<f32> = vec![1.0f32; config.tile_size as usize];
    let mut weight_buf = AllocatedBuffer::new_staging_with_data(
        &ctx.device,
        &ctx.allocator,
        &ones,
        "weight_tile",
    )?;

    let desc_set = pipeline.bind_weight_buffer(
        &ctx.device,
        weight_buf.buffer,
        weight_buf.size,
    )?;

    // Command pool + buffer.
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

    // Timestamp query pool (2 slots: begin + end).
    let query_pool_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(2);
    let query_pool = unsafe { ctx.device.create_query_pool(&query_pool_info, None)? };

    // Fence for synchronization.
    let fence_info = vk::FenceCreateInfo::default();
    let fence = unsafe { ctx.device.create_fence(&fence_info, None)? };

    let num_tiles = (config.total_params + config.tile_size as u64 - 1) / config.tile_size as u64;
    let workgroup_size = 256u32;

    log::info!(
        "Benchmarking: {} params in {} tiles of {} weights",
        config.total_params, num_tiles, config.tile_size
    );

    let mut total_ns: f64 = 0.0;

    for tile_idx in 0..num_tiles {
        let count = if tile_idx == num_tiles - 1 {
            let remaining = config.total_params - (tile_idx * config.tile_size as u64);
            remaining.min(config.tile_size as u64) as u32
        } else {
            config.tile_size
        };

        let push = PushConstants {
            seed: config.seed.wrapping_add(tile_idx as u32),
            count,
        };

        let dispatch_groups = (count + workgroup_size - 1) / workgroup_size;

        // Record commands.
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            ctx.device.begin_command_buffer(cmd, &begin_info)?;
            ctx.device.cmd_reset_query_pool(cmd, query_pool, 0, 2);
            ctx.device.cmd_write_timestamp(
                cmd, vk::PipelineStageFlags::TOP_OF_PIPE, query_pool, 0,
            );
            ctx.device.cmd_bind_pipeline(
                cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline,
            );
            ctx.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout, 0, &[desc_set], &[],
            );
            ctx.device.cmd_push_constants(
                cmd, pipeline.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE, 0,
                bytemuck::bytes_of(&push),
            );
            ctx.device.cmd_dispatch(cmd, dispatch_groups, 1, 1);
            ctx.device.cmd_write_timestamp(
                cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, query_pool, 1,
            );
            ctx.device.end_command_buffer(cmd)?;
        }

        // Submit.
        let cmd_bufs_submit = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs_submit);
        unsafe {
            ctx.device.reset_fences(&[fence])?;
            ctx.device.queue_submit(ctx.compute_queue, &[submit_info], fence)?;
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        }

        // Read timestamps.
        let mut timestamps = [0u64; 2];
        unsafe {
            ctx.device.get_query_pool_results(
                query_pool,
                0,
                &mut timestamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }

        let delta_ticks = timestamps[1].wrapping_sub(timestamps[0]);
        let delta_ns = delta_ticks as f64 * ctx.timestamp_period as f64;
        total_ns += delta_ns;

        if tile_idx % 50 == 0 && tile_idx > 0 {
            log::info!(
                "  tile {}/{} — cumulative {:.2} ms",
                tile_idx, num_tiles, total_ns / 1e6
            );
        }
    }

    let total_ms = total_ns / 1e6;
    let total_bytes = config.total_params * 4; // 4 bytes per f32 weight
    let throughput_gbs = if total_ms > 0.0 {
        (total_bytes as f64 / 1e9) / (total_ms / 1000.0)
    } else {
        0.0
    };
    let pcie_ratio = throughput_gbs / PCIE3_X16_GBS;

    // Cleanup.
    weight_buf.destroy(&ctx.device, &ctx.allocator);
    pipeline.destroy(&ctx.device);
    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_query_pool(query_pool, None);
        ctx.device.destroy_command_pool(cmd_pool, None);
    }

    Ok(BenchResult {
        gpu_name: ctx.gpu_name(),
        total_params: config.total_params,
        tile_size: config.tile_size,
        num_tiles,
        total_gpu_time_ms: total_ms,
        throughput_gbs,
        pcie_ratio,
        effective_sign_bits_per_param: seed_engine::effective_sign_bits_per_param(
            config.total_params,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_small_smoke() {
        env_logger::try_init().ok();
        let config = BenchConfig::new(4096, 4096);
        match run(&config) {
            Ok(result) => {
                println!("{}", result);
                assert!(result.total_gpu_time_ms > 0.0);
                assert!(result.throughput_gbs > 0.0);
            }
            Err(e) => {
                eprintln!("Skipping bench smoke test (no GPU): {}", e);
            }
        }
    }
}
