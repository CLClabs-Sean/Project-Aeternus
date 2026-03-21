//! # Fused VQ+Sign Benchmark
//!
//! Benchmarks the Phase 1 fused kernel: packed 2-bit magnitudes + PCG sign
//! reconstruction in a single shader dispatch. Compares against Phase 0
//! baseline and PCIe 3.0 bandwidth.

use ash::vk;
use crate::codebook::{self, Codebook};
use crate::vulkan_fabric::VulkanContext;
use crate::vulkan_fabric::buffer::AllocatedBuffer;
use crate::vulkan_fabric::fused_pipeline::{FusedPipeline, FusedPushConstants};

const PCIE3_X16_GBS: f64 = 15.754;

pub struct FusedBenchConfig {
    pub total_params: u64,
    pub tile_size: u32,
    pub seed: u32,
    pub codebook: Codebook,
}

impl FusedBenchConfig {
    pub fn new(total_params: u64, tile_size: u32) -> Self {
        Self {
            total_params,
            tile_size,
            seed: 0xCAFE_BABE,
            codebook: Codebook::default(),
        }
    }
}

#[derive(Debug)]
pub struct FusedBenchResult {
    pub gpu_name: String,
    pub total_params: u64,
    pub tile_size: u32,
    pub num_tiles: u64,
    pub total_gpu_time_ms: f64,
    /// Throughput measured against output bytes (f32 weights written).
    pub throughput_output_gbs: f64,
    /// Throughput measured against input bytes (packed u32 words read).
    pub throughput_input_gbs: f64,
    pub pcie_ratio: f64,
    pub compression_ratio: f64,
    pub bits_per_param: f64,
}

impl std::fmt::Display for FusedBenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "")?;
        writeln!(f, "================================================================")?;
        writeln!(f, "  AETERNUS Phase 1 — Fused VQ+Sign Benchmark")?;
        writeln!(f, "================================================================")?;
        writeln!(f, "  GPU:                {:>38}", self.gpu_name)?;
        writeln!(f, "  Total Parameters:   {:>35}B",
                 self.total_params / 1_000_000_000)?;
        writeln!(f, "  Tile Size:          {:>32} weights", self.tile_size)?;
        writeln!(f, "  Tiles Dispatched:   {:>38}", self.num_tiles)?;
        writeln!(f, "  Bits/Param:         {:>38.1}", self.bits_per_param)?;
        writeln!(f, "  Compression:        {:>37.1}x", self.compression_ratio)?;
        writeln!(f, "----------------------------------------------------------------")?;
        writeln!(f, "  GPU Wall Time:      {:>34.2} ms", self.total_gpu_time_ms)?;
        writeln!(f, "  Output Throughput:  {:>32.3} GB/s", self.throughput_output_gbs)?;
        writeln!(f, "  Input  Throughput:  {:>32.3} GB/s", self.throughput_input_gbs)?;
        writeln!(f, "  PCIe 3.0 x16:       {:>32.3} GB/s", PCIE3_X16_GBS)?;
        writeln!(f, "  PWR / PCIe Ratio:   {:>35.2}x", self.pcie_ratio)?;
        writeln!(f, "----------------------------------------------------------------")?;
        if self.pcie_ratio > 1.0 {
            writeln!(f, "  >> FUSED PWR VALIDATED")?;
        } else {
            writeln!(f, "  >> Below PCIe — kernel optimization needed")?;
        }
        writeln!(f, "================================================================")?;
        Ok(())
    }
}

pub fn run(config: &FusedBenchConfig) -> Result<FusedBenchResult, Box<dyn std::error::Error>> {
    log::info!("Initializing Vulkan...");
    let ctx = VulkanContext::new()?;

    log::info!("Building fused pipeline...");
    let pipeline = FusedPipeline::new(&ctx.device)?;

    // Generate synthetic packed weights for the tile.
    let packed_data = codebook::generate_synthetic_packed(config.seed, config.tile_size as usize);
    let mut packed_buf = AllocatedBuffer::new_staging_with_data(
        &ctx.device, &ctx.allocator, &packed_data, "packed_weights",
    )?;

    // Upload codebook (4 floats = 16 bytes).
    let mut codebook_buf = AllocatedBuffer::new_staging_with_data(
        &ctx.device, &ctx.allocator, &config.codebook.magnitudes, "codebook",
    )?;

    // Output buffer (f32 per weight).
    let output_size = (config.tile_size as u64) * 4;
    let mut output_buf = AllocatedBuffer::new_storage(
        &ctx.device, &ctx.allocator, output_size,
        gpu_allocator::MemoryLocation::GpuOnly, "output_weights",
    )?;

    let desc_set = pipeline.bind_buffers(
        &ctx.device,
        packed_buf.buffer, packed_buf.size,
        codebook_buf.buffer, codebook_buf.size,
        output_buf.buffer, output_buf.size,
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

    let query_pool_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(2);
    let query_pool = unsafe { ctx.device.create_query_pool(&query_pool_info, None)? };
    let fence = unsafe { ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)? };

    let num_tiles = (config.total_params + config.tile_size as u64 - 1) / config.tile_size as u64;
    let workgroup_size = 256u32;

    log::info!(
        "Fused benchmark: {} params, {} tiles of {} weights, {:.1} bits/param",
        config.total_params, num_tiles, config.tile_size, codebook::bits_per_param()
    );

    let mut total_ns: f64 = 0.0;

    for tile_idx in 0..num_tiles {
        let count = if tile_idx == num_tiles - 1 {
            let remaining = config.total_params - (tile_idx * config.tile_size as u64);
            remaining.min(config.tile_size as u64) as u32
        } else {
            config.tile_size
        };

        let push = FusedPushConstants {
            seed: config.seed.wrapping_add(tile_idx as u32),
            count,
        };
        let dispatch_groups = (count + workgroup_size - 1) / workgroup_size;

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

        let cmd_bufs_submit = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs_submit);
        unsafe {
            ctx.device.reset_fences(&[fence])?;
            ctx.device.queue_submit(ctx.compute_queue, &[submit_info], fence)?;
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        }

        let mut timestamps = [0u64; 2];
        unsafe {
            ctx.device.get_query_pool_results(
                query_pool, 0, &mut timestamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }

        let delta_ticks = timestamps[1].wrapping_sub(timestamps[0]);
        total_ns += delta_ticks as f64 * ctx.timestamp_period as f64;

        if tile_idx % 50 == 0 && tile_idx > 0 {
            log::info!("  tile {}/{} — {:.2} ms", tile_idx, num_tiles, total_ns / 1e6);
        }
    }

    let total_ms = total_ns / 1e6;
    let output_bytes = config.total_params * 4; // f32 output
    let input_bytes = (config.total_params + 15) / 16 * 4; // packed u32 input
    let throughput_output = if total_ms > 0.0 {
        (output_bytes as f64 / 1e9) / (total_ms / 1000.0)
    } else { 0.0 };
    let throughput_input = if total_ms > 0.0 {
        (input_bytes as f64 / 1e9) / (total_ms / 1000.0)
    } else { 0.0 };
    let pcie_ratio = throughput_output / PCIE3_X16_GBS;
    let compression_ratio = 32.0 / codebook::bits_per_param(); // f32 vs packed

    // Cleanup.
    output_buf.destroy(&ctx.device, &ctx.allocator);
    codebook_buf.destroy(&ctx.device, &ctx.allocator);
    packed_buf.destroy(&ctx.device, &ctx.allocator);
    pipeline.destroy(&ctx.device);
    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_query_pool(query_pool, None);
        ctx.device.destroy_command_pool(cmd_pool, None);
    }

    Ok(FusedBenchResult {
        gpu_name: ctx.gpu_name(),
        total_params: config.total_params,
        tile_size: config.tile_size,
        num_tiles,
        total_gpu_time_ms: total_ms,
        throughput_output_gbs: throughput_output,
        throughput_input_gbs: throughput_input,
        pcie_ratio,
        compression_ratio,
        bits_per_param: codebook::bits_per_param(),
    })
}

/// Sweep tile sizes to find the L2 cache sweet spot.
pub fn sweep(config: &FusedBenchConfig) -> Result<(), Box<dyn std::error::Error>> {
    let tile_sizes: Vec<u32> = vec![
        65_536,      // 64K
        131_072,     // 128K
        262_144,     // 256K
        524_288,     // 512K
        1_048_576,   // 1M
        2_097_152,   // 2M
        4_194_304,   // 4M
        8_388_608,   // 8M
        16_777_216,  // 16M
    ];

    println!("\n================================================================");
    println!("  AETERNUS — Tile Size Sweep (L2 Cache Optimization)");
    println!("================================================================");
    println!("  {:>12}  {:>12}  {:>12}  {:>8}", "Tile Size", "Time (ms)", "GB/s", "PCIe×");
    println!("  {:->12}  {:->12}  {:->12}  {:->8}", "", "", "", "");

    let mut best_throughput = 0.0f64;
    let mut best_tile = 0u32;

    for &tile_size in &tile_sizes {
        // Use a smaller param count for sweep to keep it fast.
        let sweep_params = (config.total_params).min(100_000_000); // 100M max per sweep point
        let mut sweep_config = FusedBenchConfig::new(sweep_params, tile_size);
        sweep_config.seed = config.seed;
        sweep_config.codebook = config.codebook;

        match run(&sweep_config) {
            Ok(result) => {
                let marker = if result.throughput_output_gbs > best_throughput { " <<<" } else { "" };
                if result.throughput_output_gbs > best_throughput {
                    best_throughput = result.throughput_output_gbs;
                    best_tile = tile_size;
                }
                println!(
                    "  {:>12}  {:>12.2}  {:>12.3}  {:>7.2}x{}",
                    tile_size, result.total_gpu_time_ms,
                    result.throughput_output_gbs, result.pcie_ratio, marker
                );
            }
            Err(e) => {
                println!("  {:>12}  ERROR: {}", tile_size, e);
            }
        }
    }

    println!("  {:->12}  {:->12}  {:->12}  {:->8}", "", "", "", "");
    println!("  Best tile: {} ({:.3} GB/s, {:.2}x PCIe)",
             best_tile, best_throughput, best_throughput / PCIE3_X16_GBS);
    println!("================================================================\n");

    Ok(())
}
