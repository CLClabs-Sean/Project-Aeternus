//! # Tile Prefetcher — Double-Buffer DMA Overlap
//!
//! Ping-pong between two GPU buffers so compute and DMA overlap:
//!
//! ```text
//! GPU:  [GEMV tile 0] ──────── [GEMV tile 1] ────────
//! DMA:            [upload tile 1] ──── [upload tile 2]
//! ```

use ash::vk;
use crate::vulkan_fabric::VulkanContext;
use crate::vulkan_fabric::buffer::AllocatedBuffer;

pub struct TilePrefetcher {
    pub buffers: [AllocatedBuffer; 2],
    active: usize,   // which buffer the compute shader reads from
    tile_bytes: u64,
    cmd_pool: vk::CommandPool,
    cmd_buf: vk::CommandBuffer,
    fence: vk::Fence,
    upload_pending: bool,
}

impl TilePrefetcher {
    pub fn new(
        ctx: &VulkanContext,
        tile_bytes: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let buf_a = AllocatedBuffer::new_storage(
            &ctx.device, &ctx.allocator, tile_bytes,
            gpu_allocator::MemoryLocation::CpuToGpu, "prefetch_A",
        )?;
        let buf_b = AllocatedBuffer::new_storage(
            &ctx.device, &ctx.allocator, tile_bytes,
            gpu_allocator::MemoryLocation::CpuToGpu, "prefetch_B",
        )?;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.compute_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = unsafe { ctx.device.create_command_pool(&pool_info, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_bufs = unsafe { ctx.device.allocate_command_buffers(&alloc_info)? };

        let fence = unsafe { ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)? };

        Ok(Self {
            buffers: [buf_a, buf_b],
            active: 0,
            tile_bytes,
            cmd_pool,
            cmd_buf: cmd_bufs[0],
            fence,
            upload_pending: false,
        })
    }

    /// Currently active buffer (compute reads from this).
    pub fn active_buffer(&self) -> vk::Buffer {
        self.buffers[self.active].buffer
    }

    /// Inactive buffer (DMA writes to this).
    pub fn inactive_buffer(&self) -> vk::Buffer {
        self.buffers[1 - self.active].buffer
    }

    /// Active buffer's byte size.
    pub fn active_size(&self) -> u64 {
        self.buffers[self.active].size
    }

    /// Upload data into the ACTIVE buffer synchronously (initial load).
    pub fn upload_initial<T: bytemuck::Pod>(
        &self,
        data: &[T],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = bytemuck::cast_slice::<T, u8>(data);
        let alloc = self.buffers[self.active].allocation.as_ref()
            .ok_or("Active buffer has no allocation")?;
        let mapped = alloc.mapped_ptr()
            .ok_or("Active buffer not mappable")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                mapped.as_ptr() as *mut u8,
                bytes.len().min(self.tile_bytes as usize),
            );
        }
        Ok(())
    }

    /// Start an async upload of the next tile into the inactive buffer.
    /// This returns immediately — call `wait_upload` before swapping.
    pub fn upload_next<T: bytemuck::Pod>(
        &mut self,
        data: &[T],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let inactive = 1 - self.active;
        let bytes = bytemuck::cast_slice::<T, u8>(data);
        let alloc = self.buffers[inactive].allocation.as_ref()
            .ok_or("Inactive buffer has no allocation")?;
        let mapped = alloc.mapped_ptr()
            .ok_or("Inactive buffer not mappable")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                mapped.as_ptr() as *mut u8,
                bytes.len().min(self.tile_bytes as usize),
            );
        }
        self.upload_pending = true;
        Ok(())
    }

    /// Wait for any pending upload to complete.
    pub fn wait_upload(&self) {
        // For host-visible buffers, the memcpy is synchronous.
        // This method exists for the future GPU-only DMA path.
    }

    /// Swap active and inactive buffers. The previously-uploaded
    /// tile becomes the active one for the next compute dispatch.
    pub fn swap(&mut self) {
        if self.upload_pending {
            self.wait_upload();
        }
        self.active = 1 - self.active;
        self.upload_pending = false;
    }

    pub fn destroy(mut self, ctx: &VulkanContext) {
        self.buffers[0].destroy(&ctx.device, &ctx.allocator);
        self.buffers[1].destroy(&ctx.device, &ctx.allocator);
        unsafe {
            ctx.device.destroy_fence(self.fence, None);
            ctx.device.destroy_command_pool(self.cmd_pool, None);
        }
    }
}

/// Benchmark: process N tiles with vs without prefetch.
pub struct PrefetchBenchResult {
    pub tile_bytes: u64,
    pub num_tiles: usize,
    pub serial_ms: f64,
    pub overlap_ms: f64,
    pub speedup: f64,
}

impl std::fmt::Display for PrefetchBenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "")?;
        writeln!(f, "================================================================")?;
        writeln!(f, "  AETERNUS — Async Tile Prefetch Benchmark")?;
        writeln!(f, "================================================================")?;
        writeln!(f, "  Tile size:          {:>34} bytes", self.tile_bytes)?;
        writeln!(f, "  Tiles processed:    {:>38}", self.num_tiles)?;
        writeln!(f, "  Serial time:        {:>34.3} ms", self.serial_ms)?;
        writeln!(f, "  Overlap time:       {:>34.3} ms", self.overlap_ms)?;
        writeln!(f, "  Speedup:            {:>35.2}x", self.speedup)?;
        writeln!(f, "================================================================")?;
        Ok(())
    }
}

pub fn bench_prefetch(num_tiles: usize, tile_elements: usize) -> Result<PrefetchBenchResult, Box<dyn std::error::Error>> {
    let tile_bytes = (tile_elements * 4) as u64;

    // Generate synthetic tiles.
    let tiles: Vec<Vec<u32>> = (0..num_tiles).map(|i| {
        (0..tile_elements).map(|j| (i * tile_elements + j) as u32).collect()
    }).collect();

    let ctx = VulkanContext::new()?;
    let mut prefetcher = TilePrefetcher::new(&ctx, tile_bytes)?;

    // --- Serial: upload + "process" each tile sequentially ---
    let serial_start = std::time::Instant::now();
    for tile in &tiles {
        prefetcher.upload_initial(tile)?;
        // Simulate compute work.
        std::thread::sleep(std::time::Duration::from_micros(100));
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    // --- Overlapped: upload next tile while "processing" current ---
    let overlap_start = std::time::Instant::now();
    prefetcher.upload_initial(&tiles[0])?;
    for i in 0..num_tiles {
        // Start uploading next tile while current tile is being "used".
        if i + 1 < num_tiles {
            prefetcher.upload_next(&tiles[i + 1])?;
        }
        // Simulate compute work on current tile.
        std::thread::sleep(std::time::Duration::from_micros(100));
        // Swap buffers for next iteration.
        if i + 1 < num_tiles {
            prefetcher.swap();
        }
    }
    let overlap_ms = overlap_start.elapsed().as_secs_f64() * 1000.0;

    let speedup = serial_ms / overlap_ms;

    prefetcher.destroy(&ctx);

    Ok(PrefetchBenchResult {
        tile_bytes,
        num_tiles,
        serial_ms,
        overlap_ms,
        speedup,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefetch_swap_reads_correct_data() {
        let ctx = VulkanContext::new().expect("Vulkan init");
        let tile_size = 256 * 4; // 256 u32s = 1KB tiles
        let mut pf = TilePrefetcher::new(&ctx, tile_size).expect("prefetcher");

        // Upload tile A.
        let tile_a: Vec<u32> = (0..256).collect();
        pf.upload_initial(&tile_a).expect("upload A");

        // Upload tile B into inactive buffer.
        let tile_b: Vec<u32> = (256..512).collect();
        pf.upload_next(&tile_b).expect("upload B");

        // Swap: B becomes active.
        pf.swap();

        // Read back from active buffer (should be tile B).
        let readback: Vec<u32> = pf.buffers[pf.active]
            .read_back::<u32>(&ctx.device, &ctx.allocator, 256)
            .expect("readback");
        assert_eq!(readback[0], 256);
        assert_eq!(readback[255], 511);

        pf.destroy(&ctx);
    }

    #[test]
    fn double_swap_returns_to_original() {
        let ctx = VulkanContext::new().expect("Vulkan init");
        let tile_size = 64 * 4;
        let mut pf = TilePrefetcher::new(&ctx, tile_size).expect("prefetcher");

        let original_active = pf.active;
        pf.upload_next(&vec![0u32; 64]).ok();
        pf.swap();
        pf.upload_next(&vec![0u32; 64]).ok();
        pf.swap();
        assert_eq!(pf.active, original_active);

        pf.destroy(&ctx);
    }
}
