//! # Buffer Management
//!
//! Vulkan storage buffer creation for the PIM weight tile pipeline.
//! Designed for 2 GB VRAM constraint — all operations are tile-based.

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;
use std::sync::{Arc, Mutex};

/// A Vulkan buffer with its backing allocation.
pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: u64,
}

impl AllocatedBuffer {
    /// Create a new storage buffer.
    pub fn new_storage(
        device: &ash::Device,
        allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
        size: u64,
        location: MemoryLocation,
        label: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator.lock().unwrap().allocate(&AllocationCreateDesc {
            name: label,
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::DedicatedBuffer(buffer),
        })?;

        unsafe {
            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }

        Ok(Self {
            buffer,
            allocation: Some(allocation),
            size,
        })
    }

    /// Create a host-visible buffer and upload data into it.
    pub fn new_staging_with_data<T: bytemuck::Pod>(
        device: &ash::Device,
        allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
        data: &[T],
        label: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = bytemuck::cast_slice::<T, u8>(data);
        let size = bytes.len() as u64;

        let buf = Self::new_storage(device, allocator, size, MemoryLocation::CpuToGpu, label)?;

        // Map and copy.
        if let Some(ref allocation) = buf.allocation {
            if let Some(mapped) = allocation.mapped_ptr() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        mapped.as_ptr() as *mut u8,
                        bytes.len(),
                    );
                }
            } else {
                return Err("Failed to map staging buffer".into());
            }
        }

        Ok(buf)
    }

    /// Read back data from a host-visible buffer.
    pub fn read_back<T: bytemuck::Pod>(
        &self,
        _device: &ash::Device,
        _allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
        count: usize,
    ) -> Result<Vec<T>, Box<dyn std::error::Error>> {
        if let Some(ref allocation) = self.allocation {
            if let Some(mapped) = allocation.mapped_ptr() {
                let src = mapped.as_ptr() as *const T;
                let mut result = vec![T::zeroed(); count];
                unsafe {
                    std::ptr::copy_nonoverlapping(src, result.as_mut_ptr(), count);
                }
                Ok(result)
            } else {
                Err("Buffer is not mapped — must be CpuToGpu or GpuToCpu".into())
            }
        } else {
            Err("Buffer has no allocation".into())
        }
    }

    /// Destroy and free.
    pub fn destroy(
        &mut self,
        device: &ash::Device,
        allocator: &Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
    ) {
        if let Some(alloc) = self.allocation.take() {
            allocator.lock().unwrap().free(alloc).ok();
        }
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
    }
}
