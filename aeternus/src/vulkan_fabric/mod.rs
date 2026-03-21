//! # Vulkan Fabric
//!
//! Core Vulkan device bootstrapping, memory allocation, and compute-pipeline
//! management for the AETERNUS engine.

pub mod buffer;
pub mod pipeline;
pub mod fused_pipeline;
pub mod swar_pipeline;
pub mod gemv_pipeline;
pub mod activation_pipeline;

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use std::sync::{Arc, Mutex};

/// Top-level Vulkan context.  Owns the instance, device, and allocator.
pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub compute_queue: vk::Queue,
    pub compute_queue_family: u32,
    pub allocator: Arc<Mutex<Allocator>>,
    pub device_properties: vk::PhysicalDeviceProperties,
    pub timestamp_period: f32,
}

impl VulkanContext {
    /// Initialize Vulkan: pick the best discrete GPU, create a compute queue.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let entry = unsafe { ash::Entry::load()? };

        // --- Instance ---
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"AETERNUS")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"PWR")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_2);

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info);

        let instance = unsafe { entry.create_instance(&create_info, None)? };

        // --- Physical Device ---
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        if physical_devices.is_empty() {
            return Err("No Vulkan-capable GPU found".into());
        }

        // Prefer discrete GPU.
        let physical_device = physical_devices
            .iter()
            .copied()
            .find(|&pd| {
                let props = unsafe { instance.get_physical_device_properties(pd) };
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .unwrap_or(physical_devices[0]);

        let device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe {
            std::ffi::CStr::from_ptr(device_properties.device_name.as_ptr())
                .to_string_lossy()
                .to_string()
        };
        log::info!("Selected GPU: {}", device_name);
        log::info!(
            "VRAM (device-local heaps): reported via allocator at runtime"
        );

        let timestamp_period = device_properties.limits.timestamp_period;

        // --- Queue Family (compute) ---
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let compute_family = queue_families
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
            .ok_or("No compute queue family found")?;

        // --- Logical Device ---
        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_family)
            .queue_priorities(&queue_priorities);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info));

        let device =
            unsafe { instance.create_device(physical_device, &device_create_info, None)? };
        let compute_queue = unsafe { device.get_device_queue(compute_family, 0) };

        // --- GPU Allocator ---
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            compute_queue,
            compute_queue_family: compute_family,
            allocator: Arc::new(Mutex::new(allocator)),
            device_properties,
            timestamp_period,
        })
    }

    /// GPU name as a string.
    pub fn gpu_name(&self) -> String {
        unsafe {
            std::ffi::CStr::from_ptr(self.device_properties.device_name.as_ptr())
                .to_string_lossy()
                .to_string()
        }
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            // Allocator must be dropped before device.
            // Arc<Mutex<Allocator>> handles this at ref-count zero.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vulkan_device_init() {
        env_logger::try_init().ok();
        let ctx = VulkanContext::new();
        match ctx {
            Ok(ctx) => {
                println!("GPU: {}", ctx.gpu_name());
                assert!(ctx.timestamp_period > 0.0);
            }
            Err(e) => {
                // CI environments may not have Vulkan — skip gracefully.
                eprintln!("Skipping Vulkan test (no GPU): {}", e);
            }
        }
    }
}
