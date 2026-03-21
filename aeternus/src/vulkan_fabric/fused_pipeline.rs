//! # Fused VQ+Sign Pipeline
//!
//! Compute pipeline for the fused magnitude-sign reconstruction kernel.
//! Reads packed 2-bit weights, looks up VQ codebook, applies PCG sign,
//! writes full f32 weights — all in one thread.

use ash::vk;

/// Pre-compiled SPIR-V for the fused VQ+sign kernel.
const FUSED_VQ_SIGN_SPV: &[u8] = include_bytes!("../../shaders/fused_vq_sign.spv");

/// Push constants for the fused kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FusedPushConstants {
    pub seed: u32,
    pub count: u32,
}

unsafe impl bytemuck::Pod for FusedPushConstants {}
unsafe impl bytemuck::Zeroable for FusedPushConstants {}

/// Fused VQ+Sign compute pipeline.
pub struct FusedPipeline {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_pool: vk::DescriptorPool,
    pub shader_module: vk::ShaderModule,
}

impl FusedPipeline {
    pub fn new(device: &ash::Device) -> Result<Self, Box<dyn std::error::Error>> {
        assert!(FUSED_VQ_SIGN_SPV.len() % 4 == 0);
        let spirv: &[u32] = bytemuck::cast_slice(FUSED_VQ_SIGN_SPV);

        let shader_create_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        let shader_module = unsafe { device.create_shader_module(&shader_create_info, None)? };

        // 3 bindings: packed weights (SSBO), codebook (SSBO), output (SSBO)
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<FusedPushConstants>() as u32);

        let layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| e)?[0]
        };

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(48)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(16)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        Ok(Self {
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            descriptor_pool,
            shader_module,
        })
    }

    /// Bind the packed weights, codebook, and output buffers.
    pub fn bind_buffers(
        &self,
        device: &ash::Device,
        packed_buf: vk::Buffer,
        packed_size: u64,
        codebook_buf: vk::Buffer,
        codebook_size: u64,
        output_buf: vk::Buffer,
        output_size: u64,
    ) -> Result<vk::DescriptorSet, Box<dyn std::error::Error>> {
        let layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        let sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };
        let set = sets[0];

        let packed_info = vk::DescriptorBufferInfo::default()
            .buffer(packed_buf).offset(0).range(packed_size);
        let codebook_info = vk::DescriptorBufferInfo::default()
            .buffer(codebook_buf).offset(0).range(codebook_size);
        let output_info = vk::DescriptorBufferInfo::default()
            .buffer(output_buf).offset(0).range(output_size);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set).dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&packed_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set).dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&codebook_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set).dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&output_info)),
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]) };
        Ok(set)
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_shader_module(self.shader_module, None);
        }
    }
}
