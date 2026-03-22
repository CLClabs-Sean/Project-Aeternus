//! # Fused Reconstruct-GEMV Pipeline
//!
//! Compute pipeline for y = W·x where W is never materialized.
//! Packed 2-bit magnitudes → BFE → codebook → PCG sign → MAC in registers.

use ash::vk;

const FUSED_GEMV_SPV: &[u8] = include_bytes!("../../shaders/fused_gemv.spv");

/// Push constants for the GEMV kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GemvPushConstants {
    pub seed: u32,
    pub m: u32,  // rows (output dimension)
    pub k: u32,  // cols (reduction dimension)
}

unsafe impl bytemuck::Pod for GemvPushConstants {}
unsafe impl bytemuck::Zeroable for GemvPushConstants {}

pub struct GemvPipeline {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_pool: vk::DescriptorPool,
    pub shader_module: vk::ShaderModule,
}

impl GemvPipeline {
    pub fn new(device: &ash::Device) -> Result<Self, Box<dyn std::error::Error>> {
        let spirv = super::load_spirv_aligned(FUSED_GEMV_SPV);

        let shader_create_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        let shader_module = unsafe { device.create_shader_module(&shader_create_info, None)? };

        // 4 bindings: packed_w, codebook, x, y
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<GemvPushConstants>() as u32);

        let layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(std::slice::from_ref(&push_range));
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
            .descriptor_count(64)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(16)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        Ok(Self {
            descriptor_set_layout, pipeline_layout, pipeline,
            descriptor_pool, shader_module,
        })
    }

    pub fn bind_buffers(
        &self,
        device: &ash::Device,
        packed_w: vk::Buffer, packed_w_size: u64,
        codebook: vk::Buffer, codebook_size: u64,
        input_x: vk::Buffer, input_x_size: u64,
        output_y: vk::Buffer, output_y_size: u64,
    ) -> Result<vk::DescriptorSet, Box<dyn std::error::Error>> {
        let layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        let sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };
        let set = sets[0];

        let info0 = vk::DescriptorBufferInfo::default()
            .buffer(packed_w).offset(0).range(packed_w_size);
        let info1 = vk::DescriptorBufferInfo::default()
            .buffer(codebook).offset(0).range(codebook_size);
        let info2 = vk::DescriptorBufferInfo::default()
            .buffer(input_x).offset(0).range(input_x_size);
        let info3 = vk::DescriptorBufferInfo::default()
            .buffer(output_y).offset(0).range(output_y_size);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set).dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&info0)),
            vk::WriteDescriptorSet::default()
                .dst_set(set).dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&info1)),
            vk::WriteDescriptorSet::default()
                .dst_set(set).dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&info2)),
            vk::WriteDescriptorSet::default()
                .dst_set(set).dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&info3)),
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]) };
        Ok(set)
    }

    /// Create a pipeline with a larger descriptor pool for streaming (80+ layer) models.
    /// The pool is designed to be reset between layers, so max_sets=4 is enough.
    pub fn new_large(device: &ash::Device) -> Result<Self, Box<dyn std::error::Error>> {
        let spirv = super::load_spirv_aligned(FUSED_GEMV_SPV);

        let shader_create_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
        let shader_module = unsafe { device.create_shader_module(&shader_create_info, None)? };

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<GemvPushConstants>() as u32);

        let layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(std::slice::from_ref(&push_range));
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

        // Pool sized for streaming: reset between layers, max 4 sets at once.
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(16)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(4)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        Ok(Self {
            descriptor_set_layout, pipeline_layout, pipeline,
            descriptor_pool, shader_module,
        })
    }

    /// Create a pipeline with a custom-sized descriptor pool for batched
    /// (VRAM-resident) forward passes where all sets are allocated upfront.
    pub fn new_batch(device: &ash::Device, max_sets: u32, descriptor_count: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let spirv = super::load_spirv_aligned(FUSED_GEMV_SPV);

        let shader_create_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        let shader_module = unsafe { device.create_shader_module(&shader_create_info, None)? };

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<GemvPushConstants>() as u32);

        let layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(std::slice::from_ref(&push_range));
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
            .descriptor_count(descriptor_count)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        Ok(Self {
            descriptor_set_layout, pipeline_layout, pipeline,
            descriptor_pool, shader_module,
        })
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
