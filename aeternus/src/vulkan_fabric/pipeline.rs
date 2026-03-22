//! # Compute Pipeline
//!
//! Loads the pre-compiled PIM sign regeneration SPIR-V and constructs a
//! Vulkan compute pipeline.  Zero build-time dependencies — raw bytecode.

use ash::vk;

/// Pre-compiled SPIR-V bytes (compiled once via naga-cli, embedded forever).
const PIM_SIGN_REGEN_SPV: &[u8] = include_bytes!("../../shaders/pim_sign_regen.spv");

/// Push constant layout matching the shader's `Params` block.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PushConstants {
    pub seed: u32,
    pub count: u32,
}

unsafe impl bytemuck::Pod for PushConstants {}
unsafe impl bytemuck::Zeroable for PushConstants {}

/// All pipeline objects needed to dispatch the sign reconstruction kernel.
pub struct SignReconstructPipeline {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_pool: vk::DescriptorPool,
    pub shader_module: vk::ShaderModule,
}

impl SignReconstructPipeline {
    /// Build the compute pipeline from the pre-compiled SPIR-V.
    pub fn new(device: &ash::Device) -> Result<Self, Box<dyn std::error::Error>> {
        // --- Load SPIR-V ---
        assert!(
            PIM_SIGN_REGEN_SPV.len() % 4 == 0,
            "SPIR-V binary length must be a multiple of 4"
        );
        let spirv = super::load_spirv_aligned(PIM_SIGN_REGEN_SPV);

        // --- Shader Module ---
        let shader_create_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
        let shader_module = unsafe { device.create_shader_module(&shader_create_info, None)? };

        // --- Descriptor Set Layout ---
        // binding 0: storage buffer (weights — in-place mutation)
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        // --- Pipeline Layout (with push constants) ---
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32);

        let layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        // --- Compute Pipeline ---
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

        // --- Descriptor Pool ---
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(16),
        ];
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

    /// Allocate a descriptor set and bind the weight buffer.
    pub fn bind_weight_buffer(
        &self,
        device: &ash::Device,
        weight_buffer: vk::Buffer,
        weight_buffer_size: u64,
    ) -> Result<vk::DescriptorSet, Box<dyn std::error::Error>> {
        let layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        let sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };
        let set = sets[0];

        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(weight_buffer)
            .offset(0)
            .range(weight_buffer_size);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info)),
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]) };

        Ok(set)
    }

    /// Destroy all pipeline objects.
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
