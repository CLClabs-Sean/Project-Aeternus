//! # Activation Pipeline
//!
//! Element-wise activation (ReLU / SiLU) applied in-place to a float buffer.

use ash::vk;

const ACTIVATION_SPV: &[u8] = include_bytes!("../../shaders/activation.spv");

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ActivationPushConstants {
    pub count: u32,
    pub mode: u32,  // 0 = ReLU, 1 = SiLU
}

unsafe impl bytemuck::Pod for ActivationPushConstants {}
unsafe impl bytemuck::Zeroable for ActivationPushConstants {}

pub struct ActivationPipeline {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_pool: vk::DescriptorPool,
    pub shader_module: vk::ShaderModule,
}

impl ActivationPipeline {
    pub fn new(device: &ash::Device) -> Result<Self, Box<dyn std::error::Error>> {
        assert!(ACTIVATION_SPV.len() % 4 == 0);
        let spirv: &[u32] = bytemuck::cast_slice(ACTIVATION_SPV);

        let shader_create_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        let shader_module = unsafe { device.create_shader_module(&shader_create_info, None)? };

        // 1 binding: data buffer (in-place)
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<ActivationPushConstants>() as u32);

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
            .descriptor_count(16)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(16)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        Ok(Self {
            descriptor_set_layout, pipeline_layout, pipeline,
            descriptor_pool, shader_module,
        })
    }

    pub fn bind_buffer(
        &self,
        device: &ash::Device,
        buffer: vk::Buffer,
        size: u64,
    ) -> Result<vk::DescriptorSet, Box<dyn std::error::Error>> {
        let layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        let sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };
        let set = sets[0];

        let buf_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer).offset(0).range(size);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(set).dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buf_info));

        unsafe { device.update_descriptor_sets(&[write], &[]) };
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
