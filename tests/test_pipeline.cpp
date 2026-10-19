// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gpu.h"
#include "layer_shader_type.h"
#include "option.h"
#include "pipeline.h"

#include <stdio.h>
#include <string.h>

static int test_created_subgroup_size(ncnn::VulkanDevice* vkdev)
{
    ncnn::Pipeline pipeline(vkdev);
    std::vector<VkPipelineExecutablePropertiesKHR> properties(1);
    if (pipeline.subgroup_size() != 0 || pipeline.executable_subgroup_size() != 0
            || pipeline.get_executable_properties(properties) != -1 || !properties.empty())
        return -1;

    ncnn::Option opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_subgroup_ops = false;
    opt.use_cooperative_matrix = false;
    opt.vulkan_device_index = vkdev->info.device_index();

    const char source[] = "#version 450\nvoid main() {}\n";
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(source, sizeof(source) - 1, opt, spirv) != 0)
        return -1;

    std::vector<ncnn::vk_specialization_type> specializations;
    pipeline.set_local_size_xyz(1, 1, 1);
    if (pipeline.create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations) != 0)
        return -1;

    const bool fixed_subgroup = vkdev->info.support_subgroup_size_control() || (spirv[1] < 0x00010600 && (vkdev->info.support_subgroup_ops() & VK_SUBGROUP_FEATURE_BASIC_BIT));
    const uint32_t expected = fixed_subgroup ? vkdev->info.subgroup_size() : 0;
    if (pipeline.subgroup_size() != expected)
        return -1;

    // changing the request must not change the already-created pipeline's reported size
    pipeline.set_subgroup_size(vkdev->info.min_subgroup_size());
    if (pipeline.subgroup_size() != expected)
        return -1;

    if (vkdev->info.support_subgroup_size_control())
    {
        pipeline.set_local_size_xyz(1, 1, 1);
        if (pipeline.create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations) != 0
                || pipeline.subgroup_size() != vkdev->info.min_subgroup_size())
            return -1;
    }

    // a second object must report the correct size on a pipeline cache hit
    ncnn::Pipeline cached(vkdev);
    cached.set_local_size_xyz(1, 1, 1);
    if (cached.create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations) != 0
            || cached.subgroup_size() != expected)
        return -1;

    const int ret = cached.get_executable_properties(properties);
    if (vkdev->info.support_pipeline_executable_info())
    {
        if (!vkdev->vkGetPipelineExecutablePropertiesKHR || ret != 0)
            return -1;
    }
    else if (ret != -1 || !properties.empty() || cached.executable_subgroup_size() != 0)
    {
        return -1;
    }

    // force a failure before calling the driver and reject stale pipeline information
    specializations.resize(1);
    if (pipeline.create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations) == 0
            || pipeline.subgroup_size() != 0 || pipeline.executable_subgroup_size() != 0)
        return -1;

#if NCNN_TEST_SIGMOID
    ncnn::Pipeline indexed(vkdev);
    indexed.set_local_size_xyz(1, 1, 1);
    specializations[0].u32 = 4;
    if (indexed.create(ncnn::LayerShaderType::sigmoid, opt, specializations) != 0
            || indexed.subgroup_size() != expected)
        return -1;
    ncnn::Pipeline indexed_cached(vkdev);
    indexed_cached.set_local_size_xyz(1, 1, 1);
    if (indexed_cached.create(ncnn::LayerShaderType::sigmoid, opt, specializations) != 0
            || indexed_cached.subgroup_size() != expected)
        return -1;
#endif

    return 0;
}

static int query_mode;
static int query_calls;
static VkPipeline expected_pipeline;

static VKAPI_ATTR VkResult VKAPI_CALL mock_get_pipeline_executable_properties(VkDevice, const VkPipelineInfoKHR* info, uint32_t* count, VkPipelineExecutablePropertiesKHR* properties)
{
    query_calls++;
    if (info->sType != VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR || info->pNext || info->pipeline != expected_pipeline)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (query_mode == 1 || (query_mode == 2 && properties))
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    if (!properties)
    {
        *count = 2;
        if (query_mode == 3)
            *count = 0;
        if (query_mode == 4 && query_calls == 1)
            *count = 1;
        return VK_SUCCESS;
    }

    for (uint32_t i = 0; i < *count; i++)
    {
        if (properties[i].sType != VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR || properties[i].pNext)
            return VK_ERROR_INITIALIZATION_FAILED;
        properties[i].stages = i == 0 ? VK_SHADER_STAGE_COMPUTE_BIT : 0;
        properties[i].subgroupSize = 32;
        if (query_mode == 5 && i == 1)
            properties[i].subgroupSize = 16;
        if (query_mode == 6)
            properties[i].subgroupSize = 0;
        snprintf(properties[i].name, sizeof(properties[i].name), "executable %u", i);
    }

    if (query_mode == 4 && query_calls == 2)
        return VK_INCOMPLETE;
    if (query_mode == 7)
        *count = 1;
    return VK_SUCCESS;
}

class ExecutableQueryOverride
{
public:
    explicit ExecutableQueryOverride(ncnn::VulkanDevice* device)
        : vkdev(device), saved(device->vkGetPipelineExecutablePropertiesKHR)
    {
        vkdev->vkGetPipelineExecutablePropertiesKHR = mock_get_pipeline_executable_properties;
    }
    ~ExecutableQueryOverride()
    {
        vkdev->vkGetPipelineExecutablePropertiesKHR = saved;
    }

private:
    ncnn::VulkanDevice* vkdev;
    PFN_vkGetPipelineExecutablePropertiesKHR saved;
};

static int test_executable_queries(ncnn::VulkanDevice* vkdev)
{
    ncnn::Option opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.vulkan_device_index = vkdev->info.device_index();
    const char source[] = "#version 450\nvoid main() {}\n";
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(source, sizeof(source) - 1, opt, spirv) != 0)
        return -1;

    ncnn::Pipeline pipeline(vkdev);
    pipeline.set_local_size_xyz(1, 1, 1);
    std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline.create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations) != 0)
        return -1;

    expected_pipeline = pipeline.pipeline();
    const uint32_t fixed_subgroup_size = pipeline.subgroup_size();
    ExecutableQueryOverride override_query(vkdev);

    // exercise enumeration errors, changing counts and ambiguous sizes without requiring the extension
    for (query_mode = 0; query_mode <= 7; query_mode++)
    {
        query_calls = 0;
        std::vector<VkPipelineExecutablePropertiesKHR> properties(3);
        const int ret = pipeline.get_executable_properties(properties);
        const int expected_ret = query_mode == 1 || query_mode == 2 ? -1 : 0;
        const size_t expected_count = query_mode == 1 || query_mode == 2 || query_mode == 3 ? 0 : query_mode == 7 ? 1 : 2;
        if (ret != expected_ret || properties.size() != expected_count
                || (query_mode == 4 && query_calls != 4))
        {
            fprintf(stderr, "executable enumeration failed mode=%d ret=%d count=%zu calls=%d\n", query_mode, ret, properties.size(), query_calls);
            return -1;
        }

        query_calls = 0;
        const uint32_t expected_size = query_mode == 0 || query_mode == 4 || query_mode == 7 ? 32 : 0;
        if (pipeline.executable_subgroup_size() != expected_size || pipeline.subgroup_size() != fixed_subgroup_size)
        {
            fprintf(stderr, "executable subgroup size failed mode=%d\n", query_mode);
            return -1;
        }
    }

    vkdev->vkGetPipelineExecutablePropertiesKHR = 0;
    std::vector<VkPipelineExecutablePropertiesKHR> properties(1);
    if (pipeline.get_executable_properties(properties) != -1 || !properties.empty()
            || pipeline.executable_subgroup_size() != 0)
        return -1;

    return 0;
}

int main()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    ncnn::VulkanDevice vkdev;
    if (!vkdev.is_valid())
        return -1;

    int ret = test_created_subgroup_size(&vkdev);
    if (ret != 0)
    {
        fprintf(stderr, "created subgroup size test failed\n");
        return ret;
    }

    return test_executable_queries(&vkdev);
}
