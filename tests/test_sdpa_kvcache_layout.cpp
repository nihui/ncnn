// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "net.h"
#include "testutil.h"

#if NCNN_VULKAN
#include "gpu.h"
#include "command.h"
#endif

static const char sdpa_param[] = "7767517\n"
                                 "5 8\n"
                                 "Input q_input 0 1 q\n"
                                 "Input k_input 0 1 k\n"
                                 "Input v_input 0 1 v\n"
                                 "Input cache_input 0 2 past_k past_v\n"
                                 "SDPA sdpa 5 3 q k v past_k past_v out out_k out_v 7=1\n";

static ncnn::Layer* create_sdpa_naive(void*)
{
    return ncnn::create_layer_naive("SDPA");
}

static int test_sdpa_kvcache_layout(int head_dim, int value_dim, int heads, int kv_heads, int bf16, int cache_type, int trim, int empty_type, bool vulkan, bool naive = false)
{
    ncnn::Net net;
    net.opt.num_threads = 2;
    net.opt.use_vulkan_compute = vulkan;
    net.opt.use_fp16_storage = bf16 == 2;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = bf16 == 1;
    net.opt.kvcache_max_seqlen_hint = 32;
    if (naive)
        net.register_custom_layer("SDPA", create_sdpa_naive);
    const unsigned int empty_model = 0;
    if (net.load_param_mem(sdpa_param) != 0 || net.load_model((const unsigned char*)&empty_model) != 0)
        return -1;

    ncnn::Layer* reference = ncnn::create_layer_naive("SDPA");
    ncnn::ParamDict pd;
    pd.set(7, 1);
    reference->load_param(pd);
    ncnn::Option opt;
    opt.num_threads = 1;
    reference->create_pipeline(opt);

    ncnn::Mat key_cache;
    ncnn::Mat value_cache;
    if (empty_type)
    {
        key_cache.create(head_dim, empty_type == 1 ? 0 : 17, kv_heads);
        value_cache.create(value_dim, empty_type == 1 ? 0 : 17, kv_heads);
        key_cache.h = value_cache.h = 0;
    }
    ncnn::Mat reference_key;
    ncnn::Mat reference_value;
    const int lengths[] = {13, 1, 5, 49};
    const float epsilon = bf16 ? 0.03f : 0.001f;
    int ret = 0;
    const char* operation = "forward";
    int step = 0;
    for (; step < 4 && ret == 0; step++)
    {
        ncnn::Mat query = RandomMat(head_dim, lengths[step], heads);
        ncnn::Mat key = RandomMat(head_dim, lengths[step], kv_heads);
        ncnn::Mat value = RandomMat(value_dim, lengths[step], kv_heads);
        std::vector<ncnn::Mat> reference_inputs(5);
        reference_inputs[0] = query;
        reference_inputs[1] = key;
        reference_inputs[2] = value;
        reference_inputs[3] = reference_key;
        reference_inputs[4] = reference_value;
        std::vector<ncnn::Mat> reference_outputs(3);
        ret = reference->forward(reference_inputs, reference_outputs, opt);
        if (ret != 0)
            break;

        ncnn::Mat key_before;
        ncnn::Mat value_before;
        if ((cache_type == 0 || trim) && !key_cache.empty())
        {
            key_before = key_cache.clone();
            value_before = value_cache.clone();
        }

        ncnn::Extractor ex = net.create_extractor();
        ex.input("q", query);
        ex.input("k", key);
        ex.input("v", value);
        ex.input("past_k", key_cache);
        ex.input("past_v", value_cache);
        ncnn::Mat output;
        operation = "forward";
        ret = ex.extract("out", output);
        if (ret != 0 || CompareMat(reference_outputs[0], output, epsilon) != 0)
        {
            ret = -1;
            break;
        }
        if (!key_before.empty() && (CompareMat(key_before, key_cache, 0.f) != 0 || CompareMat(value_before, value_cache, 0.f) != 0))
        {
            ret = -1;
            break;
        }

        operation = "materialize";
        ncnn::Mat materialized_key;
        ncnn::Mat materialized_value;
        ret = ex.extract("out_k", materialized_key);
        if (ret == 0)
            ret = ex.extract("out_v", materialized_value, 0);
        if (ret != 0)
            break;
        if (materialized_key.dims != 3 || materialized_value.dims != 3 || materialized_key.elempack != 1 || materialized_value.elempack != 1 || materialized_key.elemsize != 4u || materialized_value.elemsize != 4u || CompareMat(reference_outputs[1], materialized_key, epsilon) != 0 || CompareMat(reference_outputs[2], materialized_value, epsilon) != 0)
        {
            ret = -1;
            break;
        }

        // editing a materialized result must leave the extractor's cache intact
        for (int q = 0; q < kv_heads; q++)
        {
            materialized_key.channel(q).row(0)[0] += 0.25f;
            materialized_value.channel(q).row(0)[0] -= 0.5f;
        }
        operation = "independent storage";
        if (materialized_key.cstep != ncnn::alignSize((size_t)materialized_key.w * materialized_key.h * 4, 16) / 4 || materialized_value.cstep != ncnn::alignSize((size_t)materialized_value.w * materialized_value.h * 4, 16) / 4)
        {
            ret = -1;
            break;
        }

        ncnn::Mat original_key;
        ncnn::Mat original_value;
        ret = ex.extract("out_k", original_key);
        if (ret == 0)
            ret = ex.extract("out_v", original_value);
        if (ret != 0 || CompareMat(reference_outputs[1], original_key, epsilon) != 0 || CompareMat(reference_outputs[2], original_value, epsilon) != 0)
        {
            ret = -1;
            break;
        }

        if (cache_type == 1 && !trim)
        {
            ret = ex.extract("out_k", key_cache, 1);
            if (ret == 0)
                ret = ex.extract("out_v", value_cache, 1);
            if (ret != 0 || key_cache.empty() || value_cache.empty())
            {
                ret = -1;
                break;
            }
            if (step == 1)
            {
                ncnn::Mat key_copy = key_cache.clone();
                ncnn::Mat value_copy = value_cache.clone();
                if (key_copy.empty() || value_copy.empty() || key_copy.data == key_cache.data || value_copy.data == value_cache.data)
                    ret = -1;
                key_cache = key_copy;
                value_cache = value_copy;
            }
            reference_key = original_key;
            reference_value = original_value;
        }
        else
        {
            key_cache = materialized_key;
            value_cache = materialized_value;
            if (trim && (step == 1 || step == 2))
            {
                const int length = step == 1 ? 7 : 0;
                key_cache.h = value_cache.h = length;
                if (trim == 2 && length)
                {
                    ncnn::Mat trimmed_key(head_dim, length, kv_heads);
                    ncnn::Mat trimmed_value(value_dim, length, kv_heads);
                    for (int q = 0; q < kv_heads; q++)
                    {
                        memcpy(trimmed_key.channel(q), key_cache.channel(q), (size_t)head_dim * length * sizeof(float));
                        memcpy(trimmed_value.channel(q), value_cache.channel(q), (size_t)value_dim * length * sizeof(float));
                    }
                    key_cache = trimmed_key;
                    value_cache = trimmed_value;
                }
            }
            reference_key = key_cache;
            reference_value = value_cache;
        }
    }

    reference->destroy_pipeline(opt);
    delete reference;
    if (ret != 0)
        fprintf(stderr, "test_sdpa_kvcache_layout failed dim=%d value_dim=%d heads=%d kv_heads=%d bf16=%d type=%d trim=%d empty=%d vulkan=%d step=%d operation=%s\n", head_dim, value_dim, heads, kv_heads, bf16, cache_type, trim, empty_type, vulkan, step, operation);
    return ret;
}

class KvcacheTestAllocator : public ncnn::Allocator
{
public:
    KvcacheTestAllocator()
        : fail(false)
    {
    }

    virtual void* fastMalloc(size_t size)
    {
        return fail ? 0 : ncnn::fastMalloc(size);
    }

    virtual void fastFree(void* ptr)
    {
        ncnn::fastFree(ptr);
    }

    bool fail;
};

static int test_sdpa_kvcache_ownership(bool naive)
{
    KvcacheTestAllocator allocator;
    KvcacheTestAllocator output_allocator;
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_bf16_storage = false;
    net.opt.num_threads = 1;
    net.opt.kvcache_allocator = &allocator;
    net.opt.kvcache_max_seqlen_hint = 32;
    if (naive)
        net.register_custom_layer("SDPA", create_sdpa_naive);
    const unsigned int empty_model = 0;
    if (net.load_param_mem(sdpa_param) || net.load_model((const unsigned char*)&empty_model))
        return -1;

    ncnn::Mat key = RandomMat(8, 32, 2).clone(&allocator);
    ncnn::Mat value = RandomMat(5, 32, 2).clone(&allocator);
    ncnn::Mat original_key = key.clone();
    ncnn::Mat original_value = value.clone();
    key.h = value.h = 13;
    ncnn::Mat query = RandomMat(8, 1, 4);
    ncnn::Mat current_key = RandomMat(8, 1, 2);
    ncnn::Mat current_value = RandomMat(5, 1, 2);
    ncnn::Extractor first = net.create_extractor();
    first.input("q", query);
    first.input("k", current_key);
    first.input("v", current_value);
    first.input("past_k", key);
    first.input("past_v", value);
    ncnn::Mat first_output;
    ncnn::Mat first_key;
    ncnn::Mat first_value;
    if (first.extract("out", first_output) || first.extract("out_k", first_key) || first.extract("out_v", first_value))
        return -1;

    ncnn::Extractor second = net.create_extractor();
    second.input("q", query);
    second.input("k", current_key);
    second.input("v", current_value);
    second.input("past_k", key);
    second.input("past_v", value);
    ncnn::Mat second_output;
    if (second.extract("out", second_output) || CompareMat(first_output, second_output))
        return -1;

    // even the reserved rows of a materialized input belong to the caller
    key.h = value.h = 32;
    if (CompareMat(key, original_key) || CompareMat(value, original_value))
        return -1;
    ncnn::Mat check_key;
    ncnn::Mat check_value;
    if (first.extract("out_k", check_key) || first.extract("out_v", check_value) || CompareMat(first_key, check_key) || CompareMat(first_value, check_value))
        return -1;

    allocator.fail = true;
    ncnn::Extractor failed = net.create_extractor();
    failed.input("q", query);
    failed.input("k", current_key);
    failed.input("v", current_value);
    failed.input("past_k", first_key);
    failed.input("past_v", first_value);
    ncnn::Mat output;
    if (failed.extract("out", output) != -100)
        return -1;

    output_allocator.fail = true;
    first.set_blob_allocator(&output_allocator);
    if (first.extract("out_k", output) != -100)
        return -1;
    allocator.fail = false;

    return 0;
}

#if NCNN_VULKAN
class KvcacheTestVkStagingAllocator : public ncnn::VkStagingAllocator
{
public:
    KvcacheTestVkStagingAllocator(const ncnn::VulkanDevice* vkdev)
        : ncnn::VkStagingAllocator(vkdev), fail(false)
    {
    }

    virtual ncnn::VkBufferMemory* fastMalloc(size_t size)
    {
        return fail ? 0 : ncnn::VkStagingAllocator::fastMalloc(size);
    }

    bool fail;
};

static int test_sdpa_kvcache_download_failure(ncnn::VulkanDevice* vkdev)
{
    ncnn::VkBlobAllocator blob_allocator(vkdev);
    KvcacheTestVkStagingAllocator staging_allocator(vkdev);
    KvcacheTestAllocator allocator;
    ncnn::Net net;
    net.opt.num_threads = 1;
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_packed = false;
    net.opt.blob_vkallocator = &blob_allocator;
    net.opt.workspace_vkallocator = &blob_allocator;
    net.opt.staging_vkallocator = &staging_allocator;
    net.opt.kvcache_allocator = &allocator;
    const unsigned int empty_model = 0;
    if (net.load_param_mem(sdpa_param) || net.load_model((const unsigned char*)&empty_model))
        return -1;

    ncnn::Mat key = RandomMat(8, 3, 2);
    ncnn::Extractor ex = net.create_extractor();
    ex.input("q", RandomMat(8, 3, 4));
    ex.input("k", key);
    ex.input("v", RandomMat(5, 3, 2));
    {
        ncnn::VkCompute cmd(vkdev);
        ncnn::VkMat cache;
        if (ex.extract("out_k", cache, cmd) || cmd.submit_and_wait())
            return -1;
    }

    // exercise the staging path even on a device with host-visible memory
    const bool mappable = blob_allocator.mappable;
    blob_allocator.mappable = false;
    staging_allocator.fail = true;
    ncnn::Mat cache;
    int ret = ex.extract("out_k", cache, 1) == -100 ? 0 : -1;
    staging_allocator.fail = false;
    allocator.fail = true;
    if (ret == 0 && ex.extract("out_k", cache, 1) != -100)
        ret = -1;
    allocator.fail = false;
    if (ret == 0)
        ret = ex.extract("out_k", cache);
    blob_allocator.mappable = mappable;
    if (ret == 0)
        ret = CompareMat(key, cache);
    return ret;
}
#endif

int main()
{
    SRAND(7767517);

    if (test_sdpa_kvcache_ownership(false) || test_sdpa_kvcache_ownership(true))
        return -1;
    for (int type = 0; type < 2; type++)
    {
        if (test_sdpa_kvcache_layout(13, 7, 6, 3, 0, type, 0, 0, false, true) || test_sdpa_kvcache_layout(13, 7, 6, 3, 0, type, 1, 1, false, true))
            return -1;
    }

    for (int empty_type = 0; empty_type < 3; empty_type++)
    {
        for (int trim = 0; trim < 3; trim++)
        {
            for (int type = 0; type < 2; type++)
            {
                if (test_sdpa_kvcache_layout(37, 29, 6, 3, 0, type, trim, empty_type, false) != 0)
                    return -1;
#if NCNN_BF16
                if (test_sdpa_kvcache_layout(37, 29, 6, 3, 1, type, trim, empty_type, false) != 0)
                    return -1;
#endif
            }
        }
    }
#if NCNN_VULKAN
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (vkdev)
    {
        if (test_sdpa_kvcache_download_failure(vkdev))
            return -1;
        for (int storage = 0; storage < 3; storage++)
        {
            if (storage == 1 && !vkdev->info.support_bf16_storage())
                continue;
            if (storage == 2 && !vkdev->info.support_fp16_storage())
                continue;
            for (int type = 0; type < 2; type++)
            {
                for (int trim = 0; trim < 3; trim++)
                {
                    if (test_sdpa_kvcache_layout(13, 7, 8, 4, storage, type, trim, trim, true) != 0)
                        return -1;
                }
            }
        }
    }
#endif
    return test_sdpa_kvcache_layout(8, 5, 1, 1, 0, 1, 0, 0, false);
}
