// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include "net.h"

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif // NCNN_VULKAN

#include <float.h>

static int test_multiheadattention_kvcache_allocator(const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, int qdim, int max_seqlen_hint, int use_kvcache_allocator)
{
    ncnn::Layer* reference = ncnn::create_layer_naive("MultiHeadAttention");
    ncnn::Layer* op = ncnn::create_layer_cpu("MultiHeadAttention");
    if (!reference || !op)
    {
        delete reference;
        delete op;
        return -1;
    }

    reference->load_param(pd);
    reference->load_model(ncnn::ModelBinFromMatArray(weights.data()));
    op->load_param(pd);
    op->load_model(ncnn::ModelBinFromMatArray(weights.data()));

    ncnn::UnlockedPoolAllocator kvcache_allocator;
    ncnn::Option reference_opt;
    ncnn::Option opt;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.kvcache_allocator = use_kvcache_allocator ? &kvcache_allocator : 0;
    opt.kvcache_max_seqlen_hint = max_seqlen_hint;

    int ret = reference->create_pipeline(reference_opt);
    if (ret == 0)
        ret = op->create_pipeline(opt);

    ncnn::Mat reference_key;
    ncnn::Mat reference_value;
    ncnn::Mat key_cache;
    ncnn::Mat value_cache;
    const int append_lengths[] = {15, 2, 1};
    void* key_data = 0;
    void* value_data = 0;

    for (int i = 0; ret == 0 && i < 3; i++)
    {
        ncnn::Mat q = RandomMat(qdim, append_lengths[i]);

        std::vector<ncnn::Mat> reference_bottoms(3);
        reference_bottoms[0] = q;
        reference_bottoms[1] = reference_key;
        reference_bottoms[2] = reference_value;
        reference_key.release();
        reference_value.release();

        std::vector<ncnn::Mat> reference_tops(3);
        ret = reference->forward(reference_bottoms, reference_tops, reference_opt);
        if (ret != 0)
            break;

        std::vector<ncnn::Mat> bottoms(3);
        bottoms[0] = q;
        bottoms[1] = key_cache;
        bottoms[2] = value_cache;
        key_cache.release();
        value_cache.release();

        std::vector<ncnn::Mat> tops(3);
        ret = op->forward(bottoms, tops, opt);
        if (ret != 0)
            break;

        if (CompareMat(reference_tops[0], tops[0], 0.001) != 0)
            ret = -1;
        if (tops[1].allocator != opt.kvcache_allocator || tops[2].allocator != opt.kvcache_allocator)
            ret = -1;

        if (i == 0 || (i == 1 && max_seqlen_hint == 0))
        {
            if (i == 1 && (tops[1].data == key_data || tops[2].data == value_data))
                ret = -1;
            key_data = tops[1].data;
            value_data = tops[2].data;
        }
        if (tops[1].data != key_data || tops[2].data != value_data)
            ret = -1;

        reference_key = reference_tops[1];
        reference_value = reference_tops[2];
        reference_tops[1].release();
        reference_tops[2].release();

        key_cache = tops[1];
        value_cache = tops[2];
        tops[1].release();
        tops[2].release();
    }

    reference->destroy_pipeline(reference_opt);
    op->destroy_pipeline(opt);
    delete reference;
    delete op;

    if (ret != 0)
        fprintf(stderr, "test_multiheadattention_kvcache_allocator failed ret=%d\n", ret);

    return ret;
}

static int test_multiheadattention_kvcache_layout(int storage, int type, bool vulkan)
{
    ncnn::UnlockedPoolAllocator allocator;
    ncnn::Net net;
    net.opt.num_threads = 2;
    net.opt.use_vulkan_compute = vulkan;
    net.opt.use_fp16_storage = storage == 2;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = storage == 1;
    net.opt.kvcache_allocator = &allocator;
    net.opt.kvcache_max_seqlen_hint = 32;
    const char param[] = "7767517\n"
                         "3 6\n"
                         "Input input 0 1 q\n"
                         "Input cache 0 2 past_k past_v\n"
                         "MultiHeadAttention mha 3 3 q past_k past_v out out_k out_v 0=16 1=4 2=192 3=12 4=12 7=1\n";
    if (net.load_param_mem(param) != 0)
        return -1;

    std::vector<ncnn::Mat> weights(8);
    std::vector<float> model;
    for (int i = 0; i < 8; i++)
    {
        weights[i] = RandomMat(i % 2 == 0 ? 192 : i == 7 ? 12 : 16, -0.2f, 0.2f);
        if (i % 2 == 0)
            model.push_back(0.f);
        const float* ptr = weights[i];
        model.insert(model.end(), ptr, ptr + weights[i].w);
    }
    if (net.load_model((const unsigned char*)model.data()) != model.size() * sizeof(float))
        return -1;

    ncnn::Layer* reference = ncnn::create_layer_naive("MultiHeadAttention");
    ncnn::ParamDict pd;
    pd.set(0, 16);
    pd.set(1, 4);
    pd.set(2, 192);
    pd.set(3, 12);
    pd.set(4, 12);
    pd.set(7, 1);
    reference->load_param(pd);
    reference->load_model(ncnn::ModelBinFromMatArray(weights.data()));
    ncnn::Option opt;
    opt.num_threads = 1;
    reference->create_pipeline(opt);

    ncnn::Mat key;
    ncnn::Mat value;
    ncnn::Mat reference_key;
    ncnn::Mat reference_value;
    int ret = 0;
    const float epsilon = storage ? 0.03f : 0.001f;
    for (int step = 0; step < 4 && ret == 0; step++)
    {
        ncnn::Mat query = RandomMat(12, step == 0 ? 13 : step == 3 ? 40 : 1);
        std::vector<ncnn::Mat> bottoms(3);
        bottoms[0] = query;
        bottoms[1] = reference_key;
        bottoms[2] = reference_value;
        std::vector<ncnn::Mat> tops(3);
        ret = reference->forward(bottoms, tops, opt);
        if (ret != 0)
            break;

        ncnn::Mat key_before = type == 0 ? key.clone() : ncnn::Mat();
        ncnn::Mat value_before = type == 0 ? value.clone() : ncnn::Mat();
        ncnn::Extractor ex = net.create_extractor();
        ex.input("q", query);
        ex.input("past_k", key);
        ex.input("past_v", value);
        ncnn::Mat output;
        ret = ex.extract("out", output);
        if (ret != 0 || CompareMat(tops[0], output, epsilon) != 0)
        {
            ret = -1;
            break;
        }
        if (type == 0 && !key_before.empty() && (CompareMat(key, key_before) || CompareMat(value, value_before)))
        {
            ret = -1;
            break;
        }

        ncnn::Mat materialized_key;
        ncnn::Mat materialized_value;
        ret = ex.extract("out_k", materialized_key) || ex.extract("out_v", materialized_value);
        if (ret != 0 || materialized_key.dims != 3 || materialized_key.w != 4 || materialized_key.c != 4 || materialized_key.elempack != 1 || materialized_key.elemsize != 4u || CompareMat(tops[1], materialized_key, epsilon) || CompareMat(tops[2], materialized_value, epsilon))
        {
            ret = -1;
            break;
        }

        if (type == 1)
        {
            ret = ex.extract("out_k", key, 1) || ex.extract("out_v", value, 1);
            if (key.empty() || value.empty())
                ret = -1;
        }
        else
        {
            key = materialized_key;
            value = materialized_value;
            key.h = value.h = step == 1 ? 7 : step == 2 ? 0 : key.h;
            if (key.h > 0)
            {
                key.channel(0).row(0)[0] += 0.25f;
                value.channel(0).row(0)[0] -= 0.5f;
            }
        }
        reference_key = materialized_key;
        reference_value = materialized_value;
        if (type == 0)
        {
            reference_key.h = key.h;
            reference_value.h = value.h;
        }
    }

    reference->destroy_pipeline(opt);
    delete reference;
    if (ret != 0)
        fprintf(stderr, "test_multiheadattention_kvcache_layout failed storage=%d type=%d vulkan=%d\n", storage, type, vulkan);
    return ret;
}

static int test_multiheadattention_kvcache_allocator()
{
    const int qdim = 12;
    const int embed_dim = 16;
    const int num_heads = 4;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, qdim);
    pd.set(4, qdim);
    pd.set(6, 0.7f / sqrtf(embed_dim / num_heads));
    pd.set(7, 1); // kv_cache

    std::vector<ncnn::Mat> weights(8);
    weights[0] = RandomMat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomMat(embed_dim * qdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomMat(embed_dim * qdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomMat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);

    return 0
           || test_multiheadattention_kvcache_allocator(pd, weights, qdim, 0, 0)
           || test_multiheadattention_kvcache_allocator(pd, weights, qdim, 32, 0)
           || test_multiheadattention_kvcache_allocator(pd, weights, qdim, 0, 1)
           || test_multiheadattention_kvcache_allocator(pd, weights, qdim, 32, 1);
}

#if NCNN_VULKAN
static int test_multiheadattention_vulkan_kvcache_allocator()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return 0;

    const int qdim = 12;
    const int embed_dim = 16;
    const int num_heads = 4;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, qdim);
    pd.set(4, qdim);
    pd.set(6, 0.7f / sqrtf(embed_dim / num_heads));
    pd.set(7, 1); // kv_cache

    std::vector<ncnn::Mat> weights(8);
    weights[0] = RandomMat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomMat(embed_dim * qdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomMat(embed_dim * qdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomMat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);

    ncnn::VkBlobAllocator blob_vkallocator(vkdev);
    ncnn::VkBlobAllocator kvcache_vkallocator(vkdev);
    ncnn::VkStagingAllocator staging_vkallocator(vkdev);
    ncnn::VkWeightAllocator weight_vkallocator(vkdev);
    ncnn::VkWeightStagingAllocator weight_staging_vkallocator(vkdev);

    ncnn::Layer* reference = ncnn::create_layer_naive("MultiHeadAttention");
    ncnn::Layer* op = ncnn::create_layer_vulkan("MultiHeadAttention");
    if (!reference || !op)
    {
        delete reference;
        delete op;
        return -1;
    }

    ncnn::Option reference_opt;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.blob_vkallocator = &blob_vkallocator;
    opt.workspace_vkallocator = &blob_vkallocator;
    opt.staging_vkallocator = &staging_vkallocator;
    opt.kvcache_vkallocator = &kvcache_vkallocator;
    opt.kvcache_max_seqlen_hint = 0;

    reference->load_param(pd);
    reference->load_model(ncnn::ModelBinFromMatArray(weights.data()));
    op->vkdev = vkdev;
    op->load_param(pd);
    op->load_model(ncnn::ModelBinFromMatArray(weights.data()));

    int ret = reference->create_pipeline(reference_opt);
    if (ret == 0)
        ret = op->create_pipeline(opt);
    if (ret == 0)
    {
        ncnn::VkTransfer upload_cmd(vkdev);
        ncnn::Option upload_opt = opt;
        upload_opt.blob_vkallocator = &weight_vkallocator;
        upload_opt.workspace_vkallocator = &weight_vkallocator;
        upload_opt.staging_vkallocator = &weight_staging_vkallocator;
        op->upload_model(upload_cmd, upload_opt);
        ret = upload_cmd.submit_and_wait();
    }

    ncnn::Mat reference_key;
    ncnn::Mat reference_value;
    ncnn::VkMat key_cache;
    ncnn::VkMat value_cache;
    const int append_lengths[] = {15, 2, 1};
    ncnn::VkBufferMemory* key_data = 0;
    ncnn::VkBufferMemory* value_data = 0;

    for (int i = 0; ret == 0 && i < 3; i++)
    {
        ncnn::Mat q = RandomMat(qdim, append_lengths[i]);

        std::vector<ncnn::Mat> reference_bottoms(3);
        reference_bottoms[0] = q;
        reference_bottoms[1] = reference_key;
        reference_bottoms[2] = reference_value;
        reference_key.release();
        reference_value.release();
        std::vector<ncnn::Mat> reference_tops(3);
        ret = reference->forward(reference_bottoms, reference_tops, reference_opt);
        if (ret != 0)
            break;

        ncnn::VkCompute cmd(vkdev);
        ncnn::VkMat q_gpu;
        cmd.record_clone(q, q_gpu, opt);
        std::vector<ncnn::VkMat> bottoms(3);
        bottoms[0] = q_gpu;
        bottoms[1] = key_cache;
        bottoms[2] = value_cache;
        key_cache.release();
        value_cache.release();

        std::vector<ncnn::VkMat> tops(3);
        ret = op->forward(bottoms, tops, cmd, opt);

        ncnn::Mat output;
        if (ret == 0)
            cmd.record_clone(tops[0], output, opt);
        if (ret == 0)
            ret = cmd.submit_and_wait();
        if (ret != 0)
            break;

        if (CompareMat(reference_tops[0], output, 0.005f) != 0)
            ret = -1;
        if (tops[1].allocator != &kvcache_vkallocator || tops[2].allocator != &kvcache_vkallocator)
            ret = -1;

        if (i == 0)
        {
            key_data = tops[1].data;
            value_data = tops[2].data;
        }
        if (i == 1)
        {
            if (tops[1].data == key_data || tops[2].data == value_data)
                ret = -1;

            key_data = tops[1].data;
            value_data = tops[2].data;
        }
        if (i == 2 && (tops[1].data != key_data || tops[2].data != value_data))
            ret = -1;

        reference_key = reference_tops[1];
        reference_value = reference_tops[2];
        reference_tops[1].release();
        reference_tops[2].release();

        key_cache = tops[1];
        value_cache = tops[2];
        tops[1].release();
        tops[2].release();
    }

    reference->destroy_pipeline(reference_opt);
    op->destroy_pipeline(opt);
    delete reference;
    delete op;
    if (ret != 0)
        fprintf(stderr, "test_multiheadattention_vulkan_kvcache_allocator failed ret=%d\n", ret);

    return ret;
}
#endif // NCNN_VULKAN

int main()
{
    SRAND(7767517);

    for (int type = 0; type < 2; type++)
    {
        if (test_multiheadattention_kvcache_layout(0, type, false))
            return -1;
#if NCNN_BF16
        if (test_multiheadattention_kvcache_layout(1, type, false))
            return -1;
#endif
#if NCNN_VULKAN
        ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
        if (vkdev)
        {
            if (test_multiheadattention_kvcache_layout(0, type, true))
                return -1;
            if (vkdev->info.support_fp16_storage() && test_multiheadattention_kvcache_layout(2, type, true))
                return -1;
        }
#endif
    }

    return 0
           || test_multiheadattention_kvcache_allocator()
#if NCNN_VULKAN
           || test_multiheadattention_vulkan_kvcache_allocator()
#endif
           ;
}
