// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int packing_cpu_naive(const ncnn::Mat& a, ncnn::Mat& b, int out_elempack)
{
    ncnn::ParamDict pd;
    pd.set(0, out_elempack);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;

    ncnn::Layer* op = ncnn::create_layer_naive("Packing");

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    op->forward(a, b, opt);

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

static int test_packing_cpu_fp32(const ncnn::Mat& a, int in_elempack, int out_elempack)
{
    ncnn::ParamDict pd;
    pd.set(0, out_elempack);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_int8_inference = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer_cpu("Packing");

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    ncnn::Mat ap;
    ncnn::convert_packing(a, ap, in_elempack, opt);

    ncnn::Mat b;
    packing_cpu_naive(ap, b, out_elempack);

    ncnn::Mat c;
    op->forward(ap, c, opt);

    op->destroy_pipeline(opt);

    delete op;

    if (CompareMat(b, c, 0.001) != 0)
    {
        fprintf(stderr, "test_packing_cpu_fp32 failed a.dims=%d a=(%d %d %d %d) in_elempack=%d out_elempack=%d\n", a.dims, a.w, a.h, a.d, a.c, in_elempack, out_elempack);
        return -1;
    }

    return 0;
}

static int test_packing_cpu_fp16(const ncnn::Mat& a, int in_elempack, int out_elempack)
{
    ncnn::ParamDict pd;
    pd.set(0, out_elempack);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_int8_inference = false;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer_cpu("Packing");

    if (!op->support_fp16_storage)
    {
        delete op;
        return 0;
    }

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    ncnn::Mat a16;
    ncnn::cast_float32_to_float16(a, a16, opt);

    ncnn::Mat ap;
    ncnn::convert_packing(a16, ap, in_elempack, opt);

    ncnn::Mat b;
    packing_cpu_naive(ap, b, out_elempack);

    ncnn::Mat c;
    op->forward(ap, c, opt);

    op->destroy_pipeline(opt);

    delete op;

    ncnn::Mat c32;
    ncnn::cast_float16_to_float32(c, c32, opt);

    if (CompareMat(b, c32, 0.001) != 0)
    {
        fprintf(stderr, "test_packing_cpu_fp16 failed a.dims=%d a=(%d %d %d %d) in_elempack=%d out_elempack=%d\n", a.dims, a.w, a.h, a.d, a.c, in_elempack, out_elempack);
        return -1;
    }

    return 0;
}

static int test_packing_cpu_int8(const ncnn::Mat& a, int in_elempack, int out_elempack)
{
    ncnn::ParamDict pd;
    pd.set(0, out_elempack);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_int8_inference = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer_cpu("Packing");

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    ncnn::Mat a8;
    if (a.dims == 1) a8 = RandomS8Mat(a.w);
    if (a.dims == 2) a8 = RandomS8Mat(a.w, a.h);
    if (a.dims == 3) a8 = RandomS8Mat(a.w, a.h, a.c);
    if (a.dims == 4) a8 = RandomS8Mat(a.w, a.h, a.d, a.c);

    ncnn::Mat ap;
    ncnn::convert_packing(a8, ap, in_elempack, opt);

    ncnn::Mat b;
    packing_cpu_naive(ap, b, out_elempack);

    ncnn::Mat c;
    op->forward(ap, c, opt);

    op->destroy_pipeline(opt);

    delete op;

    ncnn::Mat b32;
    ncnn::cast_int8_to_float32(b, b32, opt);

    ncnn::Mat c32;
    ncnn::cast_int8_to_float32(c, c32, opt);

    if (CompareMat(b32, c32, 0.001) != 0)
    {
        fprintf(stderr, "test_packing_cpu_int8 failed a.dims=%d a=(%d %d %d %d) in_elempack=%d out_elempack=%d\n", a.dims, a.w, a.h, a.d, a.c, in_elempack, out_elempack);
        return -1;
    }

    return 0;
}

static int test_packing_cpu(const ncnn::Mat& a, int in_elempack, int out_elempack)
{
    return 0
           || test_packing_cpu_fp32(a, in_elempack, out_elempack)
           || test_packing_cpu_fp16(a, in_elempack, out_elempack)
           || test_packing_cpu_int8(a, in_elempack, out_elempack);
}

#if NCNN_VULKAN
static int test_packing_gpu_cstep(int dims, int in_elempack, int out_elempack, int cast_type_from, int cast_type_to, bool storage, bool reserved_output)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    const bool fp16 = cast_type_from == 2 || cast_type_to == 2;
    const bool bf16 = cast_type_from == 5 || cast_type_to == 5;
    const bool int8 = cast_type_from == 4 || cast_type_to == 4;
    if (storage && ((fp16 && !vkdev->info.support_fp16_storage()) || (bf16 && !vkdev->info.support_bf16_storage()) || (int8 && !vkdev->info.support_int8_storage())))
        return 0;

    ncnn::VkBlobAllocator blob_allocator(vkdev);
    ncnn::VkStagingAllocator staging_allocator(vkdev);
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_fp16_packed = fp16;
    opt.use_fp16_storage = fp16 && storage;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = bf16;
    opt.use_bf16_storage = bf16 && storage;
    opt.use_int8_packed = int8;
    opt.use_int8_storage = int8 && storage;
    opt.use_int8_arithmetic = false;
    opt.blob_vkallocator = &blob_allocator;
    opt.staging_vkallocator = &staging_allocator;

    ncnn::Mat a;
    ncnn::Mat expected;
    for (int t = 0; t < 2; t++)
    {
        const int cast_type = t == 0 ? cast_type_from : cast_type_to;
        const int elempack = t == 0 ? in_elempack : out_elempack;
        const size_t elemsize = (cast_type == 4 ? 1u : (cast_type == 2 || cast_type == 5) ? 2u : 4u) * elempack;
        ncnn::Mat& m = t == 0 ? a : expected;
        const int h = t == 0 ? 19 : 7;
        if (dims == 3)
            m.create(5, h, 12 / elempack, elemsize, elempack, 2);
        else
            m.create(5, h, 3, 12 / elempack, elemsize, elempack, 2);

        for (int b = 0; b < m.n; b++)
        {
            ncnn::Mat mb = m.batch(b);
            for (int q = 0; q < m.c; q++)
            {
                ncnn::Mat channel = mb.channel(q);
                for (int i = 0; i < m.w * m.h * m.d; i++)
                {
                    for (int k = 0; k < elempack; k++)
                    {
                        const int v = ((q * elempack + k) * 37 + i * 13 + b * 5) % 127 - 63;
                        const int index = i * elempack + k;
                        if (cast_type == 1) ((float*)channel.data)[index] = v / 16.f;
                        if (cast_type == 2) ((unsigned short*)channel.data)[index] = ncnn::float32_to_float16(v / 16.f);
                        if (cast_type == 3) ((int*)channel.data)[index] = v;
                        if (cast_type == 4) ((signed char*)channel.data)[index] = v;
                        if (cast_type == 5) ((unsigned short*)channel.data)[index] = ncnn::float32_to_bfloat16(v / 16.f);
                    }
                }
            }
        }
    }
    a.h = 7;

    ncnn::ParamDict pd;
    pd.set(0, out_elempack);
    pd.set(2, cast_type_from);
    pd.set(3, cast_type_to);
    ncnn::Layer* op = ncnn::create_layer_vulkan("Packing");
    op->vkdev = vkdev;
    op->load_param(pd);
    if (reserved_output)
    {
        op->bottom_shapes.push_back(a);
        op->top_shapes.push_back(expected);
        op->bottom_shapes[0].cstep = ncnn::alignSize((size_t)a.w * a.h * a.d * a.elemsize, 16) / a.elemsize;
    }
    op->create_pipeline(opt);

    ncnn::Mat actual;
    ncnn::VkCompute cmd(vkdev);
    ncnn::Option opt_upload = opt;
    opt_upload.blob_vkallocator = &staging_allocator;
    ncnn::VkMat a_gpu;
    cmd.record_clone(a, a_gpu, opt_upload);
    ncnn::VkMat actual_gpu;
    if (reserved_output)
    {
        ncnn::Mat reserved;
        if (dims == 3)
            reserved.create(expected.w, 29, expected.c, expected.elemsize, out_elempack, expected.n);
        else
            reserved.create(expected.w, 29, expected.d, expected.c, expected.elemsize, out_elempack, expected.n);
        for (int b = 0; b < reserved.n; b++)
            memset(reserved.batch(b).data, 0xa5, reserved.total() * reserved.elemsize);
        cmd.record_clone(reserved, actual_gpu, opt);
        actual_gpu.h = 7;
    }
    const size_t out_cstep = actual_gpu.cstep;
    int ret = op->forward(a_gpu, actual_gpu, cmd, opt);
    if (!ret && reserved_output && actual_gpu.cstep != out_cstep)
        ret = -1;
    cmd.record_clone(actual_gpu, actual, opt);
    ret = cmd.submit_and_wait() || ret;
    op->destroy_pipeline(opt);
    delete op;

    if (!ret)
    {
        if (actual.dims != expected.dims || actual.w != expected.w || actual.h != expected.h || actual.d != expected.d || actual.c != expected.c || actual.n != expected.n || actual.elempack != out_elempack || actual.elemsize != expected.elemsize)
            ret = -1;
        else
        {
            for (int b = 0; b < expected.n; b++)
            {
                for (int q = 0; q < expected.c; q++)
                {
                    if (memcmp(expected.batch(b).channel(q).data, actual.batch(b).channel(q).data, (size_t)expected.w * expected.h * expected.d * expected.elemsize))
                        ret = -1;
                    if (reserved_output)
                    {
                        const unsigned char* ptr = actual.batch(b).channel(q);
                        const size_t size = ncnn::alignSize((size_t)actual.w * actual.h * actual.d * actual.elemsize, 16);
                        for (size_t i = size; i < actual.cstep * actual.elemsize; i++)
                        {
                            if (ptr[i] != 0xa5)
                                ret = -1;
                        }
                    }
                }
            }
        }
    }
    if (ret)
        fprintf(stderr, "test_packing_gpu_cstep failed dims=%d in_elempack=%d out_elempack=%d cast_type_from=%d cast_type_to=%d storage=%d reserved_output=%d\n", dims, in_elempack, out_elempack, cast_type_from, cast_type_to, storage, reserved_output);
    return ret;
}

static int test_packing_gpu(const ncnn::Mat& a, int in_elempack, int out_elempack, int cast_type)
{
    ncnn::ParamDict pd;
    pd.set(0, out_elempack);
    pd.set(2, cast_type); // cast_type_from
    pd.set(3, cast_type); // cast_type_to

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = cast_type == 5; // bfloat16
    opt.use_bf16_storage = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_packing_layout = true;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    if (!vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
    if (!vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;

    ncnn::Layer* op = ncnn::create_layer_vulkan("Packing");

    op->vkdev = vkdev;

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    ncnn::Mat a_cast = a;
    if (cast_type == 5)
        ncnn::cast_float32_to_bfloat16(a, a_cast, opt);

    ncnn::Mat ap;
    ncnn::convert_packing(a_cast, ap, in_elempack, opt);

    ncnn::Mat b;
    packing_cpu_naive(ap, b, out_elempack);

    ncnn::Mat d;

    // forward
    ncnn::VkCompute cmd(vkdev);

    // upload
    ncnn::VkMat a_gpu;
    cmd.record_clone(ap, a_gpu, opt);

    ncnn::VkMat d_gpu;
    op->forward(a_gpu, d_gpu, cmd, opt);

    // download
    cmd.record_clone(d_gpu, d, opt);

    cmd.submit_and_wait();

    op->destroy_pipeline(opt);

    delete op;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    if (cast_type == 5)
    {
        ncnn::Mat b32;
        ncnn::Mat d32;
        ncnn::cast_bfloat16_to_float32(b, b32, opt);
        ncnn::cast_bfloat16_to_float32(d, d32, opt);
        b = b32;
        d = d32;
    }

    if (CompareMat(b, d, cast_type == 5 ? 0.f : 0.001f) != 0)
    {
        fprintf(stderr, "test_packing_gpu failed a.dims=%d a=(%d %d %d %d) in_elempack=%d out_elempack=%d cast_type=%d\n", a.dims, a.w, a.h, a.d, a.c, in_elempack, out_elempack, cast_type);
        return -1;
    }

    return 0;
}

static int test_packing_gpu_int8(const ncnn::Mat& a, int in_elempack, int out_elempack)
{
    ncnn::ParamDict pd;
    pd.set(0, out_elempack);
    pd.set(2, 4); // cast_type_from
    pd.set(3, 4); // cast_type_to

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_packing_layout = true;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    if (!vkdev->info.support_int8_packed()) opt.use_int8_packed = false;
    if (!vkdev->info.support_int8_storage()) opt.use_int8_storage = false;

    ncnn::Layer* op = ncnn::create_layer_vulkan("Packing");

    op->vkdev = vkdev;

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    ncnn::Mat a8;
    if (a.dims == 1) a8 = RandomS8Mat(a.w);
    if (a.dims == 2) a8 = RandomS8Mat(a.w, a.h);
    if (a.dims == 3) a8 = RandomS8Mat(a.w, a.h, a.c);
    if (a.dims == 4) a8 = RandomS8Mat(a.w, a.h, a.d, a.c);

    ncnn::Mat ap;
    ncnn::convert_packing(a8, ap, in_elempack, opt);

    ncnn::Mat b;
    packing_cpu_naive(ap, b, out_elempack);

    ncnn::Mat c;

    // forward
    ncnn::VkCompute cmd(vkdev);

    // upload
    ncnn::VkMat a_gpu;
    cmd.record_clone(ap, a_gpu, opt);

    ncnn::VkMat c_gpu;
    op->forward(a_gpu, c_gpu, cmd, opt);

    // download
    cmd.record_clone(c_gpu, c, opt);

    cmd.submit_and_wait();

    op->destroy_pipeline(opt);

    delete op;

    ncnn::Mat b32;
    ncnn::cast_int8_to_float32(b, b32, opt);

    ncnn::Mat c32;
    ncnn::cast_int8_to_float32(c, c32, opt);

    if (CompareMat(b32, c32, 0.001) != 0)
    {
        fprintf(stderr, "test_packing_gpu_int8 failed a.dims=%d a=(%d %d %d %d) in_elempack=%d out_elempack=%d\n", a.dims, a.w, a.h, a.d, a.c, in_elempack, out_elempack);
        return -1;
    }

    return 0;
}

static int test_packing_gpu(const ncnn::Mat& a, int in_elempack, int out_elempack)
{
    return 0
           || test_packing_gpu(a, in_elempack, out_elempack, 1)
           || test_packing_gpu(a, in_elempack, out_elempack, 5)
           || test_packing_gpu_int8(a, in_elempack, out_elempack);
}
#endif

static int test_packing_cpu(const ncnn::Mat& a)
{
    return 0
           || test_packing_cpu(a, 1, 1)
           || test_packing_cpu(a, 4, 4)
           || test_packing_cpu(a, 4, 8)
           || test_packing_cpu(a, 1, 4)
           || test_packing_cpu(a, 4, 1)
           || test_packing_cpu(a, 1, 8)
           || test_packing_cpu(a, 8, 1)
           || test_packing_cpu(a, 4, 8)
           || test_packing_cpu(a, 8, 4)
           || test_packing_cpu(a, 1, 16)
           || test_packing_cpu(a, 16, 1)
           || test_packing_cpu(a, 4, 16)
           || test_packing_cpu(a, 16, 4)
           || test_packing_cpu(a, 8, 16)
           || test_packing_cpu(a, 16, 8);
}

#if NCNN_VULKAN
static int test_packing_gpu(const ncnn::Mat& a)
{
    return 0
           || test_packing_gpu(a, 1, 1)
           || test_packing_gpu(a, 4, 4)
           || test_packing_gpu(a, 1, 4)
           || test_packing_gpu(a, 4, 1);
}
#endif // NCNN_VULKAN

static int test_packing_0()
{
    ncnn::Mat a = RandomMat(9, 7, 10, 16);
    ncnn::Mat b = RandomMat(9, 7, 10, 3);
    return 0
           || test_packing_cpu(a)
           || test_packing_cpu(b)
#if NCNN_VULKAN
           || test_packing_gpu(a)
#endif
           ;
}

static int test_packing_1()
{
    ncnn::Mat a = RandomMat(9, 10, 16);
    ncnn::Mat b = RandomMat(9, 10, 3);
    return 0
           || test_packing_cpu(a)
           || test_packing_cpu(b)
#if NCNN_VULKAN
           || test_packing_gpu(a)
#endif
           ;
}

static int test_packing_2()
{
    ncnn::Mat a = RandomMat(19, 16);
    return 0
           || test_packing_cpu(a)
#if NCNN_VULKAN
           || test_packing_gpu(a)
#endif
           ;
}

static int test_packing_3()
{
    ncnn::Mat a = RandomMat(80);
    return 0
           || test_packing_cpu(a)
#if NCNN_VULKAN
           || test_packing_gpu(a)
#endif
           ;
}

int main()
{
    SRAND(7767517);

#if NCNN_VULKAN
    const int cast_types[][2] = {{1, 1}, {1, 2}, {2, 1}, {2, 2}, {1, 5}, {5, 1}, {5, 5}, {3, 3}, {3, 4}, {4, 3}, {4, 4}};
    for (int dims = 3; dims <= 4; dims++)
    {
        for (int in_elempack = 1; in_elempack <= 4; in_elempack *= 4)
        {
            for (int out_elempack = 1; out_elempack <= 4; out_elempack *= 4)
            {
                for (int i = 0; i < 11; i++)
                {
                    for (int storage = 0; storage < 2; storage++)
                    {
                        if (test_packing_gpu_cstep(dims, in_elempack, out_elempack, cast_types[i][0], cast_types[i][1], storage, false) || test_packing_gpu_cstep(dims, in_elempack, out_elempack, cast_types[i][0], cast_types[i][1], storage, true))
                            return -1;
                    }
                }
            }
        }
    }
#endif

    return 0
           || test_packing_0()
           || test_packing_1()
           || test_packing_2()
           || test_packing_3();
}
