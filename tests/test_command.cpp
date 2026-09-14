// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "command.h"
#include "gpu.h"
#include "mat.h"
#include "testutil.h"

static int test_command_upload_download(const ncnn::Mat& a)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.blob_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;

    if (!vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
    if (!vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;
    if (!vkdev->info.support_bf16_packed()) opt.use_bf16_packed = false;
    if (!vkdev->info.support_bf16_storage()) opt.use_bf16_storage = false;

    ncnn::Mat c;
    {
        ncnn::VkCompute cmd(vkdev);

        ncnn::VkMat b;
        cmd.record_upload(a, b, opt);
        cmd.record_download(b, c, opt);

        cmd.submit_and_wait();
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);

    if (CompareMat(a, c, 0.001) != 0)
    {
        fprintf(stderr, "test_command_upload_download image failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
        return -1;
    }

    return 0;
}

static int test_command_clone(const ncnn::Mat& a)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.blob_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;

    if (!vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
    if (!vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;
    if (!vkdev->info.support_bf16_packed()) opt.use_bf16_packed = false;
    if (!vkdev->info.support_bf16_storage()) opt.use_bf16_storage = false;

    const bool support_image_storage = vkdev->info.support_image_storage();

    ncnn::Mat d;
    ncnn::Mat e;
    {
        ncnn::VkCompute cmd(vkdev);

        ncnn::VkMat b1;
        ncnn::VkMat b2;
        ncnn::VkImageMat b3;
        ncnn::VkImageMat c1;
        ncnn::VkImageMat c2;
        ncnn::VkMat c3;
        cmd.record_clone(a, b1, opt);
        cmd.record_clone(b1, b2, opt);

        if (support_image_storage)
        {
            cmd.record_clone(a, c1, opt);
            cmd.record_clone(b2, b3, opt);
            cmd.record_clone(c1, c2, opt);
            cmd.record_clone(b3, d, opt);
            cmd.record_clone(c2, c3, opt);
            cmd.record_clone(c3, e, opt);
        }
        else
        {
            cmd.record_clone(b2, d, opt);
        }

        cmd.submit_and_wait();
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);

    if (CompareMat(a, d, 0.001) != 0)
    {
        fprintf(stderr, "test_command_clone buffer failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
        return -1;
    }

    if (support_image_storage && CompareMat(a, e, 0.001) != 0)
    {
        fprintf(stderr, "test_command_clone image failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
        return -1;
    }

    return 0;
}

static int test_command_transfer(const ncnn::Mat& a)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.blob_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;

    if (!vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
    if (!vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;
    if (!vkdev->info.support_bf16_packed()) opt.use_bf16_packed = false;
    if (!vkdev->info.support_bf16_storage()) opt.use_bf16_storage = false;

    ncnn::Mat c;
    {
        ncnn::VkTransfer cmd1(vkdev);

        ncnn::VkMat b;
        cmd1.record_upload(a, b, opt, false);

        cmd1.submit_and_wait();

        ncnn::VkCompute cmd2(vkdev);

        cmd2.record_download(b, c, opt);

        cmd2.submit_and_wait();
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);

    if (CompareMat(a, c, 0.001) != 0)
    {
        fprintf(stderr, "test_command_transfer buffer failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
        return -1;
    }

    return 0;
}

static int test_command_cstep(int dims, int elempack, bool fp16, bool staging)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (fp16 && !vkdev->info.support_fp16_storage())
        return 0;

    ncnn::VkBlobAllocator blob_allocator(vkdev);
    ncnn::VkStagingAllocator staging_allocator(vkdev);
    if (staging)
        blob_allocator.mappable = false;

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_fp16_storage = fp16;
    opt.use_fp16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_bf16_packed = false;
    opt.blob_vkallocator = &blob_allocator;
    opt.staging_vkallocator = &staging_allocator;

    ncnn::Mat a = dims == 3 ? RandomMat(5, 19, 12) : RandomMat(5, 19, 3, 12);
    ncnn::Mat packed;
    ncnn::convert_packing(a, packed, elempack, opt);
    a = packed;
    if (fp16)
    {
        ncnn::Mat converted;
        ncnn::cast_float32_to_float16(a, converted, opt);
        a = converted;
    }
#if NCNN_BATCH
    ncnn::Mat batched;
    batched.create_like(a, 2);
    for (int b = 0; b < batched.n; b++)
        memcpy(batched.batch(b).data, a.data, a.total() * a.elemsize);
    a = batched;
#endif

    ncnn::Mat cloned;
    ncnn::Mat copied;
    ncnn::Mat reuploaded;
    ncnn::Mat downloaded;
    ncnn::Mat uploaded;
    {
        ncnn::VkCompute cmd(vkdev);
        ncnn::VkMat reserved;
        cmd.record_clone(a, reserved, opt);
        if (reserved.empty())
            return -1;

        // keep the allocation stride while shortening the logical shape
        a.h = reserved.h = 7;
        ncnn::VkMat shape;
        shape.create_like(reserved, &blob_allocator);
        if (shape.cstep != reserved.cstep)
            return -1;
        ncnn::VkBufferMemory* data = shape.data;
        shape.create_like(reserved, &blob_allocator);
        if (shape.data != data)
            return -1;
        shape.release();
        if (dims == 3)
            shape.create(a.w, a.h, a.c, a.elemsize, a.elempack, a.n, &blob_allocator);
        else
            shape.create(a.w, a.h, a.d, a.c, a.elemsize, a.elempack, a.n, &blob_allocator);
        if (shape.cstep != ncnn::alignSize((size_t)a.w * a.h * a.d * a.elemsize, 16) / a.elemsize)
            return -1;

        ncnn::VkMat compact;
        ncnn::VkMat from_host;
        cmd.record_clone(reserved, cloned, opt);
        cmd.record_clone(reserved, compact, opt);
        cmd.record_clone(compact, copied, opt);
        ncnn::VkMat raw_upload;
        cmd.record_clone(a, raw_upload, opt);
        cmd.record_clone(raw_upload, reuploaded, opt);
        cmd.record_download(reserved, downloaded, opt);
        cmd.record_upload(a, from_host, opt);
        cmd.record_download(from_host, uploaded, opt);
        if (cmd.submit_and_wait())
            return -1;
    }

    ncnn::Mat transferred;
    {
        ncnn::VkTransfer transfer(vkdev);
        ncnn::VkMat device;
        transfer.record_upload(a, device, opt, false);
        if (transfer.submit_and_wait())
            return -1;
        ncnn::VkCompute cmd(vkdev);
        cmd.record_clone(device, transferred, opt);
        if (cmd.submit_and_wait())
            return -1;
    }

    if (cloned.cstep != a.cstep || copied.cstep != a.cstep || reuploaded.cstep != a.cstep)
        return -1;
    for (int b = 0; b < a.n; b++)
    {
        if (memcmp(a.batch(b).data, cloned.batch(b).data, a.total() * a.elemsize) || memcmp(a.batch(b).data, copied.batch(b).data, a.total() * a.elemsize) || memcmp(a.batch(b).data, reuploaded.batch(b).data, a.total() * a.elemsize))
        {
            fprintf(stderr, "test_command_cstep clone changed storage dims=%d elempack=%d fp16=%d staging=%d\n", dims, elempack, fp16, staging);
            return -1;
        }
    }

    if (cloned.elempack != a.elempack || cloned.elemsize != a.elemsize || copied.elempack != a.elempack || copied.elemsize != a.elemsize || CompareMat(a, cloned, 0.f) || CompareMat(a, copied, 0.f) || CompareMat(a, downloaded, 0.001f) || CompareMat(a, uploaded, 0.001f) || CompareMat(a, transferred, 0.f))
    {
        fprintf(stderr, "test_command_cstep failed dims=%d elempack=%d fp16=%d staging=%d\n", dims, elempack, fp16, staging);
        return -1;
    }

    return 0;
}

static int test_command_0()
{
    return 0
           || test_command_upload_download(RandomMat(5, 7, 24))
           || test_command_upload_download(RandomMat(7, 9, 12))
           || test_command_upload_download(RandomMat(3, 5, 13))
           || test_command_upload_download(RandomMat(15, 24))
           || test_command_upload_download(RandomMat(19, 12))
           || test_command_upload_download(RandomMat(17, 15))
           || test_command_upload_download(RandomMat(128))
           || test_command_upload_download(RandomMat(124))
           || test_command_upload_download(RandomMat(127));
}

static int test_command_1()
{
    return 0
           || test_command_clone(RandomMat(5, 7, 24))
           || test_command_clone(RandomMat(7, 9, 12))
           || test_command_clone(RandomMat(3, 5, 13))
           || test_command_clone(RandomMat(15, 24))
           || test_command_clone(RandomMat(19, 12))
           || test_command_clone(RandomMat(17, 15))
           || test_command_clone(RandomMat(128))
           || test_command_clone(RandomMat(124))
           || test_command_clone(RandomMat(127));
}

static int test_command_2()
{
    return 0
           || test_command_transfer(RandomMat(5, 7, 24))
           || test_command_transfer(RandomMat(7, 9, 12))
           || test_command_transfer(RandomMat(3, 5, 13))
           || test_command_transfer(RandomMat(15, 24))
           || test_command_transfer(RandomMat(19, 12))
           || test_command_transfer(RandomMat(17, 15))
           || test_command_transfer(RandomMat(128))
           || test_command_transfer(RandomMat(124))
           || test_command_transfer(RandomMat(127));
}

int main()
{
    SRAND(7767517);

    for (int dims = 3; dims <= 4; dims++)
    {
        for (int elempack = 1; elempack <= 4; elempack *= 4)
        {
            for (int fp16 = 0; fp16 < 2; fp16++)
            {
                if (test_command_cstep(dims, elempack, fp16, false) || test_command_cstep(dims, elempack, fp16, true))
                    return -1;
            }
        }
    }

    return test_command_0() || test_command_1() || test_command_2();
}
