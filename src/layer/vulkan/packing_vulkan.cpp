// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "packing_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Packing_vulkan::Packing_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_packing = 0;
    pipeline_packing_pack1to4 = 0;
    pipeline_packing_pack4to1 = 0;
}

int Packing_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    const int dims = shape.dims;

    const int local_size_x = std::max(16, (int)vkdev->info.subgroup_size());

    bool use_int8_shader = cast_type_from == 4 || cast_type_to == 4;

    std::vector<vk_specialization_type> specializations(2 + 4);
    specializations[0].i = cast_type_from;
    specializations[1].i = cast_type_to;

    if (shape.dims == 0 || shape.elempack == out_elempack)
    {
        const size_t n = (size_t)shape.w * shape.h * shape.d * shape.elempack;
        const size_t c = shape.c;

        specializations[2 + 0].u32 = (n + 3) / 4;
        specializations[2 + 1].u32 = c;

        pipeline_packing = new Pipeline(vkdev);
        pipeline_packing->set_optimal_local_size_xyz(local_size_x, 1, 1);
        if (use_int8_shader)
        {
            pipeline_packing->create(LayerShaderType::packing_int8, opt, specializations);
        }
        else
        {
            pipeline_packing->create(LayerShaderType::packing, opt, specializations);
        }
    }
    if (shape.dims == 0 || shape.elempack < out_elempack)
    {
        size_t n = 0;
        size_t c = 0;
        if (dims == 1)
        {
            n = 1;
            c = out_shape.w;
        }
        if (dims == 2)
        {
            n = out_shape.w;
            c = out_shape.h;
        }
        if (dims == 3 || dims == 4)
        {
            n = (size_t)out_shape.w * out_shape.h * out_shape.d;
            c = out_shape.c;
        }

        if (shape.dims == 0 || (shape.elempack == 1 && out_elempack == 4))
        {
            // pack1to4
            specializations[2 + 0].u32 = n;
            specializations[2 + 1].u32 = c;

            pipeline_packing_pack1to4 = new Pipeline(vkdev);
            pipeline_packing_pack1to4->set_optimal_local_size_xyz(local_size_x, 1, 1);
            if (use_int8_shader)
            {
                pipeline_packing_pack1to4->create(LayerShaderType::packing_pack1to4_int8, opt, specializations);
            }
            else
            {
                pipeline_packing_pack1to4->create(LayerShaderType::packing_pack1to4, opt, specializations);
            }
        }
    }
    if (shape.dims == 0 || shape.elempack > out_elempack)
    {
        size_t n = 0;
        size_t c = 0;
        if (dims == 1)
        {
            n = 1;
            c = shape.w;
        }
        if (dims == 2)
        {
            n = shape.w;
            c = shape.h;
        }
        if (dims == 3 || dims == 4)
        {
            n = (size_t)shape.w * shape.h * shape.d;
            c = shape.c;
        }

        if (shape.dims == 0 || (shape.elempack == 4 && out_elempack == 1))
        {
            // pack4to1
            specializations[2 + 0].u32 = n;
            specializations[2 + 1].u32 = c;

            pipeline_packing_pack4to1 = new Pipeline(vkdev);
            pipeline_packing_pack4to1->set_optimal_local_size_xyz(local_size_x, 1, 1);
            if (use_int8_shader)
            {
                pipeline_packing_pack4to1->create(LayerShaderType::packing_pack4to1_int8, opt, specializations);
            }
            else
            {
                pipeline_packing_pack4to1->create(LayerShaderType::packing_pack4to1, opt, specializations);
            }
        }
    }

    return 0;
}

int Packing_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_packing;
    pipeline_packing = 0;

    delete pipeline_packing_pack1to4;
    pipeline_packing_pack1to4 = 0;

    delete pipeline_packing_pack4to1;
    pipeline_packing_pack4to1 = 0;

    return 0;
}

int Packing_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int elempack = bottom_blob.elempack;
    const int B = bottom_blob.n;
    // NCNN_LOGE("Packing_vulkan b2b %d %d   %d %d  n=%d", elempack, out_elempack, cast_type_from, cast_type_to, B);

    if (elempack == out_elempack && cast_type_from == cast_type_to && bottom_blob.allocator == opt.blob_vkallocator)
    {
        top_blob = bottom_blob;
        return 0;
    }

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int dims = bottom_blob.dims;

    if (!use_padding)
    {
        // identity if use_padding not allowed
        if (dims == 1 && w * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if ((dims == 3 || dims == 4) && channels * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    size_t out_elemsize;
    if (cast_type_to == 0)
    {
        if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
        {
            out_elemsize = out_elempack * 2u;
        }
        else
        {
            out_elemsize = out_elempack * 4u;
        }
    }
    else if (cast_type_to == 1 || cast_type_to == 3)
    {
        out_elemsize = out_elempack * 4u;
    }
    else if (cast_type_to == 2 || cast_type_to == 5)
    {
        out_elemsize = out_elempack * 2u;
    }
    else // if (cast_type_to == 4)
    {
        out_elemsize = out_elempack * 1u;
    }

    if (dims == 1)
    {
        if (out_elempack == 1 && cast_type_from == cast_type_to && bottom_blob.allocator == opt.blob_vkallocator)
        {
            top_blob = bottom_blob;
            top_blob.w = w * elempack;
            top_blob.cstep = bottom_blob.cstep * elempack;
            top_blob.elemsize = bottom_blob.elemsize / elempack;
            top_blob.elempack = out_elempack;
#if NCNN_BATCH
            top_blob.nstep = bottom_blob.nstep * bottom_blob.elemsize / top_blob.elemsize;
#endif
            return 0;
        }

        int outw = (w * elempack + out_elempack - 1) / out_elempack;

        if (B > 1)
            top_blob.create(outw, out_elemsize, out_elempack, B, opt.blob_vkallocator);
        else
            top_blob.create(outw, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }

    if (dims == 2)
    {
        int outh = (h * elempack + out_elempack - 1) / out_elempack;

        if (B > 1)
            top_blob.create(w, outh, out_elemsize, out_elempack, B, opt.blob_vkallocator);
        else
            top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }

    if (dims == 3)
    {
        int outc = (channels * elempack + out_elempack - 1) / out_elempack;

        if (B > 1)
            top_blob.create(w, h, outc, out_elemsize, out_elempack, B, opt.blob_vkallocator);
        else
            top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }

    if (dims == 4)
    {
        int outc = (channels * elempack + out_elempack - 1) / out_elempack;

        if (B > 1)
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, B, opt.blob_vkallocator);
        else
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }

    // dispatch per batch, writing directly to batch sub-views
    for (int b = 0; b < B; b++)
    {
        const VkMat bottom_b = B > 1 ? bottom_blob.batch(b) : bottom_blob;
        const VkMat top_b = B > 1 ? top_blob.batch(b) : top_blob;

        std::vector<VkMat> buffer_bindings(4);
        buffer_bindings[0] = bottom_b;
        buffer_bindings[1] = bottom_b;
        buffer_bindings[2] = top_b;
        buffer_bindings[3] = top_b;

        if (elempack == out_elempack)
        {
            const size_t n = (size_t)w * h * d * elempack;

            std::vector<vk_constant_type> constants(4);
            constants[0].u32 = (n + 3) / 4;
            constants[1].u32 = channels;
            constants[2].u32 = bottom_b.cstep * elempack / 4;
            constants[3].u32 = top_b.cstep * out_elempack / 4;

            VkMat dispatcher;
            dispatcher.w = (n + 3) / 4;
            dispatcher.h = channels;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_packing, buffer_bindings, constants, dispatcher);
        }
        if (elempack < out_elempack)
        {
            size_t n = 0;
            size_t c = 0;
            size_t in_stride = 0;
            size_t out_stride = 0;
            if (dims == 1)
            {
                n = 1;
                c = top_b.w;
                in_stride = 1;
                out_stride = 1;
            }
            if (dims == 2)
            {
                n = w;
                c = top_b.h;
                in_stride = w;
                out_stride = w;
            }
            if (dims == 3 || dims == 4)
            {
                n = (size_t)w * h * d;
                c = top_b.c;
                in_stride = bottom_b.cstep;
                out_stride = top_b.cstep;
            }

            std::vector<vk_constant_type> constants(4);
            constants[0].u32 = n;
            constants[1].u32 = c;
            constants[2].u32 = in_stride;
            constants[3].u32 = out_stride;

            VkMat dispatcher;
            dispatcher.w = n;
            dispatcher.h = c;
            dispatcher.c = 1;

            if (elempack == 1 && out_elempack == 4)
            {
                cmd.record_pipeline(pipeline_packing_pack1to4, buffer_bindings, constants, dispatcher);
            }
        }
        if (elempack > out_elempack)
        {
            size_t n = 0;
            size_t c = 0;
            size_t in_stride = 0;
            size_t out_stride = 0;
            if (dims == 1)
            {
                n = 1;
                c = bottom_b.w;
                in_stride = 1;
                out_stride = 1;
            }
            if (dims == 2)
            {
                n = w;
                c = bottom_b.h;
                in_stride = w;
                out_stride = w;
            }
            if (dims == 3 || dims == 4)
            {
                n = (size_t)w * h * d;
                c = bottom_b.c;
                in_stride = bottom_b.cstep;
                out_stride = top_b.cstep;
            }

            std::vector<vk_constant_type> constants(4);
            constants[0].u32 = n;
            constants[1].u32 = c;
            constants[2].u32 = in_stride;
            constants[3].u32 = out_stride;

            VkMat dispatcher;
            dispatcher.w = n;
            dispatcher.h = c;
            dispatcher.c = 1;

            if (elempack == 4 && out_elempack == 1)
            {
                cmd.record_pipeline(pipeline_packing_pack4to1, buffer_bindings, constants, dispatcher);
            }
        }
    }

    return 0;
}

} // namespace ncnn
