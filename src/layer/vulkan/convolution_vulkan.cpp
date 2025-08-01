// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

Convolution_vulkan::Convolution_vulkan()
{
    support_vulkan = true;

    padding = 0;

    pipeline_convolution = 0;
    pipeline_convolution_1x1s1d1 = 0;

    pipeline_convolution_gemm = 0;

    pipeline_convolution_3x3s1d1_winograd23_transform_input = 0;
    pipeline_convolution_3x3s1d1_winograd23_gemm = 0;
    pipeline_convolution_3x3s1d1_winograd23_transform_output = 0;

    pipeline_convolution_3x3s1d1_winograd43_transform_input = 0;
    pipeline_convolution_3x3s1d1_winograd43_gemm = 0;
    pipeline_convolution_3x3s1d1_winograd43_transform_output = 0;

    reshape_1x1xw = 0;
    reshape_w = 0;

    use_cooperative_matrix = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    UNROLL_SG_M = 1;
    UNROLL_SG_N = 1;
    UNROLL_SG_K = 1;
    UNROLL_WG_M = 1;
    UNROLL_WG_N = 1;
}

int Convolution_vulkan::load_param(const ParamDict& pd)
{
    int ret = Convolution::load_param(pd);

    if (dynamic_weight)
    {
        support_vulkan = false;
    }

    return ret;
}

int Convolution_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

    // the shape after padding
    Mat shape_bordered;
    if (shape.dims != 0)
    {
        if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
        {
            shape_bordered = Mat(shape.w + pad_left + pad_right, shape.h + pad_top + pad_bottom, shape.c, (void*)0);
        }
        else if ((pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
                 || (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234))
        {
            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
            const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

            int wpad = kernel_extent_w + (shape.w - 1) / stride_w * stride_w - shape.w;
            int hpad = kernel_extent_h + (shape.h - 1) / stride_h * stride_h - shape.h;
            if (wpad > 0 || hpad > 0)
            {
                shape_bordered = Mat(shape.w + wpad, shape.h + hpad, shape.c, (void*)0);
            }
        }
        else
        {
            shape_bordered = shape;
        }
    }

    int elempack = opt.use_shader_pack8 && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;

    size_t elemsize;
    size_t out_elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed)
    {
        elemsize = elempack * 2u;
        out_elemsize = out_elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
        out_elemsize = out_elempack * 4u;
    }

    Mat shape_bordered_packed;
    if (shape_bordered.dims == 3) shape_bordered_packed = Mat(shape_bordered.w, shape_bordered.h, num_input / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_packed;
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, num_output / out_elempack, (void*)0, out_elemsize, out_elempack);

    // fc
    if (kernel_w == 1 && kernel_h == 1)
    {
        {
            reshape_1x1xw = ncnn::create_layer_vulkan(ncnn::LayerType::Reshape);
            reshape_1x1xw->vkdev = vkdev;

            reshape_1x1xw->bottom_shapes.resize(1);
            reshape_1x1xw->bottom_shapes[0] = Mat(num_input, (void*)0);
            reshape_1x1xw->top_shapes.resize(1);
            reshape_1x1xw->top_shapes[0] = Mat(1, 1, num_input, (void*)0);

            ncnn::ParamDict pd;
            pd.set(0, 1);         // w
            pd.set(1, 1);         // h
            pd.set(2, num_input); // c

            reshape_1x1xw->load_param(pd);

            reshape_1x1xw->create_pipeline(opt);
        }

        {
            reshape_w = ncnn::create_layer_vulkan(ncnn::LayerType::Reshape);
            reshape_w->vkdev = vkdev;

            reshape_w->bottom_shapes.resize(1);
            reshape_w->bottom_shapes[0] = Mat(1, 1, num_output, (void*)0);
            reshape_w->top_shapes.resize(1);
            reshape_w->top_shapes[0] = Mat(num_output, (void*)0);

            ncnn::ParamDict pd;
            pd.set(0, num_output); // w

            reshape_w->load_param(pd);

            reshape_w->create_pipeline(opt);
        }
    }

    bool is_conv1x1s1d1 = kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;
    bool is_conv3x3s1d1 = kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;

    {
        padding = ncnn::create_layer_vulkan(ncnn::LayerType::Padding);
        padding->vkdev = vkdev;

        padding->bottom_shapes.resize(1);
        padding->bottom_shapes[0] = shape;
        padding->top_shapes.resize(1);
        padding->top_shapes[0] = shape_bordered;

        ncnn::ParamDict pd;
        pd.set(0, pad_top);
        pd.set(1, pad_bottom);
        pd.set(2, pad_left);
        pd.set(3, pad_right);
        pd.set(4, 0);
        pd.set(5, pad_value);

        padding->load_param(pd);

        padding->create_pipeline(opt);
    }

    if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && is_conv3x3s1d1 && num_input >= 16 && num_output >= 16)
    {
        use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && !opt.use_shader_pack8 && (opt.use_fp16_storage || opt.use_fp16_packed);

        if (use_cooperative_matrix)
        {
            int size = 1024;
            // f43 and f23 share the same size parameter, set zero for dynamic dispatch
            // if (out_shape.dims != 0)
            // {
            //     int block_x = (out_shape.w + 3) / 4;
            //     int block_y = (out_shape.h + 3) / 4;
            //     int block_x = (out_shape.w + 1) / 2;
            //     int block_y = (out_shape.h + 1) / 2;
            //     size = block_x * block_y;
            // }

            vkdev->info.get_optimal_cooperative_matrix_mnk(size, num_output, num_input, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K);

            // assert coopmat_M != 0 && coopmat_N != 0 && coopmat_K != 0

            UNROLL_SG_M = std::min((size + coopmat_M - 1) / coopmat_M, 2);
            UNROLL_SG_N = std::min((num_output + coopmat_N - 1) / coopmat_N, 2);
            UNROLL_SG_K = std::min((num_input + coopmat_K - 1) / coopmat_K, 2);

            UNROLL_WG_M = std::min((size + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
            UNROLL_WG_N = std::min((num_output + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);
        }

        // winograd43 transform kernel
        if (opt.use_winograd43_convolution)
        {
            Mat weight_data_tm;
            weight_data_tm.create(6 * 6, num_input, num_output);

            const float sq2 = 1.41421356237f;
            const float ktm[6][3] = {
                {1.0f, 0.0f, 0.0f},
                {-2.0f / 3, -sq2 / 3, -1.0f / 3},
                {-2.0f / 3, sq2 / 3, -1.0f / 3},
                {1.0f / 6, sq2 / 6, 1.0f / 3},
                {1.0f / 6, -sq2 / 6, 1.0f / 3},
                {0.0f, 0.0f, 1.0f}
            };

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                for (int q = 0; q < num_input; q++)
                {
                    const float* kernel0 = (const float*)weight_data + p * num_input * 9 + q * 9;
                    float* kernel_tm0 = weight_data_tm.channel(p).row(q);

                    // transform kernel
                    const float* k0 = kernel0;
                    const float* k1 = kernel0 + 3;
                    const float* k2 = kernel0 + 6;

                    // h
                    float tmp[6][3];
                    for (int i = 0; i < 6; i++)
                    {
                        tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                        tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                        tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                    }

                    // U
                    for (int j = 0; j < 6; j++)
                    {
                        float* tmpp = &tmp[j][0];

                        for (int i = 0; i < 6; i++)
                        {
                            kernel_tm0[j * 6 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                        }
                    }
                }
            }

            if (use_cooperative_matrix)
            {
                // from 36-inch-outch to inch-outch-36
                Mat weight_data_tm_r2(num_input, num_output, 36);
                for (int k = 0; k < 36; k++)
                {
                    float* g00 = weight_data_tm_r2.channel(k);

                    for (int q = 0; q < num_output; q++)
                    {
                        for (int p = 0; p < num_input; p++)
                        {
                            *g00++ = weight_data_tm[(q * num_input + p) * 36 + k];
                        }
                    }
                }

                //        +-N-+
                //        K   |
                //        +SG_UN
                //        |   |
                //     ^  +---+
                //     |  |   |
                //   SG_UK+- -+
                //     |  |   |
                //   ^ v  +---+
                //   |    |   |
                //   |    +- -+
                //   |    |   |
                // WG_UN  +---+
                //   |    |   |
                //   |    +- -+
                //   |    |   |
                //   v    +---+

                //      +-N-+
                //      K   |
                //      +SG_UN
                //      |   |
                //   ^  +---+
                //   |  |   |
                // WG_UN+- -+
                //   |  |   |
                //   v  +---+

                const int blocks_n = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
                // const int blocks_k = (num_input + coopmat_K * UNROLL_SG_K - 1) / (coopmat_K * UNROLL_SG_K);
                const int kk = (num_input + coopmat_K - 1) / coopmat_K;

                weight_winograd43_data_packed.create(coopmat_N * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk, blocks_n, 36);
                for (int b = 0; b < 36; b++)
                {
                    for (int bn = 0; bn < blocks_n; bn++)
                    {
                        float* p = weight_winograd43_data_packed.channel(b).row(bn);

                        int k = 0;
                        for (; k + UNROLL_SG_K - 1 < kk; k += UNROLL_SG_K)
                        {
                            // const int ki = k * coopmat_K;

                            for (int wn = 0; wn < UNROLL_WG_N; wn++)
                            {
                                for (int zk = 0; zk < UNROLL_SG_K; zk++)
                                {
                                    for (int zn = 0; zn < UNROLL_SG_N; zn++)
                                    {
                                        for (int i = 0; i < coopmat_K; i++)
                                        {
                                            for (int j = 0; j < coopmat_N; j++)
                                            {
                                                const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                                const int gki = (k + zk) * coopmat_K + i;

                                                if (gni < num_output && gki < num_input)
                                                {
                                                    *p++ = weight_data_tm_r2.channel(b)[gni * num_input + gki];
                                                }
                                                else
                                                {
                                                    *p++ = 0.f;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        for (; k < kk; k++)
                        {
                            // const int ki = k * coopmat_K;

                            for (int wn = 0; wn < UNROLL_WG_N; wn++)
                            {
                                // for (int zk = 0; zk < UNROLL_SG_K; zk++)
                                {
                                    for (int zn = 0; zn < UNROLL_SG_N; zn++)
                                    {
                                        for (int i = 0; i < coopmat_K; i++)
                                        {
                                            for (int j = 0; j < coopmat_N; j++)
                                            {
                                                const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                                // const int gki = (k + zk) * coopmat_K + i;
                                                const int gki = k * coopmat_K + i;

                                                if (gni < num_output && gki < num_input)
                                                {
                                                    *p++ = weight_data_tm_r2.channel(b)[gni * num_input + gki];
                                                }
                                                else
                                                {
                                                    *p++ = 0.f;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                // src = 36-inch-outch
                // dst = 8a-8b-inch/8a-outch/8b-36
                weight_winograd43_data_packed.create(num_input / elempack, num_output / out_elempack, 36, (size_t)4 * elempack * out_elempack, elempack * out_elempack);

                for (int k = 0; k < 36; k++)
                {
                    float* g00 = weight_winograd43_data_packed.channel(k);

                    for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
                    {
                        for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
                        {
                            for (int i = 0; i < out_elempack; i++)
                            {
                                const Mat k0 = weight_data_tm.channel(q + i);

                                for (int j = 0; j < elempack; j++)
                                {
                                    const float* k00 = k0.row(p + j);
                                    g00[0] = k00[k];
                                    g00++;
                                }
                            }
                        }
                    }
                }
            }
        }

        // winograd43
        if (opt.use_winograd43_convolution)
        {
            int block_x = 0;
            int block_y = 0;
            Mat shape_winograd_input_transformed;
            Mat shape_winograd_gemm;
            Mat shape_winograd_input_transformed_packed;
            Mat shape_winograd_gemm_packed;

            if (out_shape.dims != 0)
            {
                block_x = (out_shape.w + 3) / 4;
                block_y = (out_shape.h + 3) / 4;

                shape_winograd_input_transformed = Mat(block_x * block_y, 1, shape.c * 36, (void*)0);
                shape_winograd_gemm = Mat(block_x * block_y, 1, out_shape.c * 36, (void*)0);
            }

            if (shape_winograd_input_transformed.dims == 3) shape_winograd_input_transformed_packed = Mat(shape_winograd_input_transformed.w, 1, shape_winograd_input_transformed.h / elempack * 36, (void*)0, elemsize, elempack);

            if (shape_winograd_gemm.dims == 3) shape_winograd_gemm_packed = Mat(shape_winograd_gemm.w, 1, shape_winograd_gemm.h / out_elempack * 36, (void*)0, out_elemsize, out_elempack);

            {
                std::vector<vk_specialization_type> specializations(1 + 6);
                specializations[0].i = num_input / elempack;
                specializations[1 + 0].i = shape_bordered_packed.w;
                specializations[1 + 1].i = shape_bordered_packed.h;
                specializations[1 + 2].i = shape_bordered_packed.cstep;
                specializations[1 + 3].i = shape_winograd_input_transformed_packed.cstep;
                specializations[1 + 4].i = block_x;
                specializations[1 + 5].i = block_y;

                int shader_type_index = -1;
                if (elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd43_transform_input;
                if (elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd43_transform_input;
                if (elempack == 8) shader_type_index = LayerShaderType::convolution_pack8_3x3s1d1_winograd43_transform_input;

                pipeline_convolution_3x3s1d1_winograd43_transform_input = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd43_transform_input->set_local_size_xyz(4, 4, 1);
                pipeline_convolution_3x3s1d1_winograd43_transform_input->create(shader_type_index, opt, specializations);
            }

            if (use_cooperative_matrix)
            {
                Mat weight_winograd43_data_packed_fp16 = Mat(weight_winograd43_data_packed.w, weight_winograd43_data_packed.h, weight_winograd43_data_packed.c, (void*)0, 2u, 1);

                std::vector<vk_specialization_type> specializations(14 + 3);
                specializations[0].u32 = 36; //batch
                specializations[1].u32 = coopmat_M;
                specializations[2].u32 = coopmat_N;
                specializations[3].u32 = coopmat_K;
                specializations[4].u32 = UNROLL_SG_M;
                specializations[5].u32 = UNROLL_SG_N;
                specializations[6].u32 = UNROLL_SG_K;
                specializations[7].u32 = UNROLL_WG_M;
                specializations[8].u32 = UNROLL_WG_N;
                specializations[9].u32 = num_input;
                specializations[10].u32 = num_output;
                specializations[11].u32 = elempack;
                specializations[12].u32 = out_elempack;
                specializations[13].u32 = weight_winograd43_data_packed_fp16.cstep;
                specializations[14 + 0].u32 = shape_winograd_input_transformed_packed.w;
                specializations[14 + 1].u32 = shape_winograd_input_transformed_packed.cstep;
                specializations[14 + 2].u32 = shape_winograd_gemm_packed.cstep;

                const int subgroup_size = vkdev->info.subgroup_size();

                pipeline_convolution_3x3s1d1_winograd43_gemm = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd43_gemm->set_subgroup_size(subgroup_size);
                pipeline_convolution_3x3s1d1_winograd43_gemm->set_local_size_xyz(subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
                pipeline_convolution_3x3s1d1_winograd43_gemm->create(LayerShaderType::convolution_winograd_gemm_cm, opt, specializations);
            }
            else
            {
                std::vector<vk_specialization_type> specializations(3 + 3);
                specializations[0].i = 36;
                specializations[1].i = num_input / elempack;
                specializations[2].i = num_output / out_elempack;
                specializations[3 + 0].i = shape_winograd_input_transformed_packed.cstep;
                specializations[3 + 1].i = shape_winograd_gemm_packed.w;
                specializations[3 + 2].i = shape_winograd_gemm_packed.cstep;

                int shader_type_index = -1;
                if (elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd_gemm;
                if (elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd_gemm;
                if (elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack1to4_3x3s1d1_winograd_gemm;
                if (elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack4to1_3x3s1d1_winograd_gemm;
                if (elempack == 8 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack8_3x3s1d1_winograd_gemm;
                if (elempack == 1 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack1to8_3x3s1d1_winograd_gemm;
                if (elempack == 8 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack8to1_3x3s1d1_winograd_gemm;
                if (elempack == 4 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack4to8_3x3s1d1_winograd_gemm;
                if (elempack == 8 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack8to4_3x3s1d1_winograd_gemm;

                pipeline_convolution_3x3s1d1_winograd43_gemm = new Pipeline(vkdev);
                if (opt.use_shader_local_memory)
                {
                    pipeline_convolution_3x3s1d1_winograd43_gemm->set_local_size_xyz(8, 8, 1);
                }
                else
                {
                    pipeline_convolution_3x3s1d1_winograd43_gemm->set_local_size_xyz(4, std::min(4, num_output / out_elempack), 4);
                }
                pipeline_convolution_3x3s1d1_winograd43_gemm->create(shader_type_index, opt, specializations);
            }

            {
                std::vector<vk_specialization_type> specializations(5 + 6);
                specializations[0].i = bias_term;
                specializations[1].i = activation_type;
                specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
                specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
                specializations[4].i = num_output / out_elempack;
                specializations[5 + 0].i = shape_winograd_gemm_packed.cstep;
                specializations[5 + 1].i = block_x;
                specializations[5 + 2].i = block_y;
                specializations[5 + 3].i = out_shape_packed.w;
                specializations[5 + 4].i = out_shape_packed.h;
                specializations[5 + 5].i = out_shape_packed.cstep;

                int shader_type_index = -1;
                if (out_elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd43_transform_output;
                if (out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd43_transform_output;
                if (out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack8_3x3s1d1_winograd43_transform_output;

                pipeline_convolution_3x3s1d1_winograd43_transform_output = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd43_transform_output->set_local_size_xyz(4, 4, 1);
                pipeline_convolution_3x3s1d1_winograd43_transform_output->create(shader_type_index, opt, specializations);
            }
        }

        // winograd23 transform kernel
        if (opt.use_winograd23_convolution)
        {
            Mat weight_data_tm;
            weight_data_tm.create(4 * 4, num_input, num_output);

            // G
            const float ktm[4][3] = {
                {1.0f, 0.0f, 0.0f},
                {1.0f / 2, 1.0f / 2, 1.0f / 2},
                {1.0f / 2, -1.0f / 2, 1.0f / 2},
                {0.0f, 0.0f, 1.0f}
            };

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                for (int q = 0; q < num_input; q++)
                {
                    const float* kernel0 = (const float*)weight_data + p * num_input * 9 + q * 9;
                    float* kernel_tm0 = weight_data_tm.channel(p).row(q);

                    // transform kernel
                    const float* k0 = kernel0;
                    const float* k1 = kernel0 + 3;
                    const float* k2 = kernel0 + 6;

                    // h
                    float tmp[4][3];
                    for (int i = 0; i < 4; i++)
                    {
                        tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                        tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                        tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                    }

                    // U
                    for (int j = 0; j < 4; j++)
                    {
                        float* tmpp = &tmp[j][0];

                        for (int i = 0; i < 4; i++)
                        {
                            kernel_tm0[j * 4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                        }
                    }
                }
            }

            if (use_cooperative_matrix)
            {
                // from 16-inch-outch to inch-outch-16
                Mat weight_data_tm_r2(num_input, num_output, 16);
                for (int k = 0; k < 16; k++)
                {
                    float* g00 = weight_data_tm_r2.channel(k);

                    for (int q = 0; q < num_output; q++)
                    {
                        for (int p = 0; p < num_input; p++)
                        {
                            *g00++ = weight_data_tm[(q * num_input + p) * 16 + k];
                        }
                    }
                }

                //        +-N-+
                //        K   |
                //        +SG_UN
                //        |   |
                //     ^  +---+
                //     |  |   |
                //   SG_UK+- -+
                //     |  |   |
                //   ^ v  +---+
                //   |    |   |
                //   |    +- -+
                //   |    |   |
                // WG_UN  +---+
                //   |    |   |
                //   |    +- -+
                //   |    |   |
                //   v    +---+

                //      +-N-+
                //      K   |
                //      +SG_UN
                //      |   |
                //   ^  +---+
                //   |  |   |
                // WG_UN+- -+
                //   |  |   |
                //   v  +---+

                const int blocks_n = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
                // const int blocks_k = (num_input + coopmat_K * UNROLL_SG_K - 1) / (coopmat_K * UNROLL_SG_K);
                const int kk = (num_input + coopmat_K - 1) / coopmat_K;

                weight_winograd23_data_packed.create(coopmat_N * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk, blocks_n, 16);
                for (int b = 0; b < 16; b++)
                {
                    for (int bn = 0; bn < blocks_n; bn++)
                    {
                        float* p = weight_winograd23_data_packed.channel(b).row(bn);

                        int k = 0;
                        for (; k + UNROLL_SG_K - 1 < kk; k += UNROLL_SG_K)
                        {
                            // const int ki = k * coopmat_K;

                            for (int wn = 0; wn < UNROLL_WG_N; wn++)
                            {
                                for (int zk = 0; zk < UNROLL_SG_K; zk++)
                                {
                                    for (int zn = 0; zn < UNROLL_SG_N; zn++)
                                    {
                                        for (int i = 0; i < coopmat_K; i++)
                                        {
                                            for (int j = 0; j < coopmat_N; j++)
                                            {
                                                const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                                const int gki = (k + zk) * coopmat_K + i;

                                                if (gni < num_output && gki < num_input)
                                                {
                                                    *p++ = weight_data_tm_r2.channel(b)[gni * num_input + gki];
                                                }
                                                else
                                                {
                                                    *p++ = 0.f;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        for (; k < kk; k++)
                        {
                            // const int ki = k * coopmat_K;

                            for (int wn = 0; wn < UNROLL_WG_N; wn++)
                            {
                                // for (int zk = 0; zk < UNROLL_SG_K; zk++)
                                {
                                    for (int zn = 0; zn < UNROLL_SG_N; zn++)
                                    {
                                        for (int i = 0; i < coopmat_K; i++)
                                        {
                                            for (int j = 0; j < coopmat_N; j++)
                                            {
                                                const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                                // const int gki = (k + zk) * coopmat_K + i;
                                                const int gki = k * coopmat_K + i;

                                                if (gni < num_output && gki < num_input)
                                                {
                                                    *p++ = weight_data_tm_r2.channel(b)[gni * num_input + gki];
                                                }
                                                else
                                                {
                                                    *p++ = 0.f;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                // src = 16-inch-outch
                // dst = 8a-8b-inch/8a-outch/8b-16
                weight_winograd23_data_packed.create(num_input / elempack, num_output / out_elempack, 16, (size_t)4 * elempack * out_elempack, elempack * out_elempack);

                for (int k = 0; k < 16; k++)
                {
                    float* g00 = weight_winograd23_data_packed.channel(k);

                    for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
                    {
                        for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
                        {
                            for (int i = 0; i < out_elempack; i++)
                            {
                                const Mat k0 = weight_data_tm.channel(q + i);

                                for (int j = 0; j < elempack; j++)
                                {
                                    const float* k00 = k0.row(p + j);
                                    g00[0] = k00[k];
                                    g00++;
                                }
                            }
                        }
                    }
                }
            }
        }

        // winograd23
        if (opt.use_winograd23_convolution)
        {
            int block_x = 0;
            int block_y = 0;
            Mat shape_winograd_input_transformed;
            Mat shape_winograd_gemm;
            Mat shape_winograd_input_transformed_packed;
            Mat shape_winograd_gemm_packed;

            if (out_shape.dims != 0)
            {
                block_x = (out_shape.w + 1) / 2;
                block_y = (out_shape.h + 1) / 2;

                shape_winograd_input_transformed = Mat(block_x * block_y, 1, shape.c * 16, (void*)0);
                shape_winograd_gemm = Mat(block_x * block_y, 1, out_shape.c * 16, (void*)0);
            }

            if (shape_winograd_input_transformed.dims == 3) shape_winograd_input_transformed_packed = Mat(shape_winograd_input_transformed.w, 1, shape_winograd_input_transformed.h / elempack * 16, (void*)0, elemsize, elempack);

            if (shape_winograd_gemm.dims == 3) shape_winograd_gemm_packed = Mat(shape_winograd_gemm.w, 1, shape_winograd_gemm.h / out_elempack * 16, (void*)0, out_elemsize, out_elempack);

            {
                std::vector<vk_specialization_type> specializations(1 + 6);
                specializations[0].i = num_input / elempack;
                specializations[1 + 0].i = shape_bordered_packed.w;
                specializations[1 + 1].i = shape_bordered_packed.h;
                specializations[1 + 2].i = shape_bordered_packed.cstep;
                specializations[1 + 3].i = shape_winograd_input_transformed_packed.cstep;
                specializations[1 + 4].i = block_x;
                specializations[1 + 5].i = block_y;

                int shader_type_index = -1;
                if (elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd23_transform_input;
                if (elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd23_transform_input;
                if (elempack == 8) shader_type_index = LayerShaderType::convolution_pack8_3x3s1d1_winograd23_transform_input;

                pipeline_convolution_3x3s1d1_winograd23_transform_input = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd23_transform_input->set_local_size_xyz(8, 8, 1);
                pipeline_convolution_3x3s1d1_winograd23_transform_input->create(shader_type_index, opt, specializations);
            }

            if (use_cooperative_matrix)
            {
                Mat weight_winograd23_data_packed_fp16 = Mat(weight_winograd23_data_packed.w, weight_winograd23_data_packed.h, weight_winograd23_data_packed.c, (void*)0, 2u, 1);

                std::vector<vk_specialization_type> specializations(14 + 3);
                specializations[0].u32 = 16; //batch
                specializations[1].u32 = coopmat_M;
                specializations[2].u32 = coopmat_N;
                specializations[3].u32 = coopmat_K;
                specializations[4].u32 = UNROLL_SG_M;
                specializations[5].u32 = UNROLL_SG_N;
                specializations[6].u32 = UNROLL_SG_K;
                specializations[7].u32 = UNROLL_WG_M;
                specializations[8].u32 = UNROLL_WG_N;
                specializations[9].u32 = num_input;
                specializations[10].u32 = num_output;
                specializations[11].u32 = elempack;
                specializations[12].u32 = out_elempack;
                specializations[13].u32 = weight_winograd23_data_packed_fp16.cstep;
                specializations[14 + 0].u32 = shape_winograd_input_transformed_packed.w;
                specializations[14 + 1].u32 = shape_winograd_input_transformed_packed.cstep;
                specializations[14 + 2].u32 = shape_winograd_gemm_packed.cstep;

                const int subgroup_size = vkdev->info.subgroup_size();

                pipeline_convolution_3x3s1d1_winograd23_gemm = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd23_gemm->set_subgroup_size(subgroup_size);
                pipeline_convolution_3x3s1d1_winograd23_gemm->set_local_size_xyz(subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
                pipeline_convolution_3x3s1d1_winograd23_gemm->create(LayerShaderType::convolution_winograd_gemm_cm, opt, specializations);
            }
            else
            {
                std::vector<vk_specialization_type> specializations(3 + 3);
                specializations[0].i = 16;
                specializations[1].i = num_input / elempack;
                specializations[2].i = num_output / out_elempack;
                specializations[3 + 0].i = shape_winograd_input_transformed_packed.cstep;
                specializations[3 + 1].i = shape_winograd_gemm_packed.w;
                specializations[3 + 2].i = shape_winograd_gemm_packed.cstep;

                int shader_type_index = -1;
                if (elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd_gemm;
                if (elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd_gemm;
                if (elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack1to4_3x3s1d1_winograd_gemm;
                if (elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack4to1_3x3s1d1_winograd_gemm;
                if (elempack == 8 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack8_3x3s1d1_winograd_gemm;
                if (elempack == 1 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack1to8_3x3s1d1_winograd_gemm;
                if (elempack == 8 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack8to1_3x3s1d1_winograd_gemm;
                if (elempack == 4 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack4to8_3x3s1d1_winograd_gemm;
                if (elempack == 8 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack8to4_3x3s1d1_winograd_gemm;

                pipeline_convolution_3x3s1d1_winograd23_gemm = new Pipeline(vkdev);
                if (opt.use_shader_local_memory)
                {
                    pipeline_convolution_3x3s1d1_winograd23_gemm->set_local_size_xyz(8, 8, 1);
                }
                else
                {
                    pipeline_convolution_3x3s1d1_winograd23_gemm->set_local_size_xyz(4, std::min(4, num_output / out_elempack), 4);
                }
                pipeline_convolution_3x3s1d1_winograd23_gemm->create(shader_type_index, opt, specializations);
            }

            {
                std::vector<vk_specialization_type> specializations(5 + 6);
                specializations[0].i = bias_term;
                specializations[1].i = activation_type;
                specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
                specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
                specializations[4].i = num_output / out_elempack;
                specializations[5 + 0].i = shape_winograd_gemm_packed.cstep;
                specializations[5 + 1].i = block_x;
                specializations[5 + 2].i = block_y;
                specializations[5 + 3].i = out_shape_packed.w;
                specializations[5 + 4].i = out_shape_packed.h;
                specializations[5 + 5].i = out_shape_packed.cstep;

                int shader_type_index = -1;
                if (out_elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd23_transform_output;
                if (out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd23_transform_output;
                if (out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack8_3x3s1d1_winograd23_transform_output;

                pipeline_convolution_3x3s1d1_winograd23_transform_output = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd23_transform_output->set_local_size_xyz(8, 8, 1);
                pipeline_convolution_3x3s1d1_winograd23_transform_output->create(shader_type_index, opt, specializations);
            }
        }
    }
    else if (opt.use_sgemm_convolution && !is_conv1x1s1d1 && num_input * maxk >= 8 && num_output >= 8)
    {
        use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && !opt.use_shader_pack8 && (opt.use_fp16_storage || opt.use_fp16_packed);

        if (use_cooperative_matrix)
        {
            int size = 1024;
            if (out_shape_packed.dims == 3)
                size = out_shape_packed.w * out_shape_packed.h;

            vkdev->info.get_optimal_cooperative_matrix_mnk(size, num_output, num_input * maxk, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K);

            // assert coopmat_M != 0 && coopmat_N != 0 && coopmat_K != 0

            UNROLL_SG_M = std::min((size + coopmat_M - 1) / coopmat_M, 2);
            UNROLL_SG_N = std::min((num_output + coopmat_N - 1) / coopmat_N, 2);
            UNROLL_SG_K = std::min((num_input * maxk + coopmat_K - 1) / coopmat_K, 2);

            UNROLL_WG_M = std::min((size + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
            UNROLL_WG_N = std::min((num_output + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

            Mat weight_data_r2;

            if (elempack == 4)
            {
                // from maxk-inch-outch to 4-maxk-inch/4-outch
                weight_data_r2.create(4 * maxk * (num_input / 4) * num_output);
                for (int i = 0; i < num_output; i++)
                {
                    for (int j = 0; j < num_input / 4; j++)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            weight_data_r2[((i * (num_input / 4) + j) * maxk + k) * 4] = weight_data[(i * num_input + j * 4) * maxk + k];
                            weight_data_r2[((i * (num_input / 4) + j) * maxk + k) * 4 + 1] = weight_data[(i * num_input + j * 4 + 1) * maxk + k];
                            weight_data_r2[((i * (num_input / 4) + j) * maxk + k) * 4 + 2] = weight_data[(i * num_input + j * 4 + 2) * maxk + k];
                            weight_data_r2[((i * (num_input / 4) + j) * maxk + k) * 4 + 3] = weight_data[(i * num_input + j * 4 + 3) * maxk + k];
                        }
                    }
                }
            }
            else
            {
                weight_data_r2 = weight_data;
            }

            const int blocks_n = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
            // const int blocks_k = (num_input / 4 * maxk + coopmat_K * UNROLL_SG_K - 1) / (coopmat_K * UNROLL_SG_K);
            const int kk = (num_input * maxk + coopmat_K - 1) / coopmat_K;

            weight_data_packed.create(coopmat_N * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk, blocks_n);
            for (int bn = 0; bn < blocks_n; bn++)
            {
                //        +-N-+
                //        K   |
                //        +SG_UN
                //        |   |
                //     ^  +---+
                //     |  |   |
                //   SG_UK+- -+
                //     |  |   |
                //   ^ v  +---+
                //   |    |   |
                //   |    +- -+
                //   |    |   |
                // WG_UN  +---+
                //   |    |   |
                //   |    +- -+
                //   |    |   |
                //   v    +---+

                //      +-N-+
                //      K   |
                //      +SG_UN
                //      |   |
                //   ^  +---+
                //   |  |   |
                // WG_UN+- -+
                //   |  |   |
                //   v  +---+

                float* p = weight_data_packed.row(bn);

                int k = 0;
                for (; k + UNROLL_SG_K - 1 < kk; k += UNROLL_SG_K)
                {
                    // const int ki = k * coopmat_K;

                    for (int wn = 0; wn < UNROLL_WG_N; wn++)
                    {
                        for (int zk = 0; zk < UNROLL_SG_K; zk++)
                        {
                            for (int zn = 0; zn < UNROLL_SG_N; zn++)
                            {
                                for (int i = 0; i < coopmat_K; i++)
                                {
                                    for (int j = 0; j < coopmat_N; j++)
                                    {
                                        const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                        const int gki = (k + zk) * coopmat_K + i;

                                        if (gni < num_output && gki < num_input * maxk)
                                        {
                                            *p++ = weight_data_r2[gni * num_input * maxk + gki];
                                        }
                                        else
                                        {
                                            *p++ = 0.f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for (; k < kk; k++)
                {
                    // const int ki = k * coopmat_K;

                    for (int wn = 0; wn < UNROLL_WG_N; wn++)
                    {
                        // for (int zk = 0; zk < UNROLL_SG_K; zk++)
                        {
                            for (int zn = 0; zn < UNROLL_SG_N; zn++)
                            {
                                for (int i = 0; i < coopmat_K; i++)
                                {
                                    for (int j = 0; j < coopmat_N; j++)
                                    {
                                        const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                        // const int gki = (k + zk) * coopmat_K + i;
                                        const int gki = k * coopmat_K + i;

                                        if (gni < num_output && gki < num_input * maxk)
                                        {
                                            *p++ = weight_data_r2[gni * num_input * maxk + gki];
                                        }
                                        else
                                        {
                                            *p++ = 0.f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            weight_data_packed.create(maxk * num_input / elempack, num_output / out_elempack, (size_t)4 * elempack * out_elempack, elempack * out_elempack);

            for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
            {
                float* g00 = weight_data_packed.row(q / out_elempack);

                for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
                {
                    for (int k = 0; k < maxk; k++)
                    {
                        for (int i = 0; i < out_elempack; i++)
                        {
                            const Mat k0 = weight_data_r2.channel(q + i);

                            for (int j = 0; j < elempack; j++)
                            {
                                const float* k00 = k0.row(p + j);
                                g00[0] = k00[k];
                                g00++;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (is_conv1x1s1d1)
    {
        use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && !opt.use_shader_pack8 && (opt.use_fp16_storage || opt.use_fp16_packed) && num_input >= 8 && num_output >= 8;

        if (use_cooperative_matrix)
        {
            int size = 1024;
            if (shape_bordered_packed.dims == 3)
                size = shape_bordered_packed.w * shape_bordered_packed.h;

            vkdev->info.get_optimal_cooperative_matrix_mnk(size, num_output, num_input, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K);

            // assert coopmat_M != 0 && coopmat_N != 0 && coopmat_K != 0

            UNROLL_SG_M = std::min((size + coopmat_M - 1) / coopmat_M, 2);
            UNROLL_SG_N = std::min((num_output + coopmat_N - 1) / coopmat_N, 2);
            UNROLL_SG_K = std::min((num_input + coopmat_K - 1) / coopmat_K, 2);

            UNROLL_WG_M = std::min((size + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
            UNROLL_WG_N = std::min((num_output + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

            //        +-N-+
            //        K   |
            //        +SG_UN
            //        |   |
            //     ^  +---+
            //     |  |   |
            //   SG_UK+- -+
            //     |  |   |
            //   ^ v  +---+
            //   |    |   |
            //   |    +- -+
            //   |    |   |
            // WG_UN  +---+
            //   |    |   |
            //   |    +- -+
            //   |    |   |
            //   v    +---+

            //      +-N-+
            //      K   |
            //      +SG_UN
            //      |   |
            //   ^  +---+
            //   |  |   |
            // WG_UN+- -+
            //   |  |   |
            //   v  +---+

            const int blocks_n = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
            // const int blocks_k = (num_input + coopmat_K * UNROLL_SG_K - 1) / (coopmat_K * UNROLL_SG_K);
            const int kk = (num_input + coopmat_K - 1) / coopmat_K;

            weight_data_packed.create(coopmat_N * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk, blocks_n);
            for (int bn = 0; bn < blocks_n; bn++)
            {
                float* p = weight_data_packed.row(bn);

                int k = 0;
                for (; k + UNROLL_SG_K - 1 < kk; k += UNROLL_SG_K)
                {
                    // const int ki = k * coopmat_K;

                    for (int wn = 0; wn < UNROLL_WG_N; wn++)
                    {
                        for (int zk = 0; zk < UNROLL_SG_K; zk++)
                        {
                            for (int zn = 0; zn < UNROLL_SG_N; zn++)
                            {
                                for (int i = 0; i < coopmat_K; i++)
                                {
                                    for (int j = 0; j < coopmat_N; j++)
                                    {
                                        const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                        const int gki = (k + zk) * coopmat_K + i;

                                        if (gni < num_output && gki < num_input)
                                        {
                                            *p++ = weight_data[gni * num_input + gki];
                                        }
                                        else
                                        {
                                            *p++ = 0.f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for (; k < kk; k++)
                {
                    // const int ki = k * coopmat_K;

                    for (int wn = 0; wn < UNROLL_WG_N; wn++)
                    {
                        // for (int zk = 0; zk < UNROLL_SG_K; zk++)
                        {
                            for (int zn = 0; zn < UNROLL_SG_N; zn++)
                            {
                                for (int i = 0; i < coopmat_K; i++)
                                {
                                    for (int j = 0; j < coopmat_N; j++)
                                    {
                                        const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                        // const int gki = (k + zk) * coopmat_K + i;
                                        const int gki = k * coopmat_K + i;

                                        if (gni < num_output && gki < num_input)
                                        {
                                            *p++ = weight_data[gni * num_input + gki];
                                        }
                                        else
                                        {
                                            *p++ = 0.f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            weight_data_packed.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)4 * elempack * out_elempack, elempack * out_elempack);

            for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
            {
                float* g00 = weight_data_packed.channel(q / out_elempack);

                for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
                {
                    for (int k = 0; k < maxk; k++)
                    {
                        for (int i = 0; i < out_elempack; i++)
                        {
                            const Mat k0 = weight_data_r2.channel(q + i);

                            for (int j = 0; j < elempack; j++)
                            {
                                const float* k00 = k0.row(p + j);
                                g00[0] = k00[k];
                                g00++;
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        weight_data_packed.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)4 * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_packed.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < out_elempack; i++)
                    {
                        const Mat k0 = weight_data_r2.channel(q + i);

                        for (int j = 0; j < elempack; j++)
                        {
                            const float* k00 = k0.row(p + j);
                            g00[0] = k00[k];
                            g00++;
                        }
                    }
                }
            }
        }
    }

    if (bias_term)
    {
        convert_packing(bias_data, bias_data_packed, out_elempack, opt);
    }

    if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && is_conv3x3s1d1 && num_input >= 16 && num_output >= 16)
    {
        // pass
    }
    else if (opt.use_sgemm_convolution && !is_conv1x1s1d1 && num_input * maxk >= 8 && num_output >= 8)
    {
        if (use_cooperative_matrix)
        {
            std::vector<vk_specialization_type> specializations(22 + 6);
            specializations[0].u32 = kernel_w;
            specializations[1].u32 = kernel_h;
            specializations[2].u32 = dilation_w;
            specializations[3].u32 = dilation_h;
            specializations[4].u32 = stride_w;
            specializations[5].u32 = stride_h;
            specializations[6].i = bias_term;
            specializations[7].i = activation_type;
            specializations[8].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[9].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[10].u32 = coopmat_M;
            specializations[11].u32 = coopmat_N;
            specializations[12].u32 = coopmat_K;
            specializations[13].u32 = UNROLL_SG_M;
            specializations[14].u32 = UNROLL_SG_N;
            specializations[15].u32 = UNROLL_SG_K;
            specializations[16].u32 = UNROLL_WG_M;
            specializations[17].u32 = UNROLL_WG_N;
            specializations[18].u32 = num_input;
            specializations[19].u32 = num_output;
            specializations[20].u32 = elempack;
            specializations[21].u32 = out_elempack;
            specializations[22 + 0].u32 = shape_bordered_packed.w;
            specializations[22 + 1].u32 = shape_bordered_packed.h;
            specializations[22 + 2].u32 = shape_bordered_packed.cstep;
            specializations[22 + 3].u32 = out_shape_packed.w;
            specializations[22 + 4].u32 = out_shape_packed.h;
            specializations[22 + 5].u32 = out_shape_packed.cstep;

            const int subgroup_size = vkdev->info.subgroup_size();

            pipeline_convolution_gemm = new Pipeline(vkdev);
            pipeline_convolution_gemm->set_subgroup_size(subgroup_size);
            pipeline_convolution_gemm->set_local_size_xyz(subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_convolution_gemm->create(LayerShaderType::convolution_gemm_cm, opt, specializations);
        }
        else
        {
            std::vector<vk_specialization_type> specializations(10 + 8);
            specializations[0].i = kernel_w;
            specializations[1].i = kernel_h;
            specializations[2].i = dilation_w;
            specializations[3].i = dilation_h;
            specializations[4].i = stride_w;
            specializations[5].i = stride_h;
            specializations[6].i = bias_term;
            specializations[7].i = activation_type;
            specializations[8].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[9].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[10 + 0].i = shape_bordered_packed.w;
            specializations[10 + 1].i = shape_bordered_packed.h;
            specializations[10 + 2].i = shape_bordered_packed.c;
            specializations[10 + 3].i = shape_bordered_packed.cstep;
            specializations[10 + 4].i = out_shape_packed.w;
            specializations[10 + 5].i = out_shape_packed.h;
            specializations[10 + 6].i = out_shape_packed.c;
            specializations[10 + 7].i = out_shape_packed.cstep;

            Mat local_size_xyz(16, std::min(4, num_output / out_elempack), 1, (void*)0);
            if (out_shape_packed.dims != 0)
            {
                local_size_xyz.w = std::min(16, out_shape_packed.w * out_shape_packed.h);
                local_size_xyz.h = std::min(4, out_shape_packed.c);
            }

            int shader_type_index = -1;
            if (elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_gemm;
            if (elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_gemm;
            if (elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack1to4_gemm;
            if (elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack4to1_gemm;
            if (elempack == 8 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack8_gemm;
            if (elempack == 1 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack1to8_gemm;
            if (elempack == 8 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack8to1_gemm;
            if (elempack == 4 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack4to8_gemm;
            if (elempack == 8 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack8to4_gemm;

            pipeline_convolution_gemm = new Pipeline(vkdev);
            if (opt.use_shader_local_memory)
            {
                pipeline_convolution_gemm->set_local_size_xyz(8, 8, 1);
            }
            else
            {
                pipeline_convolution_gemm->set_optimal_local_size_xyz(local_size_xyz);
            }
            pipeline_convolution_gemm->create(shader_type_index, opt, specializations);
        }
    }
    else if (is_conv1x1s1d1)
    {
        if (use_cooperative_matrix)
        {
            std::vector<vk_specialization_type> specializations(16 + 3);
            specializations[0].i = bias_term;
            specializations[1].i = activation_type;
            specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[4].u32 = coopmat_M;
            specializations[5].u32 = coopmat_N;
            specializations[6].u32 = coopmat_K;
            specializations[7].u32 = UNROLL_SG_M;
            specializations[8].u32 = UNROLL_SG_N;
            specializations[9].u32 = UNROLL_SG_K;
            specializations[10].u32 = UNROLL_WG_M;
            specializations[11].u32 = UNROLL_WG_N;
            specializations[12].u32 = num_input;
            specializations[13].u32 = num_output;
            specializations[14].u32 = elempack;
            specializations[15].u32 = out_elempack;
            specializations[16 + 0].u32 = shape_bordered_packed.w * shape_bordered_packed.h;
            specializations[16 + 1].u32 = shape_bordered_packed.cstep;
            specializations[16 + 2].u32 = out_shape_packed.cstep;

            const int subgroup_size = vkdev->info.subgroup_size();

            pipeline_convolution_1x1s1d1 = new Pipeline(vkdev);
            pipeline_convolution_1x1s1d1->set_subgroup_size(subgroup_size);
            pipeline_convolution_1x1s1d1->set_local_size_xyz(subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_convolution_1x1s1d1->create(LayerShaderType::convolution_1x1s1d1_cm, opt, specializations);
        }
        else
        {
            std::vector<vk_specialization_type> specializations(4 + 8);
            specializations[0].i = bias_term;
            specializations[1].i = activation_type;
            specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[4 + 0].i = shape_bordered_packed.w;
            specializations[4 + 1].i = shape_bordered_packed.h;
            specializations[4 + 2].i = shape_bordered_packed.c;
            specializations[4 + 3].i = shape_bordered_packed.cstep;
            specializations[4 + 4].i = out_shape_packed.w;
            specializations[4 + 5].i = out_shape_packed.h;
            specializations[4 + 6].i = out_shape_packed.c;
            specializations[4 + 7].i = out_shape_packed.cstep;

            int shader_type_index = -1;
            if (elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_1x1s1d1;
            if (elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_1x1s1d1;
            if (elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack1to4_1x1s1d1;
            if (elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack4to1_1x1s1d1;
            if (elempack == 8 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack8_1x1s1d1;
            if (elempack == 1 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack1to8_1x1s1d1;
            if (elempack == 8 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack8to1_1x1s1d1;
            if (elempack == 4 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack4to8_1x1s1d1;
            if (elempack == 8 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack8to4_1x1s1d1;

            pipeline_convolution_1x1s1d1 = new Pipeline(vkdev);
            if (opt.use_shader_local_memory)
            {
                pipeline_convolution_1x1s1d1->set_local_size_xyz(8, 8, 1);
            }
            else
            {
                pipeline_convolution_1x1s1d1->set_local_size_xyz(8, std::min(8, num_output / out_elempack), 1);
            }
            pipeline_convolution_1x1s1d1->create(shader_type_index, opt, specializations);
        }
    }
    else
    {
        std::vector<vk_specialization_type> specializations(10 + 10);
        specializations[0].i = kernel_w;
        specializations[1].i = kernel_h;
        specializations[2].i = dilation_w;
        specializations[3].i = dilation_h;
        specializations[4].i = stride_w;
        specializations[5].i = stride_h;
        specializations[6].i = bias_term;
        specializations[7].i = activation_type;
        specializations[8].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specializations[9].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specializations[10 + 0].i = shape_bordered_packed.dims;
        specializations[10 + 1].i = shape_bordered_packed.w;
        specializations[10 + 2].i = shape_bordered_packed.h;
        specializations[10 + 3].i = shape_bordered_packed.c;
        specializations[10 + 4].i = shape_bordered_packed.cstep;
        specializations[10 + 5].i = out_shape_packed.dims;
        specializations[10 + 6].i = out_shape_packed.w;
        specializations[10 + 7].i = out_shape_packed.h;
        specializations[10 + 8].i = out_shape_packed.c;
        specializations[10 + 9].i = out_shape_packed.cstep;

        Mat local_size_xyz(8, 8, std::min(4, (num_output / out_elempack + 1) / 2), (void*)0);
        if (out_shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(8, out_shape_packed.w);
            local_size_xyz.h = std::min(8, out_shape_packed.h);
            local_size_xyz.c = std::min(4, (out_shape_packed.c + 1) / 2);
        }

        int shader_type_index = -1;
        if (elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::convolution;
        if (elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack4;
        if (elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack1to4;
        if (elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack4to1;
        if (elempack == 8 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack8;
        if (elempack == 1 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack1to8;
        if (elempack == 8 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack8to1;
        if (elempack == 4 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack4to8;
        if (elempack == 8 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack8to4;

        pipeline_convolution = new Pipeline(vkdev);
        pipeline_convolution->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolution->create(shader_type_index, opt, specializations);
    }

    if (opt.lightmode)
    {
        weight_data.release();
        bias_data.release();
    }

    return 0;
}

int Convolution_vulkan::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    delete pipeline_convolution;
    pipeline_convolution = 0;

    delete pipeline_convolution_1x1s1d1;
    pipeline_convolution_1x1s1d1 = 0;

    delete pipeline_convolution_gemm;
    pipeline_convolution_gemm = 0;

    delete pipeline_convolution_3x3s1d1_winograd23_transform_input;
    delete pipeline_convolution_3x3s1d1_winograd23_gemm;
    delete pipeline_convolution_3x3s1d1_winograd23_transform_output;
    pipeline_convolution_3x3s1d1_winograd23_transform_input = 0;
    pipeline_convolution_3x3s1d1_winograd23_gemm = 0;
    pipeline_convolution_3x3s1d1_winograd23_transform_output = 0;

    delete pipeline_convolution_3x3s1d1_winograd43_transform_input;
    delete pipeline_convolution_3x3s1d1_winograd43_gemm;
    delete pipeline_convolution_3x3s1d1_winograd43_transform_output;
    pipeline_convolution_3x3s1d1_winograd43_transform_input = 0;
    pipeline_convolution_3x3s1d1_winograd43_gemm = 0;
    pipeline_convolution_3x3s1d1_winograd43_transform_output = 0;

    // fc
    if (reshape_1x1xw)
    {
        reshape_1x1xw->destroy_pipeline(opt);
        delete reshape_1x1xw;
        reshape_1x1xw = 0;
    }

    if (reshape_w)
    {
        reshape_w->destroy_pipeline(opt);
        delete reshape_w;
        reshape_w = 0;
    }

    use_cooperative_matrix = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    UNROLL_SG_M = 1;
    UNROLL_SG_N = 1;
    UNROLL_SG_K = 1;
    UNROLL_WG_M = 1;
    UNROLL_WG_N = 1;

    return 0;
}

int Convolution_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (padding)
    {
        padding->upload_model(cmd, opt);
    }

    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

    bool is_conv3x3s1d1 = kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;

    if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && is_conv3x3s1d1 && num_input >= 16 && num_output >= 16)
    {
        // winograd43
        if (opt.use_winograd43_convolution)
        {
            cmd.record_upload(weight_winograd43_data_packed, weight_data_gpu_tm_winograd43, opt);

            weight_winograd43_data_packed.release();
        }

        // winograd23
        if (opt.use_winograd23_convolution)
        {
            cmd.record_upload(weight_winograd23_data_packed, weight_data_gpu_tm_winograd23, opt);

            weight_winograd23_data_packed.release();
        }
    }
    else
    {
        cmd.record_upload(weight_data_packed, weight_data_gpu, opt);

        weight_data_packed.release();
    }

    if (bias_term)
    {
        cmd.record_upload(bias_data_packed, bias_data_gpu, opt);

        bias_data_packed.release();
    }

    return 0;
}

int Convolution_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    // flattened blob, implement as InnerProduct
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        int num_input = weight_data_size / num_output;
        if (bottom_blob.w * bottom_blob.elempack == num_input)
        {
            VkMat bottom_blob_1x1xw;
            {
                Option opt_reshape = opt;
                opt_reshape.blob_vkallocator = opt.workspace_vkallocator;
                reshape_1x1xw->forward(bottom_blob, bottom_blob_1x1xw, cmd, opt_reshape);
            }

            if (bottom_blob_1x1xw.empty())
                return -100;

            VkMat top_blob_1x1xw;
            int ret = forward(bottom_blob_1x1xw, top_blob_1x1xw, cmd, opt);
            if (ret != 0)
                return ret;

            return reshape_w->forward(top_blob_1x1xw, top_blob, cmd, opt);
        }
    }

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    VkMat bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad / 2;
            padding_params[1] = hpad - hpad / 2;
            padding_params[2] = wpad / 2;
            padding_params[3] = wpad - wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;
            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad - hpad / 2;
            padding_params[1] = hpad / 2;
            padding_params[2] = wpad - wpad / 2;
            padding_params[3] = wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;

            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    const int maxk = kernel_w * kernel_h;
    const int num_input = channels * elempack;

    bool is_conv1x1s1d1 = kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;
    bool is_conv3x3s1d1 = kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;

    if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && is_conv3x3s1d1 && num_input >= 16 && num_output >= 16)
    {
        bool pre_winograd43 = opt.use_winograd43_convolution;
        if (opt.use_winograd23_convolution)
        {
            if (vkdev->info.type() == 0 && ((w <= 18 && h <= 18) || ((w >= 23 && w <= 24) && (h >= 23 && h <= 24))))
                pre_winograd43 = false;
            if (vkdev->info.type() != 0 && (w <= 12 && h <= 12))
                pre_winograd43 = false;

            if (use_cooperative_matrix && (w <= 18 && h <= 18))
                pre_winograd43 = false;
        }

        if (pre_winograd43)
        {
            // winograd43
            int block_x = (outw + 3) / 4;
            int block_y = (outh + 3) / 4;

            // transform input
            VkMat bottom_tm_blob;
            {
                bottom_tm_blob.create(block_x * block_y, 1, channels * 36, elemsize, elempack, opt.workspace_vkallocator);
                if (bottom_tm_blob.empty())
                    return -100;

                std::vector<VkMat> bindings(2);
                bindings[0] = bottom_blob_bordered;
                bindings[1] = bottom_tm_blob;

                std::vector<vk_constant_type> constants(6);
                constants[0].i = bottom_blob_bordered.w;
                constants[1].i = bottom_blob_bordered.h;
                constants[2].i = bottom_blob_bordered.cstep;
                constants[3].i = bottom_tm_blob.cstep;
                constants[4].i = block_x;
                constants[5].i = block_y;

                VkMat dispatcher;
                dispatcher.w = block_x;
                dispatcher.h = block_y;
                dispatcher.c = channels;

                cmd.record_pipeline(pipeline_convolution_3x3s1d1_winograd43_transform_input, bindings, constants, dispatcher);
            }

            // gemm
            VkMat top_tm_blob;
            {
                top_tm_blob.create(block_x * block_y, 1, num_output / out_elempack * 36, out_elemsize, out_elempack, opt.workspace_vkallocator);
                if (top_tm_blob.empty())
                    return -100;

                if (use_cooperative_matrix)
                {
                    std::vector<VkMat> bindings(3);
                    bindings[0] = bottom_tm_blob;
                    bindings[1] = top_tm_blob;
                    bindings[2] = weight_data_gpu_tm_winograd43;

                    std::vector<vk_constant_type> constants(3);
                    constants[0].i = bottom_tm_blob.w;
                    constants[1].i = bottom_tm_blob.cstep;
                    constants[2].i = top_tm_blob.cstep;

                    const int blocks_x = (bottom_tm_blob.w + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
                    const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

                    const int subgroup_size = vkdev->info.subgroup_size();

                    VkMat dispatcher;
                    dispatcher.w = (blocks_x * blocks_y) * (subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
                    dispatcher.h = 1;
                    dispatcher.c = 36;

                    cmd.record_pipeline(pipeline_convolution_3x3s1d1_winograd43_gemm, bindings, constants, dispatcher);
                }
                else
                {
                    std::vector<VkMat> bindings(3);
                    bindings[0] = bottom_tm_blob;
                    bindings[1] = top_tm_blob;
                    bindings[2] = weight_data_gpu_tm_winograd43;

                    std::vector<vk_constant_type> constants(3);
                    constants[0].i = bottom_tm_blob.cstep;
                    constants[1].i = top_tm_blob.w;
                    constants[2].i = top_tm_blob.cstep;

                    VkMat dispatcher;
                    dispatcher.w = (top_tm_blob.w + 3) / 4;
                    dispatcher.h = num_output / out_elempack;
                    dispatcher.c = 36;

                    cmd.record_pipeline(pipeline_convolution_3x3s1d1_winograd43_gemm, bindings, constants, dispatcher);
                }
            }

            // transform output
            {
                top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
                if (top_blob.empty())
                    return -100;

                std::vector<VkMat> bindings(3);
                bindings[0] = top_tm_blob;
                bindings[1] = top_blob;
                bindings[2] = bias_data_gpu;

                std::vector<vk_constant_type> constants(6);
                constants[0].i = top_tm_blob.cstep;
                constants[1].i = block_x;
                constants[2].i = block_y;
                constants[3].i = top_blob.w;
                constants[4].i = top_blob.h;
                constants[5].i = top_blob.cstep;

                VkMat dispatcher;
                dispatcher.w = block_x;
                dispatcher.h = block_y;
                dispatcher.c = top_blob.c;

                cmd.record_pipeline(pipeline_convolution_3x3s1d1_winograd43_transform_output, bindings, constants, dispatcher);
            }
        }
        else
        {
            // winograd23
            int block_x = (outw + 1) / 2;
            int block_y = (outh + 1) / 2;

            // transform input
            VkMat bottom_tm_blob;
            {
                bottom_tm_blob.create(block_x * block_y, 1, channels * 16, elemsize, elempack, opt.workspace_vkallocator);
                if (bottom_tm_blob.empty())
                    return -100;

                std::vector<VkMat> bindings(2);
                bindings[0] = bottom_blob_bordered;
                bindings[1] = bottom_tm_blob;

                std::vector<vk_constant_type> constants(6);
                constants[0].i = bottom_blob_bordered.w;
                constants[1].i = bottom_blob_bordered.h;
                constants[2].i = bottom_blob_bordered.cstep;
                constants[3].i = bottom_tm_blob.cstep;
                constants[4].i = block_x;
                constants[5].i = block_y;

                VkMat dispatcher;
                dispatcher.w = block_x;
                dispatcher.h = block_y;
                dispatcher.c = channels;

                cmd.record_pipeline(pipeline_convolution_3x3s1d1_winograd23_transform_input, bindings, constants, dispatcher);
            }

            // gemm
            VkMat top_tm_blob;
            {
                top_tm_blob.create(block_x * block_y, 1, num_output / out_elempack * 16, out_elemsize, out_elempack, opt.workspace_vkallocator);
                if (top_tm_blob.empty())
                    return -100;

                if (use_cooperative_matrix)
                {
                    std::vector<VkMat> bindings(3);
                    bindings[0] = bottom_tm_blob;
                    bindings[1] = top_tm_blob;
                    bindings[2] = weight_data_gpu_tm_winograd23;

                    std::vector<vk_constant_type> constants(3);
                    constants[0].i = bottom_tm_blob.w;
                    constants[1].i = bottom_tm_blob.cstep;
                    constants[2].i = top_tm_blob.cstep;

                    const int blocks_x = (bottom_tm_blob.w + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
                    const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

                    const int subgroup_size = vkdev->info.subgroup_size();

                    VkMat dispatcher;
                    dispatcher.w = (blocks_x * blocks_y) * (subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
                    dispatcher.h = 1;
                    dispatcher.c = 16;

                    cmd.record_pipeline(pipeline_convolution_3x3s1d1_winograd23_gemm, bindings, constants, dispatcher);
                }
                else
                {
                    std::vector<VkMat> bindings(3);
                    bindings[0] = bottom_tm_blob;
                    bindings[1] = top_tm_blob;
                    bindings[2] = weight_data_gpu_tm_winograd23;

                    std::vector<vk_constant_type> constants(3);
                    constants[0].i = bottom_tm_blob.cstep;
                    constants[1].i = top_tm_blob.w;
                    constants[2].i = top_tm_blob.cstep;

                    VkMat dispatcher;
                    dispatcher.w = (top_tm_blob.w + 3) / 4;
                    dispatcher.h = num_output / out_elempack;
                    dispatcher.c = 16;

                    cmd.record_pipeline(pipeline_convolution_3x3s1d1_winograd23_gemm, bindings, constants, dispatcher);
                }
            }

            // transform output
            {
                top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
                if (top_blob.empty())
                    return -100;

                std::vector<VkMat> bindings(3);
                bindings[0] = top_tm_blob;
                bindings[1] = top_blob;
                bindings[2] = bias_data_gpu;

                std::vector<vk_constant_type> constants(6);
                constants[0].i = top_tm_blob.cstep;
                constants[1].i = block_x;
                constants[2].i = block_y;
                constants[3].i = top_blob.w;
                constants[4].i = top_blob.h;
                constants[5].i = top_blob.cstep;

                VkMat dispatcher;
                dispatcher.w = block_x;
                dispatcher.h = block_y;
                dispatcher.c = top_blob.c;

                cmd.record_pipeline(pipeline_convolution_3x3s1d1_winograd23_transform_output, bindings, constants, dispatcher);
            }
        }

        return 0;
    }
    if (opt.use_sgemm_convolution && !is_conv1x1s1d1 && num_input * maxk >= 8 && num_output >= 8)
    {
        // gemm
        top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        if (use_cooperative_matrix)
        {
            std::vector<VkMat> bindings(4);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = weight_data_gpu;
            bindings[3] = bias_data_gpu;

            std::vector<vk_constant_type> constants(6);
            constants[0].u32 = bottom_blob_bordered.w;
            constants[1].u32 = bottom_blob_bordered.h;
            constants[2].u32 = bottom_blob_bordered.cstep;
            constants[3].u32 = top_blob.w;
            constants[4].u32 = top_blob.h;
            constants[5].u32 = top_blob.cstep;

            const int blocks_x = (top_blob.w * top_blob.h + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            const int subgroup_size = vkdev->info.subgroup_size();

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution_gemm, bindings, constants, dispatcher);
        }
        else
        {
            std::vector<VkMat> bindings(4);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = weight_data_gpu;
            bindings[3] = bias_data_gpu;

            std::vector<vk_constant_type> constants(8);
            constants[0].i = bottom_blob_bordered.w;
            constants[1].i = bottom_blob_bordered.h;
            constants[2].i = bottom_blob_bordered.c;
            constants[3].i = bottom_blob_bordered.cstep;
            constants[4].i = top_blob.w;
            constants[5].i = top_blob.h;
            constants[6].i = top_blob.c;
            constants[7].i = top_blob.cstep;

            VkMat dispatcher;
            dispatcher.w = (top_blob.w * top_blob.h + 3) / 4;
            dispatcher.h = top_blob.c;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution_gemm, bindings, constants, dispatcher);
        }

        return 0;
    }
    else if (is_conv1x1s1d1)
    {
        top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        if (use_cooperative_matrix)
        {
            std::vector<VkMat> bindings(4);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = weight_data_gpu;
            bindings[3] = bias_data_gpu;

            std::vector<vk_constant_type> constants(3);
            constants[0].u32 = bottom_blob_bordered.w * bottom_blob_bordered.h;
            constants[1].u32 = bottom_blob_bordered.cstep;
            constants[2].u32 = top_blob.cstep;

            const int blocks_x = (top_blob.w * top_blob.h + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            const int subgroup_size = vkdev->info.subgroup_size();

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution_1x1s1d1, bindings, constants, dispatcher);
        }
        else
        {
            std::vector<VkMat> bindings(4);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = weight_data_gpu;
            bindings[3] = bias_data_gpu;

            std::vector<vk_constant_type> constants(8);
            constants[0].i = bottom_blob_bordered.w;
            constants[1].i = bottom_blob_bordered.h;
            constants[2].i = bottom_blob_bordered.c;
            constants[3].i = bottom_blob_bordered.cstep;
            constants[4].i = top_blob.w;
            constants[5].i = top_blob.h;
            constants[6].i = top_blob.c;
            constants[7].i = top_blob.cstep;

            VkMat dispatcher;
            dispatcher.w = (top_blob.w * top_blob.h + 3) / 4;
            dispatcher.h = top_blob.c;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution_1x1s1d1, bindings, constants, dispatcher);
        }

        return 0;
    }

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob_bordered;
    bindings[1] = top_blob;
    bindings[2] = weight_data_gpu;
    bindings[3] = bias_data_gpu;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_bordered.dims;
    constants[1].i = bottom_blob_bordered.w;
    constants[2].i = bottom_blob_bordered.h;
    constants[3].i = bottom_blob_bordered.c;
    constants[4].i = bottom_blob_bordered.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;

    VkMat dispatcher;
    dispatcher.w = (top_blob.w + 1) / 2;
    dispatcher.h = (top_blob.h + 1) / 2;
    dispatcher.c = (top_blob.c + 1) / 2;

    cmd.record_pipeline(pipeline_convolution, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
