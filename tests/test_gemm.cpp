// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#if NCNN_VALIDATION
static int test_gemm_load_param()
{
    ncnn::ParamDict base;
    base.set(4, 1);
    base.set(7, 8);
    base.set(9, 8);
    if (test_layer_param(ncnn::LayerType::Gemm, base, 0) != 0)
        return -1;

    return test_layer_param(ncnn::LayerType::Gemm, base, 7, -1, -1);
}
#endif // NCNN_VALIDATION

#if NCNN_VULKAN
static int test_gemm_subgroup_dispatch(int M, int N, int K, int transA, int transB)
{
    ncnn::ParamDict pd;
    pd.set(2, transA);
    pd.set(3, transB);

    std::vector<ncnn::Mat> inputs(2);
    inputs[0] = transA ? RandomMat(M, K) : RandomMat(K, M);
    inputs[1] = transB ? RandomMat(K, N) : RandomMat(N, K);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_shader_local_memory = false;
    opt.use_cooperative_matrix = false;
    opt.use_subgroup_ops = true;

    // exercise subgroup tiles and partial output blocks without the shared-memory path
    int ret = test_layer_opt("Gemm", pd, std::vector<ncnn::Mat>(), opt, inputs);
    if (ret != 0)
        fprintf(stderr, "test_gemm_subgroup_dispatch failed M=%d N=%d K=%d transA=%d transB=%d\n", M, N, K, transA, transB);
    return ret;
}
#endif // NCNN_VULKAN

int main()
{
    SRAND(7767517);

    return 0
#if NCNN_VALIDATION
           || test_gemm_load_param()
#endif // NCNN_VALIDATION
#if NCNN_VULKAN
           || test_gemm_subgroup_dispatch(7, 13, 11, 0, 0)
           || test_gemm_subgroup_dispatch(7, 13, 11, 0, 1)
           || test_gemm_subgroup_dispatch(7, 13, 11, 1, 0)
           || test_gemm_subgroup_dispatch(7, 13, 11, 1, 1)
           || test_gemm_subgroup_dispatch(16, 20, 12, 0, 0)
           || test_gemm_subgroup_dispatch(16, 20, 12, 0, 1)
           || test_gemm_subgroup_dispatch(16, 20, 12, 1, 0)
           || test_gemm_subgroup_dispatch(16, 20, 12, 1, 1)
#endif // NCNN_VULKAN
           ;
}
