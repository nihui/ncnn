// Copyright 2021 Xavier Hsinyuan <me@lstlx.com>
// SPDX-License-Identifier: BSD-3-Clause

#if __riscv_vector
#define MAKE_FUNCTION(NAME, LMUL, IMPL, IMPLVV, IMPLVS, IMPLSV)                                                                            \
    struct NAME                                                                                                                            \
    {                                                                                                                                      \
        NCNN_FORCEINLINE float operator()(const float& x, const float& y) const                                                            \
        {                                                                                                                                  \
            return IMPL;                                                                                                                   \
        }                                                                                                                                  \
        NCNN_FORCEINLINE vfloat32m##LMUL##_t operator()(const vfloat32m##LMUL##_t& x, const vfloat32m##LMUL##_t& y, const size_t vl) const \
        {                                                                                                                                  \
            return IMPLVV;                                                                                                                 \
        }                                                                                                                                  \
        NCNN_FORCEINLINE vfloat32m##LMUL##_t operator()(const vfloat32m##LMUL##_t& x, const float& y, const size_t vl) const               \
        {                                                                                                                                  \
            return IMPLVS;                                                                                                                 \
        }                                                                                                                                  \
        NCNN_FORCEINLINE vfloat32m##LMUL##_t operator()(const float& x, const vfloat32m##LMUL##_t& y, const size_t vl) const               \
        {                                                                                                                                  \
            return IMPLSV;                                                                                                                 \
        }                                                                                                                                  \
    };
#else
#define MAKE_FUNCTION(NAME, LMUL, IMPL, IMPLVV, IMPLVS, IMPLSV)                 \
    struct NAME                                                                 \
    {                                                                           \
        NCNN_FORCEINLINE float operator()(const float& x, const float& y) const \
        {                                                                       \
            return IMPL;                                                        \
        }                                                                       \
    };
#endif

// clang-format off
// *INDENT-OFF*
#define _RVV_BINARY_OPS(LMUL) \
MAKE_FUNCTION(binary_op_add_m##LMUL, LMUL, x + y, __riscv_vfadd_vv_f32m##LMUL(x, y, vl), __riscv_vfadd_vf_f32m##LMUL(x, y, vl), __riscv_vfadd_vf_f32m##LMUL(y, x, vl)) \
MAKE_FUNCTION(binary_op_sub_m##LMUL, LMUL, x - y, __riscv_vfsub_vv_f32m##LMUL(x, y, vl), __riscv_vfsub_vf_f32m##LMUL(x, y, vl), __riscv_vfrsub_vf_f32m##LMUL(y, x, vl)) \
MAKE_FUNCTION(binary_op_mul_m##LMUL, LMUL, x * y, __riscv_vfmul_vv_f32m##LMUL(x, y, vl), __riscv_vfmul_vf_f32m##LMUL(x, y, vl), __riscv_vfmul_vf_f32m##LMUL(y, x, vl)) \
MAKE_FUNCTION(binary_op_div_m##LMUL, LMUL, x / y, __riscv_vfdiv_vv_f32m##LMUL(x, y, vl), __riscv_vfdiv_vf_f32m##LMUL(x, y, vl), __riscv_vfrdiv_vf_f32m##LMUL(y, x, vl)) \
MAKE_FUNCTION(binary_op_max_m##LMUL, LMUL, std::max(x, y), __riscv_vfmax_vv_f32m##LMUL(x, y, vl), __riscv_vfmax_vf_f32m##LMUL(x, y, vl), __riscv_vfmax_vf_f32m##LMUL(y, x, vl)) \
MAKE_FUNCTION(binary_op_min_m##LMUL, LMUL, std::min(x, y), __riscv_vfmin_vv_f32m##LMUL(x, y, vl), __riscv_vfmin_vf_f32m##LMUL(x, y, vl), __riscv_vfmin_vf_f32m##LMUL(y, x, vl)) \
MAKE_FUNCTION(binary_op_pow_m##LMUL, LMUL, (float)powf(x, y), pow_ps(x, y, vl), pow_ps(x, __riscv_vfmv_v_f_f32m##LMUL(y, vl), vl), pow_ps(__riscv_vfmv_v_f_f32m##LMUL(x, vl), y, vl)) \
MAKE_FUNCTION(binary_op_rsub_m##LMUL, LMUL, y - x, __riscv_vfsub_vv_f32m##LMUL(y, x, vl), __riscv_vfrsub_vf_f32m##LMUL(x, y, vl), __riscv_vfsub_vf_f32m##LMUL(y, x, vl)) \
MAKE_FUNCTION(binary_op_rdiv_m##LMUL, LMUL, y / x, __riscv_vfdiv_vv_f32m##LMUL(y, x, vl), __riscv_vfrdiv_vf_f32m##LMUL(x, y, vl), __riscv_vfdiv_vf_f32m##LMUL(y, x, vl)) \
MAKE_FUNCTION(binary_op_rpow_m##LMUL, LMUL, (float)powf(y, x), pow_ps(y, x, vl), pow_ps(__riscv_vfmv_v_f_f32m##LMUL(y, vl), x, vl), pow_ps(y, __riscv_vfmv_v_f_f32m##LMUL(x, vl), vl)) \
MAKE_FUNCTION(binary_op_atan2_m##LMUL, LMUL, (float)atan2f(x, y), atan2_ps(x, y, vl), atan2_ps(x, __riscv_vfmv_v_f_f32m##LMUL(y, vl), vl), atan2_ps(__riscv_vfmv_v_f_f32m##LMUL(x, vl), y, vl)) \
MAKE_FUNCTION(binary_op_ratan2_m##LMUL, LMUL, (float)atan2f(y, x), atan2_ps(y, x, vl), atan2_ps(__riscv_vfmv_v_f_f32m##LMUL(y, vl), x, vl), atan2_ps(y, __riscv_vfmv_v_f_f32m##LMUL(x, vl), vl)) \
MAKE_FUNCTION(binary_op_fmod_m##LMUL, LMUL, (float)fmodf(x, y), fmod_ps(x, y, vl), fmod_ps(x, __riscv_vfmv_v_f_f32m##LMUL(y, vl), vl), fmod_ps(__riscv_vfmv_v_f_f32m##LMUL(x, vl), y, vl)) \
MAKE_FUNCTION(binary_op_rfmod_m##LMUL, LMUL, (float)fmodf(y, x), fmod_ps(y, x, vl), fmod_ps(__riscv_vfmv_v_f_f32m##LMUL(y, vl), x, vl), fmod_ps(y, __riscv_vfmv_v_f_f32m##LMUL(x, vl), vl)) \
MAKE_FUNCTION(binary_op_logaddexp_m##LMUL, LMUL, (float)(std::max(x, y) + log1pf(expf(std::min(x, y) - std::max(x, y)))), logaddexp_ps(x, y, vl), logaddexp_ps(x, __riscv_vfmv_v_f_f32m##LMUL(y, vl), vl), logaddexp_ps(__riscv_vfmv_v_f_f32m##LMUL(x, vl), y, vl)) \
MAKE_FUNCTION(binary_op_floor_divide_m##LMUL, LMUL, (float)floorf(x / y), floor_divide_ps(x, y, vl), floor_divide_ps(x, __riscv_vfmv_v_f_f32m##LMUL(y, vl), vl), floor_divide_ps(__riscv_vfmv_v_f_f32m##LMUL(x, vl), y, vl)) \
MAKE_FUNCTION(binary_op_rfloor_divide_m##LMUL, LMUL, (float)floorf(y / x), floor_divide_ps(y, x, vl), floor_divide_ps(__riscv_vfmv_v_f_f32m##LMUL(y, vl), x, vl), floor_divide_ps(y, __riscv_vfmv_v_f_f32m##LMUL(x, vl), vl)) \
MAKE_FUNCTION(binary_op_remainder_m##LMUL, LMUL, (float)remainderf(x, y), remainder_ps(x, y, vl), remainder_ps(x, __riscv_vfmv_v_f_f32m##LMUL(y, vl), vl), remainder_ps(__riscv_vfmv_v_f_f32m##LMUL(x, vl), y, vl)) \
MAKE_FUNCTION(binary_op_rremainder_m##LMUL, LMUL, (float)remainderf(y, x), remainder_ps(y, x, vl), remainder_ps(__riscv_vfmv_v_f_f32m##LMUL(y, vl), x, vl), remainder_ps(y, __riscv_vfmv_v_f_f32m##LMUL(x, vl), vl))

_RVV_BINARY_OPS(2)
_RVV_BINARY_OPS(4)
_RVV_BINARY_OPS(8)

#undef _RVV_BINARY_OPS
// *INDENT-ON*
// clang-format on

#undef MAKE_FUNCTION
