// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "binaryop.h"

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

#if __riscv_vector
#include "rvv_mathfun.h"
#endif

namespace ncnn {

#if NCNN_BF16

namespace BinaryOp_riscv_zvfbfmin_functor {

#include "binaryop_functor.h"

} // namespace BinaryOp_riscv_zvfbfmin_functor

#include "binaryop_bf16s.h"

void binary_op_vector_bf16s_zvfbfmin(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int aw, int bw, int ap, int bp, int op_type)
{
    using namespace BinaryOp_riscv_zvfbfmin_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_vector_bf16s<binary_op_add_m8, binary_op_add_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_vector_bf16s<binary_op_sub_m8, binary_op_sub_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_vector_bf16s<binary_op_mul_m8, binary_op_mul_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_vector_bf16s<binary_op_div_m8, binary_op_div_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_vector_bf16s<binary_op_max_m8, binary_op_max_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_vector_bf16s<binary_op_min_m8, binary_op_min_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_POW) return binary_op_vector_bf16s<binary_op_pow_m8, binary_op_pow_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_vector_bf16s<binary_op_rsub_m8, binary_op_rsub_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_vector_bf16s<binary_op_rdiv_m8, binary_op_rdiv_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_vector_bf16s<binary_op_rpow_m8, binary_op_rpow_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_vector_bf16s<binary_op_atan2_m8, binary_op_atan2_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_vector_bf16s<binary_op_ratan2_m8, binary_op_ratan2_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_FMOD) return binary_op_vector_bf16s<binary_op_fmod_m8, binary_op_fmod_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RFMOD) return binary_op_vector_bf16s<binary_op_rfmod_m8, binary_op_rfmod_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_LOGADDEXP) return binary_op_vector_bf16s<binary_op_logaddexp_m8, binary_op_logaddexp_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_FLOOR_DIVIDE) return binary_op_vector_bf16s<binary_op_floor_divide_m8, binary_op_floor_divide_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RFLOOR_DIVIDE) return binary_op_vector_bf16s<binary_op_rfloor_divide_m8, binary_op_rfloor_divide_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_REMAINDER) return binary_op_vector_bf16s<binary_op_remainder_m8, binary_op_remainder_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RREMAINDER) return binary_op_vector_bf16s<binary_op_rremainder_m8, binary_op_rremainder_m2>(ptr, ptr1, outptr, aw, bw, ap, bp);

    // should never reach here
}

void binary_op_vector_scalar_b_bf16s_zvfbfmin(const unsigned short* ptr, float b, unsigned short* outptr, int size, int op_type)
{
    using namespace BinaryOp_riscv_zvfbfmin_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_vector_scalar_b_bf16s<binary_op_add_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_vector_scalar_b_bf16s<binary_op_sub_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_vector_scalar_b_bf16s<binary_op_mul_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_vector_scalar_b_bf16s<binary_op_div_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_vector_scalar_b_bf16s<binary_op_max_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_vector_scalar_b_bf16s<binary_op_min_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_POW) return binary_op_vector_scalar_b_bf16s<binary_op_pow_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_vector_scalar_b_bf16s<binary_op_rsub_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_vector_scalar_b_bf16s<binary_op_rdiv_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_vector_scalar_b_bf16s<binary_op_rpow_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_vector_scalar_b_bf16s<binary_op_atan2_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_vector_scalar_b_bf16s<binary_op_ratan2_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_FMOD) return binary_op_vector_scalar_b_bf16s<binary_op_fmod_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_RFMOD) return binary_op_vector_scalar_b_bf16s<binary_op_rfmod_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_LOGADDEXP) return binary_op_vector_scalar_b_bf16s<binary_op_logaddexp_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_FLOOR_DIVIDE) return binary_op_vector_scalar_b_bf16s<binary_op_floor_divide_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_RFLOOR_DIVIDE) return binary_op_vector_scalar_b_bf16s<binary_op_rfloor_divide_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_REMAINDER) return binary_op_vector_scalar_b_bf16s<binary_op_remainder_m8>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_RREMAINDER) return binary_op_vector_scalar_b_bf16s<binary_op_rremainder_m8>(ptr, b, outptr, size);

    // should never reach here
}

#endif // NCNN_BF16

} // namespace ncnn
