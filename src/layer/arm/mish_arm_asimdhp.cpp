// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "mish_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#include "neon_mathfun.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "neon_mathfun_fp16s.h"
#endif
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int Mish_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        float32x4_t _one = vdupq_n_f32(1.f);

        int i = 0;
        for (; i + 15 < size; i += 16)
        {
            float16x8_t _p01 = vld1q_f16(ptr);
            float16x8_t _p23 = vld1q_f16(ptr + 8);
            float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p01));
            float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p01));
            float32x4_t _p2 = vcvt_f32_f16(vget_low_f16(_p23));
            float32x4_t _p3 = vcvt_f32_f16(vget_high_f16(_p23));
            _p0 = vmulq_f32(_p0, tanh_ps(log_ps(vaddq_f32(exp_ps(_p0), vdupq_n_f32(1.f)))));
            _p1 = vmulq_f32(_p1, tanh_ps(log_ps(vaddq_f32(exp_ps(_p1), vdupq_n_f32(1.f)))));
            _p2 = vmulq_f32(_p2, tanh_ps(log_ps(vaddq_f32(exp_ps(_p2), vdupq_n_f32(1.f)))));
            _p3 = vmulq_f32(_p3, tanh_ps(log_ps(vaddq_f32(exp_ps(_p3), vdupq_n_f32(1.f)))));
            _p01 = vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1));
            _p23 = vcombine_f16(vcvt_f16_f32(_p2), vcvt_f16_f32(_p3));
            vst1q_f16(ptr, _p01);
            vst1q_f16(ptr + 8, _p23);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
            float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
            _p0 = vmulq_f32(_p0, tanh_ps(log_ps(vaddq_f32(exp_ps(_p0), vdupq_n_f32(1.f)))));
            _p1 = vmulq_f32(_p1, tanh_ps(log_ps(vaddq_f32(exp_ps(_p1), vdupq_n_f32(1.f)))));
            _p = vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1));
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            _p = vmulq_f32(_p, tanh_ps(log_ps(vaddq_f32(exp_ps(_p), vdupq_n_f32(1.f)))));
            vst1_f16(ptr, vcvt_f16_f32(_p));
            ptr += 4;
        }
        for (; i < size; i++)
        {
            float v = (float)*ptr;
            v = v * tanhf(logf(expf(v) + 1.f));
            *ptr = (__fp16)v;

            ptr++;
        }
    }

    return 0;
}

int Mish_arm::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        float16x8_t _one = vdupq_n_f16(1.f);

        int i = 0;
        for (; i + 31 < size; i += 32)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);
            _p0 = vmulq_f16(_p0, tanh_ps_f16(log_ps_f16(vaddq_f16(exp_ps_f16(_p0), vdupq_n_f16(1.f)))));
            _p1 = vmulq_f16(_p1, tanh_ps_f16(log_ps_f16(vaddq_f16(exp_ps_f16(_p1), vdupq_n_f16(1.f)))));
            _p2 = vmulq_f16(_p2, tanh_ps_f16(log_ps_f16(vaddq_f16(exp_ps_f16(_p2), vdupq_n_f16(1.f)))));
            _p3 = vmulq_f16(_p3, tanh_ps_f16(log_ps_f16(vaddq_f16(exp_ps_f16(_p3), vdupq_n_f16(1.f)))));
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            vst1q_f16(ptr + 16, _p2);
            vst1q_f16(ptr + 24, _p3);
            ptr += 32;
        }
        for (; i + 15 < size; i += 16)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            _p0 = vmulq_f16(_p0, tanh_ps_f16(log_ps_f16(vaddq_f16(exp_ps_f16(_p0), vdupq_n_f16(1.f)))));
            _p1 = vmulq_f16(_p1, tanh_ps_f16(log_ps_f16(vaddq_f16(exp_ps_f16(_p1), vdupq_n_f16(1.f)))));
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = vmulq_f16(_p, tanh_ps_f16(log_ps_f16(vaddq_f16(exp_ps_f16(_p), vdupq_n_f16(1.f)))));
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = vmul_f16(_p, tanh_ps_f16(log_ps_f16(vadd_f16(exp_ps_f16(_p), vdup_n_f16(1.f)))));
            vst1_f16(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            __fp16 v = *ptr;
            v = v * (__fp16)tanhf(logf(expf(v) + 1.f));
            *ptr = v;

            ptr++;
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
