// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

static void deconv4x4s2_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int group = bottom_blob.c;

    int outw = top_blob.w;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        const float* inptr = bottom_blob.channel(g);
        const float* kptr = (const float*)kernel.row(g);
        Mat out = top_blob.channel(g);

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + g * 4) : vdupq_n_f32(0.f);

        out.fill(_bias0);

        float32x4_t _k0 = vld1q_f32(kptr);
        float32x4_t _k1 = vld1q_f32(kptr + 4);
        float32x4_t _k2 = vld1q_f32(kptr + 8);
        float32x4_t _k3 = vld1q_f32(kptr + 12);
        float32x4_t _k4 = vld1q_f32(kptr + 16);
        float32x4_t _k5 = vld1q_f32(kptr + 20);
        float32x4_t _k6 = vld1q_f32(kptr + 24);
        float32x4_t _k7 = vld1q_f32(kptr + 28);
        float32x4_t _k8 = vld1q_f32(kptr + 32);
        float32x4_t _k9 = vld1q_f32(kptr + 36);
        float32x4_t _k10 = vld1q_f32(kptr + 40);
        float32x4_t _k11 = vld1q_f32(kptr + 44);
        float32x4_t _k12 = vld1q_f32(kptr + 48);
        float32x4_t _k13 = vld1q_f32(kptr + 52);
        float32x4_t _k14 = vld1q_f32(kptr + 56);
        float32x4_t _k15 = vld1q_f32(kptr + 60);

        float* outptr0 = out.row(0);
        float* outptr1 = out.row(1);
        float* outptr2 = out.row(2);
        float* outptr3 = out.row(3);

        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                float32x4_t _val = vld1q_f32(inptr);
                inptr += 4;

                vst1q_f32(outptr0, vmlaq_f32(vld1q_f32(outptr0), _val, _k0));
                vst1q_f32(outptr0 + 4, vmlaq_f32(vld1q_f32(outptr0 + 4), _val, _k1));
                vst1q_f32(outptr0 + 8, vmlaq_f32(vld1q_f32(outptr0 + 8), _val, _k2));
                vst1q_f32(outptr0 + 12, vmlaq_f32(vld1q_f32(outptr0 + 12), _val, _k3));

                vst1q_f32(outptr1, vmlaq_f32(vld1q_f32(outptr1), _val, _k4));
                vst1q_f32(outptr1 + 4, vmlaq_f32(vld1q_f32(outptr1 + 4), _val, _k5));
                vst1q_f32(outptr1 + 8, vmlaq_f32(vld1q_f32(outptr1 + 8), _val, _k6));
                vst1q_f32(outptr1 + 12, vmlaq_f32(vld1q_f32(outptr1 + 12), _val, _k7));

                vst1q_f32(outptr2, vmlaq_f32(vld1q_f32(outptr2), _val, _k8));
                vst1q_f32(outptr2 + 4, vmlaq_f32(vld1q_f32(outptr2 + 4), _val, _k9));
                vst1q_f32(outptr2 + 8, vmlaq_f32(vld1q_f32(outptr2 + 8), _val, _k10));
                vst1q_f32(outptr2 + 12, vmlaq_f32(vld1q_f32(outptr2 + 12), _val, _k11));

                vst1q_f32(outptr3, vmlaq_f32(vld1q_f32(outptr3), _val, _k12));
                vst1q_f32(outptr3 + 4, vmlaq_f32(vld1q_f32(outptr3 + 4), _val, _k13));
                vst1q_f32(outptr3 + 8, vmlaq_f32(vld1q_f32(outptr3 + 8), _val, _k14));
                vst1q_f32(outptr3 + 12, vmlaq_f32(vld1q_f32(outptr3 + 12), _val, _k15));

                outptr0 += 8;
                outptr1 += 8;
                outptr2 += 8;
                outptr3 += 8;
            }

            outptr0 += (outw + 2) * 4;
            outptr1 += (outw + 2) * 4;
            outptr2 += (outw + 2) * 4;
            outptr3 += (outw + 2) * 4;
        }
    }
}
