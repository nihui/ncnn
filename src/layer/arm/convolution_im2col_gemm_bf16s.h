// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

static void convolution_im2col_pack_A_tile_bf16s(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, int maxk, int inch, int outch)
{
    // A = (pa, maxk, inch/pa), outch
    // const int elempack = A.elempack;
    const int A_hstep = maxk * inch;

    unsigned short* pp = AT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;
            const unsigned short* p2 = (const unsigned short*)A + (i + ii + 2) * A_hstep + k;
            const unsigned short* p3 = (const unsigned short*)A + (i + ii + 3) * A_hstep + k;
            const unsigned short* p4 = (const unsigned short*)A + (i + ii + 4) * A_hstep + k;
            const unsigned short* p5 = (const unsigned short*)A + (i + ii + 5) * A_hstep + k;
            const unsigned short* p6 = (const unsigned short*)A + (i + ii + 6) * A_hstep + k;
            const unsigned short* p7 = (const unsigned short*)A + (i + ii + 7) * A_hstep + k;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8_t _r0 = vld1q_u16(p0);
                uint16x8_t _r1 = vld1q_u16(p1);
                uint16x8_t _r2 = vld1q_u16(p2);
                uint16x8_t _r3 = vld1q_u16(p3);
                uint16x8_t _r4 = vld1q_u16(p4);
                uint16x8_t _r5 = vld1q_u16(p5);
                uint16x8_t _r6 = vld1q_u16(p6);
                uint16x8_t _r7 = vld1q_u16(p7);
                transpose8x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                vst1q_u16(pp, _r0);
                vst1q_u16(pp + 8, _r1);
                vst1q_u16(pp + 8 * 2, _r2);
                vst1q_u16(pp + 8 * 3, _r3);
                vst1q_u16(pp + 8 * 4, _r4);
                vst1q_u16(pp + 8 * 5, _r5);
                vst1q_u16(pp + 8 * 6, _r6);
                vst1q_u16(pp + 8 * 7, _r7);
                pp += 64;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
                p4 += 8;
                p5 += 8;
                p6 += 8;
                p7 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[4] = p4[0];
                pp[5] = p5[0];
                pp[6] = p6[0];
                pp[7] = p7[0];
                pp += 8;
                p0++;
                p1++;
                p2++;
                p3++;
                p4++;
                p5++;
                p6++;
                p7++;
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;
            const unsigned short* p2 = (const unsigned short*)A + (i + ii + 2) * A_hstep + k;
            const unsigned short* p3 = (const unsigned short*)A + (i + ii + 3) * A_hstep + k;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x4_t _r0123;
                _r0123.val[0] = vld1q_u16(p0);
                _r0123.val[1] = vld1q_u16(p1);
                _r0123.val[2] = vld1q_u16(p2);
                _r0123.val[3] = vld1q_u16(p3);
                vst4q_u16(pp, _r0123);
                pp += 32;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4x4_t _r0123;
                _r0123.val[0] = vld1_u16(p0);
                _r0123.val[1] = vld1_u16(p1);
                _r0123.val[2] = vld1_u16(p2);
                _r0123.val[3] = vld1_u16(p3);
                vst4_u16(pp, _r0123);
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp += 4;
                p0++;
                p1++;
                p2++;
                p3++;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;

            int kk = 0;
#if __ARM_NEON
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x2_t _r01;
                _r01.val[0] = vld1q_u16(p0);
                _r01.val[1] = vld1q_u16(p1);
                vst2q_u16(pp, _r01);
                pp += 16;
                p0 += 8;
                p1 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4x2_t _r01;
                _r01.val[0] = vld1_u16(p0);
                _r01.val[1] = vld1_u16(p1);
                vst2_u16(pp, _r01);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
#endif // __ARM_NEON
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0++;
                p1++;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

            int kk = 0;
#if __ARM_NEON
            for (; kk + 7 < max_kk; kk += 8)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                pp += 8;
                p0 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += 4;
            }
#endif // __ARM_NEON
            for (; kk < max_kk; kk++)
            {
                pp[0] = (unsigned short)p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void convolution_gemm_transB_packed_tile_bf16s(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    NCNN_LOGE("convolution_gemm_transB_packed_tile_bf16s %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const int out_elempack = top_blob.elempack;
    const int out_hstep = (int)top_blob.cstep;

    const unsigned short* pAT = AT_tile;
    const unsigned short* pBT = BT_tile;
    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            const unsigned short* pA = pAT;

#if 0  //NCNN_GNU_INLINE_ASM
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                "subs   %0, %0, #320                \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "add    x4, %8, #16                 \n"
                "ld1    {v8.4s}, [%8]               \n"
                "ld1    {v20.4s}, [x4]              \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v8.16b, v8.16b, v8.16b      \n"
                "eor    v20.16b, v20.16b, v20.16b   \n"

                "2:                                 \n"
                "mov    v9.16b, v8.16b              \n"
                "mov    v10.16b, v8.16b             \n"
                "mov    v11.16b, v8.16b             \n"
                "mov    v12.16b, v8.16b             \n"
                "mov    v13.16b, v8.16b             \n"
                "mov    v14.16b, v8.16b             \n"
                "mov    v15.16b, v8.16b             \n"
                "mov    v16.16b, v8.16b             \n"
                "mov    v17.16b, v8.16b             \n"
                "mov    v18.16b, v8.16b             \n"
                "mov    v19.16b, v8.16b             \n"

                "mov    v21.16b, v20.16b            \n"
                "mov    v22.16b, v20.16b            \n"
                "mov    v23.16b, v20.16b            \n"
                "mov    v24.16b, v20.16b            \n"
                "mov    v25.16b, v20.16b            \n"
                "mov    v26.16b, v20.16b            \n"
                "mov    v27.16b, v20.16b            \n"
                "mov    v28.16b, v20.16b            \n"
                "mov    v29.16b, v20.16b            \n"
                "mov    v30.16b, v20.16b            \n"
                "mov    v31.16b, v20.16b            \n"

                "3:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "4:                                 \n"
                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                "fmla   v22.4s, v5.4s, v0.s[2]      \n"
                "fmla   v23.4s, v5.4s, v0.s[3]      \n"

                "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                "fmla   v25.4s, v5.4s, v1.s[1]      \n"
                "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                "fmla   v27.4s, v5.4s, v1.s[3]      \n"

                "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                "fmla   v19.4s, v4.4s, v2.s[3]      \n"
                "fmla   v28.4s, v5.4s, v2.s[0]      \n"
                "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                "fmla   v8.4s, v6.4s, v3.s[0]       \n"
                "fmla   v9.4s, v6.4s, v3.s[1]       \n"
                "fmla   v10.4s, v6.4s, v3.s[2]      \n"
                "fmla   v11.4s, v6.4s, v3.s[3]      \n"
                "fmla   v20.4s, v7.4s, v3.s[0]      \n"
                "fmla   v21.4s, v7.4s, v3.s[1]      \n"
                "fmla   v22.4s, v7.4s, v3.s[2]      \n"
                "fmla   v23.4s, v7.4s, v3.s[3]      \n"

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                "fmla   v12.4s, v6.4s, v0.s[0]      \n"
                "fmla   v13.4s, v6.4s, v0.s[1]      \n"
                "fmla   v14.4s, v6.4s, v0.s[2]      \n"
                "fmla   v15.4s, v6.4s, v0.s[3]      \n"
                "fmla   v24.4s, v7.4s, v0.s[0]      \n"
                "fmla   v25.4s, v7.4s, v0.s[1]      \n"
                "fmla   v26.4s, v7.4s, v0.s[2]      \n"
                "fmla   v27.4s, v7.4s, v0.s[3]      \n"

                "fmla   v16.4s, v6.4s, v1.s[0]      \n"
                "fmla   v17.4s, v6.4s, v1.s[1]      \n"
                "fmla   v18.4s, v6.4s, v1.s[2]      \n"
                "fmla   v19.4s, v6.4s, v1.s[3]      \n"
                "fmla   v28.4s, v7.4s, v1.s[0]      \n"
                "fmla   v29.4s, v7.4s, v1.s[1]      \n"
                "fmla   v30.4s, v7.4s, v1.s[2]      \n"
                "fmla   v31.4s, v7.4s, v1.s[3]      \n"

                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"

                "fmla   v8.4s, v4.4s, v2.s[0]       \n"
                "fmla   v9.4s, v4.4s, v2.s[1]       \n"
                "fmla   v10.4s, v4.4s, v2.s[2]      \n"
                "fmla   v11.4s, v4.4s, v2.s[3]      \n"
                "fmla   v20.4s, v5.4s, v2.s[0]      \n"
                "fmla   v21.4s, v5.4s, v2.s[1]      \n"
                "fmla   v22.4s, v5.4s, v2.s[2]      \n"
                "fmla   v23.4s, v5.4s, v2.s[3]      \n"

                "fmla   v12.4s, v4.4s, v3.s[0]      \n"
                "fmla   v13.4s, v4.4s, v3.s[1]      \n"
                "fmla   v14.4s, v4.4s, v3.s[2]      \n"
                "fmla   v15.4s, v4.4s, v3.s[3]      \n"
                "fmla   v24.4s, v5.4s, v3.s[0]      \n"
                "fmla   v25.4s, v5.4s, v3.s[1]      \n"
                "fmla   v26.4s, v5.4s, v3.s[2]      \n"
                "fmla   v27.4s, v5.4s, v3.s[3]      \n"

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                "fmla   v16.4s, v4.4s, v0.s[0]      \n"
                "fmla   v17.4s, v4.4s, v0.s[1]      \n"
                "fmla   v18.4s, v4.4s, v0.s[2]      \n"
                "fmla   v19.4s, v4.4s, v0.s[3]      \n"
                "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                "fmla   v31.4s, v5.4s, v0.s[3]      \n"

                "fmla   v8.4s, v6.4s, v1.s[0]       \n"
                "fmla   v9.4s, v6.4s, v1.s[1]       \n"
                "fmla   v10.4s, v6.4s, v1.s[2]      \n"
                "fmla   v11.4s, v6.4s, v1.s[3]      \n"
                "fmla   v20.4s, v7.4s, v1.s[0]      \n"
                "fmla   v21.4s, v7.4s, v1.s[1]      \n"
                "fmla   v22.4s, v7.4s, v1.s[2]      \n"
                "fmla   v23.4s, v7.4s, v1.s[3]      \n"

                "fmla   v12.4s, v6.4s, v2.s[0]      \n"
                "fmla   v13.4s, v6.4s, v2.s[1]      \n"
                "fmla   v14.4s, v6.4s, v2.s[2]      \n"
                "fmla   v15.4s, v6.4s, v2.s[3]      \n"
                "fmla   v24.4s, v7.4s, v2.s[0]      \n"
                "fmla   v25.4s, v7.4s, v2.s[1]      \n"
                "fmla   v26.4s, v7.4s, v2.s[2]      \n"
                "fmla   v27.4s, v7.4s, v2.s[3]      \n"

                "subs   w4, w4, #1                  \n"

                "fmla   v16.4s, v6.4s, v3.s[0]      \n"
                "fmla   v17.4s, v6.4s, v3.s[1]      \n"
                "fmla   v18.4s, v6.4s, v3.s[2]      \n"
                "fmla   v19.4s, v6.4s, v3.s[3]      \n"
                "fmla   v28.4s, v7.4s, v3.s[0]      \n"
                "fmla   v29.4s, v7.4s, v3.s[1]      \n"
                "fmla   v30.4s, v7.4s, v3.s[2]      \n"
                "fmla   v31.4s, v7.4s, v3.s[3]      \n"

                "bne    4b                          \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1    {v0.4s, v1.4s, v2.4s}, [%2], #48 \n"
                "ld1    {v4.4s, v5.4s}, [%1], #32   \n"

                "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                "fmla   v19.4s, v4.4s, v2.s[3]      \n"

                "subs   w4, w4, #1                  \n"

                "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                "fmla   v22.4s, v5.4s, v0.s[2]      \n"
                "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                "fmla   v25.4s, v5.4s, v1.s[1]      \n"
                "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                "fmla   v27.4s, v5.4s, v1.s[3]      \n"
                "fmla   v28.4s, v5.4s, v2.s[0]      \n"
                "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                "bne    6b                          \n"

                "7:                                 \n"
                "tst    %w11, #255                  \n"
                "beq    10f                         \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    8f                          \n"

                "lsl    w4, %w13, #2                \n"
                "add    x4, %3, w4, sxtw 2          \n"
                "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%3], #64 \n"
                "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%3], #64 \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%3], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [x4], #64 \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [x4], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [x4] \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // transpose8x12
                "zip1   v6.4s, v8.4s, v9.4s         \n"
                "zip2   v7.4s, v8.4s, v9.4s         \n"
                "zip1   v8.4s, v10.4s, v11.4s       \n"
                "zip2   v9.4s, v10.4s, v11.4s       \n"
                "zip1   v10.4s, v12.4s, v13.4s      \n"
                "zip2   v11.4s, v12.4s, v13.4s      \n"
                "zip1   v12.4s, v14.4s, v15.4s      \n"
                "zip2   v13.4s, v14.4s, v15.4s      \n"
                "zip1   v14.4s, v16.4s, v17.4s      \n"
                "zip2   v15.4s, v16.4s, v17.4s      \n"
                "zip1   v16.4s, v18.4s, v19.4s      \n"
                "zip2   v17.4s, v18.4s, v19.4s      \n"

                "zip1   v18.4s, v20.4s, v21.4s      \n"
                "zip2   v19.4s, v20.4s, v21.4s      \n"
                "zip1   v20.4s, v22.4s, v23.4s      \n"
                "zip2   v21.4s, v22.4s, v23.4s      \n"
                "zip1   v22.4s, v24.4s, v25.4s      \n"
                "zip2   v23.4s, v24.4s, v25.4s      \n"
                "zip1   v24.4s, v26.4s, v27.4s      \n"
                "zip2   v25.4s, v26.4s, v27.4s      \n"
                "zip1   v26.4s, v28.4s, v29.4s      \n"
                "zip2   v27.4s, v28.4s, v29.4s      \n"
                "zip1   v28.4s, v30.4s, v31.4s      \n"
                "zip2   v29.4s, v30.4s, v31.4s      \n"

                "zip1   v0.2d, v6.2d, v8.2d         \n"
                "zip2   v3.2d, v6.2d, v8.2d         \n"
                "zip1   v1.2d, v10.2d, v12.2d       \n"
                "zip2   v4.2d, v10.2d, v12.2d       \n"
                "zip1   v2.2d, v14.2d, v16.2d       \n"
                "zip2   v5.2d, v14.2d, v16.2d       \n"

                "zip1   v6.2d, v7.2d, v9.2d         \n"
                "zip2   v9.2d, v7.2d, v9.2d         \n"
                "zip1   v7.2d, v11.2d, v13.2d       \n"
                "zip2   v10.2d, v11.2d, v13.2d      \n"
                "zip1   v8.2d, v15.2d, v17.2d       \n"
                "zip2   v11.2d, v15.2d, v17.2d      \n"

                "zip1   v12.2d, v18.2d, v20.2d      \n"
                "zip2   v15.2d, v18.2d, v20.2d      \n"
                "zip1   v13.2d, v22.2d, v24.2d      \n"
                "zip2   v16.2d, v22.2d, v24.2d      \n"
                "zip1   v14.2d, v26.2d, v28.2d      \n"
                "zip2   v17.2d, v26.2d, v28.2d      \n"

                "zip1   v18.2d, v19.2d, v21.2d      \n"
                "zip2   v21.2d, v19.2d, v21.2d      \n"
                "zip1   v19.2d, v23.2d, v25.2d      \n"
                "zip2   v22.2d, v23.2d, v25.2d      \n"
                "zip1   v20.2d, v27.2d, v29.2d      \n"
                "zip2   v23.2d, v27.2d, v29.2d      \n"

                "add    x4, %3, %w13, sxtw 2        \n"
                "st1    {v0.4s, v1.4s, v2.4s}, [%3], #48 \n"
                "st1    {v3.4s, v4.4s, v5.4s}, [x4] \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v6.4s, v7.4s, v8.4s}, [x4] \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v9.4s, v10.4s, v11.4s}, [x4] \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v12.4s, v13.4s, v14.4s}, [x4] \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v15.4s, v16.4s, v17.4s}, [x4] \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v18.4s, v19.4s, v20.4s}, [x4] \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v21.4s, v22.4s, v23.4s}, [x4] \n"

                "9:                                 \n"
                "add    %0, %0, #384                \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum20;
            float32x4_t _sum21;
            float32x4_t _sum30;
            float32x4_t _sum31;
            float32x4_t _sum40;
            float32x4_t _sum41;
            float32x4_t _sum50;
            float32x4_t _sum51;
            float32x4_t _sum60;
            float32x4_t _sum61;
            float32x4_t _sum70;
            float32x4_t _sum71;
            float32x4_t _sum80;
            float32x4_t _sum81;
            float32x4_t _sum90;
            float32x4_t _sum91;
            float32x4_t _suma0;
            float32x4_t _suma1;
            float32x4_t _sumb0;
            float32x4_t _sumb1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                    _sum10 = _sum00;
                    _sum11 = _sum01;
                    _sum20 = _sum00;
                    _sum21 = _sum01;
                    _sum30 = _sum00;
                    _sum31 = _sum01;
                    _sum40 = _sum00;
                    _sum41 = _sum01;
                    _sum50 = _sum00;
                    _sum51 = _sum01;
                    _sum60 = _sum00;
                    _sum61 = _sum01;
                    _sum70 = _sum00;
                    _sum71 = _sum01;
                    _sum80 = _sum00;
                    _sum81 = _sum01;
                    _sum90 = _sum00;
                    _sum91 = _sum01;
                    _suma0 = _sum00;
                    _suma1 = _sum01;
                    _sumb0 = _sum00;
                    _sumb1 = _sum01;
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                    _sum10 = vdupq_n_f32(0.f);
                    _sum11 = vdupq_n_f32(0.f);
                    _sum20 = vdupq_n_f32(0.f);
                    _sum21 = vdupq_n_f32(0.f);
                    _sum30 = vdupq_n_f32(0.f);
                    _sum31 = vdupq_n_f32(0.f);
                    _sum40 = vdupq_n_f32(0.f);
                    _sum41 = vdupq_n_f32(0.f);
                    _sum50 = vdupq_n_f32(0.f);
                    _sum51 = vdupq_n_f32(0.f);
                    _sum60 = vdupq_n_f32(0.f);
                    _sum61 = vdupq_n_f32(0.f);
                    _sum70 = vdupq_n_f32(0.f);
                    _sum71 = vdupq_n_f32(0.f);
                    _sum80 = vdupq_n_f32(0.f);
                    _sum81 = vdupq_n_f32(0.f);
                    _sum90 = vdupq_n_f32(0.f);
                    _sum91 = vdupq_n_f32(0.f);
                    _suma0 = vdupq_n_f32(0.f);
                    _suma1 = vdupq_n_f32(0.f);
                    _sumb0 = vdupq_n_f32(0.f);
                    _sumb1 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
                _sum20 = vld1q_f32(outptr + 4 * 4);
                _sum21 = vld1q_f32(outptr + 4 * 5);
                _sum30 = vld1q_f32(outptr + 4 * 6);
                _sum31 = vld1q_f32(outptr + 4 * 7);
                _sum40 = vld1q_f32(outptr + 4 * 8);
                _sum41 = vld1q_f32(outptr + 4 * 9);
                _sum50 = vld1q_f32(outptr + 4 * 10);
                _sum51 = vld1q_f32(outptr + 4 * 11);
                _sum60 = vld1q_f32(outptr + 4 * 12);
                _sum61 = vld1q_f32(outptr + 4 * 13);
                _sum70 = vld1q_f32(outptr + 4 * 14);
                _sum71 = vld1q_f32(outptr + 4 * 15);
                _sum80 = vld1q_f32(outptr + 4 * 16);
                _sum81 = vld1q_f32(outptr + 4 * 17);
                _sum90 = vld1q_f32(outptr + 4 * 18);
                _sum91 = vld1q_f32(outptr + 4 * 19);
                _suma0 = vld1q_f32(outptr + 4 * 20);
                _suma1 = vld1q_f32(outptr + 4 * 21);
                _sumb0 = vld1q_f32(outptr + 4 * 22);
                _sumb1 = vld1q_f32(outptr + 4 * 23);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = bfloat2float(vld1_u16(pA));
                float32x4_t _pA1 = bfloat2float(vld1_u16(pA + 4));

                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));
                float32x4_t _pB2 = bfloat2float(vld1_u16(pB + 8));

                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32(_sum71, _pA1, _pB1, 3);
                _sum80 = vfmaq_laneq_f32(_sum80, _pA0, _pB2, 0);
                _sum81 = vfmaq_laneq_f32(_sum81, _pA1, _pB2, 0);
                _sum90 = vfmaq_laneq_f32(_sum90, _pA0, _pB2, 1);
                _sum91 = vfmaq_laneq_f32(_sum91, _pA1, _pB2, 1);
                _suma0 = vfmaq_laneq_f32(_suma0, _pA0, _pB2, 2);
                _suma1 = vfmaq_laneq_f32(_suma1, _pA1, _pB2, 2);
                _sumb0 = vfmaq_laneq_f32(_sumb0, _pA0, _pB2, 3);
                _sumb1 = vfmaq_laneq_f32(_sumb1, _pA1, _pB2, 3);

                pA += 8;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum10));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum20));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum30));
                    vst1_u16(outptr0 + 4 * 4, float2bfloat(_sum40));
                    vst1_u16(outptr0 + 4 * 5, float2bfloat(_sum50));
                    vst1_u16(outptr0 + 4 * 6, float2bfloat(_sum60));
                    vst1_u16(outptr0 + 4 * 7, float2bfloat(_sum70));
                    vst1_u16(outptr0 + 4 * 8, float2bfloat(_sum80));
                    vst1_u16(outptr0 + 4 * 9, float2bfloat(_sum90));
                    vst1_u16(outptr0 + 4 * 10, float2bfloat(_suma0));
                    vst1_u16(outptr0 + 4 * 11, float2bfloat(_sumb0));

                    vst1_u16(outptr0 + out_hstep * 4, float2bfloat(_sum01));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, float2bfloat(_sum11));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 2, float2bfloat(_sum21));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 3, float2bfloat(_sum31));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 4, float2bfloat(_sum41));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 5, float2bfloat(_sum51));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 6, float2bfloat(_sum61));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 7, float2bfloat(_sum71));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 8, float2bfloat(_sum81));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 9, float2bfloat(_sum91));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 10, float2bfloat(_suma1));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 11, float2bfloat(_sumb1));

                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    uint16x8_t _t0 = vcombine_u16(float2bfloat(_sum00), float2bfloat(_sum01));
                    uint16x8_t _t1 = vcombine_u16(float2bfloat(_sum10), float2bfloat(_sum11));
                    uint16x8_t _t2 = vcombine_u16(float2bfloat(_sum20), float2bfloat(_sum21));
                    uint16x8_t _t3 = vcombine_u16(float2bfloat(_sum30), float2bfloat(_sum31));
                    uint16x8_t _t4 = vcombine_u16(float2bfloat(_sum40), float2bfloat(_sum41));
                    uint16x8_t _t5 = vcombine_u16(float2bfloat(_sum50), float2bfloat(_sum51));
                    uint16x8_t _t6 = vcombine_u16(float2bfloat(_sum60), float2bfloat(_sum61));
                    uint16x8_t _t7 = vcombine_u16(float2bfloat(_sum70), float2bfloat(_sum71));
                    uint16x8_t _t8 = vcombine_u16(float2bfloat(_sum80), float2bfloat(_sum81));
                    uint16x8_t _t9 = vcombine_u16(float2bfloat(_sum90), float2bfloat(_sum91));
                    uint16x8_t _ta = vcombine_u16(float2bfloat(_suma0), float2bfloat(_suma1));
                    uint16x8_t _tb = vcombine_u16(float2bfloat(_sumb0), float2bfloat(_sumb1));
                    transpose8x12_u16(_t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9, _ta, _tb);

                    vst1_u16(outptr0, vget_low_u16(_t0));
                    vst1_u16(outptr0 + 4, vget_high_u16(_t0));
                    vst1_u16(outptr0 + 8, vget_low_u16(_t1));
                    vst1_u16(outptr0 + out_hstep, vget_high_u16(_t1));
                    vst1_u16(outptr0 + out_hstep + 4, vget_low_u16(_t2));
                    vst1_u16(outptr0 + out_hstep + 8, vget_high_u16(_t2));
                    vst1_u16(outptr0 + out_hstep * 2, vget_low_u16(_t3));
                    vst1_u16(outptr0 + out_hstep * 2 + 4, vget_high_u16(_t3));
                    vst1_u16(outptr0 + out_hstep * 2 + 8, vget_low_u16(_t4));
                    vst1_u16(outptr0 + out_hstep * 3, vget_high_u16(_t4));
                    vst1_u16(outptr0 + out_hstep * 3 + 4, vget_low_u16(_t5));
                    vst1_u16(outptr0 + out_hstep * 3 + 8, vget_high_u16(_t5));
                    vst1_u16(outptr0 + out_hstep * 4, vget_low_u16(_t6));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, vget_high_u16(_t6));
                    vst1_u16(outptr0 + out_hstep * 4 + 8, vget_low_u16(_t7));
                    vst1_u16(outptr0 + out_hstep * 5, vget_high_u16(_t7));
                    vst1_u16(outptr0 + out_hstep * 5 + 4, vget_low_u16(_t8));
                    vst1_u16(outptr0 + out_hstep * 5 + 8, vget_high_u16(_t8));
                    vst1_u16(outptr0 + out_hstep * 6, vget_low_u16(_t9));
                    vst1_u16(outptr0 + out_hstep * 6 + 4, vget_high_u16(_t9));
                    vst1_u16(outptr0 + out_hstep * 6 + 8, vget_low_u16(_ta));
                    vst1_u16(outptr0 + out_hstep * 7, vget_high_u16(_ta));
                    vst1_u16(outptr0 + out_hstep * 7 + 4, vget_low_u16(_tb));
                    vst1_u16(outptr0 + out_hstep * 7 + 8, vget_high_u16(_tb));

                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                vst1q_f32(outptr + 4 * 4, _sum20);
                vst1q_f32(outptr + 4 * 5, _sum21);
                vst1q_f32(outptr + 4 * 6, _sum30);
                vst1q_f32(outptr + 4 * 7, _sum31);
                vst1q_f32(outptr + 4 * 8, _sum40);
                vst1q_f32(outptr + 4 * 9, _sum41);
                vst1q_f32(outptr + 4 * 10, _sum50);
                vst1q_f32(outptr + 4 * 11, _sum51);
                vst1q_f32(outptr + 4 * 12, _sum60);
                vst1q_f32(outptr + 4 * 13, _sum61);
                vst1q_f32(outptr + 4 * 14, _sum70);
                vst1q_f32(outptr + 4 * 15, _sum71);
                vst1q_f32(outptr + 4 * 16, _sum80);
                vst1q_f32(outptr + 4 * 17, _sum81);
                vst1q_f32(outptr + 4 * 18, _sum90);
                vst1q_f32(outptr + 4 * 19, _sum91);
                vst1q_f32(outptr + 4 * 20, _suma0);
                vst1q_f32(outptr + 4 * 21, _suma1);
                vst1q_f32(outptr + 4 * 22, _sumb0);
                vst1q_f32(outptr + 4 * 23, _sumb1);
            }

            outptr += 96;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

#if 0  //NCNN_GNU_INLINE_ASM
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                "subs   %0, %0, #192                \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "add    x4, %8, #16                 \n"
                "ld1    {v16.4s}, [%8]              \n"
                "ld1    {v24.4s}, [x4]              \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v16.16b, v16.16b, v16.16b   \n"
                "eor    v24.16b, v24.16b, v24.16b   \n"

                "2:                                 \n"
                "mov    v17.16b, v16.16b            \n"
                "mov    v18.16b, v16.16b            \n"
                "mov    v19.16b, v16.16b            \n"
                "mov    v20.16b, v16.16b            \n"
                "mov    v21.16b, v16.16b            \n"
                "mov    v22.16b, v16.16b            \n"
                "mov    v23.16b, v16.16b            \n"

                "mov    v25.16b, v24.16b            \n"
                "mov    v26.16b, v24.16b            \n"
                "mov    v27.16b, v24.16b            \n"
                "mov    v28.16b, v24.16b            \n"
                "mov    v29.16b, v24.16b            \n"
                "mov    v30.16b, v24.16b            \n"
                "mov    v31.16b, v24.16b            \n"

                "3:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "4:                                 \n"
                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"
                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%1], #64 \n"
                "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                "fmla   v17.4s, v8.4s, v0.s[1]      \n"
                "fmla   v18.4s, v8.4s, v0.s[2]      \n"
                "fmla   v19.4s, v8.4s, v0.s[3]      \n"
                "fmla   v20.4s, v8.4s, v1.s[0]      \n"
                "fmla   v21.4s, v8.4s, v1.s[1]      \n"
                "fmla   v22.4s, v8.4s, v1.s[2]      \n"
                "fmla   v23.4s, v8.4s, v1.s[3]      \n"
                "fmla   v24.4s, v9.4s, v0.s[0]      \n"
                "fmla   v25.4s, v9.4s, v0.s[1]      \n"
                "fmla   v26.4s, v9.4s, v0.s[2]      \n"
                "fmla   v27.4s, v9.4s, v0.s[3]      \n"
                "fmla   v28.4s, v9.4s, v1.s[0]      \n"
                "fmla   v29.4s, v9.4s, v1.s[1]      \n"
                "fmla   v30.4s, v9.4s, v1.s[2]      \n"
                "fmla   v31.4s, v9.4s, v1.s[3]      \n"
                "fmla   v16.4s, v10.4s, v2.s[0]     \n"
                "fmla   v17.4s, v10.4s, v2.s[1]     \n"
                "fmla   v18.4s, v10.4s, v2.s[2]     \n"
                "fmla   v19.4s, v10.4s, v2.s[3]     \n"
                "fmla   v20.4s, v10.4s, v3.s[0]     \n"
                "fmla   v21.4s, v10.4s, v3.s[1]     \n"
                "fmla   v22.4s, v10.4s, v3.s[2]     \n"
                "fmla   v23.4s, v10.4s, v3.s[3]     \n"
                "fmla   v24.4s, v11.4s, v2.s[0]     \n"
                "fmla   v25.4s, v11.4s, v2.s[1]     \n"
                "fmla   v26.4s, v11.4s, v2.s[2]     \n"
                "fmla   v27.4s, v11.4s, v2.s[3]     \n"
                "fmla   v28.4s, v11.4s, v3.s[0]     \n"
                "fmla   v29.4s, v11.4s, v3.s[1]     \n"
                "fmla   v30.4s, v11.4s, v3.s[2]     \n"
                "fmla   v31.4s, v11.4s, v3.s[3]     \n"
                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2], #64 \n"
                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"
                "fmla   v16.4s, v12.4s, v4.s[0]     \n"
                "fmla   v17.4s, v12.4s, v4.s[1]     \n"
                "fmla   v18.4s, v12.4s, v4.s[2]     \n"
                "fmla   v19.4s, v12.4s, v4.s[3]     \n"
                "fmla   v20.4s, v12.4s, v5.s[0]     \n"
                "fmla   v21.4s, v12.4s, v5.s[1]     \n"
                "fmla   v22.4s, v12.4s, v5.s[2]     \n"
                "fmla   v23.4s, v12.4s, v5.s[3]     \n"
                "fmla   v24.4s, v13.4s, v4.s[0]     \n"
                "fmla   v25.4s, v13.4s, v4.s[1]     \n"
                "fmla   v26.4s, v13.4s, v4.s[2]     \n"
                "fmla   v27.4s, v13.4s, v4.s[3]     \n"
                "fmla   v28.4s, v13.4s, v5.s[0]     \n"
                "fmla   v29.4s, v13.4s, v5.s[1]     \n"
                "fmla   v30.4s, v13.4s, v5.s[2]     \n"
                "fmla   v31.4s, v13.4s, v5.s[3]     \n"
                "fmla   v16.4s, v14.4s, v6.s[0]     \n"
                "fmla   v17.4s, v14.4s, v6.s[1]     \n"
                "fmla   v18.4s, v14.4s, v6.s[2]     \n"
                "fmla   v19.4s, v14.4s, v6.s[3]     \n"
                "fmla   v20.4s, v14.4s, v7.s[0]     \n"
                "fmla   v21.4s, v14.4s, v7.s[1]     \n"
                "fmla   v22.4s, v14.4s, v7.s[2]     \n"
                "fmla   v23.4s, v14.4s, v7.s[3]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v24.4s, v15.4s, v6.s[0]     \n"
                "fmla   v25.4s, v15.4s, v6.s[1]     \n"
                "fmla   v26.4s, v15.4s, v6.s[2]     \n"
                "fmla   v27.4s, v15.4s, v6.s[3]     \n"
                "fmla   v28.4s, v15.4s, v7.s[0]     \n"
                "fmla   v29.4s, v15.4s, v7.s[1]     \n"
                "fmla   v30.4s, v15.4s, v7.s[2]     \n"
                "fmla   v31.4s, v15.4s, v7.s[3]     \n"
                "bne    4b                          \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1    {v0.4s, v1.4s}, [%2], #32   \n"
                "ld1    {v4.4s, v5.4s}, [%1], #32   \n"

                "fmla   v16.4s, v4.4s, v0.s[0]      \n"
                "fmla   v17.4s, v4.4s, v0.s[1]      \n"
                "fmla   v18.4s, v4.4s, v0.s[2]      \n"
                "fmla   v19.4s, v4.4s, v0.s[3]      \n"
                "fmla   v20.4s, v4.4s, v1.s[0]      \n"
                "fmla   v21.4s, v4.4s, v1.s[1]      \n"
                "fmla   v22.4s, v4.4s, v1.s[2]      \n"
                "fmla   v23.4s, v4.4s, v1.s[3]      \n"

                "subs   w4, w4, #1                  \n"

                "fmla   v24.4s, v5.4s, v0.s[0]      \n"
                "fmla   v25.4s, v5.4s, v0.s[1]      \n"
                "fmla   v26.4s, v5.4s, v0.s[2]      \n"
                "fmla   v27.4s, v5.4s, v0.s[3]      \n"
                "fmla   v28.4s, v5.4s, v1.s[0]      \n"
                "fmla   v29.4s, v5.4s, v1.s[1]      \n"
                "fmla   v30.4s, v5.4s, v1.s[2]      \n"
                "fmla   v31.4s, v5.4s, v1.s[3]      \n"

                "bne    6b                          \n"

                "7:                                 \n"
                "tst    %w11, #255                  \n"
                "beq    10f                         \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    8f                          \n"

                "lsl    w4, %w13, #2                \n"
                "add    x4, %3, w4, sxtw 2          \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%3], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%3], #64 \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [x4], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [x4] \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // transpose8x8
                "zip1   v14.4s, v16.4s, v17.4s      \n"
                "zip2   v15.4s, v16.4s, v17.4s      \n"
                "zip1   v16.4s, v18.4s, v19.4s      \n"
                "zip2   v17.4s, v18.4s, v19.4s      \n"
                "zip1   v18.4s, v20.4s, v21.4s      \n"
                "zip2   v19.4s, v20.4s, v21.4s      \n"
                "zip1   v20.4s, v22.4s, v23.4s      \n"
                "zip2   v21.4s, v22.4s, v23.4s      \n"

                "zip1   v22.4s, v24.4s, v25.4s      \n"
                "zip2   v23.4s, v24.4s, v25.4s      \n"
                "zip1   v24.4s, v26.4s, v27.4s      \n"
                "zip2   v25.4s, v26.4s, v27.4s      \n"
                "zip1   v26.4s, v28.4s, v29.4s      \n"
                "zip2   v27.4s, v28.4s, v29.4s      \n"
                "zip1   v28.4s, v30.4s, v31.4s      \n"
                "zip2   v29.4s, v30.4s, v31.4s      \n"

                "zip1   v0.2d, v14.2d, v16.2d       \n"
                "zip2   v2.2d, v14.2d, v16.2d       \n"
                "zip1   v4.2d, v15.2d, v17.2d       \n"
                "zip2   v6.2d, v15.2d, v17.2d       \n"
                "zip1   v1.2d, v18.2d, v20.2d       \n"
                "zip2   v3.2d, v18.2d, v20.2d       \n"
                "zip1   v5.2d, v19.2d, v21.2d       \n"
                "zip2   v7.2d, v19.2d, v21.2d       \n"

                "zip1   v8.2d, v22.2d, v24.2d       \n"
                "zip2   v10.2d, v22.2d, v24.2d      \n"
                "zip1   v12.2d, v23.2d, v25.2d      \n"
                "zip2   v14.2d, v23.2d, v25.2d      \n"
                "zip1   v9.2d, v26.2d, v28.2d       \n"
                "zip2   v11.2d, v26.2d, v28.2d      \n"
                "zip1   v13.2d, v27.2d, v29.2d      \n"
                "zip2   v15.2d, v27.2d, v29.2d      \n"

                "add    x4, %3, %w13, sxtw 2        \n"
                "st1    {v0.4s, v1.4s}, [%3], #32   \n"
                "st1    {v2.4s, v3.4s}, [x4]        \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v4.4s, v5.4s}, [x4]        \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v6.4s, v7.4s}, [x4]        \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v8.4s, v9.4s}, [x4]        \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v10.4s, v11.4s}, [x4]      \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v12.4s, v13.4s}, [x4]      \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v14.4s, v15.4s}, [x4]      \n"

                "9:                                 \n"
                "add    %0, %0, #256                \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum20;
            float32x4_t _sum21;
            float32x4_t _sum30;
            float32x4_t _sum31;
            float32x4_t _sum40;
            float32x4_t _sum41;
            float32x4_t _sum50;
            float32x4_t _sum51;
            float32x4_t _sum60;
            float32x4_t _sum61;
            float32x4_t _sum70;
            float32x4_t _sum71;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                    _sum10 = _sum00;
                    _sum11 = _sum01;
                    _sum20 = _sum00;
                    _sum21 = _sum01;
                    _sum30 = _sum00;
                    _sum31 = _sum01;
                    _sum40 = _sum00;
                    _sum41 = _sum01;
                    _sum50 = _sum00;
                    _sum51 = _sum01;
                    _sum60 = _sum00;
                    _sum61 = _sum01;
                    _sum70 = _sum00;
                    _sum71 = _sum01;
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                    _sum10 = vdupq_n_f32(0.f);
                    _sum11 = vdupq_n_f32(0.f);
                    _sum20 = vdupq_n_f32(0.f);
                    _sum21 = vdupq_n_f32(0.f);
                    _sum30 = vdupq_n_f32(0.f);
                    _sum31 = vdupq_n_f32(0.f);
                    _sum40 = vdupq_n_f32(0.f);
                    _sum41 = vdupq_n_f32(0.f);
                    _sum50 = vdupq_n_f32(0.f);
                    _sum51 = vdupq_n_f32(0.f);
                    _sum60 = vdupq_n_f32(0.f);
                    _sum61 = vdupq_n_f32(0.f);
                    _sum70 = vdupq_n_f32(0.f);
                    _sum71 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
                _sum20 = vld1q_f32(outptr + 4 * 4);
                _sum21 = vld1q_f32(outptr + 4 * 5);
                _sum30 = vld1q_f32(outptr + 4 * 6);
                _sum31 = vld1q_f32(outptr + 4 * 7);
                _sum40 = vld1q_f32(outptr + 4 * 8);
                _sum41 = vld1q_f32(outptr + 4 * 9);
                _sum50 = vld1q_f32(outptr + 4 * 10);
                _sum51 = vld1q_f32(outptr + 4 * 11);
                _sum60 = vld1q_f32(outptr + 4 * 12);
                _sum61 = vld1q_f32(outptr + 4 * 13);
                _sum70 = vld1q_f32(outptr + 4 * 14);
                _sum71 = vld1q_f32(outptr + 4 * 15);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = bfloat2float(vld1_u16(pA));
                float32x4_t _pA1 = bfloat2float(vld1_u16(pA + 4));

                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));

                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32(_sum71, _pA1, _pB1, 3);

                pA += 8;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum10));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum20));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum30));
                    vst1_u16(outptr0 + 4 * 4, float2bfloat(_sum40));
                    vst1_u16(outptr0 + 4 * 5, float2bfloat(_sum50));
                    vst1_u16(outptr0 + 4 * 6, float2bfloat(_sum60));
                    vst1_u16(outptr0 + 4 * 7, float2bfloat(_sum70));

                    vst1_u16(outptr0 + out_hstep * 4, float2bfloat(_sum01));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, float2bfloat(_sum11));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 2, float2bfloat(_sum21));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 3, float2bfloat(_sum31));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 4, float2bfloat(_sum41));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 5, float2bfloat(_sum51));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 6, float2bfloat(_sum61));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 7, float2bfloat(_sum71));

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    uint16x8_t _t0 = vcombine_u16(float2bfloat(_sum00), float2bfloat(_sum01));
                    uint16x8_t _t1 = vcombine_u16(float2bfloat(_sum10), float2bfloat(_sum11));
                    uint16x8_t _t2 = vcombine_u16(float2bfloat(_sum20), float2bfloat(_sum21));
                    uint16x8_t _t3 = vcombine_u16(float2bfloat(_sum30), float2bfloat(_sum31));
                    uint16x8_t _t4 = vcombine_u16(float2bfloat(_sum40), float2bfloat(_sum41));
                    uint16x8_t _t5 = vcombine_u16(float2bfloat(_sum50), float2bfloat(_sum51));
                    uint16x8_t _t6 = vcombine_u16(float2bfloat(_sum60), float2bfloat(_sum61));
                    uint16x8_t _t7 = vcombine_u16(float2bfloat(_sum70), float2bfloat(_sum71));
                    transpose8x8_u16(_t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7);

                    vst1q_u16(outptr0, _t0);
                    vst1q_u16(outptr0 + out_hstep, _t1);
                    vst1q_u16(outptr0 + out_hstep * 2, _t2);
                    vst1q_u16(outptr0 + out_hstep * 3, _t3);
                    vst1q_u16(outptr0 + out_hstep * 4, _t4);
                    vst1q_u16(outptr0 + out_hstep * 5, _t5);
                    vst1q_u16(outptr0 + out_hstep * 6, _t6);
                    vst1q_u16(outptr0 + out_hstep * 7, _t7);

                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                vst1q_f32(outptr + 4 * 4, _sum20);
                vst1q_f32(outptr + 4 * 5, _sum21);
                vst1q_f32(outptr + 4 * 6, _sum30);
                vst1q_f32(outptr + 4 * 7, _sum31);
                vst1q_f32(outptr + 4 * 8, _sum40);
                vst1q_f32(outptr + 4 * 9, _sum41);
                vst1q_f32(outptr + 4 * 10, _sum50);
                vst1q_f32(outptr + 4 * 11, _sum51);
                vst1q_f32(outptr + 4 * 12, _sum60);
                vst1q_f32(outptr + 4 * 13, _sum61);
                vst1q_f32(outptr + 4 * 14, _sum70);
                vst1q_f32(outptr + 4 * 15, _sum71);
            }

            outptr += 64;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

#if 0  //NCNN_GNU_INLINE_ASM
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                "subs   %0, %0, #64                 \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "add    x4, %8, #16                 \n"
                "ld1    {v24.4s}, [%8]              \n"
                "ld1    {v28.4s}, [x4]              \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v24.16b, v24.16b, v24.16b   \n"
                "eor    v28.16b, v28.16b, v28.16b   \n"

                "2:                                 \n"
                "mov    v25.16b, v24.16b            \n"
                "mov    v26.16b, v24.16b            \n"
                "mov    v27.16b, v24.16b            \n"

                "mov    v29.16b, v28.16b            \n"
                "mov    v30.16b, v28.16b            \n"
                "mov    v31.16b, v28.16b            \n"

                "3:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "4:                                 \n"
                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"
                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"
                "fmla   v24.4s, v4.4s, v0.s[0]      \n"
                "fmla   v25.4s, v4.4s, v0.s[1]      \n"
                "fmla   v26.4s, v4.4s, v0.s[2]      \n"
                "fmla   v27.4s, v4.4s, v0.s[3]      \n"
                "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                "fmla   v31.4s, v5.4s, v0.s[3]      \n"
                "fmla   v24.4s, v6.4s, v1.s[0]      \n"
                "fmla   v25.4s, v6.4s, v1.s[1]      \n"
                "fmla   v26.4s, v6.4s, v1.s[2]      \n"
                "fmla   v27.4s, v6.4s, v1.s[3]      \n"
                "fmla   v28.4s, v7.4s, v1.s[0]      \n"
                "fmla   v29.4s, v7.4s, v1.s[1]      \n"
                "fmla   v30.4s, v7.4s, v1.s[2]      \n"
                "fmla   v31.4s, v7.4s, v1.s[3]      \n"
                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%1], #64 \n"
                "fmla   v24.4s, v8.4s, v2.s[0]      \n"
                "fmla   v25.4s, v8.4s, v2.s[1]      \n"
                "fmla   v26.4s, v8.4s, v2.s[2]      \n"
                "fmla   v27.4s, v8.4s, v2.s[3]      \n"
                "fmla   v28.4s, v9.4s, v2.s[0]      \n"
                "fmla   v29.4s, v9.4s, v2.s[1]      \n"
                "fmla   v30.4s, v9.4s, v2.s[2]      \n"
                "fmla   v31.4s, v9.4s, v2.s[3]      \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v24.4s, v10.4s, v3.s[0]     \n"
                "fmla   v25.4s, v10.4s, v3.s[1]     \n"
                "fmla   v26.4s, v10.4s, v3.s[2]     \n"
                "fmla   v27.4s, v10.4s, v3.s[3]     \n"
                "fmla   v28.4s, v11.4s, v3.s[0]     \n"
                "fmla   v29.4s, v11.4s, v3.s[1]     \n"
                "fmla   v30.4s, v11.4s, v3.s[2]     \n"
                "fmla   v31.4s, v11.4s, v3.s[3]     \n"
                "bne    4b                          \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1    {v0.4s}, [%2], #16          \n"
                "ld1    {v4.4s, v5.4s}, [%1], #32   \n"
                "fmla   v24.4s, v4.4s, v0.s[0]      \n"
                "fmla   v25.4s, v4.4s, v0.s[1]      \n"
                "fmla   v26.4s, v4.4s, v0.s[2]      \n"
                "fmla   v27.4s, v4.4s, v0.s[3]      \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                "fmla   v31.4s, v5.4s, v0.s[3]      \n"
                "bne    6b                          \n"

                "7:                                 \n"
                "tst    %w11, #255                  \n"
                "beq    10f                         \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    8f                          \n"

                "lsl    w4, %w13, #2                \n"
                "add    x4, %3, w4, sxtw 2          \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%3], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [x4] \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // transpose8x4
                "zip1   v22.4s, v24.4s, v25.4s      \n"
                "zip2   v23.4s, v24.4s, v25.4s      \n"
                "zip1   v24.4s, v26.4s, v27.4s      \n"
                "zip2   v25.4s, v26.4s, v27.4s      \n"
                "zip1   v26.4s, v28.4s, v29.4s      \n"
                "zip2   v27.4s, v28.4s, v29.4s      \n"
                "zip1   v28.4s, v30.4s, v31.4s      \n"
                "zip2   v29.4s, v30.4s, v31.4s      \n"

                "zip1   v0.2d, v22.2d, v24.2d       \n"
                "zip2   v1.2d, v22.2d, v24.2d       \n"
                "zip1   v2.2d, v23.2d, v25.2d       \n"
                "zip2   v3.2d, v23.2d, v25.2d       \n"
                "zip1   v4.2d, v26.2d, v28.2d       \n"
                "zip2   v5.2d, v26.2d, v28.2d       \n"
                "zip1   v6.2d, v27.2d, v29.2d       \n"
                "zip2   v7.2d, v27.2d, v29.2d       \n"

                "add    x4, %3, %w13, sxtw 2        \n"
                "st1    {v0.4s}, [%3], #16          \n"
                "st1    {v1.4s}, [x4]               \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v2.4s}, [x4]               \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v3.4s}, [x4]               \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v4.4s}, [x4]               \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v5.4s}, [x4]               \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v6.4s}, [x4]               \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v7.4s}, [x4]               \n"

                "9:                                 \n"
                "add    %0, %0, #128                \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum20;
            float32x4_t _sum21;
            float32x4_t _sum30;
            float32x4_t _sum31;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                    _sum10 = _sum00;
                    _sum11 = _sum01;
                    _sum20 = _sum00;
                    _sum21 = _sum01;
                    _sum30 = _sum00;
                    _sum31 = _sum01;
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                    _sum10 = vdupq_n_f32(0.f);
                    _sum11 = vdupq_n_f32(0.f);
                    _sum20 = vdupq_n_f32(0.f);
                    _sum21 = vdupq_n_f32(0.f);
                    _sum30 = vdupq_n_f32(0.f);
                    _sum31 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
                _sum20 = vld1q_f32(outptr + 4 * 4);
                _sum21 = vld1q_f32(outptr + 4 * 5);
                _sum30 = vld1q_f32(outptr + 4 * 6);
                _sum31 = vld1q_f32(outptr + 4 * 7);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = bfloat2float(vld1_u16(pA));
                float32x4_t _pA1 = bfloat2float(vld1_u16(pA + 4));

                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));

                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);

                pA += 8;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum10));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum20));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum30));

                    vst1_u16(outptr0 + out_hstep * 4, float2bfloat(_sum01));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, float2bfloat(_sum11));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 2, float2bfloat(_sum21));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 3, float2bfloat(_sum31));

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    uint16x8_t _t0 = vcombine_u16(float2bfloat(_sum00), float2bfloat(_sum01));
                    uint16x8_t _t1 = vcombine_u16(float2bfloat(_sum10), float2bfloat(_sum11));
                    uint16x8_t _t2 = vcombine_u16(float2bfloat(_sum20), float2bfloat(_sum21));
                    uint16x8_t _t3 = vcombine_u16(float2bfloat(_sum30), float2bfloat(_sum31));
                    transpose8x4_u16(_t0, _t1, _t2, _t3);

                    vst1_u16(outptr0, vget_low_u16(_t0));
                    vst1_u16(outptr0 + out_hstep * 1, vget_high_u16(_t0));
                    vst1_u16(outptr0 + out_hstep * 2, vget_low_u16(_t1));
                    vst1_u16(outptr0 + out_hstep * 3, vget_high_u16(_t1));
                    vst1_u16(outptr0 + out_hstep * 4, vget_low_u16(_t2));
                    vst1_u16(outptr0 + out_hstep * 5, vget_high_u16(_t2));
                    vst1_u16(outptr0 + out_hstep * 6, vget_low_u16(_t3));
                    vst1_u16(outptr0 + out_hstep * 7, vget_high_u16(_t3));

                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                vst1q_f32(outptr + 4 * 4, _sum20);
                vst1q_f32(outptr + 4 * 5, _sum21);
                vst1q_f32(outptr + 4 * 6, _sum30);
                vst1q_f32(outptr + 4 * 7, _sum31);
            }

            outptr += 32;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

#if 0  //NCNN_GNU_INLINE_ASM
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "add    x4, %8, #16                 \n"
                "ld1    {v28.4s}, [%8]              \n"
                "ld1    {v30.4s}, [x4]              \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"

                "2:                                 \n"
                "mov    v29.16b, v28.16b            \n"
                "mov    v31.16b, v30.16b            \n"

                "3:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "4:                                 \n"
                "prfm   pldl1keep, [%2, #256]       \n"
                "ld1    {v0.4s, v1.4s}, [%2], #32   \n"
                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"
                "fmla   v28.4s, v4.4s, v0.s[0]      \n"
                "fmla   v29.4s, v4.4s, v0.s[1]      \n"
                "fmla   v30.4s, v5.4s, v0.s[0]      \n"
                "fmla   v31.4s, v5.4s, v0.s[1]      \n"
                "fmla   v28.4s, v6.4s, v0.s[2]      \n"
                "fmla   v29.4s, v6.4s, v0.s[3]      \n"
                "fmla   v30.4s, v7.4s, v0.s[2]      \n"
                "fmla   v31.4s, v7.4s, v0.s[3]      \n"
                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%1], #64 \n"
                "fmla   v28.4s, v8.4s, v1.s[0]      \n"
                "fmla   v29.4s, v8.4s, v1.s[1]      \n"
                "fmla   v30.4s, v9.4s, v1.s[0]      \n"
                "fmla   v31.4s, v9.4s, v1.s[1]      \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v28.4s, v10.4s, v1.s[2]     \n"
                "fmla   v29.4s, v10.4s, v1.s[3]     \n"
                "fmla   v30.4s, v11.4s, v1.s[2]     \n"
                "fmla   v31.4s, v11.4s, v1.s[3]     \n"
                "bne    4b                          \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1    {v0.2s}, [%2], #8           \n"
                "ld1    {v4.4s, v5.4s}, [%1], #32   \n"
                "fmla   v28.4s, v4.4s, v0.s[0]      \n"
                "fmla   v29.4s, v4.4s, v0.s[1]      \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v30.4s, v5.4s, v0.s[0]      \n"
                "fmla   v31.4s, v5.4s, v0.s[1]      \n"
                "bne    6b                          \n"

                "7:                                 \n"
                "tst    %w11, #255                  \n"
                "beq    10f                         \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    8f                          \n"

                "lsl    w4, %w13, #2                \n"
                "add    x4, %3, w4, sxtw 2          \n"
                "st1    {v28.4s, v29.4s}, [%3], #32 \n"
                "st1    {v30.4s, v31.4s}, [x4]      \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // transpose8x2
                "zip1   v0.4s, v28.4s, v29.4s       \n"
                "zip2   v2.4s, v28.4s, v29.4s       \n"
                "zip1   v4.4s, v30.4s, v31.4s       \n"
                "zip2   v6.4s, v30.4s, v31.4s       \n"

                "mov    v1.d[0], v0.d[1]            \n"
                "mov    v3.d[0], v2.d[1]            \n"
                "mov    v5.d[0], v4.d[1]            \n"
                "mov    v7.d[0], v6.d[1]            \n"

                "add    x4, %3, %w13, sxtw 2        \n"
                "st1    {v0.2s}, [%3], #8           \n"
                "st1    {v1.2s}, [x4]               \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v2.2s}, [x4]               \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v3.2s}, [x4]               \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v4.2s}, [x4]               \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v5.2s}, [x4]               \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v6.2s}, [x4]               \n"
                "add    x4, x4, %w13, sxtw 2        \n"
                "st1    {v7.2s}, [x4]               \n"

                "9:                                 \n"
                "add    %0, %0, #64                 \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                    _sum10 = _sum00;
                    _sum11 = _sum01;
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                    _sum10 = vdupq_n_f32(0.f);
                    _sum11 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = bfloat2float(vld1_u16(pA));
                float32x4_t _pA1 = bfloat2float(vld1_u16(pA + 4));

                float32x2_t _pB0 = vget_low_f32(bfloat2float(vld1_u16(pB)));

                _sum00 = vfmaq_lane_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_lane_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_lane_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_lane_f32(_sum11, _pA1, _pB0, 1);

                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum10));

                    vst1_u16(outptr0 + out_hstep * 4, float2bfloat(_sum01));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, float2bfloat(_sum11));
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[8];
                    unsigned short sum1[8];
                    vst1_u16(sum0, float2bfloat(_sum00));
                    vst1_u16(sum0 + 4, float2bfloat(_sum01));
                    vst1_u16(sum1, float2bfloat(_sum10));
                    vst1_u16(sum1 + 4, float2bfloat(_sum11));

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];

                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0[out_hstep * 4 + 1] = sum1[4];
                    outptr0[out_hstep * 5 + 1] = sum1[5];
                    outptr0[out_hstep * 6 + 1] = sum1[6];
                    outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0 += 2;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
            }

            outptr += 16;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj < max_jj; jj += 1)
        {
            const unsigned short* pA = pAT;

            float32x4_t _sum00;
            float32x4_t _sum01;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = bfloat2float(vld1_u16(pA));
                float32x4_t _pA1 = bfloat2float(vld1_u16(pA + 4));

                float32x4_t _pB = bfloat2float(vld1_dup_u16(pB));

                _sum00 = vfmaq_f32(_sum00, _pA0, _pB);
                _sum01 = vfmaq_f32(_sum01, _pA1, _pB);

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + out_hstep * 4, float2bfloat(_sum01));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[8];
                    vst1_u16(sum0, float2bfloat(_sum00));
                    vst1_u16(sum0 + 4, float2bfloat(_sum01));

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep * 1] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0++;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;
            float32x4_t _sum4;
            float32x4_t _sum5;
            float32x4_t _sum6;
            float32x4_t _sum7;
            float32x4_t _sum8;
            float32x4_t _sum9;
            float32x4_t _suma;
            float32x4_t _sumb;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vld1q_f32(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                    _sum8 = _sum0;
                    _sum9 = _sum0;
                    _suma = _sum0;
                    _sumb = _sum0;
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                    _sum2 = vdupq_n_f32(0.f);
                    _sum3 = vdupq_n_f32(0.f);
                    _sum4 = vdupq_n_f32(0.f);
                    _sum5 = vdupq_n_f32(0.f);
                    _sum6 = vdupq_n_f32(0.f);
                    _sum7 = vdupq_n_f32(0.f);
                    _sum8 = vdupq_n_f32(0.f);
                    _sum9 = vdupq_n_f32(0.f);
                    _suma = vdupq_n_f32(0.f);
                    _sumb = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4 * 1);
                _sum2 = vld1q_f32(outptr + 4 * 2);
                _sum3 = vld1q_f32(outptr + 4 * 3);
                _sum4 = vld1q_f32(outptr + 4 * 4);
                _sum5 = vld1q_f32(outptr + 4 * 5);
                _sum6 = vld1q_f32(outptr + 4 * 6);
                _sum7 = vld1q_f32(outptr + 4 * 7);
                _sum8 = vld1q_f32(outptr + 4 * 8);
                _sum9 = vld1q_f32(outptr + 4 * 9);
                _suma = vld1q_f32(outptr + 4 * 10);
                _sumb = vld1q_f32(outptr + 4 * 11);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));
                float32x4_t _pB2 = bfloat2float(vld1_u16(pB + 8));

#if __aarch64__
                _sum0 = vfmaq_laneq_f32(_sum0, _pA, _pB0, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _pA, _pB0, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _pA, _pB0, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _pA, _pB0, 3);
                _sum4 = vfmaq_laneq_f32(_sum4, _pA, _pB1, 0);
                _sum5 = vfmaq_laneq_f32(_sum5, _pA, _pB1, 1);
                _sum6 = vfmaq_laneq_f32(_sum6, _pA, _pB1, 2);
                _sum7 = vfmaq_laneq_f32(_sum7, _pA, _pB1, 3);
                _sum8 = vfmaq_laneq_f32(_sum8, _pA, _pB2, 0);
                _sum9 = vfmaq_laneq_f32(_sum9, _pA, _pB2, 1);
                _suma = vfmaq_laneq_f32(_suma, _pA, _pB2, 2);
                _sumb = vfmaq_laneq_f32(_sumb, _pA, _pB2, 3);
#else  // __aarch64__
                _sum0 = vmlaq_lane_f32(_sum0, _pA, vget_low_f32(_pB0), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, vget_low_f32(_pB0), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _pA, vget_high_f32(_pB0), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _pA, vget_high_f32(_pB0), 1);
                _sum4 = vmlaq_lane_f32(_sum4, _pA, vget_low_f32(_pB1), 0);
                _sum5 = vmlaq_lane_f32(_sum5, _pA, vget_low_f32(_pB1), 1);
                _sum6 = vmlaq_lane_f32(_sum6, _pA, vget_high_f32(_pB1), 0);
                _sum7 = vmlaq_lane_f32(_sum7, _pA, vget_high_f32(_pB1), 1);
                _sum8 = vmlaq_lane_f32(_sum8, _pA, vget_low_f32(_pB2), 0);
                _sum9 = vmlaq_lane_f32(_sum9, _pA, vget_low_f32(_pB2), 1);
                _suma = vmlaq_lane_f32(_suma, _pA, vget_high_f32(_pB2), 0);
                _sumb = vmlaq_lane_f32(_sumb, _pA, vget_high_f32(_pB2), 1);
#endif // __aarch64__

                pA += 4;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum2));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum3));
                    vst1_u16(outptr0 + 4 * 4, float2bfloat(_sum4));
                    vst1_u16(outptr0 + 4 * 5, float2bfloat(_sum5));
                    vst1_u16(outptr0 + 4 * 6, float2bfloat(_sum6));
                    vst1_u16(outptr0 + 4 * 7, float2bfloat(_sum7));
                    vst1_u16(outptr0 + 4 * 8, float2bfloat(_sum8));
                    vst1_u16(outptr0 + 4 * 9, float2bfloat(_sum9));
                    vst1_u16(outptr0 + 4 * 10, float2bfloat(_suma));
                    vst1_u16(outptr0 + 4 * 11, float2bfloat(_sumb));
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    uint16x4_t _t0 = float2bfloat(_sum0);
                    uint16x4_t _t1 = float2bfloat(_sum1);
                    uint16x4_t _t2 = float2bfloat(_sum2);
                    uint16x4_t _t3 = float2bfloat(_sum3);
                    uint16x4_t _t4 = float2bfloat(_sum4);
                    uint16x4_t _t5 = float2bfloat(_sum5);
                    uint16x4_t _t6 = float2bfloat(_sum6);
                    uint16x4_t _t7 = float2bfloat(_sum7);
                    uint16x4_t _t8 = float2bfloat(_sum8);
                    uint16x4_t _t9 = float2bfloat(_sum9);
                    uint16x4_t _ta = float2bfloat(_suma);
                    uint16x4_t _tb = float2bfloat(_sumb);
                    transpose4x12_u16(_t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9, _ta, _tb);

                    vst1_u16(outptr0, _t0);
                    vst1_u16(outptr0 + 4, _t1);
                    vst1_u16(outptr0 + 8, _t2);
                    vst1_u16(outptr0 + out_hstep, _t3);
                    vst1_u16(outptr0 + out_hstep + 4, _t4);
                    vst1_u16(outptr0 + out_hstep + 8, _t5);
                    vst1_u16(outptr0 + out_hstep * 2, _t6);
                    vst1_u16(outptr0 + out_hstep * 2 + 4, _t7);
                    vst1_u16(outptr0 + out_hstep * 2 + 8, _t8);
                    vst1_u16(outptr0 + out_hstep * 3, _t9);
                    vst1_u16(outptr0 + out_hstep * 3 + 4, _ta);
                    vst1_u16(outptr0 + out_hstep * 3 + 8, _tb);
                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
                vst1q_f32(outptr + 4 * 4, _sum4);
                vst1q_f32(outptr + 4 * 5, _sum5);
                vst1q_f32(outptr + 4 * 6, _sum6);
                vst1q_f32(outptr + 4 * 7, _sum7);
                vst1q_f32(outptr + 4 * 8, _sum8);
                vst1q_f32(outptr + 4 * 9, _sum9);
                vst1q_f32(outptr + 4 * 10, _suma);
                vst1q_f32(outptr + 4 * 11, _sumb);
            }

            outptr += 48;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;
            float32x4_t _sum4;
            float32x4_t _sum5;
            float32x4_t _sum6;
            float32x4_t _sum7;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vld1q_f32(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                    _sum2 = vdupq_n_f32(0.f);
                    _sum3 = vdupq_n_f32(0.f);
                    _sum4 = vdupq_n_f32(0.f);
                    _sum5 = vdupq_n_f32(0.f);
                    _sum6 = vdupq_n_f32(0.f);
                    _sum7 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4 * 1);
                _sum2 = vld1q_f32(outptr + 4 * 2);
                _sum3 = vld1q_f32(outptr + 4 * 3);
                _sum4 = vld1q_f32(outptr + 4 * 4);
                _sum5 = vld1q_f32(outptr + 4 * 5);
                _sum6 = vld1q_f32(outptr + 4 * 6);
                _sum7 = vld1q_f32(outptr + 4 * 7);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));

#if __aarch64__
                _sum0 = vfmaq_laneq_f32(_sum0, _pA, _pB0, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _pA, _pB0, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _pA, _pB0, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _pA, _pB0, 3);
                _sum4 = vfmaq_laneq_f32(_sum4, _pA, _pB1, 0);
                _sum5 = vfmaq_laneq_f32(_sum5, _pA, _pB1, 1);
                _sum6 = vfmaq_laneq_f32(_sum6, _pA, _pB1, 2);
                _sum7 = vfmaq_laneq_f32(_sum7, _pA, _pB1, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pA, vget_low_f32(_pB0), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, vget_low_f32(_pB0), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _pA, vget_high_f32(_pB0), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _pA, vget_high_f32(_pB0), 1);
                _sum4 = vmlaq_lane_f32(_sum4, _pA, vget_low_f32(_pB1), 0);
                _sum5 = vmlaq_lane_f32(_sum5, _pA, vget_low_f32(_pB1), 1);
                _sum6 = vmlaq_lane_f32(_sum6, _pA, vget_high_f32(_pB1), 0);
                _sum7 = vmlaq_lane_f32(_sum7, _pA, vget_high_f32(_pB1), 1);
#endif

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum2));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum3));
                    vst1_u16(outptr0 + 4 * 4, float2bfloat(_sum4));
                    vst1_u16(outptr0 + 4 * 5, float2bfloat(_sum5));
                    vst1_u16(outptr0 + 4 * 6, float2bfloat(_sum6));
                    vst1_u16(outptr0 + 4 * 7, float2bfloat(_sum7));
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    uint16x4_t _t0 = float2bfloat(_sum0);
                    uint16x4_t _t1 = float2bfloat(_sum1);
                    uint16x4_t _t2 = float2bfloat(_sum2);
                    uint16x4_t _t3 = float2bfloat(_sum3);
                    uint16x4_t _t4 = float2bfloat(_sum4);
                    uint16x4_t _t5 = float2bfloat(_sum5);
                    uint16x4_t _t6 = float2bfloat(_sum6);
                    uint16x4_t _t7 = float2bfloat(_sum7);
                    transpose4x8_u16(_t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7);

                    vst1_u16(outptr0, _t0);
                    vst1_u16(outptr0 + 4, _t1);
                    vst1_u16(outptr0 + out_hstep, _t2);
                    vst1_u16(outptr0 + out_hstep + 4, _t3);
                    vst1_u16(outptr0 + out_hstep * 2, _t4);
                    vst1_u16(outptr0 + out_hstep * 2 + 4, _t5);
                    vst1_u16(outptr0 + out_hstep * 3, _t6);
                    vst1_u16(outptr0 + out_hstep * 3 + 4, _t7);
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
                vst1q_f32(outptr + 4 * 4, _sum4);
                vst1q_f32(outptr + 4 * 5, _sum5);
                vst1q_f32(outptr + 4 * 6, _sum6);
                vst1q_f32(outptr + 4 * 7, _sum7);
            }

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vld1q_f32(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                    _sum2 = vdupq_n_f32(0.f);
                    _sum3 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4 * 1);
                _sum2 = vld1q_f32(outptr + 4 * 2);
                _sum3 = vld1q_f32(outptr + 4 * 3);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                float32x4_t _pB = bfloat2float(vld1_u16(pB));

#if __aarch64__
                _sum0 = vfmaq_laneq_f32(_sum0, _pA, _pB, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _pA, _pB, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _pA, _pB, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _pA, _pB, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pA, vget_low_f32(_pB), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, vget_low_f32(_pB), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _pA, vget_high_f32(_pB), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _pA, vget_high_f32(_pB), 1);
#endif

                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum2));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum3));
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    uint16x4_t _t0 = float2bfloat(_sum0);
                    uint16x4_t _t1 = float2bfloat(_sum1);
                    uint16x4_t _t2 = float2bfloat(_sum2);
                    uint16x4_t _t3 = float2bfloat(_sum3);
                    transpose4x4_u16(_t0, _t1, _t2, _t3);

                    vst1_u16(outptr0, _t0);
                    vst1_u16(outptr0 + out_hstep * 1, _t1);
                    vst1_u16(outptr0 + out_hstep * 2, _t2);
                    vst1_u16(outptr0 + out_hstep * 3, _t3);
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
            }

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vld1q_f32(pC);
                    _sum1 = _sum0;
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                float32x2_t _pB = vget_low_f32(bfloat2float(vld1_u16(pB)));

#if __aarch64__
                _sum0 = vfmaq_lane_f32(_sum0, _pA, _pB, 0);
                _sum1 = vfmaq_lane_f32(_sum1, _pA, _pB, 1);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pA, _pB, 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, _pB, 1);
#endif

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[4];
                    unsigned short sum1[4];
                    vst1_u16(sum0, float2bfloat(_sum0));
                    vst1_u16(sum1, float2bfloat(_sum1));

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0 += 2;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x4_t _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vld1q_f32(pC);
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                float32x4_t _pB = bfloat2float(vdup_n_u16(pB[0]));

#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA, _pB);
#else
                _sum0 = vmlaq_f32(_sum0, _pA, _pB);
#endif

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[4];
                    vst1_u16(sum0, float2bfloat(_sum0));

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0++;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum02;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum12;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vdupq_n_f32(pC[0]);
                    _sum01 = vdupq_n_f32(pC[0]);
                    _sum02 = vdupq_n_f32(pC[0]);
                    _sum10 = vdupq_n_f32(pC[1]);
                    _sum11 = vdupq_n_f32(pC[1]);
                    _sum12 = vdupq_n_f32(pC[1]);
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                    _sum02 = vdupq_n_f32(0.f);
                    _sum10 = vdupq_n_f32(0.f);
                    _sum11 = vdupq_n_f32(0.f);
                    _sum12 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                float32x4x2_t _tmp01 = vld2q_f32(outptr);
                float32x4x2_t _tmp23 = vld2q_f32(outptr + 8);
                float32x4x2_t _tmp45 = vld2q_f32(outptr + 16);
                _sum00 = _tmp01.val[0];
                _sum01 = _tmp23.val[0];
                _sum02 = _tmp45.val[0];
                _sum10 = _tmp01.val[1];
                _sum11 = _tmp23.val[1];
                _sum12 = _tmp45.val[1];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));
                float32x4_t _pB2 = bfloat2float(vld1_u16(pB + 8));

                float32x2_t _pA = vget_low_f32(bfloat2float(vld1_u16(pA)));
#if __aarch64__
                _sum00 = vfmaq_lane_f32(_sum00, _pB0, _pA, 0);
                _sum01 = vfmaq_lane_f32(_sum01, _pB1, _pA, 0);
                _sum02 = vfmaq_lane_f32(_sum02, _pB2, _pA, 0);
                _sum10 = vfmaq_lane_f32(_sum10, _pB0, _pA, 1);
                _sum11 = vfmaq_lane_f32(_sum11, _pB1, _pA, 1);
                _sum12 = vfmaq_lane_f32(_sum12, _pB2, _pA, 1);
#else
                _sum00 = vmlaq_lane_f32(_sum00, _pB0, _pA, 0);
                _sum01 = vmlaq_lane_f32(_sum01, _pB1, _pA, 0);
                _sum02 = vmlaq_lane_f32(_sum02, _pB2, _pA, 0);
                _sum10 = vmlaq_lane_f32(_sum10, _pB0, _pA, 1);
                _sum11 = vmlaq_lane_f32(_sum11, _pB1, _pA, 1);
                _sum12 = vmlaq_lane_f32(_sum12, _pB2, _pA, 1);
#endif

                pA += 2;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum01));
                    vst1_u16(outptr0 + 8, float2bfloat(_sum02));
                    vst1_u16(outptr0 + out_hstep, float2bfloat(_sum10));
                    vst1_u16(outptr0 + out_hstep + 4, float2bfloat(_sum11));
                    vst1_u16(outptr0 + out_hstep + 8, float2bfloat(_sum12));
                    outptr0 += 12;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum00;
                _tmp01.val[1] = _sum10;
                float32x4x2_t _tmp23;
                _tmp23.val[0] = _sum01;
                _tmp23.val[1] = _sum11;
                float32x4x2_t _tmp45;
                _tmp45.val[0] = _sum02;
                _tmp45.val[1] = _sum12;
                vst2q_f32(outptr, _tmp01);
                vst2q_f32(outptr + 8, _tmp23);
                vst2q_f32(outptr + 16, _tmp45);
            }

            outptr += 24;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vdupq_n_f32(pC[0]);
                    _sum01 = vdupq_n_f32(pC[0]);
                    _sum10 = vdupq_n_f32(pC[1]);
                    _sum11 = vdupq_n_f32(pC[1]);
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                    _sum10 = vdupq_n_f32(0.f);
                    _sum11 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                float32x4x2_t _tmp01 = vld2q_f32(outptr);
                float32x4x2_t _tmp23 = vld2q_f32(outptr + 8);
                _sum00 = _tmp01.val[0];
                _sum01 = _tmp23.val[0];
                _sum10 = _tmp01.val[1];
                _sum11 = _tmp23.val[1];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));

                float32x2_t _pA = vget_low_f32(bfloat2float(vld1_u16(pA)));
#if __aarch64__
                _sum00 = vfmaq_lane_f32(_sum00, _pB0, _pA, 0);
                _sum01 = vfmaq_lane_f32(_sum01, _pB1, _pA, 0);
                _sum10 = vfmaq_lane_f32(_sum10, _pB0, _pA, 1);
                _sum11 = vfmaq_lane_f32(_sum11, _pB1, _pA, 1);
#else
                _sum00 = vmlaq_lane_f32(_sum00, _pB0, _pA, 0);
                _sum01 = vmlaq_lane_f32(_sum01, _pB1, _pA, 0);
                _sum10 = vmlaq_lane_f32(_sum10, _pB0, _pA, 1);
                _sum11 = vmlaq_lane_f32(_sum11, _pB1, _pA, 1);
#endif

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum01));
                    vst1_u16(outptr0 + out_hstep, float2bfloat(_sum10));
                    vst1_u16(outptr0 + out_hstep + 4, float2bfloat(_sum11));
                    outptr0 += 8;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum00;
                _tmp01.val[1] = _sum10;
                float32x4x2_t _tmp23;
                _tmp23.val[0] = _sum01;
                _tmp23.val[1] = _sum11;
                vst2q_f32(outptr, _tmp01);
                vst2q_f32(outptr + 8, _tmp23);
            }

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vdupq_n_f32(pC[0]);
                    _sum1 = vdupq_n_f32(pC[1]);
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                float32x4x2_t _tmp01 = vld2q_f32(outptr);
                _sum0 = _tmp01.val[0];
                _sum1 = _tmp01.val[1];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB = bfloat2float(vld1_u16(pB));

                float32x2_t _pA = vget_low_f32(bfloat2float(vld1_u16(pA)));
#if __aarch64__
                _sum0 = vfmaq_lane_f32(_sum0, _pB, _pA, 0);
                _sum1 = vfmaq_lane_f32(_sum1, _pB, _pA, 1);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pB, _pA, 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pB, _pA, 1);
#endif

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + out_hstep, float2bfloat(_sum1));
                    outptr0 += 4;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum0;
                _tmp01.val[1] = _sum1;
                vst2q_f32(outptr, _tmp01);
            }

            outptr += 8;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00;
            float sum01;
            float sum10;
            float sum11;

            if (k == 0)
            {
                if (pC)
                {
                    sum00 = pC[0];
                    sum01 = pC[1];
                    sum10 = pC[0];
                    sum11 = pC[1];
                }
                else
                {
                    sum00 = 0.f;
                    sum01 = 0.f;
                    sum10 = 0.f;
                    sum11 = 0.f;
                }
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum00 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                sum01 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pB[0]);
                sum10 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[1]);
                sum11 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pB[1]);

                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum00);
                    outptr0[1] = float32_to_bfloat16(sum10);
                    outptr0[out_hstep] = float32_to_bfloat16(sum01);
                    outptr0[out_hstep + 1] = float32_to_bfloat16(sum11);
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum10;
                outptr[3] = sum11;
            }

            outptr += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[1];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                sum1 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pB[0]);
                pA += 2;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum0);
                    outptr0[out_hstep] = float32_to_bfloat16(sum1);
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vdupq_n_f32(pC[0]);
                    _sum1 = vdupq_n_f32(pC[0]);
                    _sum2 = vdupq_n_f32(pC[0]);
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                    _sum2 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
                _sum2 = vld1q_f32(outptr + 8);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));
                float32x4_t _pB2 = bfloat2float(vld1_u16(pB + 8));

                float32x4_t _pA0 = bfloat2float(vdup_n_u16(pA[0]));
#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA0, _pB1);
                _sum2 = vfmaq_f32(_sum2, _pA0, _pB2);
#else
                _sum0 = vmlaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vmlaq_f32(_sum1, _pA0, _pB1);
                _sum2 = vmlaq_f32(_sum2, _pA0, _pB2);
#endif

                pA += 1;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    vst1_u16(outptr0 + 8, float2bfloat(_sum2));
                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 8, _sum2);
            }

            outptr += 12;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vdupq_n_f32(pC[0]);
                    _sum1 = vdupq_n_f32(pC[0]);
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));

                float32x4_t _pA0 = bfloat2float(vdup_n_u16(pA[0]));
#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA0, _pB1);
#else
                _sum0 = vmlaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vmlaq_f32(_sum1, _pA0, _pB1);
#endif

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum;

            if (k == 0)
            {
                if (pC)
                {
                    _sum = vdupq_n_f32(pC[0]);
                }
                else
                {
                    _sum = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum = vld1q_f32(outptr);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB = bfloat2float(vld1_u16(pB));
                float32x4_t _pA = bfloat2float(vdup_n_u16(pA[0]));

#if __aarch64__
                _sum = vfmaq_f32(_sum, _pA, _pB);
#else
                _sum = vmlaq_f32(_sum, _pA, _pB);
#endif

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, float2bfloat(_sum));
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum);
            }

            outptr += 4;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[0];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                sum1 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[1]);

                pA += 1;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum0);
                    outptr0[1] = float32_to_bfloat16(sum1);
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum;

            if (k == 0)
            {
                if (pC)
                {
                    sum = pC[0];
                }
                else
                {
                    sum = 0.f;
                }
            }
            else
            {
                sum = outptr[0];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);

                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum);
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void convolution_im2col_input_tile_conv1x1s1d1_bf16s(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;

    unsigned short* pp = B;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                // transpose4x12
#if 0  //NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0] \n"
                    "st1    {v0.4s}, [%1], #16          \n"
                    "st1    {v4.4s}, [%1], #16          \n"
                    "st1    {v8.4s}, [%1], #16          \n"
                    "sub    %0, %0, #128                \n"
                    "st1    {v1.4s}, [%1], #16          \n"
                    "st1    {v5.4s}, [%1], #16          \n"
                    "st1    {v9.4s}, [%1], #16          \n"
                    "st1    {v2.4s}, [%1], #16          \n"
                    "st1    {v6.4s}, [%1], #16          \n"
                    "st1    {v10.4s}, [%1], #16         \n"
                    "st1    {v3.4s}, [%1], #16          \n"
                    "st1    {v7.4s}, [%1], #16          \n"
                    "st1    {v11.4s}, [%1], #16         \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else  // NCNN_GNU_INLINE_ASM
                uint16x8_t _r0 = vld1q_u16(p0);
                uint16x8_t _r1 = vld1q_u16(p0 + 8);
                uint16x8_t _r2 = vld1q_u16(p0 + 8 * 2);
                uint16x8_t _r3 = vld1q_u16(p0 + 8 * 3);
                uint16x8_t _r4 = vld1q_u16(p0 + 8 * 4);
                uint16x8_t _r5 = vld1q_u16(p0 + 8 * 5);
                uint16x8_t _r6 = vld1q_u16(p0 + 8 * 6);
                uint16x8_t _r7 = vld1q_u16(p0 + 8 * 7);
                uint16x8_t _r8 = vld1q_u16(p0 + 8 * 8);
                uint16x8_t _r9 = vld1q_u16(p0 + 8 * 9);
                uint16x8_t _ra = vld1q_u16(p0 + 8 * 10);
                uint16x8_t _rb = vld1q_u16(p0 + 8 * 11);
                transpose8x12_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                vst1q_u16(pp, _r0);
                vst1q_u16(pp + 8, _r1);
                vst1q_u16(pp + 8 * 2, _r2);
                vst1q_u16(pp + 8 * 3, _r3);
                vst1q_u16(pp + 8 * 4, _r4);
                vst1q_u16(pp + 8 * 5, _r5);
                vst1q_u16(pp + 8 * 6, _r6);
                vst1q_u16(pp + 8 * 7, _r7);
                vst1q_u16(pp + 8 * 8, _r8);
                vst1q_u16(pp + 8 * 9, _r9);
                vst1q_u16(pp + 8 * 10, _ra);
                vst1q_u16(pp + 8 * 11, _rb);
                pp += 96;
#endif // NCNN_GNU_INLINE_ASM
                p0 += bottom_blob.cstep * 8;
            }
        }

        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                // transpose4x12
#if 0  //NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0] \n"
                    "st1    {v0.4s}, [%1], #16          \n"
                    "st1    {v4.4s}, [%1], #16          \n"
                    "st1    {v8.4s}, [%1], #16          \n"
                    "sub    %0, %0, #128                \n"
                    "st1    {v1.4s}, [%1], #16          \n"
                    "st1    {v5.4s}, [%1], #16          \n"
                    "st1    {v9.4s}, [%1], #16          \n"
                    "st1    {v2.4s}, [%1], #16          \n"
                    "st1    {v6.4s}, [%1], #16          \n"
                    "st1    {v10.4s}, [%1], #16         \n"
                    "st1    {v3.4s}, [%1], #16          \n"
                    "st1    {v7.4s}, [%1], #16          \n"
                    "st1    {v11.4s}, [%1], #16         \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else  // NCNN_GNU_INLINE_ASM
                uint16x4x4_t _r0 = vld4_u16(p0);
                uint16x4x4_t _r1 = vld4_u16(p0 + 16);
                uint16x4x4_t _r2 = vld4_u16(p0 + 32);
                vst1_u16(pp, _r0.val[0]);
                vst1_u16(pp + 4, _r1.val[0]);
                vst1_u16(pp + 4 * 2, _r2.val[0]);
                vst1_u16(pp + 4 * 3, _r0.val[1]);
                vst1_u16(pp + 4 * 4, _r1.val[1]);
                vst1_u16(pp + 4 * 5, _r2.val[1]);
                vst1_u16(pp + 4 * 6, _r0.val[2]);
                vst1_u16(pp + 4 * 7, _r1.val[2]);
                vst1_u16(pp + 4 * 8, _r2.val[2]);
                vst1_u16(pp + 4 * 9, _r0.val[3]);
                vst1_u16(pp + 4 * 10, _r1.val[3]);
                vst1_u16(pp + 4 * 11, _r2.val[3]);
                pp += 48;
#endif // NCNN_GNU_INLINE_ASM
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p0[4];
                pp[5] = p0[5];
                pp[6] = p0[6];
                pp[7] = p0[7];
                pp[8] = p0[8];
                pp[9] = p0[9];
                pp[10] = p0[10];
                pp[11] = p0[11];
                pp += 12;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                // transpose8x8
#if 0 //NCNN_GNU_INLINE_ASM
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0] \n"
                    "st1    {v0.4s}, [%1], #16          \n"
                    "st1    {v4.4s}, [%1], #16          \n"
                    "st1    {v1.4s}, [%1], #16          \n"
                    "st1    {v5.4s}, [%1], #16          \n"
                    "sub    %0, %0, #64                 \n"
                    "st1    {v2.4s}, [%1], #16          \n"
                    "st1    {v6.4s}, [%1], #16          \n"
                    "st1    {v3.4s}, [%1], #16          \n"
                    "st1    {v7.4s}, [%1], #16          \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else  // __aarch64__
                asm volatile(
                    "pld        [%0, #512]          \n"
                    "vldm       %0!, {d0-d7}        \n"
                    "pld        [%0, #512]          \n"
                    "vldm       %0, {d16-d23}       \n"

                    "vtrn.32    q0, q1              \n"
                    "vtrn.32    q2, q3              \n"
                    "vtrn.32    q8, q9              \n"
                    "vtrn.32    q10, q11            \n"
                    "vswp       d1, d4              \n"
                    "vswp       d3, d6              \n"
                    "vswp       d17, d20            \n"
                    "vswp       d19, d22            \n"
                    "vswp       q1, q8              \n"
                    "vswp       q3, q10             \n"

                    "vst1.f32   {d0-d3}, [%1 :128]! \n"
                    "vst1.f32   {d16-d19}, [%1 :128]! \n"
                    "sub        %0, %0, #64         \n"
                    "vst1.f32   {d4-d7}, [%1 :128]! \n"
                    "vst1.f32   {d20-d23}, [%1 :128]! \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
                uint16x8_t _r0 = vld1q_u16(p0);
                uint16x8_t _r1 = vld1q_u16(p0 + 8);
                uint16x8_t _r2 = vld1q_u16(p0 + 8 * 2);
                uint16x8_t _r3 = vld1q_u16(p0 + 8 * 3);
                uint16x8_t _r4 = vld1q_u16(p0 + 8 * 4);
                uint16x8_t _r5 = vld1q_u16(p0 + 8 * 5);
                uint16x8_t _r6 = vld1q_u16(p0 + 8 * 6);
                uint16x8_t _r7 = vld1q_u16(p0 + 8 * 7);
                transpose8x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                vst1q_u16(pp, _r0);
                vst1q_u16(pp + 8, _r1);
                vst1q_u16(pp + 8 * 2, _r2);
                vst1q_u16(pp + 8 * 3, _r3);
                vst1q_u16(pp + 8 * 4, _r4);
                vst1q_u16(pp + 8 * 5, _r5);
                vst1q_u16(pp + 8 * 6, _r6);
                vst1q_u16(pp + 8 * 7, _r7);
                pp += 64;
#endif // NCNN_GNU_INLINE_ASM
                p0 += bottom_blob.cstep * 8;
            }
        }

        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                // transpose4x8
#if 0 //NCNN_GNU_INLINE_ASM
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0] \n"
                    "st1    {v0.4s}, [%1], #16          \n"
                    "st1    {v4.4s}, [%1], #16          \n"
                    "st1    {v1.4s}, [%1], #16          \n"
                    "st1    {v5.4s}, [%1], #16          \n"
                    "sub    %0, %0, #64                 \n"
                    "st1    {v2.4s}, [%1], #16          \n"
                    "st1    {v6.4s}, [%1], #16          \n"
                    "st1    {v3.4s}, [%1], #16          \n"
                    "st1    {v7.4s}, [%1], #16          \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else  // __aarch64__
                asm volatile(
                    "pld        [%0, #512]          \n"
                    "vldm       %0!, {d0-d7}        \n"
                    "pld        [%0, #512]          \n"
                    "vldm       %0, {d16-d23}       \n"

                    "vtrn.32    q0, q1              \n"
                    "vtrn.32    q2, q3              \n"
                    "vtrn.32    q8, q9              \n"
                    "vtrn.32    q10, q11            \n"
                    "vswp       d1, d4              \n"
                    "vswp       d3, d6              \n"
                    "vswp       d17, d20            \n"
                    "vswp       d19, d22            \n"
                    "vswp       q1, q8              \n"
                    "vswp       q3, q10             \n"

                    "vst1.f32   {d0-d3}, [%1 :128]! \n"
                    "vst1.f32   {d16-d19}, [%1 :128]! \n"
                    "sub        %0, %0, #64         \n"
                    "vst1.f32   {d4-d7}, [%1 :128]! \n"
                    "vst1.f32   {d20-d23}, [%1 :128]! \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
                uint16x8x4_t _r0 = vld4q_u16(p0);
                vst1q_u16(pp, _r0.val[0]);
                vst1q_u16(pp + 8, _r0.val[1]);
                vst1q_u16(pp + 16, _r0.val[2]);
                vst1q_u16(pp + 24, _r0.val[3]);
                pp += 32;
#endif // NCNN_GNU_INLINE_ASM
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p0[4];
                pp[5] = p0[5];
                pp[6] = p0[6];
                pp[7] = p0[7];
                pp += 8;
                p0 += bottom_blob.cstep;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                // transpose8x4
#if 0 //NCNN_GNU_INLINE_ASM
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                    "st4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "v0", "v1", "v2", "v3");
#else  // __aarch64__
                asm volatile(
                    "pld        [%0, #512]          \n"
                    "vldm       %0, {d0-d7}         \n"
                    "vtrn.32    q0, q1              \n"
                    "vtrn.32    q2, q3              \n"
                    "vswp       d1, d4              \n"
                    "vswp       d3, d6              \n"
                    "vstm       %1!, {d0-d7}        \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
                uint16x8x4_t _r0;
                _r0.val[0] = vld1q_u16(p0);
                _r0.val[1] = vld1q_u16(p0 + 8);
                _r0.val[2] = vld1q_u16(p0 + 8 * 2);
                _r0.val[3] = vld1q_u16(p0 + 8 * 3);
                vst4q_u16(pp, _r0);
                pp += 32;
#endif // NCNN_GNU_INLINE_ASM
                p0 += bottom_blob.cstep * 8;
            }
        }

        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                // transpose4x4
#if 0 //NCNN_GNU_INLINE_ASM
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                    "st4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "v0", "v1", "v2", "v3");
#else  // __aarch64__
                asm volatile(
                    "pld        [%0, #512]          \n"
                    "vldm       %0, {d0-d7}         \n"
                    "vtrn.32    q0, q1              \n"
                    "vtrn.32    q2, q3              \n"
                    "vswp       d1, d4              \n"
                    "vswp       d3, d6              \n"
                    "vstm       %1!, {d0-d7}        \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
                uint16x4x4_t _r0;
                _r0.val[0] = vld1_u16(p0);
                _r0.val[1] = vld1_u16(p0 + 4);
                _r0.val[2] = vld1_u16(p0 + 4 * 2);
                _r0.val[3] = vld1_u16(p0 + 4 * 3);
                vst4_u16(pp, _r0);
                pp += 16;
#endif // NCNN_GNU_INLINE_ASM
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                // transpose8x2
                pp[0] = p0[0];
                pp[1] = p0[8];
                pp[2] = p0[1];
                pp[3] = p0[9];
                pp[4] = p0[2];
                pp[5] = p0[10];
                pp[6] = p0[3];
                pp[7] = p0[11];
                pp[8] = p0[4];
                pp[9] = p0[12];
                pp[10] = p0[5];
                pp[11] = p0[13];
                pp[12] = p0[6];
                pp[13] = p0[14];
                pp[14] = p0[7];
                pp[15] = p0[15];
                pp += 16;
                p0 += bottom_blob.cstep * 8;
            }
        }

        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                // transpose4x2
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp[2] = p0[1];
                pp[3] = p0[5];
                pp[4] = p0[2];
                pp[5] = p0[6];
                pp[6] = p0[3];
                pp[7] = p0[7];
                pp += 8;
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += bottom_blob.cstep;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p0[4];
                pp[5] = p0[5];
                pp[6] = p0[6];
                pp[7] = p0[7];
                pp += 8;
                p0 += bottom_blob.cstep * 8;
            }
        }

        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += bottom_blob.cstep;
            }
        }
    }
}

static void convolution_im2col_input_tile_bf16s(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1_bf16s(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    const int w = bottom_blob.w;
    // const int h = bottom_blob.h;
    // const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    // const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    // const int outh = (h - kernel_extent_h) / stride_h + 1;

    // j max_jj     outw*outh    split w and h

    // k max_kk     pa*maxk*(inch/pa)    split inch

    // k/max_kk shall be multiple of maxk

    const int maxk = kernel_w * kernel_h;

    // NCNN_LOGE("convolution_im2col_input_tile_bf16s %d %d %d %d   %d", j, max_jj, k, max_kk, elempack);

    unsigned short* pp = B;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dy4 = (j + jj + 4) / outw;
        int dy5 = (j + jj + 5) / outw;
        int dy6 = (j + jj + 6) / outw;
        int dy7 = (j + jj + 7) / outw;
        int dy8 = (j + jj + 8) / outw;
        int dy9 = (j + jj + 9) / outw;
        int dya = (j + jj + 10) / outw;
        int dyb = (j + jj + 11) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;
        int dx4 = (j + jj + 4) % outw;
        int dx5 = (j + jj + 5) % outw;
        int dx6 = (j + jj + 6) % outw;
        int dx7 = (j + jj + 7) % outw;
        int dx8 = (j + jj + 8) % outw;
        int dx9 = (j + jj + 9) % outw;
        int dxa = (j + jj + 10) % outw;
        int dxb = (j + jj + 11) % outw;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x0 = stride_w * dx0 + dilation_w * v;
            int x1 = stride_w * dx1 + dilation_w * v;
            int x2 = stride_w * dx2 + dilation_w * v;
            int x3 = stride_w * dx3 + dilation_w * v;
            int x4 = stride_w * dx4 + dilation_w * v;
            int x5 = stride_w * dx5 + dilation_w * v;
            int x6 = stride_w * dx6 + dilation_w * v;
            int x7 = stride_w * dx7 + dilation_w * v;
            int x8 = stride_w * dx8 + dilation_w * v;
            int x9 = stride_w * dx9 + dilation_w * v;
            int xa = stride_w * dxa + dilation_w * v;
            int xb = stride_w * dxb + dilation_w * v;

            int y0 = stride_h * dy0 + dilation_h * u;
            int y1 = stride_h * dy1 + dilation_h * u;
            int y2 = stride_h * dy2 + dilation_h * u;
            int y3 = stride_h * dy3 + dilation_h * u;
            int y4 = stride_h * dy4 + dilation_h * u;
            int y5 = stride_h * dy5 + dilation_h * u;
            int y6 = stride_h * dy6 + dilation_h * u;
            int y7 = stride_h * dy7 + dilation_h * u;
            int y8 = stride_h * dy8 + dilation_h * u;
            int y9 = stride_h * dy9 + dilation_h * u;
            int ya = stride_h * dya + dilation_h * u;
            int yb = stride_h * dyb + dilation_h * u;

            const unsigned short* sptr0 = img.row<const unsigned short>(y0) + x0 * elempack;
            const unsigned short* sptr1 = img.row<const unsigned short>(y1) + x1 * elempack;
            const unsigned short* sptr2 = img.row<const unsigned short>(y2) + x2 * elempack;
            const unsigned short* sptr3 = img.row<const unsigned short>(y3) + x3 * elempack;
            const unsigned short* sptr4 = img.row<const unsigned short>(y4) + x4 * elempack;
            const unsigned short* sptr5 = img.row<const unsigned short>(y5) + x5 * elempack;
            const unsigned short* sptr6 = img.row<const unsigned short>(y6) + x6 * elempack;
            const unsigned short* sptr7 = img.row<const unsigned short>(y7) + x7 * elempack;
            const unsigned short* sptr8 = img.row<const unsigned short>(y8) + x8 * elempack;
            const unsigned short* sptr9 = img.row<const unsigned short>(y9) + x9 * elempack;
            const unsigned short* sptra = img.row<const unsigned short>(ya) + xa * elempack;
            const unsigned short* sptrb = img.row<const unsigned short>(yb) + xb * elempack;

            if (elempack == 8)
            {
                uint16x8_t _r0 = vld1q_u16(sptr0);
                uint16x8_t _r1 = vld1q_u16(sptr1);
                uint16x8_t _r2 = vld1q_u16(sptr2);
                uint16x8_t _r3 = vld1q_u16(sptr3);
                uint16x8_t _r4 = vld1q_u16(sptr4);
                uint16x8_t _r5 = vld1q_u16(sptr5);
                uint16x8_t _r6 = vld1q_u16(sptr6);
                uint16x8_t _r7 = vld1q_u16(sptr7);
                uint16x8_t _r8 = vld1q_u16(sptr8);
                uint16x8_t _r9 = vld1q_u16(sptr9);
                uint16x8_t _ra = vld1q_u16(sptra);
                uint16x8_t _rb = vld1q_u16(sptrb);
                transpose8x12_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                vst1q_u16(pp, _r0);
                vst1q_u16(pp + 8, _r1);
                vst1q_u16(pp + 8 * 2, _r2);
                vst1q_u16(pp + 8 * 3, _r3);
                vst1q_u16(pp + 8 * 4, _r4);
                vst1q_u16(pp + 8 * 5, _r5);
                vst1q_u16(pp + 8 * 6, _r6);
                vst1q_u16(pp + 8 * 7, _r7);
                vst1q_u16(pp + 8 * 8, _r8);
                vst1q_u16(pp + 8 * 9, _r9);
                vst1q_u16(pp + 8 * 10, _ra);
                vst1q_u16(pp + 8 * 11, _rb);
                pp += 96;
            }
            if (elempack == 4)
            {
                uint16x4_t _r0 = vld1_u16(sptr0);
                uint16x4_t _r1 = vld1_u16(sptr1);
                uint16x4_t _r2 = vld1_u16(sptr2);
                uint16x4_t _r3 = vld1_u16(sptr3);
                uint16x4_t _r4 = vld1_u16(sptr4);
                uint16x4_t _r5 = vld1_u16(sptr5);
                uint16x4_t _r6 = vld1_u16(sptr6);
                uint16x4_t _r7 = vld1_u16(sptr7);
                uint16x4_t _r8 = vld1_u16(sptr8);
                uint16x4_t _r9 = vld1_u16(sptr9);
                uint16x4_t _ra = vld1_u16(sptra);
                uint16x4_t _rb = vld1_u16(sptrb);
                transpose4x12_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                vst1_u16(pp, _r0);
                vst1_u16(pp + 4, _r1);
                vst1_u16(pp + 4 * 2, _r2);
                vst1_u16(pp + 4 * 3, _r3);
                vst1_u16(pp + 4 * 4, _r4);
                vst1_u16(pp + 4 * 5, _r5);
                vst1_u16(pp + 4 * 6, _r6);
                vst1_u16(pp + 4 * 7, _r7);
                vst1_u16(pp + 4 * 8, _r8);
                vst1_u16(pp + 4 * 9, _r9);
                vst1_u16(pp + 4 * 10, _ra);
                vst1_u16(pp + 4 * 11, _rb);
                pp += 48;
            }
            if (elempack == 1)
            {
                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp[2] = sptr2[0];
                pp[3] = sptr3[0];
                pp[4] = sptr4[0];
                pp[5] = sptr5[0];
                pp[6] = sptr6[0];
                pp[7] = sptr7[0];
                pp[8] = sptr8[0];
                pp[9] = sptr9[0];
                pp[10] = sptra[0];
                pp[11] = sptrb[0];
                pp += 12;
            }
        }
    }
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dy4 = (j + jj + 4) / outw;
        int dy5 = (j + jj + 5) / outw;
        int dy6 = (j + jj + 6) / outw;
        int dy7 = (j + jj + 7) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;
        int dx4 = (j + jj + 4) % outw;
        int dx5 = (j + jj + 5) % outw;
        int dx6 = (j + jj + 6) % outw;
        int dx7 = (j + jj + 7) % outw;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x0 = stride_w * dx0 + dilation_w * v;
            int x1 = stride_w * dx1 + dilation_w * v;
            int x2 = stride_w * dx2 + dilation_w * v;
            int x3 = stride_w * dx3 + dilation_w * v;
            int x4 = stride_w * dx4 + dilation_w * v;
            int x5 = stride_w * dx5 + dilation_w * v;
            int x6 = stride_w * dx6 + dilation_w * v;
            int x7 = stride_w * dx7 + dilation_w * v;
            int y0 = stride_h * dy0 + dilation_h * u;
            int y1 = stride_h * dy1 + dilation_h * u;
            int y2 = stride_h * dy2 + dilation_h * u;
            int y3 = stride_h * dy3 + dilation_h * u;
            int y4 = stride_h * dy4 + dilation_h * u;
            int y5 = stride_h * dy5 + dilation_h * u;
            int y6 = stride_h * dy6 + dilation_h * u;
            int y7 = stride_h * dy7 + dilation_h * u;

            const unsigned short* sptr0 = img.row<const unsigned short>(y0) + x0 * elempack;
            const unsigned short* sptr1 = img.row<const unsigned short>(y1) + x1 * elempack;
            const unsigned short* sptr2 = img.row<const unsigned short>(y2) + x2 * elempack;
            const unsigned short* sptr3 = img.row<const unsigned short>(y3) + x3 * elempack;
            const unsigned short* sptr4 = img.row<const unsigned short>(y4) + x4 * elempack;
            const unsigned short* sptr5 = img.row<const unsigned short>(y5) + x5 * elempack;
            const unsigned short* sptr6 = img.row<const unsigned short>(y6) + x6 * elempack;
            const unsigned short* sptr7 = img.row<const unsigned short>(y7) + x7 * elempack;

            if (elempack == 8)
            {
                uint16x8_t _r0 = vld1q_u16(sptr0);
                uint16x8_t _r1 = vld1q_u16(sptr1);
                uint16x8_t _r2 = vld1q_u16(sptr2);
                uint16x8_t _r3 = vld1q_u16(sptr3);
                uint16x8_t _r4 = vld1q_u16(sptr4);
                uint16x8_t _r5 = vld1q_u16(sptr5);
                uint16x8_t _r6 = vld1q_u16(sptr6);
                uint16x8_t _r7 = vld1q_u16(sptr7);
                transpose8x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                vst1q_u16(pp, _r0);
                vst1q_u16(pp + 8, _r1);
                vst1q_u16(pp + 8 * 2, _r2);
                vst1q_u16(pp + 8 * 3, _r3);
                vst1q_u16(pp + 8 * 4, _r4);
                vst1q_u16(pp + 8 * 5, _r5);
                vst1q_u16(pp + 8 * 6, _r6);
                vst1q_u16(pp + 8 * 7, _r7);
                pp += 64;
            }
            if (elempack == 4)
            {
                uint16x4_t _r0 = vld1_u16(sptr0);
                uint16x4_t _r1 = vld1_u16(sptr1);
                uint16x4_t _r2 = vld1_u16(sptr2);
                uint16x4_t _r3 = vld1_u16(sptr3);
                uint16x4_t _r4 = vld1_u16(sptr4);
                uint16x4_t _r5 = vld1_u16(sptr5);
                uint16x4_t _r6 = vld1_u16(sptr6);
                uint16x4_t _r7 = vld1_u16(sptr7);
                transpose4x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                vst1_u16(pp, _r0);
                vst1_u16(pp + 4, _r1);
                vst1_u16(pp + 4 * 2, _r2);
                vst1_u16(pp + 4 * 3, _r3);
                vst1_u16(pp + 4 * 4, _r4);
                vst1_u16(pp + 4 * 5, _r5);
                vst1_u16(pp + 4 * 6, _r6);
                vst1_u16(pp + 4 * 7, _r7);
                pp += 32;
            }
            if (elempack == 1)
            {
                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp[2] = sptr2[0];
                pp[3] = sptr3[0];
                pp[4] = sptr4[0];
                pp[5] = sptr5[0];
                pp[6] = sptr6[0];
                pp[7] = sptr7[0];
                pp += 8;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x0 = stride_w * dx0 + dilation_w * v;
            int x1 = stride_w * dx1 + dilation_w * v;
            int x2 = stride_w * dx2 + dilation_w * v;
            int x3 = stride_w * dx3 + dilation_w * v;
            int y0 = stride_h * dy0 + dilation_h * u;
            int y1 = stride_h * dy1 + dilation_h * u;
            int y2 = stride_h * dy2 + dilation_h * u;
            int y3 = stride_h * dy3 + dilation_h * u;

            const unsigned short* sptr0 = img.row<const unsigned short>(y0) + x0 * elempack;
            const unsigned short* sptr1 = img.row<const unsigned short>(y1) + x1 * elempack;
            const unsigned short* sptr2 = img.row<const unsigned short>(y2) + x2 * elempack;
            const unsigned short* sptr3 = img.row<const unsigned short>(y3) + x3 * elempack;

            if (elempack == 8)
            {
                uint16x8x4_t _r0;
                _r0.val[0] = vld1q_u16(sptr0);
                _r0.val[1] = vld1q_u16(sptr1);
                _r0.val[2] = vld1q_u16(sptr2);
                _r0.val[3] = vld1q_u16(sptr3);
                vst4q_u16(pp, _r0);
                pp += 32;
            }
            if (elempack == 4)
            {
                uint16x4x4_t _r0;
                _r0.val[0] = vld1_u16(sptr0);
                _r0.val[1] = vld1_u16(sptr1);
                _r0.val[2] = vld1_u16(sptr2);
                _r0.val[3] = vld1_u16(sptr3);
                vst4_u16(pp, _r0);
                pp += 16;
            }
            if (elempack == 1)
            {
                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp[2] = sptr2[0];
                pp[3] = sptr3[0];
                pp += 4;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x0 = stride_w * dx0 + dilation_w * v;
            int x1 = stride_w * dx1 + dilation_w * v;
            int y0 = stride_h * dy0 + dilation_h * u;
            int y1 = stride_h * dy1 + dilation_h * u;

            const unsigned short* sptr0 = img.row<const unsigned short>(y0) + x0 * elempack;
            const unsigned short* sptr1 = img.row<const unsigned short>(y1) + x1 * elempack;

            if (elempack == 8)
            {
                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp[2] = sptr0[1];
                pp[3] = sptr1[1];
                pp[4] = sptr0[2];
                pp[5] = sptr1[2];
                pp[6] = sptr0[3];
                pp[7] = sptr1[3];
                pp[8+0] = sptr0[4];
                pp[8+1] = sptr1[4];
                pp[8+2] = sptr0[5];
                pp[8+3] = sptr1[5];
                pp[8+4] = sptr0[6];
                pp[8+5] = sptr1[6];
                pp[8+6] = sptr0[7];
                pp[8+7] = sptr1[7];
                pp += 16;
            }
            if (elempack == 4)
            {
                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp[2] = sptr0[1];
                pp[3] = sptr1[1];
                pp[4] = sptr0[2];
                pp[5] = sptr1[2];
                pp[6] = sptr0[3];
                pp[7] = sptr1[3];
                pp += 8;
            }
            if (elempack == 1)
            {
                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp += 2;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
        int dy = (j + jj) / outw;
        int dx = (j + jj) % outw;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x = stride_w * dx + dilation_w * v;
            int y = stride_h * dy + dilation_h * u;

            const unsigned short* sptr = img.row<const unsigned short>(y) + x * elempack;

            if (elempack == 8)
            {
                pp[0] = sptr[0];
                pp[1] = sptr[1];
                pp[2] = sptr[2];
                pp[3] = sptr[3];
                pp[4] = sptr[4];
                pp[5] = sptr[5];
                pp[6] = sptr[6];
                pp[7] = sptr[7];
                pp += 8;
            }
            if (elempack == 4)
            {
                pp[0] = sptr[0];
                pp[1] = sptr[1];
                pp[2] = sptr[2];
                pp[3] = sptr[3];
                pp += 4;
            }
            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp += 1;
            }
        }
    }
}

static void convolution_im2col_gemm_transform_kernel_bf16s(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    // NCNN_LOGE("convolution_im2col_gemm_transform_kernel");
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    // TILE_M = (M + 7) / 8 * 8;
    // TILE_K = (K + 3) / 4 * 4;
    get_optimal_tile_mnk(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    // TILE_K = 16;

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    // Mat A_data = kernel.reshape(maxk * inch, outch);
    // wrap inch elempack

    int elempack = 1;
    if (opt.use_packing_layout)
    {
        elempack = inch % 4 == 0 ? 4 : 1;
    }

    // maxk-inch-outch to pa-maxk-inch/pa-outch
    Mat A_data;
    if (maxk == 1)
    {
        cast_float32_to_bfloat16(kernel, A_data);
    }
    else
    {
        Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);

        A_data.create(maxk * inch, outch, (size_t)2u);

        for (int q = 0; q < outch; q += 1)
        {
            unsigned short* g00 = A_data.row<unsigned short>(q);

            for (int p = 0; p + (elempack - 1) < inch; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        const float* k00 = weight_data_r2.channel(q).row(p + i);
                        g00[0] = float32_to_bfloat16(k00[k]);
                        g00++;
                    }
                }
            }
        }
    }

    AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, (size_t)2u);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            convolution_im2col_pack_A_tile_bf16s(A_data, AT_tile, i, max_ii, k, max_kk, maxk, inch, outch);
        }
    }
}

static void convolution_im2col_gemm_bf16s(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    // TILE_M = (M + 7) / 8 * 8;
    // TILE_N = (N + 3) / 4 * 4;
    // TILE_K = (K + 3) / 4 * 4;
    get_optimal_tile_mnk(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    // TILE_K = 16;

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.blob_allocator);

    const int nn_NK = nn_N * nn_K;

    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min((N - j), TILE_N);
        const int max_kk = std::min((K - k), TILE_K);

        Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

        // im2col
        convolution_im2col_input_tile_bf16s(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
    }

    Mat topT_tileX;
    if (K > TILE_K)
        topT_tileX.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat topT_tile;
        if (K > TILE_K)
            topT_tile = topT_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                const Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = k + TILE_K >= K;

                convolution_gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, bias, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end);
            }
        }
    }
}