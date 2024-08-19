// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

static void pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    // NCNN_LOGE("pack_A_tile_int8");
    // assert A.elempack == 1
    // assert A.dims == 2

    signed char* pp = AT;

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = A.row<const signed char>(i + ii) + k;
        const signed char* p1 = A.row<const signed char>(i + ii + 1) + k;
        const signed char* p2 = A.row<const signed char>(i + ii + 2) + k;
        const signed char* p3 = A.row<const signed char>(i + ii + 3) + k;
        const signed char* p4 = A.row<const signed char>(i + ii + 4) + k;
        const signed char* p5 = A.row<const signed char>(i + ii + 5) + k;
        const signed char* p6 = A.row<const signed char>(i + ii + 6) + k;
        const signed char* p7 = A.row<const signed char>(i + ii + 7) + k;

        int kk = 0;
        // for (; kk + 7 < max_kk; kk += 8)
        // {
        //     int8x8_t _r0 = vld1_s8(p0);
        //     int8x8_t _r1 = vld1_s8(p1);
        //     int8x8_t _r2 = vld1_s8(p2);
        //     int8x8_t _r3 = vld1_s8(p3);
        //     int8x8_t _r4 = vld1_s8(p4);
        //     int8x8_t _r5 = vld1_s8(p5);
        //     int8x8_t _r6 = vld1_s8(p6);
        //     int8x8_t _r7 = vld1_s8(p7);
        //     // save as transpose8x8
        //     int8x8x2_t _r01 = vzip_s8(_r0, _r1);
        //     int8x8x2_t _r23 = vzip_s8(_r2, _r3);
        //     int8x8x2_t _r45 = vzip_s8(_r4, _r5);
        //     int8x8x2_t _r67 = vzip_s8(_r6, _r7);
        //     int16x8x4_t _r0246;
        //     _r0246.val[0] = vreinterpretq_s16_s8(vcombine_s8(_r01.val[0], _r01.val[1]));
        //     _r0246.val[1] = vreinterpretq_s16_s8(vcombine_s8(_r23.val[0], _r23.val[1]));
        //     _r0246.val[2] = vreinterpretq_s16_s8(vcombine_s8(_r45.val[0], _r45.val[1]));
        //     _r0246.val[3] = vreinterpretq_s16_s8(vcombine_s8(_r67.val[0], _r67.val[1]));
        //     vst4q_s16((short*)pp, _r0246);
        //     pp += 64;
        //     p0 += 8;
        //     p1 += 8;
        //     p2 += 8;
        //     p3 += 8;
        //     p4 += 8;
        //     p5 += 8;
        //     p6 += 8;
        //     p7 += 8;
        // }
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp[8] = p4[0];
            pp[9] = p4[1];
            pp[10] = p5[0];
            pp[11] = p5[1];
            pp[12] = p6[0];
            pp[13] = p6[1];
            pp[14] = p7[0];
            pp[15] = p7[1];
            pp += 16;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            p4 += 2;
            p5 += 2;
            p6 += 2;
            p7 += 2;
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
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = A.row<const signed char>(i + ii) + k;
        const signed char* p1 = A.row<const signed char>(i + ii + 1) + k;
        const signed char* p2 = A.row<const signed char>(i + ii + 2) + k;
        const signed char* p3 = A.row<const signed char>(i + ii + 3) + k;

        int kk = 0;
        // for (; kk + 7 < max_kk; kk += 8)
        // {
        //     int8x8x4_t _r0123;
        //     _r0123.val[0] = vld1_s8(p0);
        //     _r0123.val[1] = vld1_s8(p1);
        //     _r0123.val[2] = vld1_s8(p2);
        //     _r0123.val[3] = vld1_s8(p3);
        //     vst4_s8(pp, _r0123);
        //     pp += 32;
        //     p0 += 8;
        //     p1 += 8;
        //     p2 += 8;
        //     p3 += 8;
        // }
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp += 8;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
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
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = A.row<const signed char>(i + ii) + k;
        const signed char* p1 = A.row<const signed char>(i + ii + 1) + k;

        int kk = 0;
        // #if __ARM_NEON
        //         for (; kk + 7 < max_kk; kk += 8)
        //         {
        //             int8x8x2_t _r01;
        //             _r01.val[0] = vld1_s8(p0);
        //             _r01.val[1] = vld1_s8(p1);
        //             vst2_s8(pp, _r01);
        //             pp += 16;
        //             p0 += 8;
        //             p1 += 8;
        //         }
        // #endif // __ARM_NEON
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp += 4;
            p0 += 2;
            p1 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const signed char* p0 = A.row<const signed char>(i + ii) + k;

        int kk = 0;
        // #if __ARM_NEON
        //         for (; kk + 7 < max_kk; kk += 8)
        //         {
        //             vst1_s8(pp, vld1_s8(p0));
        //             pp += 8;
        //             p0 += 8;
        //         }
        // #endif // __ARM_NEON
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp += 2;
            p0 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    // NCNN_LOGE("transpose_pack_A_tile_int8");
    // assert A.elempack == 1
    // assert A.dims == 2

    signed char* pp = AT;

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
        {
            int8x8x2_t _r01;
            _r01.val[0] = vld1_s8(p0);
            _r01.val[1] = vld1_s8(p0 + A.w);
            vst2_s8(pp, _r01);
            pp += 16;
            p0 += A.w * 2;
        }
        for (; kk < max_kk; kk++)
        {
            vst1_s8(pp, vld1_s8(p0));
            pp += 8;
            p0 += A.w;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[A.w];
            pp[2] = p0[1];
            pp[3] = p0[A.w + 1];
            pp[4] = p0[2];
            pp[5] = p0[A.w + 2];
            pp[6] = p0[3];
            pp[7] = p0[A.w + 3];
            pp += 8;
            p0 += A.w * 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp += 4;
            p0 += A.w;
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[A.w];
            pp[2] = p0[1];
            pp[3] = p0[A.w + 1];
            pp += 4;
            p0 += A.w * 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp += 2;
            p0 += A.w;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[A.w];
            pp += 2;
            p0 += A.w * 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0 += A.w;
        }
    }
}

static void pack_B_tile_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    // NCNN_LOGE("pack_B_tile_int8");
    // assert B.elempack == 1
    // assert B.dims == 2

    signed char* pp = BT;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;
        const signed char* p1 = B.row<const signed char>(j + jj + 1) + k;
        const signed char* p2 = B.row<const signed char>(j + jj + 2) + k;
        const signed char* p3 = B.row<const signed char>(j + jj + 3) + k;
        const signed char* p4 = B.row<const signed char>(j + jj + 4) + k;
        const signed char* p5 = B.row<const signed char>(j + jj + 5) + k;
        const signed char* p6 = B.row<const signed char>(j + jj + 6) + k;
        const signed char* p7 = B.row<const signed char>(j + jj + 7) + k;

        int kk = 0;
        // for (; kk + 7 < max_kk; kk += 8)
        // {
        //     int8x8_t _r0 = vld1_s8(p0);
        //     int8x8_t _r1 = vld1_s8(p1);
        //     int8x8_t _r2 = vld1_s8(p2);
        //     int8x8_t _r3 = vld1_s8(p3);
        //     int8x8_t _r4 = vld1_s8(p4);
        //     int8x8_t _r5 = vld1_s8(p5);
        //     int8x8_t _r6 = vld1_s8(p6);
        //     int8x8_t _r7 = vld1_s8(p7);
        //     // save as transpose8x8
        //     int8x8x2_t _r01 = vzip_s8(_r0, _r1);
        //     int8x8x2_t _r23 = vzip_s8(_r2, _r3);
        //     int8x8x2_t _r45 = vzip_s8(_r4, _r5);
        //     int8x8x2_t _r67 = vzip_s8(_r6, _r7);
        //     int16x8x4_t _r0246;
        //     _r0246.val[0] = vreinterpretq_s16_s8(vcombine_s8(_r01.val[0], _r01.val[1]));
        //     _r0246.val[1] = vreinterpretq_s16_s8(vcombine_s8(_r23.val[0], _r23.val[1]));
        //     _r0246.val[2] = vreinterpretq_s16_s8(vcombine_s8(_r45.val[0], _r45.val[1]));
        //     _r0246.val[3] = vreinterpretq_s16_s8(vcombine_s8(_r67.val[0], _r67.val[1]));
        //     vst4q_s16((short*)pp, _r0246);
        //     pp += 64;
        //     p0 += 8;
        //     p1 += 8;
        //     p2 += 8;
        //     p3 += 8;
        //     p4 += 8;
        //     p5 += 8;
        //     p6 += 8;
        //     p7 += 8;
        // }
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp[8] = p4[0];
            pp[9] = p4[1];
            pp[10] = p5[0];
            pp[11] = p5[1];
            pp[12] = p6[0];
            pp[13] = p6[1];
            pp[14] = p7[0];
            pp[15] = p7[1];
            pp += 16;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            p4 += 2;
            p5 += 2;
            p6 += 2;
            p7 += 2;
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
#endif // __aarch64__
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;
        const signed char* p1 = B.row<const signed char>(j + jj + 1) + k;
        const signed char* p2 = B.row<const signed char>(j + jj + 2) + k;
        const signed char* p3 = B.row<const signed char>(j + jj + 3) + k;

        int kk = 0;
        // for (; kk + 7 < max_kk; kk += 8)
        // {
        //     int8x8x4_t _r0123;
        //     _r0123.val[0] = vld1_s8(p0);
        //     _r0123.val[1] = vld1_s8(p1);
        //     _r0123.val[2] = vld1_s8(p2);
        //     _r0123.val[3] = vld1_s8(p3);
        //     vst4_s8(pp, _r0123);
        //     pp += 32;
        //     p0 += 8;
        //     p1 += 8;
        //     p2 += 8;
        //     p3 += 8;
        // }
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp += 8;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
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
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;
        const signed char* p1 = B.row<const signed char>(j + jj + 1) + k;

        int kk = 0;
        // #if __ARM_NEON
        //         for (; kk + 7 < max_kk; kk += 8)
        //         {
        //             int8x8x2_t _r01;
        //             _r01.val[0] = vld1_s8(p0);
        //             _r01.val[1] = vld1_s8(p1);
        //             vst2_s8(pp, _r01);
        //             pp += 16;
        //             p0 += 8;
        //             p1 += 8;
        //         }
        // #endif // __ARM_NEON
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp += 4;
            p0 += 2;
            p1 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;

        int kk = 0;
        // #if __ARM_NEON
        //         for (; kk + 7 < max_kk; kk += 8)
        //         {
        //             vst1_s8(pp, vld1_s8(p0));
        //             pp += 8;
        //             p0 += 8;
        //         }
        // #endif // __ARM_NEON
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp += 2;
            p0 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_B_tile_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    // NCNN_LOGE("pack_B_tile_int8");
    // assert B.elempack == 1
    // assert B.dims == 2

    signed char* pp = BT;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
        {
            int8x8x2_t _r01;
            _r01.val[0] = vld1_s8(p0);
            _r01.val[1] = vld1_s8(p0 + B.w);
            vst2_s8(pp, _r01);
            pp += 16;
            p0 += B.w * 2;
        }
        for (; kk < max_kk; kk++)
        {
            vst1_s8(pp, vld1_s8(p0));
            pp += 8;
            p0 += B.w;
        }
    }
#endif // __aarch64__
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[B.w];
            pp[2] = p0[1];
            pp[3] = p0[B.w + 1];
            pp[4] = p0[2];
            pp[5] = p0[B.w + 2];
            pp[6] = p0[3];
            pp[7] = p0[B.w + 3];
            pp += 8;
            p0 += B.w * 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp += 4;
            p0 += B.w;
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[B.w];
            pp[2] = p0[1];
            pp[3] = p0[B.w + 1];
            pp += 4;
            p0 += B.w * 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp += 2;
            p0 += B.w;
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[B.w];
            pp += 2;
            p0 += B.w * 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0 += B.w;
        }
    }
}

static void pack_A_tile_fp32_to_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, float scale)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    // NCNN_LOGE("pack_A_tile_fp32_to_int8 %d %d", max_ii, elempack);

    signed char* pp = AT;

    float32x4_t _scale = vdupq_n_f32(scale);

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;
            const float* p1 = (const float*)A + (i + ii + 4) * A_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p0n = vld1q_f32(p0 + 4);
                float32x4_t _p1 = vld1q_f32(p1);
                float32x4_t _p1n = vld1q_f32(p1 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p0n = vmulq_f32(_p0n, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p1n = vmulq_f32(_p1n, _scale);

                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p0, _p1);
                _r01.val[1] = float2int8(_p0n, _p1n);

                vst2_s8(pp, _r01);

                pp += 16;
                p0 += 8;
                p1 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p1[0] * scale);
                pp[5] = float2int8(p1[1] * scale);
                pp[6] = float2int8(p1[2] * scale);
                pp[7] = float2int8(p1[3] * scale);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
            const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
            const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;
            const float* p4 = (const float*)A + (i + ii + 4) * A_hstep + k;
            const float* p5 = (const float*)A + (i + ii + 5) * A_hstep + k;
            const float* p6 = (const float*)A + (i + ii + 6) * A_hstep + k;
            const float* p7 = (const float*)A + (i + ii + 7) * A_hstep + k;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p01 = vcombine_f32(vld1_f32(p0), vld1_f32(p1));
                float32x4_t _p23 = vcombine_f32(vld1_f32(p2), vld1_f32(p3));
                float32x4_t _p45 = vcombine_f32(vld1_f32(p4), vld1_f32(p5));
                float32x4_t _p67 = vcombine_f32(vld1_f32(p6), vld1_f32(p7));

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);
                _p45 = vmulq_f32(_p45, _scale);
                _p67 = vmulq_f32(_p67, _scale);

                int8x8_t _r0 = float2int8(_p01, _p23);
                int8x8_t _r1 = float2int8(_p45, _p67);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));

                pp += 16;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
                p4 += 2;
                p5 += 2;
                p6 += 2;
                p7 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p1[0] * scale);
                pp[2] = float2int8(p2[0] * scale);
                pp[3] = float2int8(p3[0] * scale);
                pp[4] = float2int8(p4[0] * scale);
                pp[5] = float2int8(p5[0] * scale);
                pp[6] = float2int8(p6[0] * scale);
                pp[7] = float2int8(p7[0] * scale);
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
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                int8x8_t _r01 = float2int8(_p01.val[0], _p01.val[1]);

                vst1_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[4] * scale);
                // pp[2] = float2int8(p0[1] * scale);
                // pp[3] = float2int8(p0[5] * scale);
                // pp[4] = float2int8(p0[2] * scale);
                // pp[5] = float2int8(p0[6] * scale);
                // pp[6] = float2int8(p0[3] * scale);
                // pp[7] = float2int8(p0[7] * scale);
                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
            const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
            const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p01 = vcombine_f32(vld1_f32(p0), vld1_f32(p1));
                float32x4_t _p23 = vcombine_f32(vld1_f32(p2), vld1_f32(p3));

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);

                int8x8_t _r0 = float2int8(_p01, _p23);

                vst1_s8(pp, _r0);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[1] * scale);
                // pp[2] = float2int8(p1[0] * scale);
                // pp[3] = float2int8(p1[1] * scale);
                // pp[4] = float2int8(p2[0] * scale);
                // pp[5] = float2int8(p2[1] * scale);
                // pp[6] = float2int8(p3[0] * scale);
                // pp[7] = float2int8(p3[1] * scale);
                pp += 8;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p1[0] * scale);
                pp[2] = float2int8(p2[0] * scale);
                pp[3] = float2int8(p3[0] * scale);
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
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p1);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                float32x4_t _t0 = vcombine_f32(vget_low_f32(_p0), vget_low_f32(_p1));
                float32x4_t _t1 = vcombine_f32(vget_high_f32(_p0), vget_high_f32(_p1));

                int8x8_t _r0 = float2int8(_t0, _t1);

                vst1_s8(pp, _r0);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[1] * scale);
                // pp[2] = float2int8(p1[0] * scale);
                // pp[3] = float2int8(p1[1] * scale);
                // pp[4] = float2int8(p0[2] * scale);
                // pp[5] = float2int8(p0[3] * scale);
                // pp[6] = float2int8(p1[2] * scale);
                // pp[7] = float2int8(p1[3] * scale);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p1[0] * scale);
                pp[3] = float2int8(p1[1] * scale);
                pp += 4;
                p0 += 2;
                p1 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p1[0] * scale);
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
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r0 = float2int8(_p0, _p1);

                vst1_s8(pp, _r0);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[1] * scale);
                pp += 8;
                p0 += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp += 2;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_A_tile_fp32_to_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, float scale)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    // NCNN_LOGE("transpose_pack_A_tile_fp32_to_int8 %d %d", max_ii, elempack);

    signed char* pp = AT;

    float32x4_t _scale = vdupq_n_f32(scale);

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[4] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[8] * scale);
                pp[5] = float2int8(p0[9] * scale);
                pp[6] = float2int8(p0[12] * scale);
                pp[7] = float2int8(p0[13] * scale);

                pp[8 + 0] = float2int8(p0[16] * scale);
                pp[8 + 1] = float2int8(p0[17] * scale);
                pp[8 + 2] = float2int8(p0[20] * scale);
                pp[8 + 3] = float2int8(p0[21] * scale);
                pp[8 + 4] = float2int8(p0[24] * scale);
                pp[8 + 5] = float2int8(p0[25] * scale);
                pp[8 + 6] = float2int8(p0[28] * scale);
                pp[8 + 7] = float2int8(p0[29] * scale);

                pp[16 + 0] = float2int8(p0[2] * scale);
                pp[16 + 1] = float2int8(p0[3] * scale);
                pp[16 + 2] = float2int8(p0[6] * scale);
                pp[16 + 3] = float2int8(p0[7] * scale);
                pp[16 + 4] = float2int8(p0[10] * scale);
                pp[16 + 5] = float2int8(p0[11] * scale);
                pp[16 + 6] = float2int8(p0[14] * scale);
                pp[16 + 7] = float2int8(p0[15] * scale);

                pp[24 + 0] = float2int8(p0[18] * scale);
                pp[24 + 1] = float2int8(p0[19] * scale);
                pp[24 + 2] = float2int8(p0[22] * scale);
                pp[24 + 3] = float2int8(p0[23] * scale);
                pp[24 + 4] = float2int8(p0[26] * scale);
                pp[24 + 5] = float2int8(p0[27] * scale);
                pp[24 + 6] = float2int8(p0[30] * scale);
                pp[24 + 7] = float2int8(p0[31] * scale);

                pp += 32;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + A_hstep);
                float32x4_t _p3 = vld1q_f32(p0 + A_hstep + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p0, _p1);
                _r01.val[1] = float2int8(_p2, _p3);

                vst2_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[A_hstep + 0] * scale);
                // pp[2] = float2int8(p0[1] * scale);
                // pp[3] = float2int8(p0[A_hstep + 1] * scale);
                // pp[4] = float2int8(p0[2] * scale);
                // pp[5] = float2int8(p0[A_hstep + 2] * scale);
                // pp[6] = float2int8(p0[3] * scale);
                // pp[7] = float2int8(p0[A_hstep + 3] * scale);
                // pp[8] = float2int8(p0[4] * scale);
                // pp[9] = float2int8(p0[A_hstep + 4] * scale);
                // pp[10] = float2int8(p0[5] * scale);
                // pp[11] = float2int8(p0[A_hstep + 5] * scale);
                // pp[12] = float2int8(p0[6] * scale);
                // pp[13] = float2int8(p0[A_hstep + 6] * scale);
                // pp[14] = float2int8(p0[7] * scale);
                // pp[15] = float2int8(p0[A_hstep + 7] * scale);

                pp += 16;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[4] * scale);
                pp[5] = float2int8(p0[5] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);
                pp += 8;
                p0 += A_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[4] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[8] * scale);
                pp[5] = float2int8(p0[9] * scale);
                pp[6] = float2int8(p0[12] * scale);
                pp[7] = float2int8(p0[13] * scale);
                pp[8] = float2int8(p0[2] * scale);
                pp[9] = float2int8(p0[3] * scale);
                pp[10] = float2int8(p0[6] * scale);
                pp[11] = float2int8(p0[7] * scale);
                pp[12] = float2int8(p0[10] * scale);
                pp[13] = float2int8(p0[11] * scale);
                pp[14] = float2int8(p0[14] * scale);
                pp[15] = float2int8(p0[15] * scale);

                pp += 16;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + A_hstep);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                int8x8_t _r01 = float2int8(_p01.val[0], _p01.val[1]);

                vst1_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[A_hstep + 0] * scale);
                // pp[2] = float2int8(p0[1] * scale);
                // pp[3] = float2int8(p0[A_hstep + 1] * scale);
                // pp[4] = float2int8(p0[2] * scale);
                // pp[5] = float2int8(p0[A_hstep + 2] * scale);
                // pp[6] = float2int8(p0[3] * scale);
                // pp[7] = float2int8(p0[A_hstep + 3] * scale);
                pp += 8;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[4] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[3] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);
                pp += 8;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            // for (; kk + 3 < max_kk; kk += 4)
            // {
            //     float32x2_t _p0 = vld1_f32(p0);
            //     float32x2_t _p1 = vld1_f32(p0 + A_hstep);
            //     float32x2_t _p2 = vld1_f32(p0 + A_hstep * 2);
            //     float32x2_t _p3 = vld1_f32(p0 + A_hstep * 3);
            //
            //     float32x4_t _p02 = vcombine_f32(_p0, _p2);
            //     float32x4_t _p13 = vcombine_f32(_p1, _p3);
            //
            //     _p02 = vmulq_f32(_p02, _scale);
            //     _p13 = vmulq_f32(_p13, _scale);
            //
            //     float32x4x2_t _p0123 = vzipq_f32(_p02, _p13);
            //
            //     int8x8_t _r01 = float2int8(_p0123.val[0], _p0123.val[1]);
            //
            //     vst1_s8(pp, _r01);
            //
            //     // pp[0] = float2int8(p0[0] * scale);
            //     // pp[1] = float2int8(p0[A_hstep + 0] * scale);
            //     // pp[2] = float2int8(p0[1] * scale);
            //     // pp[3] = float2int8(p0[A_hstep + 1] * scale);
            //     // pp[4] = float2int8(p0[A_hstep * 2] * scale);
            //     // pp[5] = float2int8(p0[A_hstep * 3 + 0] * scale);
            //     // pp[6] = float2int8(p0[A_hstep * 2] * scale);
            //     // pp[7] = float2int8(p0[A_hstep * 3 + 1] * scale);
            //     pp += 8;
            //     p0 += A_hstep * 4;
            // }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[A_hstep + 0] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[A_hstep + 1] * scale);
                pp += 4;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp += 2;
                p0 += A_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp += 4;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0;
                float32x4_t _p1;
                _p0 = vsetq_lane_f32(p0[0], _p0, 0);
                _p0 = vsetq_lane_f32(p0[A_hstep], _p0, 1);
                _p0 = vsetq_lane_f32(p0[A_hstep * 2], _p0, 2);
                _p0 = vsetq_lane_f32(p0[A_hstep * 3], _p0, 3);
                _p1 = vsetq_lane_f32(p0[A_hstep * 4], _p1, 0);
                _p1 = vsetq_lane_f32(p0[A_hstep * 5], _p1, 1);
                _p1 = vsetq_lane_f32(p0[A_hstep * 6], _p1, 2);
                _p1 = vsetq_lane_f32(p0[A_hstep * 7], _p1, 3);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r01 = float2int8(_p0, _p1);

                vst1_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[A_hstep] * scale);
                pp += 8;
                p0 += A_hstep * 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[A_hstep] * scale);
                pp += 2;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp += 1;
                p0 += A_hstep;
            }
        }
    }
}

static void pack_B_tile_fp32_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    // NCNN_LOGE("pack_B_tile_fp32_to_int8 %d %d", max_jj, elempack);

    signed char* pp = BT;

    float32x4_t _scale = vdupq_n_f32(scale);

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;
            const float* p1 = (const float*)B + (j + jj + 4) * B_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p0n = vld1q_f32(p0 + 4);
                float32x4_t _p1 = vld1q_f32(p1);
                float32x4_t _p1n = vld1q_f32(p1 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p0n = vmulq_f32(_p0n, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p1n = vmulq_f32(_p1n, _scale);

                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p0, _p1);
                _r01.val[1] = float2int8(_p0n, _p1n);

                vst2_s8(pp, _r01);

                pp += 16;
                p0 += 8;
                p1 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p1[0] * scale);
                pp[5] = float2int8(p1[1] * scale);
                pp[6] = float2int8(p1[2] * scale);
                pp[7] = float2int8(p1[3] * scale);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
            const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
            const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;
            const float* p4 = (const float*)B + (j + jj + 4) * B_hstep + k;
            const float* p5 = (const float*)B + (j + jj + 5) * B_hstep + k;
            const float* p6 = (const float*)B + (j + jj + 6) * B_hstep + k;
            const float* p7 = (const float*)B + (j + jj + 7) * B_hstep + k;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p01 = vcombine_f32(vld1_f32(p0), vld1_f32(p1));
                float32x4_t _p23 = vcombine_f32(vld1_f32(p2), vld1_f32(p3));
                float32x4_t _p45 = vcombine_f32(vld1_f32(p4), vld1_f32(p5));
                float32x4_t _p67 = vcombine_f32(vld1_f32(p6), vld1_f32(p7));

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);
                _p45 = vmulq_f32(_p45, _scale);
                _p67 = vmulq_f32(_p67, _scale);

                int8x8_t _r0 = float2int8(_p01, _p23);
                int8x8_t _r1 = float2int8(_p45, _p67);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));

                pp += 16;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
                p4 += 2;
                p5 += 2;
                p6 += 2;
                p7 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p1[0] * scale);
                pp[2] = float2int8(p2[0] * scale);
                pp[3] = float2int8(p3[0] * scale);
                pp[4] = float2int8(p4[0] * scale);
                pp[5] = float2int8(p5[0] * scale);
                pp[6] = float2int8(p6[0] * scale);
                pp[7] = float2int8(p7[0] * scale);
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
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                int8x8_t _r01 = float2int8(_p01.val[0], _p01.val[1]);

                vst1_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[4] * scale);
                // pp[2] = float2int8(p0[1] * scale);
                // pp[3] = float2int8(p0[5] * scale);
                // pp[4] = float2int8(p0[2] * scale);
                // pp[5] = float2int8(p0[6] * scale);
                // pp[6] = float2int8(p0[3] * scale);
                // pp[7] = float2int8(p0[7] * scale);
                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
            const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
            const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p01 = vcombine_f32(vld1_f32(p0), vld1_f32(p1));
                float32x4_t _p23 = vcombine_f32(vld1_f32(p2), vld1_f32(p3));

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);

                int8x8_t _r0 = float2int8(_p01, _p23);

                vst1_s8(pp, _r0);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[1] * scale);
                // pp[2] = float2int8(p1[0] * scale);
                // pp[3] = float2int8(p1[1] * scale);
                // pp[4] = float2int8(p2[0] * scale);
                // pp[5] = float2int8(p2[1] * scale);
                // pp[6] = float2int8(p3[0] * scale);
                // pp[7] = float2int8(p3[1] * scale);
                pp += 8;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p1[0] * scale);
                pp[2] = float2int8(p2[0] * scale);
                pp[3] = float2int8(p3[0] * scale);
                pp += 4;
                p0++;
                p1++;
                p2++;
                p3++;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p1[0] * scale);
                pp[3] = float2int8(p1[1] * scale);
                pp += 4;
                p0 += 2;
                p1 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p1[0] * scale);
                pp += 2;
                p0++;
                p1++;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp += 2;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_B_tile_fp32_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    // NCNN_LOGE("transpose_pack_B_tile_fp32_to_int8 %d %d", max_jj, elempack);

    signed char* pp = BT;

    float32x4_t _scale = vdupq_n_f32(scale);

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[4] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[8] * scale);
                pp[5] = float2int8(p0[9] * scale);
                pp[6] = float2int8(p0[12] * scale);
                pp[7] = float2int8(p0[13] * scale);

                pp[8 + 0] = float2int8(p0[16] * scale);
                pp[8 + 1] = float2int8(p0[17] * scale);
                pp[8 + 2] = float2int8(p0[20] * scale);
                pp[8 + 3] = float2int8(p0[21] * scale);
                pp[8 + 4] = float2int8(p0[24] * scale);
                pp[8 + 5] = float2int8(p0[25] * scale);
                pp[8 + 6] = float2int8(p0[28] * scale);
                pp[8 + 7] = float2int8(p0[29] * scale);

                pp[16 + 0] = float2int8(p0[2] * scale);
                pp[16 + 1] = float2int8(p0[3] * scale);
                pp[16 + 2] = float2int8(p0[6] * scale);
                pp[16 + 3] = float2int8(p0[7] * scale);
                pp[16 + 4] = float2int8(p0[10] * scale);
                pp[16 + 5] = float2int8(p0[11] * scale);
                pp[16 + 6] = float2int8(p0[14] * scale);
                pp[16 + 7] = float2int8(p0[15] * scale);

                pp[24 + 0] = float2int8(p0[18] * scale);
                pp[24 + 1] = float2int8(p0[19] * scale);
                pp[24 + 2] = float2int8(p0[22] * scale);
                pp[24 + 3] = float2int8(p0[23] * scale);
                pp[24 + 4] = float2int8(p0[26] * scale);
                pp[24 + 5] = float2int8(p0[27] * scale);
                pp[24 + 6] = float2int8(p0[30] * scale);
                pp[24 + 7] = float2int8(p0[31] * scale);

                pp += 32;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + B_hstep);
                float32x4_t _p3 = vld1q_f32(p0 + B_hstep + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p0, _p1);
                _r01.val[1] = float2int8(_p2, _p3);

                vst2_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[B_hstep + 0] * scale);
                // pp[2] = float2int8(p0[1] * scale);
                // pp[3] = float2int8(p0[B_hstep + 1] * scale);
                // pp[4] = float2int8(p0[2] * scale);
                // pp[5] = float2int8(p0[B_hstep + 2] * scale);
                // pp[6] = float2int8(p0[3] * scale);
                // pp[7] = float2int8(p0[B_hstep + 3] * scale);
                // pp[8] = float2int8(p0[4] * scale);
                // pp[9] = float2int8(p0[B_hstep + 4] * scale);
                // pp[10] = float2int8(p0[5] * scale);
                // pp[11] = float2int8(p0[B_hstep + 5] * scale);
                // pp[12] = float2int8(p0[6] * scale);
                // pp[13] = float2int8(p0[B_hstep + 6] * scale);
                // pp[14] = float2int8(p0[7] * scale);
                // pp[15] = float2int8(p0[B_hstep + 7] * scale);

                pp += 16;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[4] * scale);
                pp[5] = float2int8(p0[5] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);
                pp += 8;
                p0 += B_hstep;
            }
        }
    }
#endif // __aarch64__
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[4] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[8] * scale);
                pp[5] = float2int8(p0[9] * scale);
                pp[6] = float2int8(p0[12] * scale);
                pp[7] = float2int8(p0[13] * scale);
                pp[8] = float2int8(p0[2] * scale);
                pp[9] = float2int8(p0[3] * scale);
                pp[10] = float2int8(p0[6] * scale);
                pp[11] = float2int8(p0[7] * scale);
                pp[12] = float2int8(p0[10] * scale);
                pp[13] = float2int8(p0[11] * scale);
                pp[14] = float2int8(p0[14] * scale);
                pp[15] = float2int8(p0[15] * scale);

                pp += 16;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + B_hstep);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                int8x8_t _r01 = float2int8(_p01.val[0], _p01.val[1]);

                vst1_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[B_hstep + 0] * scale);
                // pp[2] = float2int8(p0[1] * scale);
                // pp[3] = float2int8(p0[B_hstep + 1] * scale);
                // pp[4] = float2int8(p0[2] * scale);
                // pp[5] = float2int8(p0[B_hstep + 2] * scale);
                // pp[6] = float2int8(p0[3] * scale);
                // pp[7] = float2int8(p0[B_hstep + 3] * scale);
                pp += 8;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[4] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[3] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x2_t _p0 = vld1_f32(p0);
                float32x2_t _p1 = vld1_f32(p0 + B_hstep);
                float32x2_t _p2 = vld1_f32(p0 + B_hstep * 2);
                float32x2_t _p3 = vld1_f32(p0 + B_hstep * 3);

                float32x4_t _p02 = vcombine_f32(_p0, _p2);
                float32x4_t _p13 = vcombine_f32(_p1, _p3);

                _p02 = vmulq_f32(_p02, _scale);
                _p13 = vmulq_f32(_p13, _scale);

                float32x4x2_t _p0123 = vzipq_f32(_p02, _p13);

                int8x8_t _r01 = float2int8(_p0123.val[0], _p0123.val[1]);

                vst1_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[B_hstep + 0] * scale);
                // pp[2] = float2int8(p0[1] * scale);
                // pp[3] = float2int8(p0[B_hstep + 1] * scale);
                // pp[4] = float2int8(p0[B_hstep * 2] * scale);
                // pp[5] = float2int8(p0[B_hstep * 3 + 0] * scale);
                // pp[6] = float2int8(p0[B_hstep * 2] * scale);
                // pp[7] = float2int8(p0[B_hstep * 3 + 1] * scale);
                pp += 8;
                p0 += B_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep + 0] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
                pp += 4;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp += 2;
                p0 += B_hstep;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0;
                float32x4_t _p1;
                _p0 = vsetq_lane_f32(p0[0], _p0, 0);
                _p0 = vsetq_lane_f32(p0[B_hstep], _p0, 1);
                _p0 = vsetq_lane_f32(p0[B_hstep * 2], _p0, 2);
                _p0 = vsetq_lane_f32(p0[B_hstep * 3], _p0, 3);
                _p1 = vsetq_lane_f32(p0[B_hstep * 4], _p1, 0);
                _p1 = vsetq_lane_f32(p0[B_hstep * 5], _p1, 1);
                _p1 = vsetq_lane_f32(p0[B_hstep * 6], _p1, 2);
                _p1 = vsetq_lane_f32(p0[B_hstep * 7], _p1, 3);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r01 = float2int8(_p0, _p1);

                vst1_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[B_hstep] * scale);
                pp += 8;
                p0 += B_hstep * 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep] * scale);
                pp += 2;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp += 1;
                p0 += B_hstep;
            }
        }
    }
}

static void unpack_output_tile_int32_to_fp32(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float descale)
{
    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    // NCNN_LOGE("unpack_output_tile_int32_to_fp32  %d %d %d %d  %d  %d", i, max_ii, j, max_jj, out_elempack, broadcast_type_C);

    // if (broadcast_type_C != -1)
    // NCNN_LOGE("unpack_output_tile_int32_to_fp32 %d %d    %d    %d %d %d", out_elempack, C.elempack, broadcast_type_C, C.w, C.h, C.c);

    // TODO __aarch64__
    // TODO __ARM_FEATURE_DOTPROD

    const int c_hstep = C.dims == 3 ? (int)C.cstep * out_elempack : C.w * out_elempack;
    const float* pC = C;

    const int* pp = topT;

    float32x4_t _descale = vdupq_n_f32(descale);

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + (i + ii) * out_hstep + j * 4;
            float* p1 = (float*)top_blob + (i + ii + 4) * out_hstep + j * 4;

            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * 4;
            }

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);
                int32x4_t _sum8 = vld1q_s32(pp + 32);
                int32x4_t _sum9 = vld1q_s32(pp + 36);
                int32x4_t _suma = vld1q_s32(pp + 40);
                int32x4_t _sumb = vld1q_s32(pp + 44);
                int32x4_t _sumc = vld1q_s32(pp + 48);
                int32x4_t _sumd = vld1q_s32(pp + 52);
                int32x4_t _sume = vld1q_s32(pp + 56);
                int32x4_t _sumf = vld1q_s32(pp + 60);

#if __ARM_FEATURE_DOTPROD
                //TODO
#else
                // from
                //      a0 b1 c2 d3
                //      e4 f5 g6 h7
                //      e0 f1 g2 h3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      g4 h5 e6 f7
                //      g0 h1 e2 f3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      e7 f6 g5 h4
                //      e3 f2 g1 h0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      g7 h6 e5 f4
                //      g3 h2 e1 f0
                //      c7 d6 a5 b4

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      a4 b4 c4 d4
                //      a5 b5 c5 d5
                //      a6 b6 c6 d6
                //      a7 b7 c7 d7
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                //      e2 f2 g2 h2
                //      e3 f3 g3 h3
                //      e4 f4 g4 h4
                //      e5 f5 g5 h5
                //      e6 f6 g6 h6
                //      e7 f7 g7 h7
                {
                    _sum8 = vrev64q_s32(_sum8);
                    _sum9 = vrev64q_s32(_sum9);
                    _suma = vrev64q_s32(_suma);
                    _sumb = vrev64q_s32(_sumb);
                    _sumc = vrev64q_s32(_sumc);
                    _sumd = vrev64q_s32(_sumd);
                    _sume = vrev64q_s32(_sume);
                    _sumf = vrev64q_s32(_sumf);
                    _sum8 = vextq_s32(_sum8, _sum8, 2);
                    _sum9 = vextq_s32(_sum9, _sum9, 2);
                    _suma = vextq_s32(_suma, _suma, 2);
                    _sumb = vextq_s32(_sumb, _sumb, 2);
                    _sumc = vextq_s32(_sumc, _sumc, 2);
                    _sumd = vextq_s32(_sumd, _sumd, 2);
                    _sume = vextq_s32(_sume, _sume, 2);
                    _sumf = vextq_s32(_sumf, _sumf, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sumc);
                    int32x4x2_t _t1 = vzipq_s32(_sum4, _sum8);
                    int32x4x2_t _t2 = vzipq_s32(_sum2, _sume);
                    int32x4x2_t _t3 = vzipq_s32(_sum6, _suma);
                    int32x4x2_t _t4 = vzipq_s32(_sum3, _sumf);
                    int32x4x2_t _t5 = vzipq_s32(_sum7, _sumb);
                    int32x4x2_t _t6 = vzipq_s32(_sum1, _sumd);
                    int32x4x2_t _t7 = vzipq_s32(_sum5, _sum9);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum4 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
                    _sum5 = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
                    _sum6 = vcombine_s32(vget_low_s32(_t5.val[1]), vget_low_s32(_t4.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t5.val[1]), vget_high_s32(_t4.val[1]));
                    _sum8 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum9 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _suma = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sumb = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sumc = vcombine_s32(vget_low_s32(_t6.val[0]), vget_low_s32(_t7.val[0]));
                    _sumd = vcombine_s32(vget_high_s32(_t6.val[0]), vget_high_s32(_t7.val[0]));
                    _sume = vcombine_s32(vget_low_s32(_t7.val[1]), vget_low_s32(_t6.val[1]));
                    _sumf = vcombine_s32(vget_high_s32(_t7.val[1]), vget_high_s32(_t6.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum7 = vrev64q_s32(_sum7);
                    _sum9 = vrev64q_s32(_sum9);
                    _sumb = vrev64q_s32(_sumb);
                    _sumd = vrev64q_s32(_sumd);
                    _sumf = vrev64q_s32(_sumf);
                }

#endif

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);
                float32x4_t _f8 = vmulq_f32(vcvtq_f32_s32(_sum8), _descale);
                float32x4_t _f9 = vmulq_f32(vcvtq_f32_s32(_sum9), _descale);
                float32x4_t _fa = vmulq_f32(vcvtq_f32_s32(_suma), _descale);
                float32x4_t _fb = vmulq_f32(vcvtq_f32_s32(_sumb), _descale);
                float32x4_t _fc = vmulq_f32(vcvtq_f32_s32(_sumc), _descale);
                float32x4_t _fd = vmulq_f32(vcvtq_f32_s32(_sumd), _descale);
                float32x4_t _fe = vmulq_f32(vcvtq_f32_s32(_sume), _descale);
                float32x4_t _ff = vmulq_f32(vcvtq_f32_s32(_sumf), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        _f4 = vaddq_f32(_f4, _c);
                        _f5 = vaddq_f32(_f5, _c);
                        _f6 = vaddq_f32(_f6, _c);
                        _f7 = vaddq_f32(_f7, _c);
                        _f8 = vaddq_f32(_f8, _c);
                        _f9 = vaddq_f32(_f9, _c);
                        _fa = vaddq_f32(_fa, _c);
                        _fb = vaddq_f32(_fb, _c);
                        _fc = vaddq_f32(_fc, _c);
                        _fd = vaddq_f32(_fd, _c);
                        _fe = vaddq_f32(_fe, _c);
                        _ff = vaddq_f32(_ff, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c0);
                        _f4 = vaddq_f32(_f4, _c0);
                        _f5 = vaddq_f32(_f5, _c0);
                        _f6 = vaddq_f32(_f6, _c0);
                        _f7 = vaddq_f32(_f7, _c0);
                        _f8 = vaddq_f32(_f8, _c1);
                        _f9 = vaddq_f32(_f9, _c1);
                        _fa = vaddq_f32(_fa, _c1);
                        _fb = vaddq_f32(_fb, _c1);
                        _fc = vaddq_f32(_fc, _c1);
                        _fd = vaddq_f32(_fd, _c1);
                        _fe = vaddq_f32(_fe, _c1);
                        _ff = vaddq_f32(_ff, _c1);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4 * 1);
                        float32x4_t _c2 = vld1q_f32(pC + 4 * 2);
                        float32x4_t _c3 = vld1q_f32(pC + 4 * 3);
                        float32x4_t _c4 = vld1q_f32(pC + 4 * 4);
                        float32x4_t _c5 = vld1q_f32(pC + 4 * 5);
                        float32x4_t _c6 = vld1q_f32(pC + 4 * 6);
                        float32x4_t _c7 = vld1q_f32(pC + 4 * 7);
                        float32x4_t _c8 = vld1q_f32(pC + c_hstep * 4);
                        float32x4_t _c9 = vld1q_f32(pC + c_hstep * 4 + 4 * 1);
                        float32x4_t _ca = vld1q_f32(pC + c_hstep * 4 + 4 * 2);
                        float32x4_t _cb = vld1q_f32(pC + c_hstep * 4 + 4 * 3);
                        float32x4_t _cc = vld1q_f32(pC + c_hstep * 4 + 4 * 4);
                        float32x4_t _cd = vld1q_f32(pC + c_hstep * 4 + 4 * 5);
                        float32x4_t _ce = vld1q_f32(pC + c_hstep * 4 + 4 * 6);
                        float32x4_t _cf = vld1q_f32(pC + c_hstep * 4 + 4 * 7);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        _f8 = vaddq_f32(_f8, _c8);
                        _f9 = vaddq_f32(_f9, _c9);
                        _fa = vaddq_f32(_fa, _ca);
                        _fb = vaddq_f32(_fb, _cb);
                        _fc = vaddq_f32(_fc, _cc);
                        _fd = vaddq_f32(_fd, _cd);
                        _fe = vaddq_f32(_fe, _ce);
                        _ff = vaddq_f32(_ff, _cf);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        float32x4_t _c2 = vdupq_n_f32(pC[2]);
                        float32x4_t _c3 = vdupq_n_f32(pC[3]);
                        float32x4_t _c4 = vdupq_n_f32(pC[4]);
                        float32x4_t _c5 = vdupq_n_f32(pC[5]);
                        float32x4_t _c6 = vdupq_n_f32(pC[6]);
                        float32x4_t _c7 = vdupq_n_f32(pC[7]);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        _f8 = vaddq_f32(_f8, _c0);
                        _f9 = vaddq_f32(_f9, _c1);
                        _fa = vaddq_f32(_fa, _c2);
                        _fb = vaddq_f32(_fb, _c3);
                        _fc = vaddq_f32(_fc, _c4);
                        _fd = vaddq_f32(_fd, _c5);
                        _fe = vaddq_f32(_fe, _c6);
                        _ff = vaddq_f32(_ff, _c7);
                        pC += 8;
                    }
                }

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                vst1q_f32(p0 + 8, _f2);
                vst1q_f32(p0 + 12, _f3);
                vst1q_f32(p0 + 16, _f4);
                vst1q_f32(p0 + 20, _f5);
                vst1q_f32(p0 + 24, _f6);
                vst1q_f32(p0 + 28, _f7);
                vst1q_f32(p1, _f8);
                vst1q_f32(p1 + 4, _f9);
                vst1q_f32(p1 + 8, _fa);
                vst1q_f32(p1 + 12, _fb);
                vst1q_f32(p1 + 16, _fc);
                vst1q_f32(p1 + 20, _fd);
                vst1q_f32(p1 + 24, _fe);
                vst1q_f32(p1 + 28, _ff);

                pp += 64;
                p0 += 32;
                p1 += 32;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);

#if __ARM_FEATURE_DOTPROD
                //TODO
#else
                // from
                //      a0 b1 c2 d3
                //      e0 f1 g2 h3
                //      c0 d1 a2 b3
                //      g0 h1 e2 f3
                //      a3 b2 c1 d0
                //      e3 f2 g1 h0
                //      c3 d2 a1 b0
                //      g3 h2 e1 f0

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                //      e2 f2 g2 h2
                //      e3 f3 g3 h3

                {
                    _sum4 = vrev64q_s32(_sum4);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum6 = vrev64q_s32(_sum6);
                    _sum7 = vrev64q_s32(_sum7);
                    _sum4 = vextq_s32(_sum4, _sum4, 2);
                    _sum5 = vextq_s32(_sum5, _sum5, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                    int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                    int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                    int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum7 = vrev64q_s32(_sum7);
                }
#endif // __ARM_FEATURE_DOTPROD

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        _f4 = vaddq_f32(_f4, _c);
                        _f5 = vaddq_f32(_f5, _c);
                        _f6 = vaddq_f32(_f6, _c);
                        _f7 = vaddq_f32(_f7, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c0);
                        _f4 = vaddq_f32(_f4, _c1);
                        _f5 = vaddq_f32(_f5, _c1);
                        _f6 = vaddq_f32(_f6, _c1);
                        _f7 = vaddq_f32(_f7, _c1);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4 * 1);
                        float32x4_t _c2 = vld1q_f32(pC + 4 * 2);
                        float32x4_t _c3 = vld1q_f32(pC + 4 * 3);
                        float32x4_t _c4 = vld1q_f32(pC + c_hstep * 4);
                        float32x4_t _c5 = vld1q_f32(pC + c_hstep * 4 + 4 * 1);
                        float32x4_t _c6 = vld1q_f32(pC + c_hstep * 4 + 4 * 2);
                        float32x4_t _c7 = vld1q_f32(pC + c_hstep * 4 + 4 * 3);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        float32x4_t _c2 = vdupq_n_f32(pC[2]);
                        float32x4_t _c3 = vdupq_n_f32(pC[3]);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c0);
                        _f5 = vaddq_f32(_f5, _c1);
                        _f6 = vaddq_f32(_f6, _c2);
                        _f7 = vaddq_f32(_f7, _c3);
                        pC += 4;
                    }
                }

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                vst1q_f32(p0 + 8, _f2);
                vst1q_f32(p0 + 12, _f3);
                vst1q_f32(p1, _f4);
                vst1q_f32(p1 + 4, _f5);
                vst1q_f32(p1 + 8, _f6);
                vst1q_f32(p1 + 12, _f7);

                pp += 32;
                p0 += 16;
                p1 += 16;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                // from
                //      a0 b1 c0 d1
                //      e0 f1 g0 h1
                //      a1 b0 c1 d0
                //      e1 f0 g1 h0

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                {
                    _sum2 = vrev64q_s32(_sum2);
                    _sum3 = vrev64q_s32(_sum3);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum2);
                    int32x4x2_t _t1 = vzipq_s32(_sum1, _sum3);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t1.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t1.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c1);
                        _f3 = vaddq_f32(_f3, _c1);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep * 4);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep * 4 + 4);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c1);
                        pC += 2;
                    }
                }

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                vst1q_f32(p1, _f2);
                vst1q_f32(p1 + 4, _f3);

                pp += 16;
                p0 += 8;
                p1 += 8;
            }
            for (; jj < max_jj; jj++)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + c_hstep * 4);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        pC += 1;
                    }
                }

                vst1q_f32(p0, _f0);
                vst1q_f32(p1, _f1);

                pp += 8;
                p0 += 4;
                p1 += 4;
            }
        }
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + (i + ii) * out_hstep + j;
            float* p1 = (float*)top_blob + (i + ii + 1) * out_hstep + j;
            float* p2 = (float*)top_blob + (i + ii + 2) * out_hstep + j;
            float* p3 = (float*)top_blob + (i + ii + 3) * out_hstep + j;
            float* p4 = (float*)top_blob + (i + ii + 4) * out_hstep + j;
            float* p5 = (float*)top_blob + (i + ii + 5) * out_hstep + j;
            float* p6 = (float*)top_blob + (i + ii + 6) * out_hstep + j;
            float* p7 = (float*)top_blob + (i + ii + 7) * out_hstep + j;

            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);
                int32x4_t _sum8 = vld1q_s32(pp + 32);
                int32x4_t _sum9 = vld1q_s32(pp + 36);
                int32x4_t _suma = vld1q_s32(pp + 40);
                int32x4_t _sumb = vld1q_s32(pp + 44);
                int32x4_t _sumc = vld1q_s32(pp + 48);
                int32x4_t _sumd = vld1q_s32(pp + 52);
                int32x4_t _sume = vld1q_s32(pp + 56);
                int32x4_t _sumf = vld1q_s32(pp + 60);

                // from
                //      a0 b1 c2 d3
                //      e4 f5 g6 h7
                //      e0 f1 g2 h3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      g4 h5 e6 f7
                //      g0 h1 e2 f3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      e7 f6 g5 h4
                //      e3 f2 g1 h0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      g7 h6 e5 f4
                //      g3 h2 e1 f0
                //      c7 d6 a5 b4

                // to
                //      a0 a1 a2 a3
                //      a4 a5 a6 a7
                //      b0 b1 b2 b3
                //      b4 b5 b6 b7
                //      c0 c1 c2 c3
                //      c4 c5 c6 c7
                //      d0 d1 d2 d3
                //      d4 d5 d6 d7
                //      e0 e1 e2 e3
                //      e4 e5 e6 e7
                //      f0 f1 f2 f3
                //      f4 f5 f6 f7
                //      g0 g1 g2 g3
                //      g4 g5 g6 g7
                //      h0 h1 h2 h3
                //      h4 h5 h6 h7
                {
                    _sum4 = vextq_s32(_sum4, _sum4, 2);
                    _sum5 = vextq_s32(_sum5, _sum5, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    _sumc = vextq_s32(_sumc, _sumc, 2);
                    _sumd = vextq_s32(_sumd, _sumd, 2);
                    _sume = vextq_s32(_sume, _sume, 2);
                    _sumf = vextq_s32(_sumf, _sumf, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sumc);
                    int32x4x2_t _t1 = vzipq_s32(_sum4, _sum8);
                    int32x4x2_t _t2 = vzipq_s32(_sum3, _sumf);
                    int32x4x2_t _t3 = vzipq_s32(_sum7, _sumb);
                    int32x4x2_t _t4 = vzipq_s32(_sum2, _sume);
                    int32x4x2_t _t5 = vzipq_s32(_sum6, _suma);
                    int32x4x2_t _t6 = vzipq_s32(_sum1, _sumd);
                    int32x4x2_t _t7 = vzipq_s32(_sum5, _sum9);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum8 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
                    _sum9 = vcombine_s32(vget_low_s32(_t6.val[0]), vget_low_s32(_t7.val[0]));
                    _suma = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
                    _sumb = vcombine_s32(vget_high_s32(_t6.val[0]), vget_high_s32(_t7.val[0]));
                    _sumc = vcombine_s32(vget_low_s32(_t5.val[1]), vget_low_s32(_t4.val[1]));
                    _sumd = vcombine_s32(vget_low_s32(_t7.val[1]), vget_low_s32(_t6.val[1]));
                    _sume = vcombine_s32(vget_high_s32(_t5.val[1]), vget_high_s32(_t4.val[1]));
                    _sumf = vcombine_s32(vget_high_s32(_t7.val[1]), vget_high_s32(_t6.val[1]));
                    _sum2 = vrev64q_s32(_sum2);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum6 = vrev64q_s32(_sum6);
                    _sum7 = vrev64q_s32(_sum7);
                    _suma = vrev64q_s32(_suma);
                    _sumb = vrev64q_s32(_sumb);
                    _sume = vrev64q_s32(_sume);
                    _sumf = vrev64q_s32(_sumf);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);
                float32x4_t _f8 = vmulq_f32(vcvtq_f32_s32(_sum8), _descale);
                float32x4_t _f9 = vmulq_f32(vcvtq_f32_s32(_sum9), _descale);
                float32x4_t _fa = vmulq_f32(vcvtq_f32_s32(_suma), _descale);
                float32x4_t _fb = vmulq_f32(vcvtq_f32_s32(_sumb), _descale);
                float32x4_t _fc = vmulq_f32(vcvtq_f32_s32(_sumc), _descale);
                float32x4_t _fd = vmulq_f32(vcvtq_f32_s32(_sumd), _descale);
                float32x4_t _fe = vmulq_f32(vcvtq_f32_s32(_sume), _descale);
                float32x4_t _ff = vmulq_f32(vcvtq_f32_s32(_sumf), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        _f4 = vaddq_f32(_f4, _c);
                        _f5 = vaddq_f32(_f5, _c);
                        _f6 = vaddq_f32(_f6, _c);
                        _f7 = vaddq_f32(_f7, _c);
                        _f8 = vaddq_f32(_f8, _c);
                        _f9 = vaddq_f32(_f9, _c);
                        _fa = vaddq_f32(_fa, _c);
                        _fb = vaddq_f32(_fb, _c);
                        _fc = vaddq_f32(_fc, _c);
                        _fd = vaddq_f32(_fd, _c);
                        _fe = vaddq_f32(_fe, _c);
                        _ff = vaddq_f32(_ff, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        float32x4_t _c2 = vdupq_n_f32(pC[2]);
                        float32x4_t _c3 = vdupq_n_f32(pC[3]);
                        float32x4_t _c4 = vdupq_n_f32(pC[4]);
                        float32x4_t _c5 = vdupq_n_f32(pC[5]);
                        float32x4_t _c6 = vdupq_n_f32(pC[6]);
                        float32x4_t _c7 = vdupq_n_f32(pC[7]);
                        float32x4_t _c8 = vdupq_n_f32(pC[8]);
                        float32x4_t _c9 = vdupq_n_f32(pC[9]);
                        float32x4_t _ca = vdupq_n_f32(pC[10]);
                        float32x4_t _cb = vdupq_n_f32(pC[11]);
                        float32x4_t _cc = vdupq_n_f32(pC[12]);
                        float32x4_t _cd = vdupq_n_f32(pC[13]);
                        float32x4_t _ce = vdupq_n_f32(pC[14]);
                        float32x4_t _cf = vdupq_n_f32(pC[15]);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        _f8 = vaddq_f32(_f8, _c8);
                        _f9 = vaddq_f32(_f9, _c9);
                        _fa = vaddq_f32(_fa, _ca);
                        _fb = vaddq_f32(_fb, _cb);
                        _fc = vaddq_f32(_fc, _cc);
                        _fd = vaddq_f32(_fd, _cd);
                        _fe = vaddq_f32(_fe, _ce);
                        _ff = vaddq_f32(_ff, _cf);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep + 4);
                        float32x4_t _c4 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c5 = vld1q_f32(pC + c_hstep * 2 + 4);
                        float32x4_t _c6 = vld1q_f32(pC + c_hstep * 3);
                        float32x4_t _c7 = vld1q_f32(pC + c_hstep * 3 + 4);
                        float32x4_t _c8 = vld1q_f32(pC + c_hstep * 4);
                        float32x4_t _c9 = vld1q_f32(pC + c_hstep * 4 + 4);
                        float32x4_t _ca = vld1q_f32(pC + c_hstep * 5);
                        float32x4_t _cb = vld1q_f32(pC + c_hstep * 5 + 4);
                        float32x4_t _cc = vld1q_f32(pC + c_hstep * 6);
                        float32x4_t _cd = vld1q_f32(pC + c_hstep * 6 + 4);
                        float32x4_t _ce = vld1q_f32(pC + c_hstep * 7);
                        float32x4_t _cf = vld1q_f32(pC + c_hstep * 7 + 4);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        _f8 = vaddq_f32(_f8, _c8);
                        _f9 = vaddq_f32(_f9, _c9);
                        _fa = vaddq_f32(_fa, _ca);
                        _fb = vaddq_f32(_fb, _cb);
                        _fc = vaddq_f32(_fc, _cc);
                        _fd = vaddq_f32(_fd, _cd);
                        _fe = vaddq_f32(_fe, _ce);
                        _ff = vaddq_f32(_ff, _cf);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        // float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        // float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        // float32x4_t _c2 = vdupq_n_f32(pC[2]);
                        // float32x4_t _c3 = vdupq_n_f32(pC[3]);
                        // float32x4_t _c4 = vdupq_n_f32(pC[4]);
                        // float32x4_t _c5 = vdupq_n_f32(pC[5]);
                        // float32x4_t _c6 = vdupq_n_f32(pC[6]);
                        // float32x4_t _c7 = vdupq_n_f32(pC[7]);
                        // _f0 = vaddq_f32(_f0, _c0);
                        // _f1 = vaddq_f32(_f1, _c1);
                        // _f2 = vaddq_f32(_f2, _c2);
                        // _f3 = vaddq_f32(_f3, _c3);
                        // _f4 = vaddq_f32(_f4, _c4);
                        // _f5 = vaddq_f32(_f5, _c5);
                        // _f6 = vaddq_f32(_f6, _c6);
                        // _f7 = vaddq_f32(_f7, _c7);
                        // _f8 = vaddq_f32(_f8, _c0);
                        // _f9 = vaddq_f32(_f9, _c1);
                        // _fa = vaddq_f32(_fa, _c2);
                        // _fb = vaddq_f32(_fb, _c3);
                        // _fc = vaddq_f32(_fc, _c4);
                        // _fd = vaddq_f32(_fd, _c5);
                        // _fe = vaddq_f32(_fe, _c6);
                        // _ff = vaddq_f32(_ff, _c7);
                        // pC += 8;

                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c1);
                        _f4 = vaddq_f32(_f4, _c0);
                        _f5 = vaddq_f32(_f5, _c1);
                        _f6 = vaddq_f32(_f6, _c0);
                        _f7 = vaddq_f32(_f7, _c1);
                        _f8 = vaddq_f32(_f8, _c0);
                        _f9 = vaddq_f32(_f9, _c1);
                        _fa = vaddq_f32(_fa, _c0);
                        _fb = vaddq_f32(_fb, _c1);
                        _fc = vaddq_f32(_fc, _c0);
                        _fd = vaddq_f32(_fd, _c1);
                        _fe = vaddq_f32(_fe, _c0);
                        _ff = vaddq_f32(_ff, _c1);
                        pC += 8;
                    }
                }

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                vst1q_f32(p1, _f2);
                vst1q_f32(p1 + 4, _f3);
                vst1q_f32(p2, _f4);
                vst1q_f32(p2 + 4, _f5);
                vst1q_f32(p3, _f6);
                vst1q_f32(p3 + 4, _f7);
                vst1q_f32(p4, _f8);
                vst1q_f32(p4 + 4, _f9);
                vst1q_f32(p5, _fa);
                vst1q_f32(p5 + 4, _fb);
                vst1q_f32(p6, _fc);
                vst1q_f32(p6 + 4, _fd);
                vst1q_f32(p7, _fe);
                vst1q_f32(p7 + 4, _ff);

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
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);

                // from
                //      a0 b1 c2 d3
                //      e0 f1 g2 h3
                //      c0 d1 a2 b3
                //      g0 h1 e2 f3
                //      a3 b2 c1 d0
                //      e3 f2 g1 h0
                //      c3 d2 a1 b0
                //      g3 h2 e1 f0

                // to
                //      a0 a1 a2 a3
                //      b0 b1 b2 b3
                //      c0 c1 c2 c3
                //      d0 d1 d2 d3
                //      e0 e1 e2 e3
                //      f0 f1 f2 f3
                //      g0 g1 g2 g3
                //      h0 h1 h2 h3
                {
                    _sum2 = vextq_s32(_sum2, _sum2, 2);
                    _sum3 = vextq_s32(_sum3, _sum3, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                    int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                    int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                    int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum7 = vrev64q_s32(_sum7);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        _f4 = vaddq_f32(_f4, _c);
                        _f5 = vaddq_f32(_f5, _c);
                        _f6 = vaddq_f32(_f6, _c);
                        _f7 = vaddq_f32(_f7, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        float32x4_t _c2 = vdupq_n_f32(pC[2]);
                        float32x4_t _c3 = vdupq_n_f32(pC[3]);
                        float32x4_t _c4 = vdupq_n_f32(pC[4]);
                        float32x4_t _c5 = vdupq_n_f32(pC[5]);
                        float32x4_t _c6 = vdupq_n_f32(pC[6]);
                        float32x4_t _c7 = vdupq_n_f32(pC[7]);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + c_hstep * 1);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep * 3);
                        float32x4_t _c4 = vld1q_f32(pC + c_hstep * 4);
                        float32x4_t _c5 = vld1q_f32(pC + c_hstep * 5);
                        float32x4_t _c6 = vld1q_f32(pC + c_hstep * 6);
                        float32x4_t _c7 = vld1q_f32(pC + c_hstep * 7);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        // float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        // float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        // float32x4_t _c2 = vdupq_n_f32(pC[2]);
                        // float32x4_t _c3 = vdupq_n_f32(pC[3]);
                        // _f0 = vaddq_f32(_f0, _c0);
                        // _f1 = vaddq_f32(_f1, _c1);
                        // _f2 = vaddq_f32(_f2, _c2);
                        // _f3 = vaddq_f32(_f3, _c3);
                        // _f4 = vaddq_f32(_f4, _c0);
                        // _f5 = vaddq_f32(_f5, _c1);
                        // _f6 = vaddq_f32(_f6, _c2);
                        // _f7 = vaddq_f32(_f7, _c3);
                        // pC += 4;

                        float32x4_t _c = vld1q_f32(pC);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        _f4 = vaddq_f32(_f4, _c);
                        _f5 = vaddq_f32(_f5, _c);
                        _f6 = vaddq_f32(_f6, _c);
                        _f7 = vaddq_f32(_f7, _c);
                        pC += 4;
                    }
                }

                vst1q_f32(p0, _f0);
                vst1q_f32(p1, _f1);
                vst1q_f32(p2, _f2);
                vst1q_f32(p3, _f3);
                vst1q_f32(p4, _f4);
                vst1q_f32(p5, _f5);
                vst1q_f32(p6, _f6);
                vst1q_f32(p7, _f7);

                pp += 32;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
                p4 += 4;
                p5 += 4;
                p6 += 4;
                p7 += 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                // from
                //      a0 b1 c0 d1
                //      e0 f1 g0 h1
                //      a1 b0 c1 d0
                //      e1 f0 g1 h0

                // to
                //      a0 a1 c0 c1
                //      b0 b1 d0 d1
                //      e0 e1 g0 g1
                //      f0 f1 h0 h1
                {
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum2);
                    int32x4x2_t _t1 = vzipq_s32(_sum1, _sum3);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t1.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t1.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c1);
                        _f3 = vaddq_f32(_f3, _c1);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep * 1);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep * 2);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        // float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        // float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        // _f0 = vaddq_f32(_f0, _c0);
                        // _f1 = vaddq_f32(_f1, _c1);
                        // _f2 = vaddq_f32(_f2, _c0);
                        // _f3 = vaddq_f32(_f3, _c1);
                        // pC += 2;

                        float32x2_t _c0 = vld1_f32(pC);
                        float32x4_t _c = vcombine_f32(_c0, _c0);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        pC += 2;
                    }
                }

                vst1_f32(p0, vget_low_f32(_f0));
                vst1_f32(p1, vget_low_f32(_f1));
                vst1_f32(p2, vget_high_f32(_f0));
                vst1_f32(p3, vget_high_f32(_f1));
                vst1_f32(p4, vget_low_f32(_f2));
                vst1_f32(p5, vget_low_f32(_f3));
                vst1_f32(p6, vget_high_f32(_f2));
                vst1_f32(p7, vget_high_f32(_f3));

                pp += 16;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
                p4 += 2;
                p5 += 2;
                p6 += 2;
                p7 += 2;
            }
            for (; jj < max_jj; jj++)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + c_hstep);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        pC += 1;
                    }
                }

                p0[0] = vgetq_lane_f32(_f0, 0);
                p1[0] = vgetq_lane_f32(_f0, 1);
                p2[0] = vgetq_lane_f32(_f0, 2);
                p3[0] = vgetq_lane_f32(_f0, 3);
                p4[0] = vgetq_lane_f32(_f1, 0);
                p5[0] = vgetq_lane_f32(_f1, 1);
                p6[0] = vgetq_lane_f32(_f1, 2);
                p7[0] = vgetq_lane_f32(_f1, 3);

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
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + (i + ii) * out_hstep + j * 4;

            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * 4;
            }

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);

                // from
                //      a0 b1 c2 d3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      c7 d6 a5 b4

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      a4 b4 c4 d4
                //      a5 b5 c5 d5
                //      a6 b6 c6 d6
                //      a7 b7 c7 d7
                {
                    _sum4 = vrev64q_s32(_sum4);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum6 = vrev64q_s32(_sum6);
                    _sum7 = vrev64q_s32(_sum7);
                    _sum4 = vextq_s32(_sum4, _sum4, 2);
                    _sum5 = vextq_s32(_sum5, _sum5, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                    int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                    int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                    int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum7 = vrev64q_s32(_sum7);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        _f4 = vaddq_f32(_f4, _c);
                        _f5 = vaddq_f32(_f5, _c);
                        _f6 = vaddq_f32(_f6, _c);
                        _f7 = vaddq_f32(_f7, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c0);
                        _f4 = vaddq_f32(_f4, _c0);
                        _f5 = vaddq_f32(_f5, _c0);
                        _f6 = vaddq_f32(_f6, _c0);
                        _f7 = vaddq_f32(_f7, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4 * 1);
                        float32x4_t _c2 = vld1q_f32(pC + 4 * 2);
                        float32x4_t _c3 = vld1q_f32(pC + 4 * 3);
                        float32x4_t _c4 = vld1q_f32(pC + 4 * 4);
                        float32x4_t _c5 = vld1q_f32(pC + 4 * 5);
                        float32x4_t _c6 = vld1q_f32(pC + 4 * 6);
                        float32x4_t _c7 = vld1q_f32(pC + 4 * 7);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        float32x4_t _c2 = vdupq_n_f32(pC[2]);
                        float32x4_t _c3 = vdupq_n_f32(pC[3]);
                        float32x4_t _c4 = vdupq_n_f32(pC[4]);
                        float32x4_t _c5 = vdupq_n_f32(pC[5]);
                        float32x4_t _c6 = vdupq_n_f32(pC[6]);
                        float32x4_t _c7 = vdupq_n_f32(pC[7]);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        pC += 8;
                    }
                }

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                vst1q_f32(p0 + 8, _f2);
                vst1q_f32(p0 + 12, _f3);
                vst1q_f32(p0 + 16, _f4);
                vst1q_f32(p0 + 20, _f5);
                vst1q_f32(p0 + 24, _f6);
                vst1q_f32(p0 + 28, _f7);

                pp += 32;
                p0 += 32;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                // from
                //      a0 b1 c2 d3
                //      c0 d1 a2 b3
                //      a3 b2 c1 d0
                //      c3 d2 a1 b0

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                {
                    _sum2 = vrev64q_s32(_sum2);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum2 = vextq_s32(_sum2, _sum2, 2);
                    _sum3 = vextq_s32(_sum3, _sum3, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum3);
                    int32x4x2_t _t1 = vzipq_s32(_sum1, _sum2);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4 * 1);
                        float32x4_t _c2 = vld1q_f32(pC + 4 * 2);
                        float32x4_t _c3 = vld1q_f32(pC + 4 * 3);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        float32x4_t _c2 = vdupq_n_f32(pC[2]);
                        float32x4_t _c3 = vdupq_n_f32(pC[3]);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        pC += 4;
                    }
                }

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                vst1q_f32(p0 + 8, _f2);
                vst1q_f32(p0 + 12, _f3);

                pp += 16;
                p0 += 16;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);

                // from
                //      a0 b1 c0 d1
                //      a1 b0 c1 d0

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                {
                    _sum1 = vrev64q_s32(_sum1);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        pC += 2;
                    }
                }

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);

                pp += 8;
                p0 += 8;
            }
            for (; jj < max_jj; jj++)
            {
                int32x4_t _sum0 = vld1q_s32(pp);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        _f0 = vaddq_f32(_f0, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        _f0 = vaddq_f32(_f0, _c0);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c0);
                        pC += 1;
                    }
                }

                vst1q_f32(p0, _f0);

                pp += 4;
                p0 += 4;
            }
        }
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + (i + ii) * out_hstep + j;
            float* p1 = (float*)top_blob + (i + ii + 1) * out_hstep + j;
            float* p2 = (float*)top_blob + (i + ii + 2) * out_hstep + j;
            float* p3 = (float*)top_blob + (i + ii + 3) * out_hstep + j;

            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);

                // from
                //      a0 b1 c2 d3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      c7 d6 a5 b4

                // to
                //      a0 a1 a2 a3
                //      a4 a5 a6 a7
                //      b0 b1 b2 b3
                //      b4 b5 b6 b7
                //      c0 c1 c2 c3
                //      c4 c5 c6 c7
                //      d0 d1 d2 d3
                //      d4 d5 d6 d7
                {
                    _sum2 = vextq_s32(_sum2, _sum2, 2);
                    _sum3 = vextq_s32(_sum3, _sum3, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                    int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                    int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                    int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum2 = vrev64q_s32(_sum2);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum6 = vrev64q_s32(_sum6);
                    _sum7 = vrev64q_s32(_sum7);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        _f4 = vaddq_f32(_f4, _c);
                        _f5 = vaddq_f32(_f5, _c);
                        _f6 = vaddq_f32(_f6, _c);
                        _f7 = vaddq_f32(_f7, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c0);
                        _f4 = vaddq_f32(_f4, _c0);
                        _f5 = vaddq_f32(_f5, _c0);
                        _f6 = vaddq_f32(_f6, _c0);
                        _f7 = vaddq_f32(_f7, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep + 4);
                        float32x4_t _c4 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c5 = vld1q_f32(pC + c_hstep * 2 + 4);
                        float32x4_t _c6 = vld1q_f32(pC + c_hstep * 3);
                        float32x4_t _c7 = vld1q_f32(pC + c_hstep * 3 + 4);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        // float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        // float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        // float32x4_t _c2 = vdupq_n_f32(pC[2]);
                        // float32x4_t _c3 = vdupq_n_f32(pC[3]);
                        // float32x4_t _c4 = vdupq_n_f32(pC[4]);
                        // float32x4_t _c5 = vdupq_n_f32(pC[5]);
                        // float32x4_t _c6 = vdupq_n_f32(pC[6]);
                        // float32x4_t _c7 = vdupq_n_f32(pC[7]);
                        // _f0 = vaddq_f32(_f0, _c0);
                        // _f1 = vaddq_f32(_f1, _c1);
                        // _f2 = vaddq_f32(_f2, _c2);
                        // _f3 = vaddq_f32(_f3, _c3);
                        // _f4 = vaddq_f32(_f4, _c4);
                        // _f5 = vaddq_f32(_f5, _c5);
                        // _f6 = vaddq_f32(_f6, _c6);
                        // _f7 = vaddq_f32(_f7, _c7);
                        // pC += 8;

                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c1);
                        _f4 = vaddq_f32(_f4, _c0);
                        _f5 = vaddq_f32(_f5, _c1);
                        _f6 = vaddq_f32(_f6, _c0);
                        _f7 = vaddq_f32(_f7, _c1);
                        pC += 8;
                    }
                }

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                vst1q_f32(p1, _f2);
                vst1q_f32(p1 + 4, _f3);
                vst1q_f32(p2, _f4);
                vst1q_f32(p2 + 4, _f5);
                vst1q_f32(p3, _f6);
                vst1q_f32(p3 + 4, _f7);

                pp += 32;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                // from
                //      a0 b1 c2 d3
                //      c0 d1 a2 b3
                //      a3 b2 c1 d0
                //      c3 d2 a1 b0

                // to
                //      a0 a1 a2 a3
                //      b0 b1 b2 b3
                //      c0 c1 c2 c3
                //      d0 d1 d2 d3
                {
                    _sum1 = vextq_s32(_sum1, _sum1, 2);
                    _sum3 = vextq_s32(_sum3, _sum3, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum3);
                    int32x4x2_t _t1 = vzipq_s32(_sum1, _sum2);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + c_hstep * 1);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep * 3);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        // float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        // float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        // float32x4_t _c2 = vdupq_n_f32(pC[2]);
                        // float32x4_t _c3 = vdupq_n_f32(pC[3]);
                        // _f0 = vaddq_f32(_f0, _c0);
                        // _f1 = vaddq_f32(_f1, _c1);
                        // _f2 = vaddq_f32(_f2, _c2);
                        // _f3 = vaddq_f32(_f3, _c3);
                        // pC += 4;

                        float32x4_t _c = vld1q_f32(pC);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        pC += 4;
                    }
                }

                vst1q_f32(p0, _f0);
                vst1q_f32(p1, _f1);
                vst1q_f32(p2, _f2);
                vst1q_f32(p3, _f3);

                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);

                // from
                //      a0 b1 c0 d1
                //      a1 b0 c1 d0

                // to
                //      a0 a1 c0 c1
                //      b0 b1 d0 d1
                {
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                    _sum1 = vrev64q_s32(_sum1);

                    int32x4_t _tmp0 = _sum0;
                    int32x4_t _tmp1 = _sum1;

                    _sum0 = vcombine_s32(vget_low_s32(_tmp0), vget_low_s32(_tmp1));
                    _sum1 = vcombine_s32(vget_high_s32(_tmp0), vget_high_s32(_tmp1));
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + c_hstep);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        // float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        // float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        // _f0 = vaddq_f32(_f0, _c0);
                        // _f1 = vaddq_f32(_f1, _c1);
                        // pC += 2;

                        float32x2_t _c0 = vld1_f32(pC);
                        float32x4_t _c = vcombine_f32(_c0, _c0);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        pC += 2;
                    }
                }

                vst1_f32(p0, vget_low_f32(_f0));
                vst1_f32(p1, vget_high_f32(_f0));
                vst1_f32(p2, vget_low_f32(_f1));
                vst1_f32(p3, vget_high_f32(_f1));

                pp += 8;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
            }
            for (; jj < max_jj; jj++)
            {
                int32x4_t _sum0 = vld1q_s32(pp);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        _f0 = vaddq_f32(_f0, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        _f0 = vaddq_f32(_f0, _c0);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c0);
                        pC += 1;
                    }
                }

                p0[0] = vgetq_lane_f32(_f0, 0);
                p1[0] = vgetq_lane_f32(_f0, 1);
                p2[0] = vgetq_lane_f32(_f0, 2);
                p3[0] = vgetq_lane_f32(_f0, 3);

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
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        // if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + (i + ii) * out_hstep + j;
            float* p1 = (float*)top_blob + (i + ii + 1) * out_hstep + j;

            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep + 4);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        // float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        // float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        // _f0 = vaddq_f32(_f0, _c0);
                        // _f1 = vaddq_f32(_f1, _c1);
                        // pC += 2;

                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c1);
                        pC += 8;
                    }
                }

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                vst1q_f32(p1, _f2);
                vst1q_f32(p1 + 4, _f3);

                pp += 16;
                p0 += 8;
                p1 += 8;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + c_hstep * 1);
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        // float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        // float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        // _f0 = vaddq_f32(_f0, _c0);
                        // _f1 = vaddq_f32(_f1, _c1);
                        // pC += 2;

                        float32x4_t _c = vld1q_f32(pC);
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        pC += 4;
                    }
                }

                vst1q_f32(p0, _f0);
                vst1q_f32(p1, _f1);

                pp += 8;
                p0 += 4;
                p1 += 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                p0[0] = pp[0] * descale;
                p0[1] = pp[1] * descale;
                p1[0] = pp[2] * descale;
                p1[1] = pp[3] * descale;

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        p0[0] += pC[0];
                        p0[1] += pC[0];
                        p1[0] += pC[0];
                        p1[1] += pC[0];
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        p0[0] += pC[0];
                        p0[1] += pC[0];
                        p1[0] += pC[1];
                        p1[1] += pC[1];
                    }
                    if (broadcast_type_C == 3)
                    {
                        p0[0] += pC[0];
                        p0[1] += pC[c_hstep];
                        p1[0] += pC[1];
                        p1[1] += pC[c_hstep + 1];
                        pC += 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        p0[0] += pC[0];
                        p0[1] += pC[1];
                        p1[0] += pC[0];
                        p1[1] += pC[1];
                        pC += 2;
                    }
                }

                // NCNN_LOGE("%f %f", p0[1], p1[0]);

                pp += 4;
                p0 += 2;
                p1 += 2;
            }
            for (; jj < max_jj; jj++)
            {
                p0[0] = pp[0] * descale;
                p1[0] = pp[1] * descale;

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        p0[0] += pC[0];
                        p1[0] += pC[0];
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        p0[0] += pC[0];
                        p1[0] += pC[0];
                    }
                    if (broadcast_type_C == 3)
                    {
                        p0[0] += pC[0];
                        p1[0] += pC[c_hstep];
                        pC += 1;
                    }
                    if (broadcast_type_C == 4)
                    {
                        p0[0] += pC[0];
                        p1[0] += pC[0];
                        pC += 1;
                    }
                }

                pp += 2;
                p0++;
                p1++;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        // if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + (i + ii) * out_hstep + j;

            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }

            int jj = 0;
            //TODO
            // for (; jj + 1 < max_jj; jj += 2)
            // {
            //     p0[0] = pp[0] * descale;
            //     p0[1] = pp[1] * descale;
            //     pp += 2;
            //     p0 += 2;
            // }
            for (; jj < max_jj; jj++)
            {
                p0[0] = pp[0] * descale;

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        p0[0] += pC[0];
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        p0[0] += pC[0];
                    }
                    if (broadcast_type_C == 3)
                    {
                        p0[0] += pC[0];
                        pC += 1;
                    }
                    if (broadcast_type_C == 4)
                    {
                        p0[0] += pC[0];
                        pC += 1;
                    }
                }

                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_unpack_output_tile_int32_to_fp32(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj, float descale)
{
    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    // NCNN_LOGE("transpose_unpack_output_tile_int32_to_fp32 %d %d %d %d    %d", i, max_ii, j, max_jj, out_elempack);

    // TODO __ARM_FEATURE_DOTPROD
    // TODO broadcast_type_C

    const int* pp = topT;

    float32x4_t _descale = vdupq_n_f32(descale);

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);
                int32x4_t _sum8 = vld1q_s32(pp + 32);
                int32x4_t _sum9 = vld1q_s32(pp + 36);
                int32x4_t _suma = vld1q_s32(pp + 40);
                int32x4_t _sumb = vld1q_s32(pp + 44);
                int32x4_t _sumc = vld1q_s32(pp + 48);
                int32x4_t _sumd = vld1q_s32(pp + 52);
                int32x4_t _sume = vld1q_s32(pp + 56);
                int32x4_t _sumf = vld1q_s32(pp + 60);

                // from
                //      a0 b1 c2 d3
                //      e4 f5 g6 h7
                //      e0 f1 g2 h3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      g4 h5 e6 f7
                //      g0 h1 e2 f3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      e7 f6 g5 h4
                //      e3 f2 g1 h0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      g7 h6 e5 f4
                //      g3 h2 e1 f0
                //      c7 d6 a5 b4

                // to
                //      a0 a1 a2 a3
                //      a4 a5 a6 a7
                //      b0 b1 b2 b3
                //      b4 b5 b6 b7
                //      c0 c1 c2 c3
                //      c4 c5 c6 c7
                //      d0 d1 d2 d3
                //      d4 d5 d6 d7
                //      e0 e1 e2 e3
                //      e4 e5 e6 e7
                //      f0 f1 f2 f3
                //      f4 f5 f6 f7
                //      g0 g1 g2 g3
                //      g4 g5 g6 g7
                //      h0 h1 h2 h3
                //      h4 h5 h6 h7
                {
                    _sum4 = vextq_s32(_sum4, _sum4, 2);
                    _sum5 = vextq_s32(_sum5, _sum5, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    _sumc = vextq_s32(_sumc, _sumc, 2);
                    _sumd = vextq_s32(_sumd, _sumd, 2);
                    _sume = vextq_s32(_sume, _sume, 2);
                    _sumf = vextq_s32(_sumf, _sumf, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sumc);
                    int32x4x2_t _t1 = vzipq_s32(_sum4, _sum8);
                    int32x4x2_t _t2 = vzipq_s32(_sum3, _sumf);
                    int32x4x2_t _t3 = vzipq_s32(_sum7, _sumb);
                    int32x4x2_t _t4 = vzipq_s32(_sum2, _sume);
                    int32x4x2_t _t5 = vzipq_s32(_sum6, _suma);
                    int32x4x2_t _t6 = vzipq_s32(_sum1, _sumd);
                    int32x4x2_t _t7 = vzipq_s32(_sum5, _sum9);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum8 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
                    _sum9 = vcombine_s32(vget_low_s32(_t6.val[0]), vget_low_s32(_t7.val[0]));
                    _suma = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
                    _sumb = vcombine_s32(vget_high_s32(_t6.val[0]), vget_high_s32(_t7.val[0]));
                    _sumc = vcombine_s32(vget_low_s32(_t5.val[1]), vget_low_s32(_t4.val[1]));
                    _sumd = vcombine_s32(vget_low_s32(_t7.val[1]), vget_low_s32(_t6.val[1]));
                    _sume = vcombine_s32(vget_high_s32(_t5.val[1]), vget_high_s32(_t4.val[1]));
                    _sumf = vcombine_s32(vget_high_s32(_t7.val[1]), vget_high_s32(_t6.val[1]));
                    _sum2 = vrev64q_s32(_sum2);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum6 = vrev64q_s32(_sum6);
                    _sum7 = vrev64q_s32(_sum7);
                    _suma = vrev64q_s32(_suma);
                    _sumb = vrev64q_s32(_sumb);
                    _sume = vrev64q_s32(_sume);
                    _sumf = vrev64q_s32(_sumf);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);
                float32x4_t _f8 = vmulq_f32(vcvtq_f32_s32(_sum8), _descale);
                float32x4_t _f9 = vmulq_f32(vcvtq_f32_s32(_sum9), _descale);
                float32x4_t _fa = vmulq_f32(vcvtq_f32_s32(_suma), _descale);
                float32x4_t _fb = vmulq_f32(vcvtq_f32_s32(_sumb), _descale);
                float32x4_t _fc = vmulq_f32(vcvtq_f32_s32(_sumc), _descale);
                float32x4_t _fd = vmulq_f32(vcvtq_f32_s32(_sumd), _descale);
                float32x4_t _fe = vmulq_f32(vcvtq_f32_s32(_sume), _descale);
                float32x4_t _ff = vmulq_f32(vcvtq_f32_s32(_sumf), _descale);

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f2);
                vst1q_f32(p0 + 8, _f4);
                vst1q_f32(p0 + 12, _f6);
                vst1q_f32(p0 + 16, _f8);
                vst1q_f32(p0 + 20, _fa);
                vst1q_f32(p0 + 24, _fc);
                vst1q_f32(p0 + 28, _fe);
                vst1q_f32(p0 + out_hstep * 4, _f1);
                vst1q_f32(p0 + out_hstep * 4 + 4, _f3);
                vst1q_f32(p0 + out_hstep * 4 + 8, _f5);
                vst1q_f32(p0 + out_hstep * 4 + 12, _f7);
                vst1q_f32(p0 + out_hstep * 4 + 16, _f9);
                vst1q_f32(p0 + out_hstep * 4 + 20, _fb);
                vst1q_f32(p0 + out_hstep * 4 + 24, _fd);
                vst1q_f32(p0 + out_hstep * 4 + 28, _ff);
                pp += 64;
                p0 += out_hstep * 8;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);

#if __ARM_FEATURE_DOTPROD
                //TODO
#else
                // from
                //      a0 b1 c2 d3
                //      e0 f1 g2 h3
                //      c0 d1 a2 b3
                //      g0 h1 e2 f3
                //      a3 b2 c1 d0
                //      e3 f2 g1 h0
                //      c3 d2 a1 b0
                //      g3 h2 e1 f0

                // to
                //      a0 a1 a2 a3
                //      b0 b1 b2 b3
                //      c0 c1 c2 c3
                //      d0 d1 d2 d3
                //      e0 e1 e2 e3
                //      f0 f1 f2 f3
                //      g0 g1 g2 g3
                //      h0 h1 h2 h3
                {
                    _sum2 = vextq_s32(_sum2, _sum2, 2);
                    _sum3 = vextq_s32(_sum3, _sum3, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                    int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                    int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                    int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum7 = vrev64q_s32(_sum7);
                }
#endif // __ARM_FEATURE_DOTPROD

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                vst1q_f32(p0 + 8, _f2);
                vst1q_f32(p0 + 12, _f3);
                vst1q_f32(p0 + 16, _f4);
                vst1q_f32(p0 + 20, _f5);
                vst1q_f32(p0 + 24, _f6);
                vst1q_f32(p0 + 28, _f7);
                pp += 32;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);
                int32x4_t _sum8 = vld1q_s32(pp + 32);
                int32x4_t _sum9 = vld1q_s32(pp + 36);
                int32x4_t _suma = vld1q_s32(pp + 40);
                int32x4_t _sumb = vld1q_s32(pp + 44);
                int32x4_t _sumc = vld1q_s32(pp + 48);
                int32x4_t _sumd = vld1q_s32(pp + 52);
                int32x4_t _sume = vld1q_s32(pp + 56);
                int32x4_t _sumf = vld1q_s32(pp + 60);

                // from
                //      a0 b1 c2 d3
                //      e4 f5 g6 h7
                //      e0 f1 g2 h3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      g4 h5 e6 f7
                //      g0 h1 e2 f3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      e7 f6 g5 h4
                //      e3 f2 g1 h0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      g7 h6 e5 f4
                //      g3 h2 e1 f0
                //      c7 d6 a5 b4

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      a4 b4 c4 d4
                //      a5 b5 c5 d5
                //      a6 b6 c6 d6
                //      a7 b7 c7 d7
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                //      e2 f2 g2 h2
                //      e3 f3 g3 h3
                //      e4 f4 g4 h4
                //      e5 f5 g5 h5
                //      e6 f6 g6 h6
                //      e7 f7 g7 h7
                {
                    _sum8 = vrev64q_s32(_sum8);
                    _sum9 = vrev64q_s32(_sum9);
                    _suma = vrev64q_s32(_suma);
                    _sumb = vrev64q_s32(_sumb);
                    _sumc = vrev64q_s32(_sumc);
                    _sumd = vrev64q_s32(_sumd);
                    _sume = vrev64q_s32(_sume);
                    _sumf = vrev64q_s32(_sumf);
                    _sum8 = vextq_s32(_sum8, _sum8, 2);
                    _sum9 = vextq_s32(_sum9, _sum9, 2);
                    _suma = vextq_s32(_suma, _suma, 2);
                    _sumb = vextq_s32(_sumb, _sumb, 2);
                    _sumc = vextq_s32(_sumc, _sumc, 2);
                    _sumd = vextq_s32(_sumd, _sumd, 2);
                    _sume = vextq_s32(_sume, _sume, 2);
                    _sumf = vextq_s32(_sumf, _sumf, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sumc);
                    int32x4x2_t _t1 = vzipq_s32(_sum4, _sum8);
                    int32x4x2_t _t2 = vzipq_s32(_sum2, _sume);
                    int32x4x2_t _t3 = vzipq_s32(_sum6, _suma);
                    int32x4x2_t _t4 = vzipq_s32(_sum3, _sumf);
                    int32x4x2_t _t5 = vzipq_s32(_sum7, _sumb);
                    int32x4x2_t _t6 = vzipq_s32(_sum1, _sumd);
                    int32x4x2_t _t7 = vzipq_s32(_sum5, _sum9);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum4 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
                    _sum5 = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
                    _sum6 = vcombine_s32(vget_low_s32(_t5.val[1]), vget_low_s32(_t4.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t5.val[1]), vget_high_s32(_t4.val[1]));
                    _sum8 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum9 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _suma = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sumb = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sumc = vcombine_s32(vget_low_s32(_t6.val[0]), vget_low_s32(_t7.val[0]));
                    _sumd = vcombine_s32(vget_high_s32(_t6.val[0]), vget_high_s32(_t7.val[0]));
                    _sume = vcombine_s32(vget_low_s32(_t7.val[1]), vget_low_s32(_t6.val[1]));
                    _sumf = vcombine_s32(vget_high_s32(_t7.val[1]), vget_high_s32(_t6.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum7 = vrev64q_s32(_sum7);
                    _sum9 = vrev64q_s32(_sum9);
                    _sumb = vrev64q_s32(_sumb);
                    _sumd = vrev64q_s32(_sumd);
                    _sumf = vrev64q_s32(_sumf);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);
                float32x4_t _f8 = vmulq_f32(vcvtq_f32_s32(_sum8), _descale);
                float32x4_t _f9 = vmulq_f32(vcvtq_f32_s32(_sum9), _descale);
                float32x4_t _fa = vmulq_f32(vcvtq_f32_s32(_suma), _descale);
                float32x4_t _fb = vmulq_f32(vcvtq_f32_s32(_sumb), _descale);
                float32x4_t _fc = vmulq_f32(vcvtq_f32_s32(_sumc), _descale);
                float32x4_t _fd = vmulq_f32(vcvtq_f32_s32(_sumd), _descale);
                float32x4_t _fe = vmulq_f32(vcvtq_f32_s32(_sume), _descale);
                float32x4_t _ff = vmulq_f32(vcvtq_f32_s32(_sumf), _descale);

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f8);
                vst1q_f32(p0 + out_hstep, _f1);
                vst1q_f32(p0 + out_hstep + 4, _f9);
                vst1q_f32(p0 + out_hstep * 2, _f2);
                vst1q_f32(p0 + out_hstep * 2 + 4, _fa);
                vst1q_f32(p0 + out_hstep * 3, _f3);
                vst1q_f32(p0 + out_hstep * 3 + 4, _fb);
                vst1q_f32(p0 + out_hstep * 4, _f4);
                vst1q_f32(p0 + out_hstep * 4 + 4, _fc);
                vst1q_f32(p0 + out_hstep * 5, _f5);
                vst1q_f32(p0 + out_hstep * 5 + 4, _fd);
                vst1q_f32(p0 + out_hstep * 6, _f6);
                vst1q_f32(p0 + out_hstep * 6 + 4, _fe);
                vst1q_f32(p0 + out_hstep * 7, _f7);
                vst1q_f32(p0 + out_hstep * 7 + 4, _ff);

                pp += 64;
                p0 += out_hstep * 8;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);

                // from
                //      a0 b1 c2 d3
                //      e0 f1 g2 h3
                //      c0 d1 a2 b3
                //      g0 h1 e2 f3
                //      a3 b2 c1 d0
                //      e3 f2 g1 h0
                //      c3 d2 a1 b0
                //      g3 h2 e1 f0

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                //      e2 f2 g2 h2
                //      e3 f3 g3 h3

                {
                    _sum4 = vrev64q_s32(_sum4);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum6 = vrev64q_s32(_sum6);
                    _sum7 = vrev64q_s32(_sum7);
                    _sum4 = vextq_s32(_sum4, _sum4, 2);
                    _sum5 = vextq_s32(_sum5, _sum5, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                    int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                    int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                    int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum7 = vrev64q_s32(_sum7);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f4);
                vst1q_f32(p0 + out_hstep, _f1);
                vst1q_f32(p0 + out_hstep + 4, _f5);
                vst1q_f32(p0 + out_hstep * 2, _f2);
                vst1q_f32(p0 + out_hstep * 2 + 4, _f6);
                vst1q_f32(p0 + out_hstep * 3, _f3);
                vst1q_f32(p0 + out_hstep * 3 + 4, _f7);

                pp += 32;
                p0 += out_hstep * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                // from
                //      a0 b1 c0 d1
                //      e0 f1 g0 h1
                //      a1 b0 c1 d0
                //      e1 f0 g1 h0

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                {
                    _sum2 = vrev64q_s32(_sum2);
                    _sum3 = vrev64q_s32(_sum3);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum2);
                    int32x4x2_t _t1 = vzipq_s32(_sum1, _sum3);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t1.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t1.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f2);
                vst1q_f32(p0 + out_hstep, _f1);
                vst1q_f32(p0 + out_hstep + 4, _f3);

                pp += 16;
                p0 += out_hstep * 2;
            }
            for (; jj < max_jj; jj += 1)
            {
                float32x4_t _r0 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(pp)), _descale);
                float32x4_t _r1 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(pp + 4)), _descale);
                vst1q_f32(p0, _r0);
                vst1q_f32(p0 + 4, _r1);
                pp += 8;
                p0 += out_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);

                // from
                //      a0 b1 c2 d3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      c7 d6 a5 b4

                // to
                //      a0 a1 a2 a3
                //      a4 a5 a6 a7
                //      b0 b1 b2 b3
                //      b4 b5 b6 b7
                //      c0 c1 c2 c3
                //      c4 c5 c6 c7
                //      d0 d1 d2 d3
                //      d4 d5 d6 d7
                {
                    _sum2 = vextq_s32(_sum2, _sum2, 2);
                    _sum3 = vextq_s32(_sum3, _sum3, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                    int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                    int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                    int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum2 = vrev64q_s32(_sum2);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum6 = vrev64q_s32(_sum6);
                    _sum7 = vrev64q_s32(_sum7);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f2);
                vst1q_f32(p0 + 8, _f4);
                vst1q_f32(p0 + 12, _f6);
                vst1q_f32(p0 + out_hstep * 4, _f1);
                vst1q_f32(p0 + out_hstep * 4 + 4, _f3);
                vst1q_f32(p0 + out_hstep * 4 + 8, _f5);
                vst1q_f32(p0 + out_hstep * 4 + 12, _f7);

                pp += 32;
                p0 += out_hstep * 8;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                // from
                //      a0 b1 c2 d3
                //      c0 d1 a2 b3
                //      a3 b2 c1 d0
                //      c3 d2 a1 b0

                // to
                //      a0 a1 a2 a3
                //      b0 b1 b2 b3
                //      c0 c1 c2 c3
                //      d0 d1 d2 d3
                {
                    _sum1 = vextq_s32(_sum1, _sum1, 2);
                    _sum3 = vextq_s32(_sum3, _sum3, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum3);
                    int32x4x2_t _t1 = vzipq_s32(_sum1, _sum2);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                vst1q_f32(p0 + 8, _f2);
                vst1q_f32(p0 + 12, _f3);

                pp += 16;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);

                // from
                //      a0 b1 c2 d3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      c7 d6 a5 b4

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      a4 b4 c4 d4
                //      a5 b5 c5 d5
                //      a6 b6 c6 d6
                //      a7 b7 c7 d7
                {
                    _sum4 = vrev64q_s32(_sum4);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum6 = vrev64q_s32(_sum6);
                    _sum7 = vrev64q_s32(_sum7);
                    _sum4 = vextq_s32(_sum4, _sum4, 2);
                    _sum5 = vextq_s32(_sum5, _sum5, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                    int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                    int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                    int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum7 = vrev64q_s32(_sum7);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + out_hstep, _f1);
                vst1q_f32(p0 + out_hstep * 2, _f2);
                vst1q_f32(p0 + out_hstep * 3, _f3);
                vst1q_f32(p0 + out_hstep * 4, _f4);
                vst1q_f32(p0 + out_hstep * 5, _f5);
                vst1q_f32(p0 + out_hstep * 6, _f6);
                vst1q_f32(p0 + out_hstep * 7, _f7);

                pp += 32;
                p0 += out_hstep * 8;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                // from
                //      a0 b1 c2 d3
                //      c0 d1 a2 b3
                //      a3 b2 c1 d0
                //      c3 d2 a1 b0

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3

                {
                    _sum2 = vrev64q_s32(_sum2);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum2 = vextq_s32(_sum2, _sum2, 2);
                    _sum3 = vextq_s32(_sum3, _sum3, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum3);
                    int32x4x2_t _t1 = vzipq_s32(_sum1, _sum2);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + out_hstep, _f1);
                vst1q_f32(p0 + out_hstep * 2, _f2);
                vst1q_f32(p0 + out_hstep * 3, _f3);

                pp += 16;
                p0 += out_hstep * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);

                // from
                //      a0 b1 c0 d1
                //      a1 b0 c1 d0

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                {
                    _sum1 = vrev64q_s32(_sum1);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + out_hstep, _f1);

                pp += 8;
                p0 += out_hstep * 2;
            }
            for (; jj < max_jj; jj += 1)
            {
                float32x4_t _r0 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(pp)), _descale);
                vst1q_f32(p0, _r0);
                pp += 4;
                p0 += out_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __ARM_NEON
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f2);
                vst1q_f32(p0 + out_hstep * 4, _f1);
                vst1q_f32(p0 + out_hstep * 4 + 4, _f3);

                pp += 16;
                p0 += out_hstep * 8;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                // a0 a1 a2 a3
                // b0 b1 b2 b3

                p0[0] = pp[0] * descale;
                p0[1] = pp[1] * descale;
                p0[2] = pp[2] * descale;
                p0[3] = pp[3] * descale;
                p0[4] = pp[4] * descale;
                p0[5] = pp[5] * descale;
                p0[6] = pp[6] * descale;
                p0[7] = pp[7] * descale;
                pp += 8;
                p0 += out_hstep * 4;
            }
        }
#endif // __ARM_NEON
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                // a0 a1 a2 a3
                // b0 b1 b2 b3
                // a4 a5 a6 a7
                // b4 b5 b6 b7

                p0[0] = pp[0] * descale;
                p0[1] = pp[4] * descale;
                p0[out_hstep] = pp[1] * descale;
                p0[out_hstep + 1] = pp[5] * descale;
                p0[out_hstep * 2] = pp[2] * descale;
                p0[out_hstep * 2 + 1] = pp[6] * descale;
                p0[out_hstep * 3] = pp[3] * descale;
                p0[out_hstep * 3 + 1] = pp[7] * descale;
                p0[out_hstep * 4] = pp[8] * descale;
                p0[out_hstep * 4 + 1] = pp[12] * descale;
                p0[out_hstep * 5] = pp[9] * descale;
                p0[out_hstep * 5 + 1] = pp[13] * descale;
                p0[out_hstep * 6] = pp[10] * descale;
                p0[out_hstep * 6 + 1] = pp[14] * descale;
                p0[out_hstep * 7] = pp[11] * descale;
                p0[out_hstep * 7 + 1] = pp[15] * descale;
                pp += 16;
                p0 += out_hstep * 8;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                // a0 a1 a2 a3
                // b0 b1 b2 b3

                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);

                int32x4x2_t _sum01 = vzipq_s32(_sum0, _sum1);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum01.val[0]), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum01.val[1]), _descale);

                vst1_f32(p0, vget_low_f32(_f0));
                vst1_f32(p0 + out_hstep, vget_high_f32(_f0));
                vst1_f32(p0 + out_hstep * 2, vget_low_f32(_f1));
                vst1_f32(p0 + out_hstep * 3, vget_high_f32(_f1));

                // p0[0] = pp[0] * descale;
                // p0[1] = pp[4] * descale;
                // p0[out_hstep] = pp[1] * descale;
                // p0[out_hstep + 1] = pp[5] * descale;
                // p0[out_hstep * 2] = pp[2] * descale;
                // p0[out_hstep * 2 + 1] = pp[6] * descale;
                // p0[out_hstep * 3] = pp[3] * descale;
                // p0[out_hstep * 3 + 1] = pp[7] * descale;
                pp += 8;
                p0 += out_hstep * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                // a0 a1 b0 b1

                p0[0] = pp[0] * descale;
                p0[1] = pp[2] * descale;
                p0[out_hstep] = pp[1] * descale;
                p0[out_hstep + 1] = pp[3] * descale;
                pp += 4;
                p0 += out_hstep * 2;
            }
            for (; jj < max_jj; jj += 1)
            {
                p0[0] = pp[0] * descale;
                p0[1] = pp[1] * descale;
                pp += 2;
                p0 += out_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
#if __ARM_NEON
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                float32x4_t _r0 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(pp)), _descale);
                vst1q_f32(p0, _r0);
                pp += 4;
                p0 += out_hstep * 4;
            }
        }
#endif // __ARM_NEON
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            int jj = 0;
            for (; jj + 3 < max_jj; jj += 4)
            {
                p0[0] = pp[0] * descale;
                p0[out_hstep] = pp[1] * descale;
                p0[out_hstep * 2] = pp[2] * descale;
                p0[out_hstep * 3] = pp[3] * descale;
                pp += 4;
                p0 += out_hstep * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                p0[0] = pp[0] * descale;
                p0[out_hstep] = pp[1] * descale;
                pp += 2;
                p0 += out_hstep * 2;
            }
            for (; jj < max_jj; jj += 1)
            {
                p0[0] = pp[0] * descale;
                pp += 1;
                p0 += out_hstep;
            }
        }
    }
}

static void pack_A_tile_bf16_to_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, float scale)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    // NCNN_LOGE("pack_A_tile_fp32_to_int8 %d %d", max_ii, elempack);

    signed char* pp = AT;

    float32x4_t _scale = vdupq_n_f32(scale);

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 4) * A_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x8_t _p00 = vld1q_u16(p0);
                uint16x8_t _p11 = vld1q_u16(p1);

                float32x4_t _p0 = bfloat2float(vget_low_u16(_p00));
                float32x4_t _p0n = bfloat2float(vget_high_u16(_p00));
                float32x4_t _p1 = bfloat2float(vget_low_u16(_p11));
                float32x4_t _p1n = bfloat2float(vget_high_u16(_p11));

                _p0 = vmulq_f32(_p0, _scale);
                _p0n = vmulq_f32(_p0n, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p1n = vmulq_f32(_p1n, _scale);

                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p0, _p1);
                _r01.val[1] = float2int8(_p0n, _p1n);

                vst2_s8(pp, _r01);

                pp += 16;
                p0 += 8;
                p1 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp[4] = float2int8(bfloat16_to_float32(p1[0]) * scale);
                pp[5] = float2int8(bfloat16_to_float32(p1[1]) * scale);
                pp[6] = float2int8(bfloat16_to_float32(p1[2]) * scale);
                pp[7] = float2int8(bfloat16_to_float32(p1[3]) * scale);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
        }
        if (elempack == 1)
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p01 = bfloat2float(vreinterpret_u16_u32(vzip_u32(vreinterpret_u32_u16(vld1_u16(p0)), vreinterpret_u32_u16(vld1_u16(p1))).val[0]));
                float32x4_t _p23 = bfloat2float(vreinterpret_u16_u32(vzip_u32(vreinterpret_u32_u16(vld1_u16(p2)), vreinterpret_u32_u16(vld1_u16(p3))).val[0]));
                float32x4_t _p45 = bfloat2float(vreinterpret_u16_u32(vzip_u32(vreinterpret_u32_u16(vld1_u16(p4)), vreinterpret_u32_u16(vld1_u16(p5))).val[0]));
                float32x4_t _p67 = bfloat2float(vreinterpret_u16_u32(vzip_u32(vreinterpret_u32_u16(vld1_u16(p6)), vreinterpret_u32_u16(vld1_u16(p7))).val[0]));

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);
                _p45 = vmulq_f32(_p45, _scale);
                _p67 = vmulq_f32(_p67, _scale);

                int8x8_t _r0 = float2int8(_p01, _p23);
                int8x8_t _r1 = float2int8(_p45, _p67);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));

                pp += 16;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
                p4 += 2;
                p5 += 2;
                p6 += 2;
                p7 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p1[0]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p2[0]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p3[0]) * scale);
                pp[4] = float2int8(bfloat16_to_float32(p4[0]) * scale);
                pp[5] = float2int8(bfloat16_to_float32(p5[0]) * scale);
                pp[6] = float2int8(bfloat16_to_float32(p6[0]) * scale);
                pp[7] = float2int8(bfloat16_to_float32(p7[0]) * scale);
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
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x8_t _p = vld1q_u16(p0);

                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                int8x8_t _r01 = float2int8(_p01.val[0], _p01.val[1]);

                vst1_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[4] * scale);
                // pp[2] = float2int8(p0[1] * scale);
                // pp[3] = float2int8(p0[5] * scale);
                // pp[4] = float2int8(p0[2] * scale);
                // pp[5] = float2int8(p0[6] * scale);
                // pp[6] = float2int8(p0[3] * scale);
                // pp[7] = float2int8(p0[7] * scale);
                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;
            const unsigned short* p2 = (const unsigned short*)A + (i + ii + 2) * A_hstep + k;
            const unsigned short* p3 = (const unsigned short*)A + (i + ii + 3) * A_hstep + k;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p01 = bfloat2float(vreinterpret_u16_u32(vzip_u32(vreinterpret_u32_u16(vld1_u16(p0)), vreinterpret_u32_u16(vld1_u16(p1))).val[0]));
                float32x4_t _p23 = bfloat2float(vreinterpret_u16_u32(vzip_u32(vreinterpret_u32_u16(vld1_u16(p2)), vreinterpret_u32_u16(vld1_u16(p3))).val[0]));

                // float32x4_t _p01 = vcombine_f32(bfloat2float(vld1_u16(p0)), bfloat2float(vld1_u16(p1)));
                // float32x4_t _p23 = vcombine_f32(bfloat2float(vld1_u16(p2)), bfloat2float(vld1_u16(p3)));

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);

                int8x8_t _r0 = float2int8(_p01, _p23);

                vst1_s8(pp, _r0);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[1] * scale);
                // pp[2] = float2int8(p1[0] * scale);
                // pp[3] = float2int8(p1[1] * scale);
                // pp[4] = float2int8(p2[0] * scale);
                // pp[5] = float2int8(p2[1] * scale);
                // pp[6] = float2int8(p3[0] * scale);
                // pp[7] = float2int8(p3[1] * scale);
                pp += 8;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p1[0]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p2[0]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p3[0]) * scale);
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
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p1));

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                float32x4_t _t0 = vcombine_f32(vget_low_f32(_p0), vget_low_f32(_p1));
                float32x4_t _t1 = vcombine_f32(vget_high_f32(_p0), vget_high_f32(_p1));

                int8x8_t _r0 = float2int8(_t0, _t1);

                vst1_s8(pp, _r0);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[1] * scale);
                // pp[2] = float2int8(p1[0] * scale);
                // pp[3] = float2int8(p1[1] * scale);
                // pp[4] = float2int8(p0[2] * scale);
                // pp[5] = float2int8(p0[3] * scale);
                // pp[6] = float2int8(p1[2] * scale);
                // pp[7] = float2int8(p1[3] * scale);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p1[0]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p1[1]) * scale);
                pp += 4;
                p0 += 2;
                p1 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p1[0]) * scale);
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
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8_t _p = vld1q_u16(p0);

                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r0 = float2int8(_p0, _p1);

                vst1_s8(pp, _r0);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[1] * scale);
                pp += 8;
                p0 += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp += 2;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_A_tile_bf16_to_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, float scale)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    // NCNN_LOGE("transpose_pack_A_tile_fp32_to_int8 %d %d", max_ii, elempack);

    signed char* pp = AT;

    float32x4_t _scale = vdupq_n_f32(scale);

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[4]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[5]) * scale);
                pp[4] = float2int8(bfloat16_to_float32(p0[8]) * scale);
                pp[5] = float2int8(bfloat16_to_float32(p0[9]) * scale);
                pp[6] = float2int8(bfloat16_to_float32(p0[12]) * scale);
                pp[7] = float2int8(bfloat16_to_float32(p0[13]) * scale);

                pp[8 + 0] = float2int8(bfloat16_to_float32(p0[16]) * scale);
                pp[8 + 1] = float2int8(bfloat16_to_float32(p0[17]) * scale);
                pp[8 + 2] = float2int8(bfloat16_to_float32(p0[20]) * scale);
                pp[8 + 3] = float2int8(bfloat16_to_float32(p0[21]) * scale);
                pp[8 + 4] = float2int8(bfloat16_to_float32(p0[24]) * scale);
                pp[8 + 5] = float2int8(bfloat16_to_float32(p0[25]) * scale);
                pp[8 + 6] = float2int8(bfloat16_to_float32(p0[28]) * scale);
                pp[8 + 7] = float2int8(bfloat16_to_float32(p0[29]) * scale);

                pp[16 + 0] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[16 + 1] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp[16 + 2] = float2int8(bfloat16_to_float32(p0[6]) * scale);
                pp[16 + 3] = float2int8(bfloat16_to_float32(p0[7]) * scale);
                pp[16 + 4] = float2int8(bfloat16_to_float32(p0[10]) * scale);
                pp[16 + 5] = float2int8(bfloat16_to_float32(p0[11]) * scale);
                pp[16 + 6] = float2int8(bfloat16_to_float32(p0[14]) * scale);
                pp[16 + 7] = float2int8(bfloat16_to_float32(p0[15]) * scale);

                pp[24 + 0] = float2int8(bfloat16_to_float32(p0[18]) * scale);
                pp[24 + 1] = float2int8(bfloat16_to_float32(p0[19]) * scale);
                pp[24 + 2] = float2int8(bfloat16_to_float32(p0[22]) * scale);
                pp[24 + 3] = float2int8(bfloat16_to_float32(p0[23]) * scale);
                pp[24 + 4] = float2int8(bfloat16_to_float32(p0[26]) * scale);
                pp[24 + 5] = float2int8(bfloat16_to_float32(p0[27]) * scale);
                pp[24 + 6] = float2int8(bfloat16_to_float32(p0[30]) * scale);
                pp[24 + 7] = float2int8(bfloat16_to_float32(p0[31]) * scale);

                pp += 32;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x8_t _p01 = vld1q_u16(p0);
                uint16x8_t _p23 = vld1q_u16(p0 + A_hstep);

                float32x4_t _p0 = bfloat2float(vget_low_u16(_p01));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p01));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_p23));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_p23));

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p0, _p1);
                _r01.val[1] = float2int8(_p2, _p3);

                vst2_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[A_hstep + 0] * scale);
                // pp[2] = float2int8(p0[1] * scale);
                // pp[3] = float2int8(p0[A_hstep + 1] * scale);
                // pp[4] = float2int8(p0[2] * scale);
                // pp[5] = float2int8(p0[A_hstep + 2] * scale);
                // pp[6] = float2int8(p0[3] * scale);
                // pp[7] = float2int8(p0[A_hstep + 3] * scale);
                // pp[8] = float2int8(p0[4] * scale);
                // pp[9] = float2int8(p0[A_hstep + 4] * scale);
                // pp[10] = float2int8(p0[5] * scale);
                // pp[11] = float2int8(p0[A_hstep + 5] * scale);
                // pp[12] = float2int8(p0[6] * scale);
                // pp[13] = float2int8(p0[A_hstep + 6] * scale);
                // pp[14] = float2int8(p0[7] * scale);
                // pp[15] = float2int8(p0[A_hstep + 7] * scale);

                pp += 16;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp[4] = float2int8(bfloat16_to_float32(p0[4]) * scale);
                pp[5] = float2int8(bfloat16_to_float32(p0[5]) * scale);
                pp[6] = float2int8(bfloat16_to_float32(p0[6]) * scale);
                pp[7] = float2int8(bfloat16_to_float32(p0[7]) * scale);
                pp += 8;
                p0 += A_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[4]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[5]) * scale);
                pp[4] = float2int8(bfloat16_to_float32(p0[8]) * scale);
                pp[5] = float2int8(bfloat16_to_float32(p0[9]) * scale);
                pp[6] = float2int8(bfloat16_to_float32(p0[12]) * scale);
                pp[7] = float2int8(bfloat16_to_float32(p0[13]) * scale);
                pp[8] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[9] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp[10] = float2int8(bfloat16_to_float32(p0[6]) * scale);
                pp[11] = float2int8(bfloat16_to_float32(p0[7]) * scale);
                pp[12] = float2int8(bfloat16_to_float32(p0[10]) * scale);
                pp[13] = float2int8(bfloat16_to_float32(p0[11]) * scale);
                pp[14] = float2int8(bfloat16_to_float32(p0[14]) * scale);
                pp[15] = float2int8(bfloat16_to_float32(p0[15]) * scale);

                pp += 16;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep));

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                int8x8_t _r01 = float2int8(_p01.val[0], _p01.val[1]);

                vst1_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[A_hstep + 0] * scale);
                // pp[2] = float2int8(p0[1] * scale);
                // pp[3] = float2int8(p0[A_hstep + 1] * scale);
                // pp[4] = float2int8(p0[2] * scale);
                // pp[5] = float2int8(p0[A_hstep + 2] * scale);
                // pp[6] = float2int8(p0[3] * scale);
                // pp[7] = float2int8(p0[A_hstep + 3] * scale);
                pp += 8;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[4]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[5]) * scale);
                pp[4] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[5] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp[6] = float2int8(bfloat16_to_float32(p0[6]) * scale);
                pp[7] = float2int8(bfloat16_to_float32(p0[7]) * scale);
                pp += 8;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // float32x2_t _p0 = bfloat2float(vld1_u16(p0));
                // float32x2_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep));
                // float32x2_t _p2 = bfloat2float(vld1_u16(p0 + A_hstep * 2));
                // float32x2_t _p3 = bfloat2float(vld1_u16(p0 + A_hstep * 3));

                // float32x4_t _p02 = vcombine_f32(_p0, _p2);
                // float32x4_t _p13 = vcombine_f32(_p1, _p3);

                float32x4_t _p02 = bfloat2float(vreinterpret_u16_u32(vzip_u32(vreinterpret_u32_u16(vld1_u16(p0)), vreinterpret_u32_u16(vld1_u16(p0 + A_hstep * 2))).val[0]));
                float32x4_t _p13 = bfloat2float(vreinterpret_u16_u32(vzip_u32(vreinterpret_u32_u16(vld1_u16(p0 + A_hstep)), vreinterpret_u32_u16(vld1_u16(p0 + A_hstep * 3))).val[0]));

                _p02 = vmulq_f32(_p02, _scale);
                _p13 = vmulq_f32(_p13, _scale);

                float32x4x2_t _p0123 = vzipq_f32(_p02, _p13);

                int8x8_t _r01 = float2int8(_p0123.val[0], _p0123.val[1]);

                vst1_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[A_hstep + 0] * scale);
                // pp[2] = float2int8(p0[1] * scale);
                // pp[3] = float2int8(p0[A_hstep + 1] * scale);
                // pp[4] = float2int8(p0[A_hstep * 2] * scale);
                // pp[5] = float2int8(p0[A_hstep * 3 + 0] * scale);
                // pp[6] = float2int8(p0[A_hstep * 2] * scale);
                // pp[7] = float2int8(p0[A_hstep * 3 + 1] * scale);
                pp += 8;
                p0 += A_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[A_hstep + 0]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[A_hstep + 1]) * scale);
                pp += 4;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp += 2;
                p0 += A_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp += 4;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0;
                float32x4_t _p1;
                _p0 = vsetq_lane_f32(bfloat16_to_float32(p0[0]), _p0, 0);
                _p0 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep]), _p0, 1);
                _p0 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 2]), _p0, 2);
                _p0 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 3]), _p0, 3);
                _p1 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 4]), _p1, 0);
                _p1 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 5]), _p1, 1);
                _p1 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 6]), _p1, 2);
                _p1 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 7]), _p1, 3);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r01 = float2int8(_p0, _p1);

                vst1_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[A_hstep] * scale);
                pp += 8;
                p0 += A_hstep * 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[A_hstep]) * scale);
                pp += 2;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp += 1;
                p0 += A_hstep;
            }
        }
    }
}

static void pack_B_tile_bf16_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    // NCNN_LOGE("pack_B_tile_fp32_to_int8 %d %d", max_jj, elempack);

    signed char* pp = BT;

    float32x4_t _scale = vdupq_n_f32(scale);

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x8_t _p00 = vld1q_u16(p0);
                uint16x8_t _p11 = vld1q_u16(p1);

                float32x4_t _p0 = bfloat2float(vget_low_u16(_p00));
                float32x4_t _p0n = bfloat2float(vget_high_u16(_p00));
                float32x4_t _p1 = bfloat2float(vget_low_u16(_p11));
                float32x4_t _p1n = bfloat2float(vget_high_u16(_p11));

                _p0 = vmulq_f32(_p0, _scale);
                _p0n = vmulq_f32(_p0n, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p1n = vmulq_f32(_p1n, _scale);

                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p0, _p1);
                _r01.val[1] = float2int8(_p0n, _p1n);

                vst2_s8(pp, _r01);

                pp += 16;
                p0 += 8;
                p1 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp[4] = float2int8(bfloat16_to_float32(p1[0]) * scale);
                pp[5] = float2int8(bfloat16_to_float32(p1[1]) * scale);
                pp[6] = float2int8(bfloat16_to_float32(p1[2]) * scale);
                pp[7] = float2int8(bfloat16_to_float32(p1[3]) * scale);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 2) * B_hstep + k;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 3) * B_hstep + k;
            const unsigned short* p4 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k;
            const unsigned short* p5 = (const unsigned short*)B + (j + jj + 5) * B_hstep + k;
            const unsigned short* p6 = (const unsigned short*)B + (j + jj + 6) * B_hstep + k;
            const unsigned short* p7 = (const unsigned short*)B + (j + jj + 7) * B_hstep + k;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p01 = bfloat2float(vreinterpret_u16_u32(vzip_u32(vreinterpret_u32_u16(vld1_u16(p0)), vreinterpret_u32_u16(vld1_u16(p1))).val[0]));
                float32x4_t _p23 = bfloat2float(vreinterpret_u16_u32(vzip_u32(vreinterpret_u32_u16(vld1_u16(p2)), vreinterpret_u32_u16(vld1_u16(p3))).val[0]));
                float32x4_t _p45 = bfloat2float(vreinterpret_u16_u32(vzip_u32(vreinterpret_u32_u16(vld1_u16(p4)), vreinterpret_u32_u16(vld1_u16(p5))).val[0]));
                float32x4_t _p67 = bfloat2float(vreinterpret_u16_u32(vzip_u32(vreinterpret_u32_u16(vld1_u16(p6)), vreinterpret_u32_u16(vld1_u16(p7))).val[0]));

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);
                _p45 = vmulq_f32(_p45, _scale);
                _p67 = vmulq_f32(_p67, _scale);

                int8x8_t _r0 = float2int8(_p01, _p23);
                int8x8_t _r1 = float2int8(_p45, _p67);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));

                pp += 16;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
                p4 += 2;
                p5 += 2;
                p6 += 2;
                p7 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p1[0]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p2[0]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p3[0]) * scale);
                pp[4] = float2int8(bfloat16_to_float32(p4[0]) * scale);
                pp[5] = float2int8(bfloat16_to_float32(p5[0]) * scale);
                pp[6] = float2int8(bfloat16_to_float32(p6[0]) * scale);
                pp[7] = float2int8(bfloat16_to_float32(p7[0]) * scale);
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
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x8_t _p = vld1q_u16(p0);

                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                int8x8_t _r01 = float2int8(_p01.val[0], _p01.val[1]);

                vst1_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[4] * scale);
                // pp[2] = float2int8(p0[1] * scale);
                // pp[3] = float2int8(p0[5] * scale);
                // pp[4] = float2int8(p0[2] * scale);
                // pp[5] = float2int8(p0[6] * scale);
                // pp[6] = float2int8(p0[3] * scale);
                // pp[7] = float2int8(p0[7] * scale);
                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 2) * B_hstep + k;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 3) * B_hstep + k;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p01 = bfloat2float(vreinterpret_u16_u32(vzip_u32(vreinterpret_u32_u16(vld1_u16(p0)), vreinterpret_u32_u16(vld1_u16(p1))).val[0]));
                float32x4_t _p23 = bfloat2float(vreinterpret_u16_u32(vzip_u32(vreinterpret_u32_u16(vld1_u16(p2)), vreinterpret_u32_u16(vld1_u16(p3))).val[0]));

                // float32x4_t _p01 = vcombine_f32(bfloat2float(vld1_u16(p0)), bfloat2float(vld1_u16(p1)));
                // float32x4_t _p23 = vcombine_f32(bfloat2float(vld1_u16(p2)), bfloat2float(vld1_u16(p3)));

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);

                int8x8_t _r0 = float2int8(_p01, _p23);

                vst1_s8(pp, _r0);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[1] * scale);
                // pp[2] = float2int8(p1[0] * scale);
                // pp[3] = float2int8(p1[1] * scale);
                // pp[4] = float2int8(p2[0] * scale);
                // pp[5] = float2int8(p2[1] * scale);
                // pp[6] = float2int8(p3[0] * scale);
                // pp[7] = float2int8(p3[1] * scale);
                pp += 8;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p1[0]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p2[0]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p3[0]) * scale);
                pp += 4;
                p0++;
                p1++;
                p2++;
                p3++;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p1));

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                float32x4_t _t0 = vcombine_f32(vget_low_f32(_p0), vget_low_f32(_p1));
                float32x4_t _t1 = vcombine_f32(vget_high_f32(_p0), vget_high_f32(_p1));

                int8x8_t _r0 = float2int8(_t0, _t1);

                vst1_s8(pp, _r0);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[1] * scale);
                // pp[2] = float2int8(p1[0] * scale);
                // pp[3] = float2int8(p1[1] * scale);
                // pp[4] = float2int8(p0[2] * scale);
                // pp[5] = float2int8(p0[3] * scale);
                // pp[6] = float2int8(p1[2] * scale);
                // pp[7] = float2int8(p1[3] * scale);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p1[0]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p1[1]) * scale);
                pp += 4;
                p0 += 2;
                p1 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p1[0]) * scale);
                pp += 2;
                p0++;
                p1++;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8_t _p = vld1q_u16(p0);

                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r0 = float2int8(_p0, _p1);

                vst1_s8(pp, _r0);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[1] * scale);
                pp += 8;
                p0 += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp += 2;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_B_tile_bf16_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    // NCNN_LOGE("transpose_pack_B_tile_fp32_to_int8 %d %d", max_jj, elempack);

    signed char* pp = BT;

    float32x4_t _scale = vdupq_n_f32(scale);

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[4]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[5]) * scale);
                pp[4] = float2int8(bfloat16_to_float32(p0[8]) * scale);
                pp[5] = float2int8(bfloat16_to_float32(p0[9]) * scale);
                pp[6] = float2int8(bfloat16_to_float32(p0[12]) * scale);
                pp[7] = float2int8(bfloat16_to_float32(p0[13]) * scale);

                pp[8 + 0] = float2int8(bfloat16_to_float32(p0[16]) * scale);
                pp[8 + 1] = float2int8(bfloat16_to_float32(p0[17]) * scale);
                pp[8 + 2] = float2int8(bfloat16_to_float32(p0[20]) * scale);
                pp[8 + 3] = float2int8(bfloat16_to_float32(p0[21]) * scale);
                pp[8 + 4] = float2int8(bfloat16_to_float32(p0[24]) * scale);
                pp[8 + 5] = float2int8(bfloat16_to_float32(p0[25]) * scale);
                pp[8 + 6] = float2int8(bfloat16_to_float32(p0[28]) * scale);
                pp[8 + 7] = float2int8(bfloat16_to_float32(p0[29]) * scale);

                pp[16 + 0] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[16 + 1] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp[16 + 2] = float2int8(bfloat16_to_float32(p0[6]) * scale);
                pp[16 + 3] = float2int8(bfloat16_to_float32(p0[7]) * scale);
                pp[16 + 4] = float2int8(bfloat16_to_float32(p0[10]) * scale);
                pp[16 + 5] = float2int8(bfloat16_to_float32(p0[11]) * scale);
                pp[16 + 6] = float2int8(bfloat16_to_float32(p0[14]) * scale);
                pp[16 + 7] = float2int8(bfloat16_to_float32(p0[15]) * scale);

                pp[24 + 0] = float2int8(bfloat16_to_float32(p0[18]) * scale);
                pp[24 + 1] = float2int8(bfloat16_to_float32(p0[19]) * scale);
                pp[24 + 2] = float2int8(bfloat16_to_float32(p0[22]) * scale);
                pp[24 + 3] = float2int8(bfloat16_to_float32(p0[23]) * scale);
                pp[24 + 4] = float2int8(bfloat16_to_float32(p0[26]) * scale);
                pp[24 + 5] = float2int8(bfloat16_to_float32(p0[27]) * scale);
                pp[24 + 6] = float2int8(bfloat16_to_float32(p0[30]) * scale);
                pp[24 + 7] = float2int8(bfloat16_to_float32(p0[31]) * scale);

                pp += 32;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x8_t _p01 = vld1q_u16(p0);
                uint16x8_t _p23 = vld1q_u16(p0 + B_hstep);

                float32x4_t _p0 = bfloat2float(vget_low_u16(_p01));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p01));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_p23));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_p23));

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p0, _p1);
                _r01.val[1] = float2int8(_p2, _p3);

                vst2_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[B_hstep + 0] * scale);
                // pp[2] = float2int8(p0[1] * scale);
                // pp[3] = float2int8(p0[B_hstep + 1] * scale);
                // pp[4] = float2int8(p0[2] * scale);
                // pp[5] = float2int8(p0[B_hstep + 2] * scale);
                // pp[6] = float2int8(p0[3] * scale);
                // pp[7] = float2int8(p0[B_hstep + 3] * scale);
                // pp[8] = float2int8(p0[4] * scale);
                // pp[9] = float2int8(p0[B_hstep + 4] * scale);
                // pp[10] = float2int8(p0[5] * scale);
                // pp[11] = float2int8(p0[B_hstep + 5] * scale);
                // pp[12] = float2int8(p0[6] * scale);
                // pp[13] = float2int8(p0[B_hstep + 6] * scale);
                // pp[14] = float2int8(p0[7] * scale);
                // pp[15] = float2int8(p0[B_hstep + 7] * scale);

                pp += 16;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp[4] = float2int8(bfloat16_to_float32(p0[4]) * scale);
                pp[5] = float2int8(bfloat16_to_float32(p0[5]) * scale);
                pp[6] = float2int8(bfloat16_to_float32(p0[6]) * scale);
                pp[7] = float2int8(bfloat16_to_float32(p0[7]) * scale);
                pp += 8;
                p0 += B_hstep;
            }
        }
    }
#endif // __aarch64__
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[4]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[5]) * scale);
                pp[4] = float2int8(bfloat16_to_float32(p0[8]) * scale);
                pp[5] = float2int8(bfloat16_to_float32(p0[9]) * scale);
                pp[6] = float2int8(bfloat16_to_float32(p0[12]) * scale);
                pp[7] = float2int8(bfloat16_to_float32(p0[13]) * scale);
                pp[8] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[9] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp[10] = float2int8(bfloat16_to_float32(p0[6]) * scale);
                pp[11] = float2int8(bfloat16_to_float32(p0[7]) * scale);
                pp[12] = float2int8(bfloat16_to_float32(p0[10]) * scale);
                pp[13] = float2int8(bfloat16_to_float32(p0[11]) * scale);
                pp[14] = float2int8(bfloat16_to_float32(p0[14]) * scale);
                pp[15] = float2int8(bfloat16_to_float32(p0[15]) * scale);

                pp += 16;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + B_hstep));

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                int8x8_t _r01 = float2int8(_p01.val[0], _p01.val[1]);

                vst1_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[B_hstep + 0] * scale);
                // pp[2] = float2int8(p0[1] * scale);
                // pp[3] = float2int8(p0[B_hstep + 1] * scale);
                // pp[4] = float2int8(p0[2] * scale);
                // pp[5] = float2int8(p0[B_hstep + 2] * scale);
                // pp[6] = float2int8(p0[3] * scale);
                // pp[7] = float2int8(p0[B_hstep + 3] * scale);
                pp += 8;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[4]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[5]) * scale);
                pp[4] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[5] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp[6] = float2int8(bfloat16_to_float32(p0[6]) * scale);
                pp[7] = float2int8(bfloat16_to_float32(p0[7]) * scale);
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // float32x2_t _p0 = bfloat2float(vld1_u16(p0));
                // float32x2_t _p1 = bfloat2float(vld1_u16(p0 + B_hstep));
                // float32x2_t _p2 = bfloat2float(vld1_u16(p0 + B_hstep * 2));
                // float32x2_t _p3 = bfloat2float(vld1_u16(p0 + B_hstep * 3));

                // float32x4_t _p02 = vcombine_f32(_p0, _p2);
                // float32x4_t _p13 = vcombine_f32(_p1, _p3);

                float32x4_t _p02 = bfloat2float(vreinterpret_u16_u32(vzip_u32(vreinterpret_u32_u16(vld1_u16(p0)), vreinterpret_u32_u16(vld1_u16(p0 + B_hstep * 2))).val[0]));
                float32x4_t _p13 = bfloat2float(vreinterpret_u16_u32(vzip_u32(vreinterpret_u32_u16(vld1_u16(p0 + B_hstep)), vreinterpret_u32_u16(vld1_u16(p0 + B_hstep * 3))).val[0]));

                _p02 = vmulq_f32(_p02, _scale);
                _p13 = vmulq_f32(_p13, _scale);

                float32x4x2_t _p0123 = vzipq_f32(_p02, _p13);

                int8x8_t _r01 = float2int8(_p0123.val[0], _p0123.val[1]);

                vst1_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[B_hstep + 0] * scale);
                // pp[2] = float2int8(p0[1] * scale);
                // pp[3] = float2int8(p0[B_hstep + 1] * scale);
                // pp[4] = float2int8(p0[B_hstep * 2] * scale);
                // pp[5] = float2int8(p0[B_hstep * 3 + 0] * scale);
                // pp[6] = float2int8(p0[B_hstep * 2] * scale);
                // pp[7] = float2int8(p0[B_hstep * 3 + 1] * scale);
                pp += 8;
                p0 += B_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[B_hstep + 0]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[B_hstep + 1]) * scale);
                pp += 4;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp += 2;
                p0 += B_hstep;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0;
                float32x4_t _p1;
                _p0 = vsetq_lane_f32(bfloat16_to_float32(p0[0]), _p0, 0);
                _p0 = vsetq_lane_f32(bfloat16_to_float32(p0[B_hstep]), _p0, 1);
                _p0 = vsetq_lane_f32(bfloat16_to_float32(p0[B_hstep * 2]), _p0, 2);
                _p0 = vsetq_lane_f32(bfloat16_to_float32(p0[B_hstep * 3]), _p0, 3);
                _p1 = vsetq_lane_f32(bfloat16_to_float32(p0[B_hstep * 4]), _p1, 0);
                _p1 = vsetq_lane_f32(bfloat16_to_float32(p0[B_hstep * 5]), _p1, 1);
                _p1 = vsetq_lane_f32(bfloat16_to_float32(p0[B_hstep * 6]), _p1, 2);
                _p1 = vsetq_lane_f32(bfloat16_to_float32(p0[B_hstep * 7]), _p1, 3);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r01 = float2int8(_p0, _p1);

                vst1_s8(pp, _r01);

                // pp[0] = float2int8(p0[0] * scale);
                // pp[1] = float2int8(p0[B_hstep] * scale);
                pp += 8;
                p0 += B_hstep * 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[B_hstep]) * scale);
                pp += 2;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp += 1;
                p0 += B_hstep;
            }
        }
    }
}

static void unpack_output_tile_int32_to_bf16(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float descale)
{
    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    // NCNN_LOGE("unpack_output_tile_int32_to_fp32  %d %d %d %d  %d  %d", i, max_ii, j, max_jj, out_elempack, broadcast_type_C);

    // if (broadcast_type_C != -1)
    // NCNN_LOGE("unpack_output_tile_int32_to_fp32 %d %d    %d    %d %d %d", out_elempack, C.elempack, broadcast_type_C, C.w, C.h, C.c);

    // TODO __aarch64__
    // TODO __ARM_FEATURE_DOTPROD

    const int c_hstep = C.dims == 3 ? (int)C.cstep * out_elempack : C.w * out_elempack;
    const unsigned short* pC = C;

    const int* pp = topT;

    float32x4_t _descale = vdupq_n_f32(descale);

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const unsigned short*)C + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const unsigned short*)C + j;
            }
        }

        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * 4;
            unsigned short* p1 = (unsigned short*)top_blob + (i + ii + 4) * out_hstep + j * 4;

            if (broadcast_type_C == 3)
            {
                pC = (const unsigned short*)C + (i + ii) * c_hstep + j * 4;
            }

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);
                int32x4_t _sum8 = vld1q_s32(pp + 32);
                int32x4_t _sum9 = vld1q_s32(pp + 36);
                int32x4_t _suma = vld1q_s32(pp + 40);
                int32x4_t _sumb = vld1q_s32(pp + 44);
                int32x4_t _sumc = vld1q_s32(pp + 48);
                int32x4_t _sumd = vld1q_s32(pp + 52);
                int32x4_t _sume = vld1q_s32(pp + 56);
                int32x4_t _sumf = vld1q_s32(pp + 60);

#if __ARM_FEATURE_DOTPROD
                //TODO
#else
                // from
                //      a0 b1 c2 d3
                //      e4 f5 g6 h7
                //      e0 f1 g2 h3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      g4 h5 e6 f7
                //      g0 h1 e2 f3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      e7 f6 g5 h4
                //      e3 f2 g1 h0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      g7 h6 e5 f4
                //      g3 h2 e1 f0
                //      c7 d6 a5 b4

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      a4 b4 c4 d4
                //      a5 b5 c5 d5
                //      a6 b6 c6 d6
                //      a7 b7 c7 d7
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                //      e2 f2 g2 h2
                //      e3 f3 g3 h3
                //      e4 f4 g4 h4
                //      e5 f5 g5 h5
                //      e6 f6 g6 h6
                //      e7 f7 g7 h7
                {
                    _sum8 = vrev64q_s32(_sum8);
                    _sum9 = vrev64q_s32(_sum9);
                    _suma = vrev64q_s32(_suma);
                    _sumb = vrev64q_s32(_sumb);
                    _sumc = vrev64q_s32(_sumc);
                    _sumd = vrev64q_s32(_sumd);
                    _sume = vrev64q_s32(_sume);
                    _sumf = vrev64q_s32(_sumf);
                    _sum8 = vextq_s32(_sum8, _sum8, 2);
                    _sum9 = vextq_s32(_sum9, _sum9, 2);
                    _suma = vextq_s32(_suma, _suma, 2);
                    _sumb = vextq_s32(_sumb, _sumb, 2);
                    _sumc = vextq_s32(_sumc, _sumc, 2);
                    _sumd = vextq_s32(_sumd, _sumd, 2);
                    _sume = vextq_s32(_sume, _sume, 2);
                    _sumf = vextq_s32(_sumf, _sumf, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sumc);
                    int32x4x2_t _t1 = vzipq_s32(_sum4, _sum8);
                    int32x4x2_t _t2 = vzipq_s32(_sum2, _sume);
                    int32x4x2_t _t3 = vzipq_s32(_sum6, _suma);
                    int32x4x2_t _t4 = vzipq_s32(_sum3, _sumf);
                    int32x4x2_t _t5 = vzipq_s32(_sum7, _sumb);
                    int32x4x2_t _t6 = vzipq_s32(_sum1, _sumd);
                    int32x4x2_t _t7 = vzipq_s32(_sum5, _sum9);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum4 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
                    _sum5 = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
                    _sum6 = vcombine_s32(vget_low_s32(_t5.val[1]), vget_low_s32(_t4.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t5.val[1]), vget_high_s32(_t4.val[1]));
                    _sum8 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum9 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _suma = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sumb = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sumc = vcombine_s32(vget_low_s32(_t6.val[0]), vget_low_s32(_t7.val[0]));
                    _sumd = vcombine_s32(vget_high_s32(_t6.val[0]), vget_high_s32(_t7.val[0]));
                    _sume = vcombine_s32(vget_low_s32(_t7.val[1]), vget_low_s32(_t6.val[1]));
                    _sumf = vcombine_s32(vget_high_s32(_t7.val[1]), vget_high_s32(_t6.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum7 = vrev64q_s32(_sum7);
                    _sum9 = vrev64q_s32(_sum9);
                    _sumb = vrev64q_s32(_sumb);
                    _sumd = vrev64q_s32(_sumd);
                    _sumf = vrev64q_s32(_sumf);
                }

#endif

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);
                float32x4_t _f8 = vmulq_f32(vcvtq_f32_s32(_sum8), _descale);
                float32x4_t _f9 = vmulq_f32(vcvtq_f32_s32(_sum9), _descale);
                float32x4_t _fa = vmulq_f32(vcvtq_f32_s32(_suma), _descale);
                float32x4_t _fb = vmulq_f32(vcvtq_f32_s32(_sumb), _descale);
                float32x4_t _fc = vmulq_f32(vcvtq_f32_s32(_sumc), _descale);
                float32x4_t _fd = vmulq_f32(vcvtq_f32_s32(_sumd), _descale);
                float32x4_t _fe = vmulq_f32(vcvtq_f32_s32(_sume), _descale);
                float32x4_t _ff = vmulq_f32(vcvtq_f32_s32(_sumf), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        _f4 = vaddq_f32(_f4, _c);
                        _f5 = vaddq_f32(_f5, _c);
                        _f6 = vaddq_f32(_f6, _c);
                        _f7 = vaddq_f32(_f7, _c);
                        _f8 = vaddq_f32(_f8, _c);
                        _f9 = vaddq_f32(_f9, _c);
                        _fa = vaddq_f32(_fa, _c);
                        _fb = vaddq_f32(_fb, _c);
                        _fc = vaddq_f32(_fc, _c);
                        _fd = vaddq_f32(_fd, _c);
                        _fe = vaddq_f32(_fe, _c);
                        _ff = vaddq_f32(_ff, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        uint16x8_t _c = vld1q_u16(pC);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c0);
                        _f4 = vaddq_f32(_f4, _c0);
                        _f5 = vaddq_f32(_f5, _c0);
                        _f6 = vaddq_f32(_f6, _c0);
                        _f7 = vaddq_f32(_f7, _c0);
                        _f8 = vaddq_f32(_f8, _c1);
                        _f9 = vaddq_f32(_f9, _c1);
                        _fa = vaddq_f32(_fa, _c1);
                        _fb = vaddq_f32(_fb, _c1);
                        _fc = vaddq_f32(_fc, _c1);
                        _fd = vaddq_f32(_fd, _c1);
                        _fe = vaddq_f32(_fe, _c1);
                        _ff = vaddq_f32(_ff, _c1);
                    }
                    if (broadcast_type_C == 3)
                    {
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + 8);
                        uint16x8_t _c45 = vld1q_u16(pC + 16);
                        uint16x8_t _c67 = vld1q_u16(pC + 24);
                        uint16x8_t _c89 = vld1q_u16(pC + c_hstep * 4);
                        uint16x8_t _cab = vld1q_u16(pC + c_hstep * 4 + 8);
                        uint16x8_t _ccd = vld1q_u16(pC + c_hstep * 4 + 16);
                        uint16x8_t _cef = vld1q_u16(pC + c_hstep * 4 + 24);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c01));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                        float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                        float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
                        float32x4_t _c4 = bfloat2float(vget_low_u16(_c45));
                        float32x4_t _c5 = bfloat2float(vget_high_u16(_c45));
                        float32x4_t _c6 = bfloat2float(vget_low_u16(_c67));
                        float32x4_t _c7 = bfloat2float(vget_high_u16(_c67));
                        float32x4_t _c8 = bfloat2float(vget_low_u16(_c89));
                        float32x4_t _c9 = bfloat2float(vget_high_u16(_c89));
                        float32x4_t _ca = bfloat2float(vget_low_u16(_cab));
                        float32x4_t _cb = bfloat2float(vget_high_u16(_cab));
                        float32x4_t _cc = bfloat2float(vget_low_u16(_ccd));
                        float32x4_t _cd = bfloat2float(vget_high_u16(_ccd));
                        float32x4_t _ce = bfloat2float(vget_low_u16(_cef));
                        float32x4_t _cf = bfloat2float(vget_high_u16(_cef));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        _f8 = vaddq_f32(_f8, _c8);
                        _f9 = vaddq_f32(_f9, _c9);
                        _fa = vaddq_f32(_fa, _ca);
                        _fb = vaddq_f32(_fb, _cb);
                        _fc = vaddq_f32(_fc, _cc);
                        _fd = vaddq_f32(_fd, _cd);
                        _fe = vaddq_f32(_fe, _ce);
                        _ff = vaddq_f32(_ff, _cf);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        float32x4_t _c1 = vdupq_n_f32(bfloat16_to_float32(pC[1]));
                        float32x4_t _c2 = vdupq_n_f32(bfloat16_to_float32(pC[2]));
                        float32x4_t _c3 = vdupq_n_f32(bfloat16_to_float32(pC[3]));
                        float32x4_t _c4 = vdupq_n_f32(bfloat16_to_float32(pC[4]));
                        float32x4_t _c5 = vdupq_n_f32(bfloat16_to_float32(pC[5]));
                        float32x4_t _c6 = vdupq_n_f32(bfloat16_to_float32(pC[6]));
                        float32x4_t _c7 = vdupq_n_f32(bfloat16_to_float32(pC[7]));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        _f8 = vaddq_f32(_f8, _c0);
                        _f9 = vaddq_f32(_f9, _c1);
                        _fa = vaddq_f32(_fa, _c2);
                        _fb = vaddq_f32(_fb, _c3);
                        _fc = vaddq_f32(_fc, _c4);
                        _fd = vaddq_f32(_fd, _c5);
                        _fe = vaddq_f32(_fe, _c6);
                        _ff = vaddq_f32(_ff, _c7);
                        pC += 8;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f1));
                vst1_u16(p0 + 8, float2bfloat(_f2));
                vst1_u16(p0 + 12, float2bfloat(_f3));
                vst1_u16(p0 + 16, float2bfloat(_f4));
                vst1_u16(p0 + 20, float2bfloat(_f5));
                vst1_u16(p0 + 24, float2bfloat(_f6));
                vst1_u16(p0 + 28, float2bfloat(_f7));
                vst1_u16(p1, float2bfloat(_f8));
                vst1_u16(p1 + 4, float2bfloat(_f9));
                vst1_u16(p1 + 8, float2bfloat(_fa));
                vst1_u16(p1 + 12, float2bfloat(_fb));
                vst1_u16(p1 + 16, float2bfloat(_fc));
                vst1_u16(p1 + 20, float2bfloat(_fd));
                vst1_u16(p1 + 24, float2bfloat(_fe));
                vst1_u16(p1 + 28, float2bfloat(_ff));

                pp += 64;
                p0 += 32;
                p1 += 32;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);

#if __ARM_FEATURE_DOTPROD
                //TODO
#else
                // from
                //      a0 b1 c2 d3
                //      e0 f1 g2 h3
                //      c0 d1 a2 b3
                //      g0 h1 e2 f3
                //      a3 b2 c1 d0
                //      e3 f2 g1 h0
                //      c3 d2 a1 b0
                //      g3 h2 e1 f0

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                //      e2 f2 g2 h2
                //      e3 f3 g3 h3

                {
                    _sum4 = vrev64q_s32(_sum4);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum6 = vrev64q_s32(_sum6);
                    _sum7 = vrev64q_s32(_sum7);
                    _sum4 = vextq_s32(_sum4, _sum4, 2);
                    _sum5 = vextq_s32(_sum5, _sum5, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                    int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                    int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                    int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum7 = vrev64q_s32(_sum7);
                }
#endif // __ARM_FEATURE_DOTPROD

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        _f4 = vaddq_f32(_f4, _c);
                        _f5 = vaddq_f32(_f5, _c);
                        _f6 = vaddq_f32(_f6, _c);
                        _f7 = vaddq_f32(_f7, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        uint16x8_t _c = vld1q_u16(pC);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c0);
                        _f4 = vaddq_f32(_f4, _c1);
                        _f5 = vaddq_f32(_f5, _c1);
                        _f6 = vaddq_f32(_f6, _c1);
                        _f7 = vaddq_f32(_f7, _c1);
                    }
                    if (broadcast_type_C == 3)
                    {
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + 8);
                        uint16x8_t _c45 = vld1q_u16(pC + c_hstep * 4);
                        uint16x8_t _c67 = vld1q_u16(pC + c_hstep * 4 + 8);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c01));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                        float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                        float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
                        float32x4_t _c4 = bfloat2float(vget_low_u16(_c45));
                        float32x4_t _c5 = bfloat2float(vget_high_u16(_c45));
                        float32x4_t _c6 = bfloat2float(vget_low_u16(_c67));
                        float32x4_t _c7 = bfloat2float(vget_high_u16(_c67));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        float32x4_t _c1 = vdupq_n_f32(bfloat16_to_float32(pC[1]));
                        float32x4_t _c2 = vdupq_n_f32(bfloat16_to_float32(pC[2]));
                        float32x4_t _c3 = vdupq_n_f32(bfloat16_to_float32(pC[3]));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c0);
                        _f5 = vaddq_f32(_f5, _c1);
                        _f6 = vaddq_f32(_f6, _c2);
                        _f7 = vaddq_f32(_f7, _c3);
                        pC += 4;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f1));
                vst1_u16(p0 + 8, float2bfloat(_f2));
                vst1_u16(p0 + 12, float2bfloat(_f3));
                vst1_u16(p1, float2bfloat(_f4));
                vst1_u16(p1 + 4, float2bfloat(_f5));
                vst1_u16(p1 + 8, float2bfloat(_f6));
                vst1_u16(p1 + 12, float2bfloat(_f7));

                pp += 32;
                p0 += 16;
                p1 += 16;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                // from
                //      a0 b1 c0 d1
                //      e0 f1 g0 h1
                //      a1 b0 c1 d0
                //      e1 f0 g1 h0

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                {
                    _sum2 = vrev64q_s32(_sum2);
                    _sum3 = vrev64q_s32(_sum3);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum2);
                    int32x4x2_t _t1 = vzipq_s32(_sum1, _sum3);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t1.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t1.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        uint16x8_t _c = vld1q_u16(pC);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c1);
                        _f3 = vaddq_f32(_f3, _c1);
                    }
                    if (broadcast_type_C == 3)
                    {
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + c_hstep * 4);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c01));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                        float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                        float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        float32x4_t _c1 = vdupq_n_f32(bfloat16_to_float32(pC[1]));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c1);
                        pC += 2;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f1));
                vst1_u16(p1, float2bfloat(_f2));
                vst1_u16(p1 + 4, float2bfloat(_f3));

                pp += 16;
                p0 += 8;
                p1 += 8;
            }
            for (; jj < max_jj; jj++)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        uint16x8_t _c = vld1q_u16(pC);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        float32x4_t _c1 = bfloat2float(vld1_u16(pC + c_hstep * 4));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        pC += 1;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p1, float2bfloat(_f1));

                pp += 8;
                p0 += 4;
                p1 += 4;
            }
        }
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;
            unsigned short* p1 = (unsigned short*)top_blob + (i + ii + 1) * out_hstep + j;
            unsigned short* p2 = (unsigned short*)top_blob + (i + ii + 2) * out_hstep + j;
            unsigned short* p3 = (unsigned short*)top_blob + (i + ii + 3) * out_hstep + j;
            unsigned short* p4 = (unsigned short*)top_blob + (i + ii + 4) * out_hstep + j;
            unsigned short* p5 = (unsigned short*)top_blob + (i + ii + 5) * out_hstep + j;
            unsigned short* p6 = (unsigned short*)top_blob + (i + ii + 6) * out_hstep + j;
            unsigned short* p7 = (unsigned short*)top_blob + (i + ii + 7) * out_hstep + j;

            if (broadcast_type_C == 3)
            {
                pC = (const unsigned short*)C + (i + ii) * c_hstep + j;
            }

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);
                int32x4_t _sum8 = vld1q_s32(pp + 32);
                int32x4_t _sum9 = vld1q_s32(pp + 36);
                int32x4_t _suma = vld1q_s32(pp + 40);
                int32x4_t _sumb = vld1q_s32(pp + 44);
                int32x4_t _sumc = vld1q_s32(pp + 48);
                int32x4_t _sumd = vld1q_s32(pp + 52);
                int32x4_t _sume = vld1q_s32(pp + 56);
                int32x4_t _sumf = vld1q_s32(pp + 60);

                // from
                //      a0 b1 c2 d3
                //      e4 f5 g6 h7
                //      e0 f1 g2 h3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      g4 h5 e6 f7
                //      g0 h1 e2 f3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      e7 f6 g5 h4
                //      e3 f2 g1 h0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      g7 h6 e5 f4
                //      g3 h2 e1 f0
                //      c7 d6 a5 b4

                // to
                //      a0 a1 a2 a3
                //      a4 a5 a6 a7
                //      b0 b1 b2 b3
                //      b4 b5 b6 b7
                //      c0 c1 c2 c3
                //      c4 c5 c6 c7
                //      d0 d1 d2 d3
                //      d4 d5 d6 d7
                //      e0 e1 e2 e3
                //      e4 e5 e6 e7
                //      f0 f1 f2 f3
                //      f4 f5 f6 f7
                //      g0 g1 g2 g3
                //      g4 g5 g6 g7
                //      h0 h1 h2 h3
                //      h4 h5 h6 h7
                {
                    _sum4 = vextq_s32(_sum4, _sum4, 2);
                    _sum5 = vextq_s32(_sum5, _sum5, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    _sumc = vextq_s32(_sumc, _sumc, 2);
                    _sumd = vextq_s32(_sumd, _sumd, 2);
                    _sume = vextq_s32(_sume, _sume, 2);
                    _sumf = vextq_s32(_sumf, _sumf, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sumc);
                    int32x4x2_t _t1 = vzipq_s32(_sum4, _sum8);
                    int32x4x2_t _t2 = vzipq_s32(_sum3, _sumf);
                    int32x4x2_t _t3 = vzipq_s32(_sum7, _sumb);
                    int32x4x2_t _t4 = vzipq_s32(_sum2, _sume);
                    int32x4x2_t _t5 = vzipq_s32(_sum6, _suma);
                    int32x4x2_t _t6 = vzipq_s32(_sum1, _sumd);
                    int32x4x2_t _t7 = vzipq_s32(_sum5, _sum9);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum8 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
                    _sum9 = vcombine_s32(vget_low_s32(_t6.val[0]), vget_low_s32(_t7.val[0]));
                    _suma = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
                    _sumb = vcombine_s32(vget_high_s32(_t6.val[0]), vget_high_s32(_t7.val[0]));
                    _sumc = vcombine_s32(vget_low_s32(_t5.val[1]), vget_low_s32(_t4.val[1]));
                    _sumd = vcombine_s32(vget_low_s32(_t7.val[1]), vget_low_s32(_t6.val[1]));
                    _sume = vcombine_s32(vget_high_s32(_t5.val[1]), vget_high_s32(_t4.val[1]));
                    _sumf = vcombine_s32(vget_high_s32(_t7.val[1]), vget_high_s32(_t6.val[1]));
                    _sum2 = vrev64q_s32(_sum2);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum6 = vrev64q_s32(_sum6);
                    _sum7 = vrev64q_s32(_sum7);
                    _suma = vrev64q_s32(_suma);
                    _sumb = vrev64q_s32(_sumb);
                    _sume = vrev64q_s32(_sume);
                    _sumf = vrev64q_s32(_sumf);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);
                float32x4_t _f8 = vmulq_f32(vcvtq_f32_s32(_sum8), _descale);
                float32x4_t _f9 = vmulq_f32(vcvtq_f32_s32(_sum9), _descale);
                float32x4_t _fa = vmulq_f32(vcvtq_f32_s32(_suma), _descale);
                float32x4_t _fb = vmulq_f32(vcvtq_f32_s32(_sumb), _descale);
                float32x4_t _fc = vmulq_f32(vcvtq_f32_s32(_sumc), _descale);
                float32x4_t _fd = vmulq_f32(vcvtq_f32_s32(_sumd), _descale);
                float32x4_t _fe = vmulq_f32(vcvtq_f32_s32(_sume), _descale);
                float32x4_t _ff = vmulq_f32(vcvtq_f32_s32(_sumf), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        _f4 = vaddq_f32(_f4, _c);
                        _f5 = vaddq_f32(_f5, _c);
                        _f6 = vaddq_f32(_f6, _c);
                        _f7 = vaddq_f32(_f7, _c);
                        _f8 = vaddq_f32(_f8, _c);
                        _f9 = vaddq_f32(_f9, _c);
                        _fa = vaddq_f32(_fa, _c);
                        _fb = vaddq_f32(_fb, _c);
                        _fc = vaddq_f32(_fc, _c);
                        _fd = vaddq_f32(_fd, _c);
                        _fe = vaddq_f32(_fe, _c);
                        _ff = vaddq_f32(_ff, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        // TODO
                        // float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        // float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        // float32x4_t _c2 = vdupq_n_f32(pC[2]);
                        // float32x4_t _c3 = vdupq_n_f32(pC[3]);
                        // float32x4_t _c4 = vdupq_n_f32(pC[4]);
                        // float32x4_t _c5 = vdupq_n_f32(pC[5]);
                        // float32x4_t _c6 = vdupq_n_f32(pC[6]);
                        // float32x4_t _c7 = vdupq_n_f32(pC[7]);
                        // float32x4_t _c8 = vdupq_n_f32(pC[8]);
                        // float32x4_t _c9 = vdupq_n_f32(pC[9]);
                        // float32x4_t _ca = vdupq_n_f32(pC[10]);
                        // float32x4_t _cb = vdupq_n_f32(pC[11]);
                        // float32x4_t _cc = vdupq_n_f32(pC[12]);
                        // float32x4_t _cd = vdupq_n_f32(pC[13]);
                        // float32x4_t _ce = vdupq_n_f32(pC[14]);
                        // float32x4_t _cf = vdupq_n_f32(pC[15]);
                        // _f0 = vaddq_f32(_f0, _c0);
                        // _f1 = vaddq_f32(_f1, _c1);
                        // _f2 = vaddq_f32(_f2, _c2);
                        // _f3 = vaddq_f32(_f3, _c3);
                        // _f4 = vaddq_f32(_f4, _c4);
                        // _f5 = vaddq_f32(_f5, _c5);
                        // _f6 = vaddq_f32(_f6, _c6);
                        // _f7 = vaddq_f32(_f7, _c7);
                        // _f8 = vaddq_f32(_f8, _c8);
                        // _f9 = vaddq_f32(_f9, _c9);
                        // _fa = vaddq_f32(_fa, _ca);
                        // _fb = vaddq_f32(_fb, _cb);
                        // _fc = vaddq_f32(_fc, _cc);
                        // _fd = vaddq_f32(_fd, _cd);
                        // _fe = vaddq_f32(_fe, _ce);
                        // _ff = vaddq_f32(_ff, _cf);
                    }
                    if (broadcast_type_C == 3)
                    {
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + c_hstep);
                        uint16x8_t _c45 = vld1q_u16(pC + c_hstep * 2);
                        uint16x8_t _c67 = vld1q_u16(pC + c_hstep * 3);
                        uint16x8_t _c89 = vld1q_u16(pC + c_hstep * 4);
                        uint16x8_t _cab = vld1q_u16(pC + c_hstep * 5);
                        uint16x8_t _ccd = vld1q_u16(pC + c_hstep * 6);
                        uint16x8_t _cef = vld1q_u16(pC + c_hstep * 7);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c01));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                        float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                        float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
                        float32x4_t _c4 = bfloat2float(vget_low_u16(_c45));
                        float32x4_t _c5 = bfloat2float(vget_high_u16(_c45));
                        float32x4_t _c6 = bfloat2float(vget_low_u16(_c67));
                        float32x4_t _c7 = bfloat2float(vget_high_u16(_c67));
                        float32x4_t _c8 = bfloat2float(vget_low_u16(_c89));
                        float32x4_t _c9 = bfloat2float(vget_high_u16(_c89));
                        float32x4_t _ca = bfloat2float(vget_low_u16(_cab));
                        float32x4_t _cb = bfloat2float(vget_high_u16(_cab));
                        float32x4_t _cc = bfloat2float(vget_low_u16(_ccd));
                        float32x4_t _cd = bfloat2float(vget_high_u16(_ccd));
                        float32x4_t _ce = bfloat2float(vget_low_u16(_cef));
                        float32x4_t _cf = bfloat2float(vget_high_u16(_cef));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        _f8 = vaddq_f32(_f8, _c8);
                        _f9 = vaddq_f32(_f9, _c9);
                        _fa = vaddq_f32(_fa, _ca);
                        _fb = vaddq_f32(_fb, _cb);
                        _fc = vaddq_f32(_fc, _cc);
                        _fd = vaddq_f32(_fd, _cd);
                        _fe = vaddq_f32(_fe, _ce);
                        _ff = vaddq_f32(_ff, _cf);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        uint16x8_t _c = vld1q_u16(pC);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c1);
                        _f4 = vaddq_f32(_f4, _c0);
                        _f5 = vaddq_f32(_f5, _c1);
                        _f6 = vaddq_f32(_f6, _c0);
                        _f7 = vaddq_f32(_f7, _c1);
                        _f8 = vaddq_f32(_f8, _c0);
                        _f9 = vaddq_f32(_f9, _c1);
                        _fa = vaddq_f32(_fa, _c0);
                        _fb = vaddq_f32(_fb, _c1);
                        _fc = vaddq_f32(_fc, _c0);
                        _fd = vaddq_f32(_fd, _c1);
                        _fe = vaddq_f32(_fe, _c0);
                        _ff = vaddq_f32(_ff, _c1);
                        pC += 8;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f1));
                vst1_u16(p1, float2bfloat(_f2));
                vst1_u16(p1 + 4, float2bfloat(_f3));
                vst1_u16(p2, float2bfloat(_f4));
                vst1_u16(p2 + 4, float2bfloat(_f5));
                vst1_u16(p3, float2bfloat(_f6));
                vst1_u16(p3 + 4, float2bfloat(_f7));
                vst1_u16(p4, float2bfloat(_f8));
                vst1_u16(p4 + 4, float2bfloat(_f9));
                vst1_u16(p5, float2bfloat(_fa));
                vst1_u16(p5 + 4, float2bfloat(_fb));
                vst1_u16(p6, float2bfloat(_fc));
                vst1_u16(p6 + 4, float2bfloat(_fd));
                vst1_u16(p7, float2bfloat(_fe));
                vst1_u16(p7 + 4, float2bfloat(_ff));

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
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);

                // from
                //      a0 b1 c2 d3
                //      e0 f1 g2 h3
                //      c0 d1 a2 b3
                //      g0 h1 e2 f3
                //      a3 b2 c1 d0
                //      e3 f2 g1 h0
                //      c3 d2 a1 b0
                //      g3 h2 e1 f0

                // to
                //      a0 a1 a2 a3
                //      b0 b1 b2 b3
                //      c0 c1 c2 c3
                //      d0 d1 d2 d3
                //      e0 e1 e2 e3
                //      f0 f1 f2 f3
                //      g0 g1 g2 g3
                //      h0 h1 h2 h3
                {
                    _sum2 = vextq_s32(_sum2, _sum2, 2);
                    _sum3 = vextq_s32(_sum3, _sum3, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                    int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                    int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                    int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum7 = vrev64q_s32(_sum7);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        _f4 = vaddq_f32(_f4, _c);
                        _f5 = vaddq_f32(_f5, _c);
                        _f6 = vaddq_f32(_f6, _c);
                        _f7 = vaddq_f32(_f7, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        // TODO
                        // float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        // float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        // float32x4_t _c2 = vdupq_n_f32(pC[2]);
                        // float32x4_t _c3 = vdupq_n_f32(pC[3]);
                        // float32x4_t _c4 = vdupq_n_f32(pC[4]);
                        // float32x4_t _c5 = vdupq_n_f32(pC[5]);
                        // float32x4_t _c6 = vdupq_n_f32(pC[6]);
                        // float32x4_t _c7 = vdupq_n_f32(pC[7]);
                        // _f0 = vaddq_f32(_f0, _c0);
                        // _f1 = vaddq_f32(_f1, _c1);
                        // _f2 = vaddq_f32(_f2, _c2);
                        // _f3 = vaddq_f32(_f3, _c3);
                        // _f4 = vaddq_f32(_f4, _c4);
                        // _f5 = vaddq_f32(_f5, _c5);
                        // _f6 = vaddq_f32(_f6, _c6);
                        // _f7 = vaddq_f32(_f7, _c7);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        float32x4_t _c1 = bfloat2float(vld1_u16(pC + c_hstep * 1));
                        float32x4_t _c2 = bfloat2float(vld1_u16(pC + c_hstep * 2));
                        float32x4_t _c3 = bfloat2float(vld1_u16(pC + c_hstep * 3));
                        float32x4_t _c4 = bfloat2float(vld1_u16(pC + c_hstep * 4));
                        float32x4_t _c5 = bfloat2float(vld1_u16(pC + c_hstep * 5));
                        float32x4_t _c6 = bfloat2float(vld1_u16(pC + c_hstep * 6));
                        float32x4_t _c7 = bfloat2float(vld1_u16(pC + c_hstep * 7));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        _f4 = vaddq_f32(_f4, _c);
                        _f5 = vaddq_f32(_f5, _c);
                        _f6 = vaddq_f32(_f6, _c);
                        _f7 = vaddq_f32(_f7, _c);
                        pC += 4;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p1, float2bfloat(_f1));
                vst1_u16(p2, float2bfloat(_f2));
                vst1_u16(p3, float2bfloat(_f3));
                vst1_u16(p4, float2bfloat(_f4));
                vst1_u16(p5, float2bfloat(_f5));
                vst1_u16(p6, float2bfloat(_f6));
                vst1_u16(p7, float2bfloat(_f7));

                pp += 32;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
                p4 += 4;
                p5 += 4;
                p6 += 4;
                p7 += 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                // from
                //      a0 b1 c0 d1
                //      e0 f1 g0 h1
                //      a1 b0 c1 d0
                //      e1 f0 g1 h0

                // to
                //      a0 a1 c0 c1
                //      b0 b1 d0 d1
                //      e0 e1 g0 g1
                //      f0 f1 h0 h1
                {
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum2);
                    int32x4x2_t _t1 = vzipq_s32(_sum1, _sum3);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t1.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t1.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        // TODO
                        // float32x4_t _c0 = vld1q_f32(pC);
                        // float32x4_t _c1 = vld1q_f32(pC + 4);
                        // _f0 = vaddq_f32(_f0, _c0);
                        // _f1 = vaddq_f32(_f1, _c0);
                        // _f2 = vaddq_f32(_f2, _c1);
                        // _f3 = vaddq_f32(_f3, _c1);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        float32x4_t _c1 = bfloat2float(vld1_u16(pC + c_hstep));
                        float32x4_t _c2 = bfloat2float(vld1_u16(pC + c_hstep * 1));
                        float32x4_t _c3 = bfloat2float(vld1_u16(pC + c_hstep * 2));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        // TODO
                        // float32x2_t _c0 = vld1_f32(pC);
                        // float32x4_t _c = vcombine_f32(_c0, _c0);
                        // _f0 = vaddq_f32(_f0, _c);
                        // _f1 = vaddq_f32(_f1, _c);
                        // _f2 = vaddq_f32(_f2, _c);
                        // _f3 = vaddq_f32(_f3, _c);
                        pC += 2;
                    }
                }

                uint16x4_t _fb0 = float2bfloat(_f0);
                uint16x4_t _fb1 = float2bfloat(_f1);
                uint16x4_t _fb2 = float2bfloat(_f2);
                uint16x4_t _fb3 = float2bfloat(_f3);

                p0[0] = vget_lane_u16(_fb0, 0);
                p0[1] = vget_lane_u16(_fb0, 1);
                p1[0] = vget_lane_u16(_fb1, 0);
                p1[1] = vget_lane_u16(_fb1, 1);
                p2[0] = vget_lane_u16(_fb0, 2);
                p2[1] = vget_lane_u16(_fb0, 3);
                p3[0] = vget_lane_u16(_fb1, 2);
                p3[1] = vget_lane_u16(_fb1, 3);
                p4[0] = vget_lane_u16(_fb2, 0);
                p4[1] = vget_lane_u16(_fb2, 1);
                p5[0] = vget_lane_u16(_fb3, 0);
                p5[1] = vget_lane_u16(_fb3, 1);
                p6[0] = vget_lane_u16(_fb2, 2);
                p6[1] = vget_lane_u16(_fb2, 3);
                p7[0] = vget_lane_u16(_fb3, 2);
                p7[1] = vget_lane_u16(_fb3, 3);

                // vst1_f32(p0, vget_low_f32(_f0));
                // vst1_f32(p1, vget_low_f32(_f1));
                // vst1_f32(p2, vget_high_f32(_f0));
                // vst1_f32(p3, vget_high_f32(_f1));
                // vst1_f32(p4, vget_low_f32(_f2));
                // vst1_f32(p5, vget_low_f32(_f3));
                // vst1_f32(p6, vget_high_f32(_f2));
                // vst1_f32(p7, vget_high_f32(_f3));

                pp += 16;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
                p4 += 2;
                p5 += 2;
                p6 += 2;
                p7 += 2;
            }
            for (; jj < max_jj; jj++)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        // TODO
                        //     float32x4_t _c0 = vld1q_f32(pC);
                        //     float32x4_t _c1 = vld1q_f32(pC + 4);
                        //     _f0 = vaddq_f32(_f0, _c0);
                        //     _f1 = vaddq_f32(_f1, _c1);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        float32x4_t _c1 = bfloat2float(vld1_u16(pC + c_hstep));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        pC += 1;
                    }
                }

                uint16x4_t _fb0 = float2bfloat(_f0);
                uint16x4_t _fb1 = float2bfloat(_f1);

                p0[0] = vget_lane_u16(_fb0, 0);
                p1[0] = vget_lane_u16(_fb0, 1);
                p2[0] = vget_lane_u16(_fb0, 2);
                p3[0] = vget_lane_u16(_fb0, 3);
                p4[0] = vget_lane_u16(_fb1, 0);
                p5[0] = vget_lane_u16(_fb1, 1);
                p6[0] = vget_lane_u16(_fb1, 2);
                p7[0] = vget_lane_u16(_fb1, 3);

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
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const unsigned short*)C + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const unsigned short*)C + j;
            }
        }

        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * 4;

            if (broadcast_type_C == 3)
            {
                pC = (const unsigned short*)C + (i + ii) * c_hstep + j * 4;
            }

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);

                // from
                //      a0 b1 c2 d3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      c7 d6 a5 b4

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      a4 b4 c4 d4
                //      a5 b5 c5 d5
                //      a6 b6 c6 d6
                //      a7 b7 c7 d7
                {
                    _sum4 = vrev64q_s32(_sum4);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum6 = vrev64q_s32(_sum6);
                    _sum7 = vrev64q_s32(_sum7);
                    _sum4 = vextq_s32(_sum4, _sum4, 2);
                    _sum5 = vextq_s32(_sum5, _sum5, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                    int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                    int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                    int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum7 = vrev64q_s32(_sum7);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        _f4 = vaddq_f32(_f4, _c);
                        _f5 = vaddq_f32(_f5, _c);
                        _f6 = vaddq_f32(_f6, _c);
                        _f7 = vaddq_f32(_f7, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c0);
                        _f4 = vaddq_f32(_f4, _c0);
                        _f5 = vaddq_f32(_f5, _c0);
                        _f6 = vaddq_f32(_f6, _c0);
                        _f7 = vaddq_f32(_f7, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + 8);
                        uint16x8_t _c45 = vld1q_u16(pC + 16);
                        uint16x8_t _c67 = vld1q_u16(pC + 24);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c01));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                        float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                        float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
                        float32x4_t _c4 = bfloat2float(vget_low_u16(_c45));
                        float32x4_t _c5 = bfloat2float(vget_high_u16(_c45));
                        float32x4_t _c6 = bfloat2float(vget_low_u16(_c67));
                        float32x4_t _c7 = bfloat2float(vget_high_u16(_c67));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        float32x4_t _c1 = vdupq_n_f32(bfloat16_to_float32(pC[1]));
                        float32x4_t _c2 = vdupq_n_f32(bfloat16_to_float32(pC[2]));
                        float32x4_t _c3 = vdupq_n_f32(bfloat16_to_float32(pC[3]));
                        float32x4_t _c4 = vdupq_n_f32(bfloat16_to_float32(pC[4]));
                        float32x4_t _c5 = vdupq_n_f32(bfloat16_to_float32(pC[5]));
                        float32x4_t _c6 = vdupq_n_f32(bfloat16_to_float32(pC[6]));
                        float32x4_t _c7 = vdupq_n_f32(bfloat16_to_float32(pC[7]));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        pC += 8;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f1));
                vst1_u16(p0 + 8, float2bfloat(_f2));
                vst1_u16(p0 + 12, float2bfloat(_f3));
                vst1_u16(p0 + 16, float2bfloat(_f4));
                vst1_u16(p0 + 20, float2bfloat(_f5));
                vst1_u16(p0 + 24, float2bfloat(_f6));
                vst1_u16(p0 + 28, float2bfloat(_f7));

                pp += 32;
                p0 += 32;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                // from
                //      a0 b1 c2 d3
                //      c0 d1 a2 b3
                //      a3 b2 c1 d0
                //      c3 d2 a1 b0

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                {
                    _sum2 = vrev64q_s32(_sum2);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum2 = vextq_s32(_sum2, _sum2, 2);
                    _sum3 = vextq_s32(_sum3, _sum3, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum3);
                    int32x4x2_t _t1 = vzipq_s32(_sum1, _sum2);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + 8);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c01));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                        float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                        float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        float32x4_t _c1 = vdupq_n_f32(bfloat16_to_float32(pC[1]));
                        float32x4_t _c2 = vdupq_n_f32(bfloat16_to_float32(pC[2]));
                        float32x4_t _c3 = vdupq_n_f32(bfloat16_to_float32(pC[3]));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        pC += 4;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f1));
                vst1_u16(p0 + 8, float2bfloat(_f2));
                vst1_u16(p0 + 12, float2bfloat(_f3));

                pp += 16;
                p0 += 16;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);

                // from
                //      a0 b1 c0 d1
                //      a1 b0 c1 d0

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                {
                    _sum1 = vrev64q_s32(_sum1);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        uint16x8_t _c01 = vld1q_u16(pC);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c01));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        float32x4_t _c1 = vdupq_n_f32(bfloat16_to_float32(pC[1]));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        pC += 2;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f1));

                pp += 8;
                p0 += 8;
            }
            for (; jj < max_jj; jj++)
            {
                int32x4_t _sum0 = vld1q_s32(pp);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c0);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c0);
                        pC += 1;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));

                pp += 4;
                p0 += 4;
            }
        }
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;
            unsigned short* p1 = (unsigned short*)top_blob + (i + ii + 1) * out_hstep + j;
            unsigned short* p2 = (unsigned short*)top_blob + (i + ii + 2) * out_hstep + j;
            unsigned short* p3 = (unsigned short*)top_blob + (i + ii + 3) * out_hstep + j;

            if (broadcast_type_C == 3)
            {
                pC = (const unsigned short*)C + (i + ii) * c_hstep + j;
            }

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);

                // from
                //      a0 b1 c2 d3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      c7 d6 a5 b4

                // to
                //      a0 a1 a2 a3
                //      a4 a5 a6 a7
                //      b0 b1 b2 b3
                //      b4 b5 b6 b7
                //      c0 c1 c2 c3
                //      c4 c5 c6 c7
                //      d0 d1 d2 d3
                //      d4 d5 d6 d7
                {
                    _sum2 = vextq_s32(_sum2, _sum2, 2);
                    _sum3 = vextq_s32(_sum3, _sum3, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                    int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                    int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                    int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum2 = vrev64q_s32(_sum2);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum6 = vrev64q_s32(_sum6);
                    _sum7 = vrev64q_s32(_sum7);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        _f4 = vaddq_f32(_f4, _c);
                        _f5 = vaddq_f32(_f5, _c);
                        _f6 = vaddq_f32(_f6, _c);
                        _f7 = vaddq_f32(_f7, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c0);
                        _f4 = vaddq_f32(_f4, _c0);
                        _f5 = vaddq_f32(_f5, _c0);
                        _f6 = vaddq_f32(_f6, _c0);
                        _f7 = vaddq_f32(_f7, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + c_hstep * 1);
                        uint16x8_t _c45 = vld1q_u16(pC + c_hstep * 2);
                        uint16x8_t _c67 = vld1q_u16(pC + c_hstep * 3);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c01));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                        float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                        float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
                        float32x4_t _c4 = bfloat2float(vget_low_u16(_c45));
                        float32x4_t _c5 = bfloat2float(vget_high_u16(_c45));
                        float32x4_t _c6 = bfloat2float(vget_low_u16(_c67));
                        float32x4_t _c7 = bfloat2float(vget_high_u16(_c67));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        uint16x8_t _c = vld1q_u16(pC);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c1);
                        _f4 = vaddq_f32(_f4, _c0);
                        _f5 = vaddq_f32(_f5, _c1);
                        _f6 = vaddq_f32(_f6, _c0);
                        _f7 = vaddq_f32(_f7, _c1);
                        pC += 8;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f1));
                vst1_u16(p1, float2bfloat(_f2));
                vst1_u16(p1 + 4, float2bfloat(_f3));
                vst1_u16(p2, float2bfloat(_f4));
                vst1_u16(p2 + 4, float2bfloat(_f5));
                vst1_u16(p3, float2bfloat(_f6));
                vst1_u16(p3 + 4, float2bfloat(_f7));

                pp += 32;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                // from
                //      a0 b1 c2 d3
                //      c0 d1 a2 b3
                //      a3 b2 c1 d0
                //      c3 d2 a1 b0

                // to
                //      a0 a1 a2 a3
                //      b0 b1 b2 b3
                //      c0 c1 c2 c3
                //      d0 d1 d2 d3
                {
                    _sum1 = vextq_s32(_sum1, _sum1, 2);
                    _sum3 = vextq_s32(_sum3, _sum3, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum3);
                    int32x4x2_t _t1 = vzipq_s32(_sum1, _sum2);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        float32x4_t _c1 = bfloat2float(vld1_u16(pC + c_hstep * 1));
                        float32x4_t _c2 = bfloat2float(vld1_u16(pC + c_hstep * 2));
                        float32x4_t _c3 = bfloat2float(vld1_u16(pC + c_hstep * 3));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                        pC += 4;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p1, float2bfloat(_f1));
                vst1_u16(p2, float2bfloat(_f2));
                vst1_u16(p3, float2bfloat(_f3));

                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);

                // from
                //      a0 b1 c0 d1
                //      a1 b0 c1 d0

                // to
                //      a0 a1 c0 c1
                //      b0 b1 d0 d1
                {
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                    _sum1 = vrev64q_s32(_sum1);

                    int32x4_t _tmp0 = _sum0;
                    int32x4_t _tmp1 = _sum1;

                    _sum0 = vcombine_s32(vget_low_s32(_tmp0), vget_low_s32(_tmp1));
                    _sum1 = vcombine_s32(vget_high_s32(_tmp0), vget_high_s32(_tmp1));
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        float32x4_t _c1 = bfloat2float(vld1_u16(pC + c_hstep));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        // float32x4_t _c0 = vdupq_n_f32(pC[0]);
                        // float32x4_t _c1 = vdupq_n_f32(pC[1]);
                        // _f0 = vaddq_f32(_f0, _c0);
                        // _f1 = vaddq_f32(_f1, _c1);
                        // pC += 2;

                        // TODO
                        // float32x2_t _c0 = vld1_f32(pC);
                        // float32x4_t _c = vcombine_f32(_c0, _c0);
                        // _f0 = vaddq_f32(_f0, _c);
                        // _f1 = vaddq_f32(_f1, _c);
                        pC += 2;
                    }
                }

                uint16x4_t _fb0 = float2bfloat(_f0);
                uint16x4_t _fb1 = float2bfloat(_f1);

                p0[0] = vget_lane_u16(_fb0, 0);
                p0[1] = vget_lane_u16(_fb0, 1);
                p1[0] = vget_lane_u16(_fb0, 2);
                p1[1] = vget_lane_u16(_fb0, 3);
                p2[0] = vget_lane_u16(_fb1, 0);
                p2[1] = vget_lane_u16(_fb1, 1);
                p3[0] = vget_lane_u16(_fb1, 2);
                p3[1] = vget_lane_u16(_fb1, 3);

                // vst1_f32(p0, vget_low_f32(_f0));
                // vst1_f32(p1, vget_high_f32(_f0));
                // vst1_f32(p2, vget_low_f32(_f1));
                // vst1_f32(p3, vget_high_f32(_f1));

                pp += 8;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
            }
            for (; jj < max_jj; jj++)
            {
                int32x4_t _sum0 = vld1q_s32(pp);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c0);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c0);
                        pC += 1;
                    }
                }

                uint16x4_t _fb0 = float2bfloat(_f0);

                p0[0] = vget_lane_u16(_fb0, 0);
                p1[0] = vget_lane_u16(_fb0, 1);
                p2[0] = vget_lane_u16(_fb0, 2);
                p3[0] = vget_lane_u16(_fb0, 3);

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
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const unsigned short*)C + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const unsigned short*)C + j;
            }
        }

        // if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;
            unsigned short* p1 = (unsigned short*)top_blob + (i + ii + 1) * out_hstep + j;

            if (broadcast_type_C == 3)
            {
                pC = (const unsigned short*)C + (i + ii) * c_hstep + j;
            }

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + c_hstep);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c01));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                        float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                        float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        uint16x8_t _c = vld1q_u16(pC);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c0);
                        _f3 = vaddq_f32(_f3, _c1);
                        pC += 8;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f1));
                vst1_u16(p1, float2bfloat(_f2));
                vst1_u16(p1 + 4, float2bfloat(_f3));

                pp += 16;
                p0 += 8;
                p1 += 8;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        float32x4_t _c1 = bfloat2float(vld1_u16(pC + c_hstep));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        pC += 4;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p1, float2bfloat(_f1));

                pp += 8;
                p0 += 4;
                p1 += 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                float p00 = pp[0] * descale;
                float p01 = pp[1] * descale;
                float p10 = pp[2] * descale;
                float p11 = pp[3] * descale;

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        p00 += bfloat16_to_float32(pC[0]);
                        p01 += bfloat16_to_float32(pC[0]);
                        p10 += bfloat16_to_float32(pC[0]);
                        p11 += bfloat16_to_float32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        p00 += bfloat16_to_float32(pC[0]);
                        p01 += bfloat16_to_float32(pC[0]);
                        p10 += bfloat16_to_float32(pC[1]);
                        p11 += bfloat16_to_float32(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        p00 += bfloat16_to_float32(pC[0]);
                        p01 += bfloat16_to_float32(pC[c_hstep]);
                        p10 += bfloat16_to_float32(pC[1]);
                        p11 += bfloat16_to_float32(pC[c_hstep + 1]);
                        pC += 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        p00 += bfloat16_to_float32(pC[0]);
                        p01 += bfloat16_to_float32(pC[1]);
                        p10 += bfloat16_to_float32(pC[0]);
                        p11 += bfloat16_to_float32(pC[1]);
                        pC += 2;
                    }
                }

                p0[0] = float32_to_bfloat16(p00);
                p0[1] = float32_to_bfloat16(p01);
                p1[0] = float32_to_bfloat16(p10);
                p1[1] = float32_to_bfloat16(p11);

                // NCNN_LOGE("%f %f", p0[1], p1[0]);

                pp += 4;
                p0 += 2;
                p1 += 2;
            }
            for (; jj < max_jj; jj++)
            {
                float p00 = pp[0] * descale;
                float p10 = pp[1] * descale;

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        p00 += bfloat16_to_float32(pC[0]);
                        p10 += bfloat16_to_float32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        p00 += bfloat16_to_float32(pC[0]);
                        p10 += bfloat16_to_float32(pC[0]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        p00 += bfloat16_to_float32(pC[0]);
                        p10 += bfloat16_to_float32(pC[c_hstep]);
                        pC += 1;
                    }
                    if (broadcast_type_C == 4)
                    {
                        p00 += bfloat16_to_float32(pC[0]);
                        p10 += bfloat16_to_float32(pC[0]);
                        pC += 1;
                    }
                }

                p0[0] = float32_to_bfloat16(p00);
                p1[0] = float32_to_bfloat16(p10);

                pp += 2;
                p0++;
                p1++;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const unsigned short*)C + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const unsigned short*)C + j;
            }
        }

        // if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;

            if (broadcast_type_C == 3)
            {
                pC = (const unsigned short*)C + (i + ii) * c_hstep + j;
            }

            int jj = 0;
            for (; jj + 15 < max_jj; jj += 16)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                        _f2 = vaddq_f32(_f2, _c);
                        _f3 = vaddq_f32(_f3, _c);
                    }
                    if (broadcast_type_C == 3)
                    {
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + 8);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c01));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                        float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                        float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + 8);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c01));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                        float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                        float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        pC += 16;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f1));
                vst1_u16(p0 + 8, float2bfloat(_f2));
                vst1_u16(p0 + 12, float2bfloat(_f3));

                pp += 16;
                p0 += 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                        _f1 = vaddq_f32(_f1, _c);
                    }
                    if (broadcast_type_C == 3)
                    {
                        uint16x8_t _c01 = vld1q_u16(pC);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c01));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        uint16x8_t _c01 = vld1q_u16(pC);
                        float32x4_t _c0 = bfloat2float(vget_low_u16(_c01));
                        float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        pC += 8;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f1));

                pp += 8;
                p0 += 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4_t _c = vdupq_n_f32(bfloat16_to_float32(pC[0]));
                        _f0 = vaddq_f32(_f0, _c);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c0);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        float32x4_t _c0 = bfloat2float(vld1_u16(pC));
                        _f0 = vaddq_f32(_f0, _c0);
                        pC += 4;
                    }
                }

                vst1_u16(p0, float2bfloat(_f0));

                pp += 4;
                p0 += 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                float p00 = pp[0] * descale;
                float p01 = pp[1] * descale;

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        p00 += bfloat16_to_float32(pC[0]);
                        p01 += bfloat16_to_float32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        p00 += bfloat16_to_float32(pC[0]);
                        p01 += bfloat16_to_float32(pC[0]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        p00 += bfloat16_to_float32(pC[0]);
                        p01 += bfloat16_to_float32(pC[1]);
                        pC += 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        p00 += bfloat16_to_float32(pC[0]);
                        p01 += bfloat16_to_float32(pC[1]);
                        pC += 2;
                    }
                }

                p0[0] = float32_to_bfloat16(p00);
                p0[1] = float32_to_bfloat16(p01);

                pp += 2;
                p0 += 2;
            }
            for (; jj < max_jj; jj++)
            {
                float p00 = pp[0] * descale;

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        p00 += bfloat16_to_float32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        p00 += bfloat16_to_float32(pC[0]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        p00 += bfloat16_to_float32(pC[0]);
                        pC += 1;
                    }
                    if (broadcast_type_C == 4)
                    {
                        p00 += bfloat16_to_float32(pC[0]);
                        pC += 1;
                    }
                }

                p0[0] = float32_to_bfloat16(p00);

                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_unpack_output_tile_int32_to_bf16(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj, float descale)
{
    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    // NCNN_LOGE("transpose_unpack_output_tile_int32_to_fp32 %d %d %d %d    %d", i, max_ii, j, max_jj, out_elempack);

    // TODO __ARM_FEATURE_DOTPROD
    // TODO broadcast_type_C

    const int* pp = topT;

    float32x4_t _descale = vdupq_n_f32(descale);

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);
                int32x4_t _sum8 = vld1q_s32(pp + 32);
                int32x4_t _sum9 = vld1q_s32(pp + 36);
                int32x4_t _suma = vld1q_s32(pp + 40);
                int32x4_t _sumb = vld1q_s32(pp + 44);
                int32x4_t _sumc = vld1q_s32(pp + 48);
                int32x4_t _sumd = vld1q_s32(pp + 52);
                int32x4_t _sume = vld1q_s32(pp + 56);
                int32x4_t _sumf = vld1q_s32(pp + 60);

                // from
                //      a0 b1 c2 d3
                //      e4 f5 g6 h7
                //      e0 f1 g2 h3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      g4 h5 e6 f7
                //      g0 h1 e2 f3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      e7 f6 g5 h4
                //      e3 f2 g1 h0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      g7 h6 e5 f4
                //      g3 h2 e1 f0
                //      c7 d6 a5 b4

                // to
                //      a0 a1 a2 a3
                //      a4 a5 a6 a7
                //      b0 b1 b2 b3
                //      b4 b5 b6 b7
                //      c0 c1 c2 c3
                //      c4 c5 c6 c7
                //      d0 d1 d2 d3
                //      d4 d5 d6 d7
                //      e0 e1 e2 e3
                //      e4 e5 e6 e7
                //      f0 f1 f2 f3
                //      f4 f5 f6 f7
                //      g0 g1 g2 g3
                //      g4 g5 g6 g7
                //      h0 h1 h2 h3
                //      h4 h5 h6 h7
                {
                    _sum4 = vextq_s32(_sum4, _sum4, 2);
                    _sum5 = vextq_s32(_sum5, _sum5, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    _sumc = vextq_s32(_sumc, _sumc, 2);
                    _sumd = vextq_s32(_sumd, _sumd, 2);
                    _sume = vextq_s32(_sume, _sume, 2);
                    _sumf = vextq_s32(_sumf, _sumf, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sumc);
                    int32x4x2_t _t1 = vzipq_s32(_sum4, _sum8);
                    int32x4x2_t _t2 = vzipq_s32(_sum3, _sumf);
                    int32x4x2_t _t3 = vzipq_s32(_sum7, _sumb);
                    int32x4x2_t _t4 = vzipq_s32(_sum2, _sume);
                    int32x4x2_t _t5 = vzipq_s32(_sum6, _suma);
                    int32x4x2_t _t6 = vzipq_s32(_sum1, _sumd);
                    int32x4x2_t _t7 = vzipq_s32(_sum5, _sum9);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum8 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
                    _sum9 = vcombine_s32(vget_low_s32(_t6.val[0]), vget_low_s32(_t7.val[0]));
                    _suma = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
                    _sumb = vcombine_s32(vget_high_s32(_t6.val[0]), vget_high_s32(_t7.val[0]));
                    _sumc = vcombine_s32(vget_low_s32(_t5.val[1]), vget_low_s32(_t4.val[1]));
                    _sumd = vcombine_s32(vget_low_s32(_t7.val[1]), vget_low_s32(_t6.val[1]));
                    _sume = vcombine_s32(vget_high_s32(_t5.val[1]), vget_high_s32(_t4.val[1]));
                    _sumf = vcombine_s32(vget_high_s32(_t7.val[1]), vget_high_s32(_t6.val[1]));
                    _sum2 = vrev64q_s32(_sum2);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum6 = vrev64q_s32(_sum6);
                    _sum7 = vrev64q_s32(_sum7);
                    _suma = vrev64q_s32(_suma);
                    _sumb = vrev64q_s32(_sumb);
                    _sume = vrev64q_s32(_sume);
                    _sumf = vrev64q_s32(_sumf);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);
                float32x4_t _f8 = vmulq_f32(vcvtq_f32_s32(_sum8), _descale);
                float32x4_t _f9 = vmulq_f32(vcvtq_f32_s32(_sum9), _descale);
                float32x4_t _fa = vmulq_f32(vcvtq_f32_s32(_suma), _descale);
                float32x4_t _fb = vmulq_f32(vcvtq_f32_s32(_sumb), _descale);
                float32x4_t _fc = vmulq_f32(vcvtq_f32_s32(_sumc), _descale);
                float32x4_t _fd = vmulq_f32(vcvtq_f32_s32(_sumd), _descale);
                float32x4_t _fe = vmulq_f32(vcvtq_f32_s32(_sume), _descale);
                float32x4_t _ff = vmulq_f32(vcvtq_f32_s32(_sumf), _descale);

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f2));
                vst1_u16(p0 + 8, float2bfloat(_f4));
                vst1_u16(p0 + 12, float2bfloat(_f6));
                vst1_u16(p0 + 16, float2bfloat(_f8));
                vst1_u16(p0 + 20, float2bfloat(_fa));
                vst1_u16(p0 + 24, float2bfloat(_fc));
                vst1_u16(p0 + 28, float2bfloat(_fe));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_f1));
                vst1_u16(p0 + out_hstep * 4 + 4, float2bfloat(_f3));
                vst1_u16(p0 + out_hstep * 4 + 8, float2bfloat(_f5));
                vst1_u16(p0 + out_hstep * 4 + 12, float2bfloat(_f7));
                vst1_u16(p0 + out_hstep * 4 + 16, float2bfloat(_f9));
                vst1_u16(p0 + out_hstep * 4 + 20, float2bfloat(_fb));
                vst1_u16(p0 + out_hstep * 4 + 24, float2bfloat(_fd));
                vst1_u16(p0 + out_hstep * 4 + 28, float2bfloat(_ff));
                pp += 64;
                p0 += out_hstep * 8;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);

#if __ARM_FEATURE_DOTPROD
                //TODO
#else
                // from
                //      a0 b1 c2 d3
                //      e0 f1 g2 h3
                //      c0 d1 a2 b3
                //      g0 h1 e2 f3
                //      a3 b2 c1 d0
                //      e3 f2 g1 h0
                //      c3 d2 a1 b0
                //      g3 h2 e1 f0

                // to
                //      a0 a1 a2 a3
                //      b0 b1 b2 b3
                //      c0 c1 c2 c3
                //      d0 d1 d2 d3
                //      e0 e1 e2 e3
                //      f0 f1 f2 f3
                //      g0 g1 g2 g3
                //      h0 h1 h2 h3
                {
                    _sum2 = vextq_s32(_sum2, _sum2, 2);
                    _sum3 = vextq_s32(_sum3, _sum3, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                    int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                    int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                    int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum7 = vrev64q_s32(_sum7);
                }
#endif // __ARM_FEATURE_DOTPROD

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f1));
                vst1_u16(p0 + 8, float2bfloat(_f2));
                vst1_u16(p0 + 12, float2bfloat(_f3));
                vst1_u16(p0 + 16, float2bfloat(_f4));
                vst1_u16(p0 + 20, float2bfloat(_f5));
                vst1_u16(p0 + 24, float2bfloat(_f6));
                vst1_u16(p0 + 28, float2bfloat(_f7));
                pp += 32;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);
                int32x4_t _sum8 = vld1q_s32(pp + 32);
                int32x4_t _sum9 = vld1q_s32(pp + 36);
                int32x4_t _suma = vld1q_s32(pp + 40);
                int32x4_t _sumb = vld1q_s32(pp + 44);
                int32x4_t _sumc = vld1q_s32(pp + 48);
                int32x4_t _sumd = vld1q_s32(pp + 52);
                int32x4_t _sume = vld1q_s32(pp + 56);
                int32x4_t _sumf = vld1q_s32(pp + 60);

                // from
                //      a0 b1 c2 d3
                //      e4 f5 g6 h7
                //      e0 f1 g2 h3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      g4 h5 e6 f7
                //      g0 h1 e2 f3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      e7 f6 g5 h4
                //      e3 f2 g1 h0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      g7 h6 e5 f4
                //      g3 h2 e1 f0
                //      c7 d6 a5 b4

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      a4 b4 c4 d4
                //      a5 b5 c5 d5
                //      a6 b6 c6 d6
                //      a7 b7 c7 d7
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                //      e2 f2 g2 h2
                //      e3 f3 g3 h3
                //      e4 f4 g4 h4
                //      e5 f5 g5 h5
                //      e6 f6 g6 h6
                //      e7 f7 g7 h7
                {
                    _sum8 = vrev64q_s32(_sum8);
                    _sum9 = vrev64q_s32(_sum9);
                    _suma = vrev64q_s32(_suma);
                    _sumb = vrev64q_s32(_sumb);
                    _sumc = vrev64q_s32(_sumc);
                    _sumd = vrev64q_s32(_sumd);
                    _sume = vrev64q_s32(_sume);
                    _sumf = vrev64q_s32(_sumf);
                    _sum8 = vextq_s32(_sum8, _sum8, 2);
                    _sum9 = vextq_s32(_sum9, _sum9, 2);
                    _suma = vextq_s32(_suma, _suma, 2);
                    _sumb = vextq_s32(_sumb, _sumb, 2);
                    _sumc = vextq_s32(_sumc, _sumc, 2);
                    _sumd = vextq_s32(_sumd, _sumd, 2);
                    _sume = vextq_s32(_sume, _sume, 2);
                    _sumf = vextq_s32(_sumf, _sumf, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sumc);
                    int32x4x2_t _t1 = vzipq_s32(_sum4, _sum8);
                    int32x4x2_t _t2 = vzipq_s32(_sum2, _sume);
                    int32x4x2_t _t3 = vzipq_s32(_sum6, _suma);
                    int32x4x2_t _t4 = vzipq_s32(_sum3, _sumf);
                    int32x4x2_t _t5 = vzipq_s32(_sum7, _sumb);
                    int32x4x2_t _t6 = vzipq_s32(_sum1, _sumd);
                    int32x4x2_t _t7 = vzipq_s32(_sum5, _sum9);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum4 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
                    _sum5 = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
                    _sum6 = vcombine_s32(vget_low_s32(_t5.val[1]), vget_low_s32(_t4.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t5.val[1]), vget_high_s32(_t4.val[1]));
                    _sum8 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum9 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _suma = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sumb = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sumc = vcombine_s32(vget_low_s32(_t6.val[0]), vget_low_s32(_t7.val[0]));
                    _sumd = vcombine_s32(vget_high_s32(_t6.val[0]), vget_high_s32(_t7.val[0]));
                    _sume = vcombine_s32(vget_low_s32(_t7.val[1]), vget_low_s32(_t6.val[1]));
                    _sumf = vcombine_s32(vget_high_s32(_t7.val[1]), vget_high_s32(_t6.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum7 = vrev64q_s32(_sum7);
                    _sum9 = vrev64q_s32(_sum9);
                    _sumb = vrev64q_s32(_sumb);
                    _sumd = vrev64q_s32(_sumd);
                    _sumf = vrev64q_s32(_sumf);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);
                float32x4_t _f8 = vmulq_f32(vcvtq_f32_s32(_sum8), _descale);
                float32x4_t _f9 = vmulq_f32(vcvtq_f32_s32(_sum9), _descale);
                float32x4_t _fa = vmulq_f32(vcvtq_f32_s32(_suma), _descale);
                float32x4_t _fb = vmulq_f32(vcvtq_f32_s32(_sumb), _descale);
                float32x4_t _fc = vmulq_f32(vcvtq_f32_s32(_sumc), _descale);
                float32x4_t _fd = vmulq_f32(vcvtq_f32_s32(_sumd), _descale);
                float32x4_t _fe = vmulq_f32(vcvtq_f32_s32(_sume), _descale);
                float32x4_t _ff = vmulq_f32(vcvtq_f32_s32(_sumf), _descale);

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f8));
                vst1_u16(p0 + out_hstep, float2bfloat(_f1));
                vst1_u16(p0 + out_hstep + 4, float2bfloat(_f9));
                vst1_u16(p0 + out_hstep * 2, float2bfloat(_f2));
                vst1_u16(p0 + out_hstep * 2 + 4, float2bfloat(_fa));
                vst1_u16(p0 + out_hstep * 3, float2bfloat(_f3));
                vst1_u16(p0 + out_hstep * 3 + 4, float2bfloat(_fb));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_f4));
                vst1_u16(p0 + out_hstep * 4 + 4, float2bfloat(_fc));
                vst1_u16(p0 + out_hstep * 5, float2bfloat(_f5));
                vst1_u16(p0 + out_hstep * 5 + 4, float2bfloat(_fd));
                vst1_u16(p0 + out_hstep * 6, float2bfloat(_f6));
                vst1_u16(p0 + out_hstep * 6 + 4, float2bfloat(_fe));
                vst1_u16(p0 + out_hstep * 7, float2bfloat(_f7));
                vst1_u16(p0 + out_hstep * 7 + 4, float2bfloat(_ff));

                pp += 64;
                p0 += out_hstep * 8;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);

                // from
                //      a0 b1 c2 d3
                //      e0 f1 g2 h3
                //      c0 d1 a2 b3
                //      g0 h1 e2 f3
                //      a3 b2 c1 d0
                //      e3 f2 g1 h0
                //      c3 d2 a1 b0
                //      g3 h2 e1 f0

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                //      e2 f2 g2 h2
                //      e3 f3 g3 h3

                {
                    _sum4 = vrev64q_s32(_sum4);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum6 = vrev64q_s32(_sum6);
                    _sum7 = vrev64q_s32(_sum7);
                    _sum4 = vextq_s32(_sum4, _sum4, 2);
                    _sum5 = vextq_s32(_sum5, _sum5, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                    int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                    int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                    int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum7 = vrev64q_s32(_sum7);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f4));
                vst1_u16(p0 + out_hstep, float2bfloat(_f1));
                vst1_u16(p0 + out_hstep + 4, float2bfloat(_f5));
                vst1_u16(p0 + out_hstep * 2, float2bfloat(_f2));
                vst1_u16(p0 + out_hstep * 2 + 4, float2bfloat(_f6));
                vst1_u16(p0 + out_hstep * 3, float2bfloat(_f3));
                vst1_u16(p0 + out_hstep * 3 + 4, float2bfloat(_f7));

                pp += 32;
                p0 += out_hstep * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                // from
                //      a0 b1 c0 d1
                //      e0 f1 g0 h1
                //      a1 b0 c1 d0
                //      e1 f0 g1 h0

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                {
                    _sum2 = vrev64q_s32(_sum2);
                    _sum3 = vrev64q_s32(_sum3);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum2);
                    int32x4x2_t _t1 = vzipq_s32(_sum1, _sum3);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t1.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t1.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f2));
                vst1_u16(p0 + out_hstep, float2bfloat(_f1));
                vst1_u16(p0 + out_hstep + 4, float2bfloat(_f3));

                pp += 16;
                p0 += out_hstep * 2;
            }
            for (; jj < max_jj; jj += 1)
            {
                float32x4_t _r0 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(pp)), _descale);
                float32x4_t _r1 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(pp + 4)), _descale);
                vst1_u16(p0, float2bfloat(_r0));
                vst1_u16(p0 + 4, float2bfloat(_r1));
                pp += 8;
                p0 += out_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);

                // from
                //      a0 b1 c2 d3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      c7 d6 a5 b4

                // to
                //      a0 a1 a2 a3
                //      a4 a5 a6 a7
                //      b0 b1 b2 b3
                //      b4 b5 b6 b7
                //      c0 c1 c2 c3
                //      c4 c5 c6 c7
                //      d0 d1 d2 d3
                //      d4 d5 d6 d7
                {
                    _sum2 = vextq_s32(_sum2, _sum2, 2);
                    _sum3 = vextq_s32(_sum3, _sum3, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                    int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                    int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                    int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum2 = vrev64q_s32(_sum2);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum6 = vrev64q_s32(_sum6);
                    _sum7 = vrev64q_s32(_sum7);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f2));
                vst1_u16(p0 + 8, float2bfloat(_f4));
                vst1_u16(p0 + 12, float2bfloat(_f6));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_f1));
                vst1_u16(p0 + out_hstep * 4 + 4, float2bfloat(_f3));
                vst1_u16(p0 + out_hstep * 4 + 8, float2bfloat(_f5));
                vst1_u16(p0 + out_hstep * 4 + 12, float2bfloat(_f7));

                pp += 32;
                p0 += out_hstep * 8;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                // from
                //      a0 b1 c2 d3
                //      c0 d1 a2 b3
                //      a3 b2 c1 d0
                //      c3 d2 a1 b0

                // to
                //      a0 a1 a2 a3
                //      b0 b1 b2 b3
                //      c0 c1 c2 c3
                //      d0 d1 d2 d3
                {
                    _sum1 = vextq_s32(_sum1, _sum1, 2);
                    _sum3 = vextq_s32(_sum3, _sum3, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum3);
                    int32x4x2_t _t1 = vzipq_s32(_sum1, _sum2);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f1));
                vst1_u16(p0 + 8, float2bfloat(_f2));
                vst1_u16(p0 + 12, float2bfloat(_f3));

                pp += 16;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);
                int32x4_t _sum4 = vld1q_s32(pp + 16);
                int32x4_t _sum5 = vld1q_s32(pp + 20);
                int32x4_t _sum6 = vld1q_s32(pp + 24);
                int32x4_t _sum7 = vld1q_s32(pp + 28);

                // from
                //      a0 b1 c2 d3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      c7 d6 a5 b4

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      a4 b4 c4 d4
                //      a5 b5 c5 d5
                //      a6 b6 c6 d6
                //      a7 b7 c7 d7
                {
                    _sum4 = vrev64q_s32(_sum4);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum6 = vrev64q_s32(_sum6);
                    _sum7 = vrev64q_s32(_sum7);
                    _sum4 = vextq_s32(_sum4, _sum4, 2);
                    _sum5 = vextq_s32(_sum5, _sum5, 2);
                    _sum6 = vextq_s32(_sum6, _sum6, 2);
                    _sum7 = vextq_s32(_sum7, _sum7, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                    int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                    int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                    int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                    _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                    _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                    _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum5 = vrev64q_s32(_sum5);
                    _sum7 = vrev64q_s32(_sum7);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
                float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
                float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
                float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
                float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + out_hstep, float2bfloat(_f1));
                vst1_u16(p0 + out_hstep * 2, float2bfloat(_f2));
                vst1_u16(p0 + out_hstep * 3, float2bfloat(_f3));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_f4));
                vst1_u16(p0 + out_hstep * 5, float2bfloat(_f5));
                vst1_u16(p0 + out_hstep * 6, float2bfloat(_f6));
                vst1_u16(p0 + out_hstep * 7, float2bfloat(_f7));

                pp += 32;
                p0 += out_hstep * 8;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                // from
                //      a0 b1 c2 d3
                //      c0 d1 a2 b3
                //      a3 b2 c1 d0
                //      c3 d2 a1 b0

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3

                {
                    _sum2 = vrev64q_s32(_sum2);
                    _sum3 = vrev64q_s32(_sum3);
                    _sum2 = vextq_s32(_sum2, _sum2, 2);
                    _sum3 = vextq_s32(_sum3, _sum3, 2);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum3);
                    int32x4x2_t _t1 = vzipq_s32(_sum1, _sum2);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                    _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                    _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                    _sum3 = vrev64q_s32(_sum3);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + out_hstep, float2bfloat(_f1));
                vst1_u16(p0 + out_hstep * 2, float2bfloat(_f2));
                vst1_u16(p0 + out_hstep * 3, float2bfloat(_f3));

                pp += 16;
                p0 += out_hstep * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);

                // from
                //      a0 b1 c0 d1
                //      a1 b0 c1 d0

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                {
                    _sum1 = vrev64q_s32(_sum1);
                    int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
                    _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                    _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                    _sum1 = vrev64q_s32(_sum1);
                }

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + out_hstep, float2bfloat(_f1));

                pp += 8;
                p0 += out_hstep * 2;
            }
            for (; jj < max_jj; jj += 1)
            {
                float32x4_t _r0 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(pp)), _descale);
                vst1_u16(p0, float2bfloat(_r0));
                pp += 4;
                p0 += out_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __ARM_NEON
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                int32x4_t _sum2 = vld1q_s32(pp + 8);
                int32x4_t _sum3 = vld1q_s32(pp + 12);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
                float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

                vst1_u16(p0, float2bfloat(_f0));
                vst1_u16(p0 + 4, float2bfloat(_f2));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_f1));
                vst1_u16(p0 + out_hstep * 4 + 4, float2bfloat(_f3));

                pp += 16;
                p0 += out_hstep * 8;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                // a0 a1 a2 a3
                // b0 b1 b2 b3

                p0[0] = float32_to_bfloat16(pp[0] * descale);
                p0[1] = float32_to_bfloat16(pp[1] * descale);
                p0[2] = float32_to_bfloat16(pp[2] * descale);
                p0[3] = float32_to_bfloat16(pp[3] * descale);
                p0[4] = float32_to_bfloat16(pp[4] * descale);
                p0[5] = float32_to_bfloat16(pp[5] * descale);
                p0[6] = float32_to_bfloat16(pp[6] * descale);
                p0[7] = float32_to_bfloat16(pp[7] * descale);
                pp += 8;
                p0 += out_hstep * 4;
            }
        }
#endif // __ARM_NEON
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            int jj = 0;
#if __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                // a0 a1 a2 a3
                // b0 b1 b2 b3
                // a4 a5 a6 a7
                // b4 b5 b6 b7

                p0[0] = float32_to_bfloat16(pp[0] * descale);
                p0[1] = float32_to_bfloat16(pp[4] * descale);
                p0[out_hstep] = float32_to_bfloat16(pp[1] * descale);
                p0[out_hstep + 1] = float32_to_bfloat16(pp[5] * descale);
                p0[out_hstep * 2] = float32_to_bfloat16(pp[2] * descale);
                p0[out_hstep * 2 + 1] = float32_to_bfloat16(pp[6] * descale);
                p0[out_hstep * 3] = float32_to_bfloat16(pp[3] * descale);
                p0[out_hstep * 3 + 1] = float32_to_bfloat16(pp[7] * descale);
                p0[out_hstep * 4] = float32_to_bfloat16(pp[8] * descale);
                p0[out_hstep * 4 + 1] = float32_to_bfloat16(pp[12] * descale);
                p0[out_hstep * 5] = float32_to_bfloat16(pp[9] * descale);
                p0[out_hstep * 5 + 1] = float32_to_bfloat16(pp[13] * descale);
                p0[out_hstep * 6] = float32_to_bfloat16(pp[10] * descale);
                p0[out_hstep * 6 + 1] = float32_to_bfloat16(pp[14] * descale);
                p0[out_hstep * 7] = float32_to_bfloat16(pp[11] * descale);
                p0[out_hstep * 7 + 1] = float32_to_bfloat16(pp[15] * descale);
                pp += 16;
                p0 += out_hstep * 8;
            }
#endif // __aarch64__
            for (; jj + 3 < max_jj; jj += 4)
            {
                // a0 a1 a2 a3
                // b0 b1 b2 b3

                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);

                int32x4x2_t _sum01 = vzipq_s32(_sum0, _sum1);

                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum01.val[0]), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum01.val[1]), _descale);

                uint16x4_t _fb0 = float2bfloat(_f0);
                uint16x4_t _fb1 = float2bfloat(_f1);

                p0[0] = vget_lane_u16(_fb0, 0);
                p0[1] = vget_lane_u16(_fb0, 1);
                p0[out_hstep] = vget_lane_u16(_fb0, 2);
                p0[out_hstep + 1] = vget_lane_u16(_fb0, 3);
                p0[out_hstep * 2] = vget_lane_u16(_fb1, 0);
                p0[out_hstep * 2 + 1] = vget_lane_u16(_fb1, 1);
                p0[out_hstep * 3] = vget_lane_u16(_fb1, 2);
                p0[out_hstep * 3 + 1] = vget_lane_u16(_fb1, 3);

                // vst1_f32(p0, vget_low_f32(_f0));
                // vst1_f32(p0 + out_hstep, vget_high_f32(_f0));
                // vst1_f32(p0 + out_hstep * 2, vget_low_f32(_f1));
                // vst1_f32(p0 + out_hstep * 3, vget_high_f32(_f1));

                // p0[0] = pp[0] * descale;
                // p0[1] = pp[4] * descale;
                // p0[out_hstep] = pp[1] * descale;
                // p0[out_hstep + 1] = pp[5] * descale;
                // p0[out_hstep * 2] = pp[2] * descale;
                // p0[out_hstep * 2 + 1] = pp[6] * descale;
                // p0[out_hstep * 3] = pp[3] * descale;
                // p0[out_hstep * 3 + 1] = pp[7] * descale;
                pp += 8;
                p0 += out_hstep * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                // a0 a1 b0 b1

                p0[0] = float32_to_bfloat16(pp[0] * descale);
                p0[1] = float32_to_bfloat16(pp[2] * descale);
                p0[out_hstep] = float32_to_bfloat16(pp[1] * descale);
                p0[out_hstep + 1] = float32_to_bfloat16(pp[3] * descale);
                pp += 4;
                p0 += out_hstep * 2;
            }
            for (; jj < max_jj; jj += 1)
            {
                p0[0] = float32_to_bfloat16(pp[0] * descale);
                p0[1] = float32_to_bfloat16(pp[1] * descale);
                pp += 2;
                p0 += out_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
#if __ARM_NEON
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                float32x4_t _r0 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(pp)), _descale);
                vst1_u16(p0, float2bfloat(_r0));
                pp += 4;
                p0 += out_hstep * 4;
            }
        }
#endif // __ARM_NEON
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            int jj = 0;
            for (; jj + 7 < max_jj; jj += 8)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                int32x4_t _sum1 = vld1q_s32(pp + 4);
                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
                uint16x4_t _fb0 = float2bfloat(_f0);
                uint16x4_t _fb1 = float2bfloat(_f1);

                p0[0] = vget_lane_u16(_fb0, 0);
                p0[out_hstep] = vget_lane_u16(_fb0, 1);
                p0[out_hstep * 2] = vget_lane_u16(_fb0, 2);
                p0[out_hstep * 3] = vget_lane_u16(_fb0, 3);
                p0[out_hstep * 4] = vget_lane_u16(_fb1, 0);
                p0[out_hstep * 5] = vget_lane_u16(_fb1, 1);
                p0[out_hstep * 6] = vget_lane_u16(_fb1, 2);
                p0[out_hstep * 7] = vget_lane_u16(_fb1, 3);
                pp += 8;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                int32x4_t _sum0 = vld1q_s32(pp);
                float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
                uint16x4_t _fb0 = float2bfloat(_f0);

                p0[0] = vget_lane_u16(_fb0, 0);
                p0[out_hstep] = vget_lane_u16(_fb0, 1);
                p0[out_hstep * 2] = vget_lane_u16(_fb0, 2);
                p0[out_hstep * 3] = vget_lane_u16(_fb0, 3);
                pp += 4;
                p0 += out_hstep * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                p0[0] = float32_to_bfloat16(pp[0] * descale);
                p0[out_hstep] = float32_to_bfloat16(pp[1] * descale);
                pp += 2;
                p0 += out_hstep * 2;
            }
            for (; jj < max_jj; jj += 1)
            {
                p0[0] = float32_to_bfloat16(pp[0] * descale);
                pp += 1;
                p0 += out_hstep;
            }
        }
    }
}

static void gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
    // NCNN_LOGE("gemm_transB_packed_tile_int8 %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const signed char* pAT = AT_tile;
    const signed char* pBT = BT_tile;

    int* outptr = topT_tile;

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

#if NCNN_GNU_INLINE_ASM
            asm volatile(
#if !__ARM_FEATURE_MATMUL_INT8
                "cmp    %w7, #0                     \n"
                "beq    0f                          \n"

                "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                "sub    %0, %0, #192                \n"
                "b      1f                          \n"

                "0:                                 \n"
                "eor    v16.16b, v16.16b, v16.16b   \n"
                "eor    v17.16b, v17.16b, v17.16b   \n"
                "eor    v18.16b, v18.16b, v18.16b   \n"
                "eor    v19.16b, v19.16b, v19.16b   \n"
                "eor    v20.16b, v20.16b, v20.16b   \n"
                "eor    v21.16b, v21.16b, v21.16b   \n"
                "eor    v22.16b, v22.16b, v22.16b   \n"
                "eor    v23.16b, v23.16b, v23.16b   \n"
                "eor    v24.16b, v24.16b, v24.16b   \n"
                "eor    v25.16b, v25.16b, v25.16b   \n"
                "eor    v26.16b, v26.16b, v26.16b   \n"
                "eor    v27.16b, v27.16b, v27.16b   \n"
                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v29.16b, v29.16b, v29.16b   \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"
                "eor    v31.16b, v31.16b, v31.16b   \n"

                "1:                                 \n"
#endif // !__ARM_FEATURE_MATMUL_INT8

#if __ARM_FEATURE_DOTPROD
                "lsr    w4, %w6, #3                 \n" // w4 = max_kk >> 3
                "cmp    w4, #0                      \n"
                "beq    101f                        \n"

#if __ARM_FEATURE_MATMUL_INT8
                "eor    v0.16b, v0.16b, v0.16b      \n"
                "eor    v1.16b, v1.16b, v1.16b      \n"
                "eor    v2.16b, v2.16b, v2.16b      \n"
                "eor    v3.16b, v3.16b, v3.16b      \n"
                "eor    v4.16b, v4.16b, v4.16b      \n"
                "eor    v5.16b, v5.16b, v5.16b      \n"
                "eor    v6.16b, v6.16b, v6.16b      \n"
                "eor    v7.16b, v7.16b, v7.16b      \n"
                "eor    v8.16b, v8.16b, v8.16b      \n"
                "eor    v9.16b, v9.16b, v9.16b      \n"
                "eor    v10.16b, v10.16b, v10.16b   \n"
                "eor    v11.16b, v11.16b, v11.16b   \n"
                "eor    v12.16b, v12.16b, v12.16b   \n"
                "eor    v13.16b, v13.16b, v13.16b   \n"
                "eor    v14.16b, v14.16b, v14.16b   \n"
                "eor    v15.16b, v15.16b, v15.16b   \n"

                "2:                                 \n"
                "ld1    {v16.16b, v17.16b, v18.16b, v19.16b}, [%1], #64 \n"
                "ld1    {v20.16b, v21.16b, v22.16b, v23.16b}, [%2], #64 \n"
                "smmla  v0.4s, v16.16b, v20.16b     \n"
                "smmla  v1.4s, v17.16b, v20.16b     \n"
                "smmla  v2.4s, v16.16b, v21.16b     \n"
                "smmla  v3.4s, v17.16b, v21.16b     \n"
                "smmla  v4.4s, v18.16b, v20.16b     \n"
                "smmla  v5.4s, v19.16b, v20.16b     \n"
                "smmla  v6.4s, v18.16b, v21.16b     \n"
                "smmla  v7.4s, v19.16b, v21.16b     \n"
                "subs   w4, w4, #1                  \n"
                "smmla  v8.4s, v16.16b, v22.16b     \n"
                "smmla  v9.4s, v17.16b, v22.16b     \n"
                "smmla  v10.4s, v16.16b, v23.16b    \n"
                "smmla  v11.4s, v17.16b, v23.16b    \n"
                "smmla  v12.4s, v18.16b, v22.16b    \n"
                "smmla  v13.4s, v19.16b, v22.16b    \n"
                "smmla  v14.4s, v18.16b, v23.16b    \n"
                "smmla  v15.4s, v19.16b, v23.16b    \n"
                "bne    2b                          \n"

                "uzp1   v16.4s, v0.4s, v1.4s        \n"
                "uzp2   v17.4s, v0.4s, v1.4s        \n"
                "uzp1   v18.4s, v2.4s, v3.4s        \n"
                "uzp2   v19.4s, v2.4s, v3.4s        \n"
                "uzp1   v20.4s, v4.4s, v5.4s        \n"
                "uzp2   v21.4s, v4.4s, v5.4s        \n"
                "uzp1   v22.4s, v6.4s, v7.4s        \n"
                "uzp2   v23.4s, v6.4s, v7.4s        \n"
                "uzp1   v24.4s, v8.4s, v9.4s        \n"
                "uzp2   v25.4s, v8.4s, v9.4s        \n"
                "uzp1   v26.4s, v10.4s, v11.4s      \n"
                "uzp2   v27.4s, v10.4s, v11.4s      \n"
                "uzp1   v28.4s, v12.4s, v13.4s      \n"
                "uzp2   v29.4s, v12.4s, v13.4s      \n"
                "uzp1   v30.4s, v14.4s, v15.4s      \n"
                "uzp2   v31.4s, v14.4s, v15.4s      \n"

                "cmp    %w7, #0                     \n"
                "beq    1f                          \n"

                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64   \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64   \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64 \n"
                "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0]    \n"
                "sub    %0, %0, #192                \n"
                "add    v16.4s, v16.4s, v0.4s       \n"
                "add    v17.4s, v17.4s, v1.4s       \n"
                "add    v18.4s, v18.4s, v2.4s       \n"
                "add    v19.4s, v19.4s, v3.4s       \n"
                "add    v20.4s, v20.4s, v4.4s       \n"
                "add    v21.4s, v21.4s, v5.4s       \n"
                "add    v22.4s, v22.4s, v6.4s       \n"
                "add    v23.4s, v23.4s, v7.4s       \n"
                "add    v24.4s, v24.4s, v8.4s       \n"
                "add    v25.4s, v25.4s, v9.4s       \n"
                "add    v26.4s, v26.4s, v10.4s      \n"
                "add    v27.4s, v27.4s, v11.4s      \n"
                "add    v28.4s, v28.4s, v12.4s      \n"
                "add    v29.4s, v29.4s, v13.4s      \n"
                "add    v30.4s, v30.4s, v14.4s      \n"
                "add    v31.4s, v31.4s, v15.4s      \n"
                "b      1f                          \n"
#else  // __ARM_FEATURE_MATMUL_INT8
                "2:                                 \n"
                "ld1    {v0.16b, v1.16b, v2.16b, v3.16b}, [%1], #64 \n"
                "ld1    {v4.16b, v5.16b, v6.16b, v7.16b}, [%2], #64 \n"
                "sdot   v16.4s, v0.16b, v4.4b[0]    \n"
                "sdot   v17.4s, v0.16b, v4.4b[1]    \n"
                "sdot   v18.4s, v0.16b, v4.4b[2]    \n"
                "sdot   v19.4s, v0.16b, v4.4b[3]    \n"
                "sdot   v20.4s, v1.16b, v4.4b[0]    \n"
                "sdot   v21.4s, v1.16b, v4.4b[1]    \n"
                "sdot   v22.4s, v1.16b, v4.4b[2]    \n"
                "sdot   v23.4s, v1.16b, v4.4b[3]    \n"
                "sdot   v24.4s, v0.16b, v5.4b[0]    \n"
                "sdot   v25.4s, v0.16b, v5.4b[1]    \n"
                "sdot   v26.4s, v0.16b, v5.4b[2]    \n"
                "sdot   v27.4s, v0.16b, v5.4b[3]    \n"
                "sdot   v28.4s, v1.16b, v5.4b[0]    \n"
                "sdot   v29.4s, v1.16b, v5.4b[1]    \n"
                "sdot   v30.4s, v1.16b, v5.4b[2]    \n"
                "sdot   v31.4s, v1.16b, v5.4b[3]    \n"
                "subs   w4, w4, #1                  \n"
                "sdot   v16.4s, v2.16b, v6.4b[0]    \n"
                "sdot   v17.4s, v2.16b, v6.4b[1]    \n"
                "sdot   v18.4s, v2.16b, v6.4b[2]    \n"
                "sdot   v19.4s, v2.16b, v6.4b[3]    \n"
                "sdot   v20.4s, v3.16b, v6.4b[0]    \n"
                "sdot   v21.4s, v3.16b, v6.4b[1]    \n"
                "sdot   v22.4s, v3.16b, v6.4b[2]    \n"
                "sdot   v23.4s, v3.16b, v6.4b[3]    \n"
                "sdot   v24.4s, v2.16b, v7.4b[0]    \n"
                "sdot   v25.4s, v2.16b, v7.4b[1]    \n"
                "sdot   v26.4s, v2.16b, v7.4b[2]    \n"
                "sdot   v27.4s, v2.16b, v7.4b[3]    \n"
                "sdot   v28.4s, v3.16b, v7.4b[0]    \n"
                "sdot   v29.4s, v3.16b, v7.4b[1]    \n"
                "sdot   v30.4s, v3.16b, v7.4b[2]    \n"
                "sdot   v31.4s, v3.16b, v7.4b[3]    \n"
                "bne    2b                          \n"
#endif // __ARM_FEATURE_MATMUL_INT8

                "101:                               \n"
#if __ARM_FEATURE_MATMUL_INT8
                "cmp    %w7, #0                     \n"
                "beq    0f                          \n"

                "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                "sub    %0, %0, #192                \n"
                "b      1f                          \n"

                "0:                                 \n"
                "eor    v16.16b, v16.16b, v16.16b   \n"
                "eor    v17.16b, v17.16b, v17.16b   \n"
                "eor    v18.16b, v18.16b, v18.16b   \n"
                "eor    v19.16b, v19.16b, v19.16b   \n"
                "eor    v20.16b, v20.16b, v20.16b   \n"
                "eor    v21.16b, v21.16b, v21.16b   \n"
                "eor    v22.16b, v22.16b, v22.16b   \n"
                "eor    v23.16b, v23.16b, v23.16b   \n"
                "eor    v24.16b, v24.16b, v24.16b   \n"
                "eor    v25.16b, v25.16b, v25.16b   \n"
                "eor    v26.16b, v26.16b, v26.16b   \n"
                "eor    v27.16b, v27.16b, v27.16b   \n"
                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v29.16b, v29.16b, v29.16b   \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"
                "eor    v31.16b, v31.16b, v31.16b   \n"
                "1:                                 \n"
#endif // __ARM_FEATURE_MATMUL_INT8

                "and    w4, %w6, #4                 \n" // w4 = remain = max_kk & 4
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                // kk += 4 part
                "ld1    {v0.16b, v1.16b}, [%1], #32 \n"
                "ld1    {v2.16b, v3.16b}, [%2], #32 \n"
                "sdot   v16.4s, v0.16b, v2.4b[0]    \n"
                "sdot   v17.4s, v0.16b, v2.4b[1]    \n"
                "sdot   v18.4s, v0.16b, v2.4b[2]    \n"
                "sdot   v19.4s, v0.16b, v2.4b[3]    \n"
                "sdot   v20.4s, v1.16b, v2.4b[0]    \n"
                "sdot   v21.4s, v1.16b, v2.4b[1]    \n"
                "sdot   v22.4s, v1.16b, v2.4b[2]    \n"
                "sdot   v23.4s, v1.16b, v2.4b[3]    \n"
                "sdot   v24.4s, v0.16b, v3.4b[0]    \n"
                "sdot   v25.4s, v0.16b, v3.4b[1]    \n"
                "sdot   v26.4s, v0.16b, v3.4b[2]    \n"
                "sdot   v27.4s, v0.16b, v3.4b[3]    \n"
                "sdot   v28.4s, v1.16b, v3.4b[0]    \n"
                "sdot   v29.4s, v1.16b, v3.4b[1]    \n"
                "sdot   v30.4s, v1.16b, v3.4b[2]    \n"
                "sdot   v31.4s, v1.16b, v3.4b[3]    \n"
#else  // __ARM_FEATURE_DOTPROD
                "lsr    w4, %w6, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                "2:                                 \n"
                "ld1    {v0.16b, v1.16b}, [%1], #32 \n"
                "ld1    {v4.16b, v5.16b}, [%2], #32 \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull2 v9.8h, v0.16b, v4.16b       \n"
                "rev64  v2.4s, v0.4s                \n"
                "smull  v10.8h, v2.8b, v4.8b        \n"
                "smull2 v11.8h, v2.16b, v4.16b      \n"
                "rev64  v6.8h, v4.8h                \n"
                "smull  v12.8h, v0.8b, v6.8b        \n"
                "smull2 v13.8h, v0.16b, v6.16b      \n"
                "rev64  v3.4s, v1.4s                \n"
                "smull  v14.8h, v2.8b, v6.8b        \n"
                "smull2 v15.8h, v2.16b, v6.16b      \n"
                "rev64  v7.8h, v5.8h                \n"
                "smlal  v8.8h, v1.8b, v5.8b         \n"
                "smlal2 v9.8h, v1.16b, v5.16b       \n"
                "smlal  v10.8h, v3.8b, v5.8b        \n"
                "smlal2 v11.8h, v3.16b, v5.16b      \n"
                "smlal  v12.8h, v1.8b, v7.8b        \n"
                "smlal2 v13.8h, v1.16b, v7.16b      \n"
                "smlal  v14.8h, v3.8b, v7.8b        \n"
                "smlal2 v15.8h, v3.16b, v7.16b      \n"
                "ext    v0.16b, v0.16b, v0.16b, #8  \n"
                "ext    v2.16b, v2.16b, v2.16b, #8  \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v20.4s, v10.8h              \n"
                "sadalp v21.4s, v11.8h              \n"
                "ext    v1.16b, v1.16b, v1.16b, #8  \n"
                "ext    v3.16b, v3.16b, v3.16b, #8  \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull2 v9.8h, v0.16b, v4.16b       \n"
                "smull  v10.8h, v2.8b, v4.8b        \n"
                "smull2 v11.8h, v2.16b, v4.16b      \n"
                "sadalp v24.4s, v12.8h              \n"
                "sadalp v25.4s, v13.8h              \n"
                "sadalp v28.4s, v14.8h              \n"
                "sadalp v29.4s, v15.8h              \n"
                "smull  v12.8h, v0.8b, v6.8b        \n"
                "smull2 v13.8h, v0.16b, v6.16b      \n"
                "smull  v14.8h, v2.8b, v6.8b        \n"
                "smull2 v15.8h, v2.16b, v6.16b      \n"
                "smlal  v8.8h, v1.8b, v5.8b         \n"
                "smlal2 v9.8h, v1.16b, v5.16b       \n"
                "smlal  v10.8h, v3.8b, v5.8b        \n"
                "smlal2 v11.8h, v3.16b, v5.16b      \n"
                "smlal  v12.8h, v1.8b, v7.8b        \n"
                "smlal2 v13.8h, v1.16b, v7.16b      \n"
                "smlal  v14.8h, v3.8b, v7.8b        \n"
                "smlal2 v15.8h, v3.16b, v7.16b      \n"
                "subs   w4, w4, #1                  \n"
                "sadalp v18.4s, v8.8h               \n"
                "sadalp v19.4s, v9.8h               \n"
                "sadalp v22.4s, v10.8h              \n"
                "sadalp v23.4s, v11.8h              \n"
                "sadalp v26.4s, v12.8h              \n"
                "sadalp v27.4s, v13.8h              \n"
                "sadalp v30.4s, v14.8h              \n"
                "sadalp v31.4s, v15.8h              \n"
                "bne    2b                          \n"
#endif // __ARM_FEATURE_DOTPROD

                "3:                                 \n"
                "and    w4, %w6, #2                 \n" // w4 = remain = max_kk & 2
                "cmp    w4, #0                      \n"
                "beq    4f                          \n"

                // kk += 2 part
#if __ARM_FEATURE_DOTPROD
                "ld1    {v0.16b}, [%1], #16         \n"
                "ld1    {v1.16b}, [%2], #16         \n"
                "dup    v4.8h, v1.h[0]              \n"
                "dup    v5.8h, v1.h[1]              \n"
                "dup    v6.8h, v1.h[2]              \n"
                "dup    v7.8h, v1.h[3]              \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull  v9.8h, v0.8b, v5.8b         \n"
                "smull  v10.8h, v0.8b, v6.8b        \n"
                "smull  v11.8h, v0.8b, v7.8b        \n"
                "smull2 v12.8h, v0.16b, v4.16b      \n"
                "smull2 v13.8h, v0.16b, v5.16b      \n"
                "smull2 v14.8h, v0.16b, v6.16b      \n"
                "smull2 v15.8h, v0.16b, v7.16b      \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
                "sadalp v20.4s, v12.8h              \n"
                "sadalp v21.4s, v13.8h              \n"
                "sadalp v22.4s, v14.8h              \n"
                "sadalp v23.4s, v15.8h              \n"
                "dup    v4.8h, v1.h[4]              \n"
                "dup    v5.8h, v1.h[5]              \n"
                "dup    v6.8h, v1.h[6]              \n"
                "dup    v7.8h, v1.h[7]              \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull  v9.8h, v0.8b, v5.8b         \n"
                "smull  v10.8h, v0.8b, v6.8b        \n"
                "smull  v11.8h, v0.8b, v7.8b        \n"
                "smull2 v12.8h, v0.16b, v4.16b      \n"
                "smull2 v13.8h, v0.16b, v5.16b      \n"
                "smull2 v14.8h, v0.16b, v6.16b      \n"
                "smull2 v15.8h, v0.16b, v7.16b      \n"
                "sadalp v24.4s, v8.8h               \n"
                "sadalp v25.4s, v9.8h               \n"
                "sadalp v26.4s, v10.8h              \n"
                "sadalp v27.4s, v11.8h              \n"
                "sadalp v28.4s, v12.8h              \n"
                "sadalp v29.4s, v13.8h              \n"
                "sadalp v30.4s, v14.8h              \n"
                "sadalp v31.4s, v15.8h              \n"
#else  // __ARM_FEATURE_DOTPROD
                "ld1    {v0.16b}, [%1], #16         \n"
                "ld1    {v2.16b}, [%2], #16         \n"
                "rev64  v1.4s, v0.4s                \n"
                "rev64  v3.8h, v2.8h                \n"
                "smull  v8.8h, v0.8b, v2.8b         \n"
                "smull2 v9.8h, v0.16b, v2.16b       \n"
                "smull  v10.8h, v1.8b, v2.8b        \n"
                "smull2 v11.8h, v1.16b, v2.16b      \n"
                "smull  v12.8h, v0.8b, v3.8b        \n"
                "smull2 v13.8h, v0.16b, v3.16b      \n"
                "smull  v14.8h, v1.8b, v3.8b        \n"
                "smull2 v15.8h, v1.16b, v3.16b      \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v20.4s, v10.8h              \n"
                "sadalp v21.4s, v11.8h              \n"
                "sadalp v24.4s, v12.8h              \n"
                "sadalp v25.4s, v13.8h              \n"
                "sadalp v28.4s, v14.8h              \n"
                "sadalp v29.4s, v15.8h              \n"
                "ext    v0.16b, v0.16b, v0.16b, #8  \n"
                "ext    v1.16b, v1.16b, v1.16b, #8  \n"
                "smull  v8.8h, v0.8b, v2.8b         \n"
                "smull2 v9.8h, v0.16b, v2.16b       \n"
                "smull  v10.8h, v1.8b, v2.8b        \n"
                "smull2 v11.8h, v1.16b, v2.16b      \n"
                "smull  v12.8h, v0.8b, v3.8b        \n"
                "smull2 v13.8h, v0.16b, v3.16b      \n"
                "smull  v14.8h, v1.8b, v3.8b        \n"
                "smull2 v15.8h, v1.16b, v3.16b      \n"
                "sadalp v18.4s, v8.8h               \n"
                "sadalp v19.4s, v9.8h               \n"
                "sadalp v22.4s, v10.8h              \n"
                "sadalp v23.4s, v11.8h              \n"
                "sadalp v26.4s, v12.8h              \n"
                "sadalp v27.4s, v13.8h              \n"
                "sadalp v30.4s, v14.8h              \n"
                "sadalp v31.4s, v15.8h              \n"
#endif // __ARM_FEATURE_DOTPROD

                "4:                                 \n"
                "and    w4, %w6, #1                 \n" // w4 = remain = max_kk & 1
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                // kk += 1 part
#if __ARM_FEATURE_DOTPROD
                "ld1    {v0.8b}, [%1], #8           \n"
                "ld1    {v1.8b}, [%2], #8           \n"
                "dup    v8.8b, v1.b[0]              \n"
                "dup    v9.8b, v1.b[1]              \n"
                "dup    v10.8b, v1.b[2]             \n"
                "dup    v11.8b, v1.b[3]             \n"
                "dup    v12.8b, v1.b[4]             \n"
                "dup    v13.8b, v1.b[5]             \n"
                "dup    v14.8b, v1.b[6]             \n"
                "dup    v15.8b, v1.b[7]             \n"
                "smull  v8.8h, v0.8b, v8.8b         \n"
                "smull  v9.8h, v0.8b, v9.8b         \n"
                "smull  v10.8h, v0.8b, v10.8b       \n"
                "smull  v11.8h, v0.8b, v11.8b       \n"
                "smull  v12.8h, v0.8b, v12.8b       \n"
                "smull  v13.8h, v0.8b, v13.8b       \n"
                "smull  v14.8h, v0.8b, v14.8b       \n"
                "smull  v15.8h, v0.8b, v15.8b       \n"
                "saddw  v16.4s, v16.4s, v8.4h       \n"
                "saddw  v17.4s, v17.4s, v9.4h       \n"
                "saddw  v18.4s, v18.4s, v10.4h      \n"
                "saddw  v19.4s, v19.4s, v11.4h      \n"
                "saddw2 v20.4s, v20.4s, v8.8h       \n"
                "saddw2 v21.4s, v21.4s, v9.8h       \n"
                "saddw2 v22.4s, v22.4s, v10.8h      \n"
                "saddw2 v23.4s, v23.4s, v11.8h      \n"
                "saddw  v24.4s, v24.4s, v12.4h      \n"
                "saddw  v25.4s, v25.4s, v13.4h      \n"
                "saddw  v26.4s, v26.4s, v14.4h      \n"
                "saddw  v27.4s, v27.4s, v15.4h      \n"
                "saddw2 v28.4s, v28.4s, v12.8h      \n"
                "saddw2 v29.4s, v29.4s, v13.8h      \n"
                "saddw2 v30.4s, v30.4s, v14.8h      \n"
                "saddw2 v31.4s, v31.4s, v15.8h      \n"
#else  // __ARM_FEATURE_DOTPROD
                "ld1    {v0.8b}, [%1], #8           \n"
                "ld1    {v4.8b}, [%2], #8           \n"
                "ext    v1.8b, v0.8b, v0.8b, #4     \n"
                "rev32  v2.4h, v0.4h                \n"
                "rev64  v3.4h, v0.4h                \n"
                "rev32  v5.8b, v4.8b                \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull  v9.8h, v1.8b, v4.8b         \n"
                "smull  v10.8h, v2.8b, v4.8b        \n"
                "smull  v11.8h, v3.8b, v4.8b        \n"
                "smull  v12.8h, v0.8b, v5.8b        \n"
                "smull  v13.8h, v1.8b, v5.8b        \n"
                "smull  v14.8h, v2.8b, v5.8b        \n"
                "smull  v15.8h, v3.8b, v5.8b        \n"
                "saddw  v16.4s, v16.4s, v8.4h       \n"
                "saddw2 v17.4s, v17.4s, v8.8h       \n"
                "saddw  v18.4s, v18.4s, v9.4h       \n"
                "saddw2 v19.4s, v19.4s, v9.8h       \n"
                "saddw  v20.4s, v20.4s, v10.4h      \n"
                "saddw2 v21.4s, v21.4s, v10.8h      \n"
                "saddw  v22.4s, v22.4s, v11.4h      \n"
                "saddw2 v23.4s, v23.4s, v11.8h      \n"
                "saddw  v24.4s, v24.4s, v12.4h      \n"
                "saddw2 v25.4s, v25.4s, v12.8h      \n"
                "saddw  v26.4s, v26.4s, v13.4h      \n"
                "saddw2 v27.4s, v27.4s, v13.8h      \n"
                "saddw  v28.4s, v28.4s, v14.4h      \n"
                "saddw2 v29.4s, v29.4s, v14.8h      \n"
                "saddw  v30.4s, v30.4s, v15.4h      \n"
                "saddw2 v31.4s, v31.4s, v15.8h      \n"
#endif // __ARM_FEATURE_DOTPROD

                "5:                                 \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB)      // %2
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "r"(max_kk), // %6
                "r"(k)       // %7
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else // NCNN_GNU_INLINE_ASM
            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;
            int32x4_t _sum4;
            int32x4_t _sum5;
            int32x4_t _sum6;
            int32x4_t _sum7;
            int32x4_t _sum8;
            int32x4_t _sum9;
            int32x4_t _suma;
            int32x4_t _sumb;
            int32x4_t _sumc;
            int32x4_t _sumd;
            int32x4_t _sume;
            int32x4_t _sumf;

#if __ARM_FEATURE_MATMUL_INT8
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
                _sum4 = vdupq_n_s32(0);
                _sum5 = vdupq_n_s32(0);
                _sum6 = vdupq_n_s32(0);
                _sum7 = vdupq_n_s32(0);
                _sum8 = vdupq_n_s32(0);
                _sum9 = vdupq_n_s32(0);
                _suma = vdupq_n_s32(0);
                _sumb = vdupq_n_s32(0);
                _sumc = vdupq_n_s32(0);
                _sumd = vdupq_n_s32(0);
                _sume = vdupq_n_s32(0);
                _sumf = vdupq_n_s32(0);
            }
#else  // __ARM_FEATURE_MATMUL_INT8
            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
                _sum4 = vdupq_n_s32(0);
                _sum5 = vdupq_n_s32(0);
                _sum6 = vdupq_n_s32(0);
                _sum7 = vdupq_n_s32(0);
                _sum8 = vdupq_n_s32(0);
                _sum9 = vdupq_n_s32(0);
                _suma = vdupq_n_s32(0);
                _sumb = vdupq_n_s32(0);
                _sumc = vdupq_n_s32(0);
                _sumd = vdupq_n_s32(0);
                _sume = vdupq_n_s32(0);
                _sumf = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
                _sum4 = vld1q_s32(outptr + 16);
                _sum5 = vld1q_s32(outptr + 20);
                _sum6 = vld1q_s32(outptr + 24);
                _sum7 = vld1q_s32(outptr + 28);
                _sum8 = vld1q_s32(outptr + 32);
                _sum9 = vld1q_s32(outptr + 36);
                _suma = vld1q_s32(outptr + 40);
                _sumb = vld1q_s32(outptr + 44);
                _sumc = vld1q_s32(outptr + 48);
                _sumd = vld1q_s32(outptr + 52);
                _sume = vld1q_s32(outptr + 56);
                _sumf = vld1q_s32(outptr + 60);
            }
#endif // __ARM_FEATURE_MATMUL_INT8

            int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
            {
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA1 = vld1q_s8(pA + 16);
                    int8x16_t _pA2 = vld1q_s8(pA + 32);
                    int8x16_t _pA3 = vld1q_s8(pA + 48);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);
                    int8x16_t _pB2 = vld1q_s8(pB + 32);
                    int8x16_t _pB3 = vld1q_s8(pB + 48);

                    _sum0 = vmmlaq_s32(_sum0, _pA0, _pB0);
                    _sum1 = vmmlaq_s32(_sum1, _pA1, _pB0);
                    _sum2 = vmmlaq_s32(_sum2, _pA0, _pB1);
                    _sum3 = vmmlaq_s32(_sum3, _pA1, _pB1);
                    _sum4 = vmmlaq_s32(_sum4, _pA2, _pB0);
                    _sum5 = vmmlaq_s32(_sum5, _pA3, _pB0);
                    _sum6 = vmmlaq_s32(_sum6, _pA2, _pB1);
                    _sum7 = vmmlaq_s32(_sum7, _pA3, _pB1);
                    _sum8 = vmmlaq_s32(_sum8, _pA0, _pB2);
                    _sum9 = vmmlaq_s32(_sum9, _pA1, _pB2);
                    _suma = vmmlaq_s32(_suma, _pA0, _pB3);
                    _sumb = vmmlaq_s32(_sumb, _pA1, _pB3);
                    _sumc = vmmlaq_s32(_sumc, _pA2, _pB2);
                    _sumd = vmmlaq_s32(_sumd, _pA3, _pB2);
                    _sume = vmmlaq_s32(_sume, _pA2, _pB3);
                    _sumf = vmmlaq_s32(_sumf, _pA3, _pB3);

                    pA += 64;
                    pB += 64;
                }

                int32x4x2_t _ss0 = vuzpq_s32(_sum0, _sum1);
                int32x4x2_t _ss1 = vuzpq_s32(_sum2, _sum3);
                int32x4x2_t _ss2 = vuzpq_s32(_sum4, _sum5);
                int32x4x2_t _ss3 = vuzpq_s32(_sum6, _sum7);
                int32x4x2_t _ss4 = vuzpq_s32(_sum8, _sum9);
                int32x4x2_t _ss5 = vuzpq_s32(_suma, _sumb);
                int32x4x2_t _ss6 = vuzpq_s32(_sumc, _sumd);
                int32x4x2_t _ss7 = vuzpq_s32(_sume, _sumf);

                if (k == 0)
                {
                    _sum0 = _ss0.val[0];
                    _sum1 = _ss0.val[1];
                    _sum2 = _ss1.val[0];
                    _sum3 = _ss1.val[1];
                    _sum4 = _ss2.val[0];
                    _sum5 = _ss2.val[1];
                    _sum6 = _ss3.val[0];
                    _sum7 = _ss3.val[1];
                    _sum8 = _ss4.val[0];
                    _sum9 = _ss4.val[1];
                    _suma = _ss5.val[0];
                    _sumb = _ss5.val[1];
                    _sumc = _ss6.val[0];
                    _sumd = _ss6.val[1];
                    _sume = _ss7.val[0];
                    _sumf = _ss7.val[1];
                }
                else
                {
                    _sum0 = vld1q_s32(outptr);
                    _sum1 = vld1q_s32(outptr + 4);
                    _sum2 = vld1q_s32(outptr + 8);
                    _sum3 = vld1q_s32(outptr + 12);
                    _sum4 = vld1q_s32(outptr + 16);
                    _sum5 = vld1q_s32(outptr + 20);
                    _sum6 = vld1q_s32(outptr + 24);
                    _sum7 = vld1q_s32(outptr + 28);
                    _sum8 = vld1q_s32(outptr + 32);
                    _sum9 = vld1q_s32(outptr + 36);
                    _suma = vld1q_s32(outptr + 40);
                    _sumb = vld1q_s32(outptr + 44);
                    _sumc = vld1q_s32(outptr + 48);
                    _sumd = vld1q_s32(outptr + 52);
                    _sume = vld1q_s32(outptr + 56);
                    _sumf = vld1q_s32(outptr + 60);

                    _sum0 = vaddq_s32(_sum0, _ss0.val[0]);
                    _sum1 = vaddq_s32(_sum1, _ss0.val[1]);
                    _sum2 = vaddq_s32(_sum2, _ss1.val[0]);
                    _sum3 = vaddq_s32(_sum3, _ss1.val[1]);
                    _sum4 = vaddq_s32(_sum4, _ss2.val[0]);
                    _sum5 = vaddq_s32(_sum5, _ss2.val[1]);
                    _sum6 = vaddq_s32(_sum6, _ss3.val[0]);
                    _sum7 = vaddq_s32(_sum7, _ss3.val[1]);
                    _sum8 = vaddq_s32(_sum8, _ss4.val[0]);
                    _sum9 = vaddq_s32(_sum9, _ss4.val[1]);
                    _suma = vaddq_s32(_suma, _ss5.val[0]);
                    _sumb = vaddq_s32(_sumb, _ss5.val[1]);
                    _sumc = vaddq_s32(_sumc, _ss6.val[0]);
                    _sumd = vaddq_s32(_sumd, _ss6.val[1]);
                    _sume = vaddq_s32(_sume, _ss7.val[0]);
                    _sumf = vaddq_s32(_sumf, _ss7.val[1]);
                }
            }
#elif __ARM_FEATURE_DOTPROD
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA1 = vld1q_s8(pA + 16);
                int8x16_t _pA2 = vld1q_s8(pA + 32);
                int8x16_t _pA3 = vld1q_s8(pA + 48);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB1 = vld1q_s8(pB + 16);
                int8x16_t _pB2 = vld1q_s8(pB + 32);
                int8x16_t _pB3 = vld1q_s8(pB + 48);

                // aaaa bbbb cccc dddd    eeee ffff gggg hhhh

                // 0000 1111 2222 3333    4444 5555 6666 7777
                _sum0 = vdotq_laneq_s32(_sum0, _pA0, _pB0, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA0, _pB0, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA0, _pB0, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA0, _pB0, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _pA1, _pB0, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _pA1, _pB0, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _pA1, _pB0, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _pA1, _pB0, 3);
                _sum8 = vdotq_laneq_s32(_sum8, _pA0, _pB1, 0);
                _sum9 = vdotq_laneq_s32(_sum9, _pA0, _pB1, 1);
                _suma = vdotq_laneq_s32(_suma, _pA0, _pB1, 2);
                _sumb = vdotq_laneq_s32(_sumb, _pA0, _pB1, 3);
                _sumc = vdotq_laneq_s32(_sumc, _pA1, _pB1, 0);
                _sumd = vdotq_laneq_s32(_sumd, _pA1, _pB1, 1);
                _sume = vdotq_laneq_s32(_sume, _pA1, _pB1, 2);
                _sumf = vdotq_laneq_s32(_sumf, _pA1, _pB1, 3);

                _sum0 = vdotq_laneq_s32(_sum0, _pA2, _pB2, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA2, _pB2, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA2, _pB2, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA2, _pB2, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _pA3, _pB2, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _pA3, _pB2, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _pA3, _pB2, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _pA3, _pB2, 3);
                _sum8 = vdotq_laneq_s32(_sum8, _pA2, _pB3, 0);
                _sum9 = vdotq_laneq_s32(_sum9, _pA2, _pB3, 1);
                _suma = vdotq_laneq_s32(_suma, _pA2, _pB3, 2);
                _sumb = vdotq_laneq_s32(_sumb, _pA2, _pB3, 3);
                _sumc = vdotq_laneq_s32(_sumc, _pA3, _pB3, 0);
                _sumd = vdotq_laneq_s32(_sumd, _pA3, _pB3, 1);
                _sume = vdotq_laneq_s32(_sume, _pA3, _pB3, 2);
                _sumf = vdotq_laneq_s32(_sumf, _pA3, _pB3, 3);

                pA += 64;
                pB += 64;
            }
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA1 = vld1q_s8(pA + 16);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB1 = vld1q_s8(pB + 16);

                // aaaa bbbb cccc dddd    eeee ffff gggg hhhh

                // 0000 1111 2222 3333    4444 5555 6666 7777
                _sum0 = vdotq_laneq_s32(_sum0, _pA0, _pB0, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA0, _pB0, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA0, _pB0, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA0, _pB0, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _pA1, _pB0, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _pA1, _pB0, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _pA1, _pB0, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _pA1, _pB0, 3);
                _sum8 = vdotq_laneq_s32(_sum8, _pA0, _pB1, 0);
                _sum9 = vdotq_laneq_s32(_sum9, _pA0, _pB1, 1);
                _suma = vdotq_laneq_s32(_suma, _pA0, _pB1, 2);
                _sumb = vdotq_laneq_s32(_sumb, _pA0, _pB1, 3);
                _sumc = vdotq_laneq_s32(_sumc, _pA1, _pB1, 0);
                _sumd = vdotq_laneq_s32(_sumd, _pA1, _pB1, 1);
                _sume = vdotq_laneq_s32(_sume, _pA1, _pB1, 2);
                _sumf = vdotq_laneq_s32(_sumf, _pA1, _pB1, 3);

#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA2 = vld1q_s8(pA + 16);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB2 = vld1q_s8(pB + 16);

                // aabbccdd eeffgghh
                // ccddaabb gghheeff

                int8x16_t _pA1 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA0)));

                // 00112233 44556677
                // 33221100 77665544

                int8x16_t _pB1 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB0)));

                // aabbccdd eeffgghh
                // ccddaabb gghheeff

                int8x16_t _pA3 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA2)));

                // 00112233 44556677
                // 33221100 77665544

                int8x16_t _pB3 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB2)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB0));
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), vget_high_s8(_pB0));
                int16x8_t _s2 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB0));
                int16x8_t _s3 = vmull_s8(vget_low_s8(_pA0), vget_high_s8(_pB0));
                int16x8_t _s4 = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB0));
                int16x8_t _s5 = vmull_s8(vget_high_s8(_pA1), vget_high_s8(_pB0));
                int16x8_t _s6 = vmull_s8(vget_high_s8(_pA1), vget_low_s8(_pB0));
                int16x8_t _s7 = vmull_s8(vget_low_s8(_pA1), vget_high_s8(_pB0));
                int16x8_t _s8 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB1));
                int16x8_t _s9 = vmull_s8(vget_high_s8(_pA0), vget_high_s8(_pB1));
                int16x8_t _sa = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB1));
                int16x8_t _sb = vmull_s8(vget_low_s8(_pA0), vget_high_s8(_pB1));
                int16x8_t _sc = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB1));
                int16x8_t _sd = vmull_s8(vget_high_s8(_pA1), vget_high_s8(_pB1));
                int16x8_t _se = vmull_s8(vget_high_s8(_pA1), vget_low_s8(_pB1));
                int16x8_t _sf = vmull_s8(vget_low_s8(_pA1), vget_high_s8(_pB1));

                _s0 = vmlal_s8(_s0, vget_low_s8(_pA2), vget_low_s8(_pB2));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA2), vget_high_s8(_pB2));
                _s2 = vmlal_s8(_s2, vget_high_s8(_pA2), vget_low_s8(_pB2));
                _s3 = vmlal_s8(_s3, vget_low_s8(_pA2), vget_high_s8(_pB2));
                _s4 = vmlal_s8(_s4, vget_low_s8(_pA3), vget_low_s8(_pB2));
                _s5 = vmlal_s8(_s5, vget_high_s8(_pA3), vget_high_s8(_pB2));
                _s6 = vmlal_s8(_s6, vget_high_s8(_pA3), vget_low_s8(_pB2));
                _s7 = vmlal_s8(_s7, vget_low_s8(_pA3), vget_high_s8(_pB2));
                _s8 = vmlal_s8(_s8, vget_low_s8(_pA2), vget_low_s8(_pB3));
                _s9 = vmlal_s8(_s9, vget_high_s8(_pA2), vget_high_s8(_pB3));
                _sa = vmlal_s8(_sa, vget_high_s8(_pA2), vget_low_s8(_pB3));
                _sb = vmlal_s8(_sb, vget_low_s8(_pA2), vget_high_s8(_pB3));
                _sc = vmlal_s8(_sc, vget_low_s8(_pA3), vget_low_s8(_pB3));
                _sd = vmlal_s8(_sd, vget_high_s8(_pA3), vget_high_s8(_pB3));
                _se = vmlal_s8(_se, vget_high_s8(_pA3), vget_low_s8(_pB3));
                _sf = vmlal_s8(_sf, vget_low_s8(_pA3), vget_high_s8(_pB3));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
                _sum8 = vpadalq_s16(_sum8, _s8);
                _sum9 = vpadalq_s16(_sum9, _s9);
                _suma = vpadalq_s16(_suma, _sa);
                _sumb = vpadalq_s16(_sumb, _sb);
                _sumc = vpadalq_s16(_sumc, _sc);
                _sumd = vpadalq_s16(_sumd, _sd);
                _sume = vpadalq_s16(_sume, _se);
                _sumf = vpadalq_s16(_sumf, _sf);
#endif // __ARM_FEATURE_DOTPROD

                pA += 32;
                pB += 32;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x16_t _pB = vld1q_s8(pB);

                // aabbccdd eeffgghh

                // 00112233 44556677

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB)), 0)));
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB)), 1)));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB)), 2)));
                int16x8_t _s3 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB)), 3)));
                int16x8_t _s4 = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB)), 0)));
                int16x8_t _s5 = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB)), 1)));
                int16x8_t _s6 = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB)), 2)));
                int16x8_t _s7 = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB)), 3)));
                int16x8_t _s8 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB)), 0)));
                int16x8_t _s9 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB)), 1)));
                int16x8_t _sa = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB)), 2)));
                int16x8_t _sb = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB)), 3)));
                int16x8_t _sc = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB)), 0)));
                int16x8_t _sd = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB)), 1)));
                int16x8_t _se = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB)), 2)));
                int16x8_t _sf = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB)), 3)));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
                _sum8 = vpadalq_s16(_sum8, _s8);
                _sum9 = vpadalq_s16(_sum9, _s9);
                _suma = vpadalq_s16(_suma, _sa);
                _sumb = vpadalq_s16(_sumb, _sb);
                _sumc = vpadalq_s16(_sumc, _sc);
                _sumd = vpadalq_s16(_sumd, _sd);
                _sume = vpadalq_s16(_sume, _se);
                _sumf = vpadalq_s16(_sumf, _sf);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pB0 = vld1q_s8(pB);

                // aabbccdd eeffgghh

                // ccddaabb gghheeff

                int8x16_t _pA1 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA0)));

                // 00112233 44556677

                // 33221100 77665544

                int8x16_t _pB1 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB0)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB0));
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), vget_high_s8(_pB0));
                int16x8_t _s2 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB0));
                int16x8_t _s3 = vmull_s8(vget_low_s8(_pA0), vget_high_s8(_pB0));
                int16x8_t _s4 = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB0));
                int16x8_t _s5 = vmull_s8(vget_high_s8(_pA1), vget_high_s8(_pB0));
                int16x8_t _s6 = vmull_s8(vget_high_s8(_pA1), vget_low_s8(_pB0));
                int16x8_t _s7 = vmull_s8(vget_low_s8(_pA1), vget_high_s8(_pB0));
                int16x8_t _s8 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB1));
                int16x8_t _s9 = vmull_s8(vget_high_s8(_pA0), vget_high_s8(_pB1));
                int16x8_t _sa = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB1));
                int16x8_t _sb = vmull_s8(vget_low_s8(_pA0), vget_high_s8(_pB1));
                int16x8_t _sc = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB1));
                int16x8_t _sd = vmull_s8(vget_high_s8(_pA1), vget_high_s8(_pB1));
                int16x8_t _se = vmull_s8(vget_high_s8(_pA1), vget_low_s8(_pB1));
                int16x8_t _sf = vmull_s8(vget_low_s8(_pA1), vget_high_s8(_pB1));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
                _sum8 = vpadalq_s16(_sum8, _s8);
                _sum9 = vpadalq_s16(_sum9, _s9);
                _suma = vpadalq_s16(_suma, _sa);
                _sumb = vpadalq_s16(_sumb, _sb);
                _sumc = vpadalq_s16(_sumc, _sc);
                _sumd = vpadalq_s16(_sumd, _sd);
                _sume = vpadalq_s16(_sume, _se);
                _sumf = vpadalq_s16(_sumf, _sf);
#endif // __ARM_FEATURE_DOTPROD

                pA += 16;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vld1_s8(pA);
                // int8x8_t _pB0 = vld1_s8(pB);

                // abcd efgh
                // 0123 4567

                int16x8_t _s01 = vmull_s8(_pA, vdup_n_s8(pB[0]));
                int16x8_t _s23 = vmull_s8(_pA, vdup_n_s8(pB[1]));
                int16x8_t _s45 = vmull_s8(_pA, vdup_n_s8(pB[2]));
                int16x8_t _s67 = vmull_s8(_pA, vdup_n_s8(pB[3]));
                int16x8_t _s89 = vmull_s8(_pA, vdup_n_s8(pB[4]));
                int16x8_t _sab = vmull_s8(_pA, vdup_n_s8(pB[5]));
                int16x8_t _scd = vmull_s8(_pA, vdup_n_s8(pB[6]));
                int16x8_t _sef = vmull_s8(_pA, vdup_n_s8(pB[7]));

                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_low_s16(_s23));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s45));
                _sum3 = vaddw_s16(_sum3, vget_low_s16(_s67));
                _sum4 = vaddw_s16(_sum4, vget_high_s16(_s01));
                _sum5 = vaddw_s16(_sum5, vget_high_s16(_s23));
                _sum6 = vaddw_s16(_sum6, vget_high_s16(_s45));
                _sum7 = vaddw_s16(_sum7, vget_high_s16(_s67));
                _sum8 = vaddw_s16(_sum8, vget_low_s16(_s89));
                _sum9 = vaddw_s16(_sum9, vget_low_s16(_sab));
                _suma = vaddw_s16(_suma, vget_low_s16(_scd));
                _sumb = vaddw_s16(_sumb, vget_low_s16(_sef));
                _sumc = vaddw_s16(_sumc, vget_high_s16(_s89));
                _sumd = vaddw_s16(_sumd, vget_high_s16(_sab));
                _sume = vaddw_s16(_sume, vget_high_s16(_scd));
                _sumf = vaddw_s16(_sumf, vget_high_s16(_sef));
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vld1_s8(pA);
                int8x8_t _pB0 = vld1_s8(pB);

                // abcd efgh
                // efgh abcd
                // cdab ghef
                // ghef cdab

                // 0123 4567
                // 3210 7654

                // abcdefgh  ->  ghefcdab  ->  cdabghef

                int8x8_t _pA1 = vext_s8(_pA0, _pA0, 4);
                int8x8_t _pA2 = vreinterpret_s8_s16(vrev32_s16(vreinterpret_s16_s8(_pA0)));
                int8x8_t _pA3 = vreinterpret_s8_s16(vrev64_s16(vreinterpret_s16_s8(_pA0)));

                // 01234567  ->  32107654

                int8x8_t _pB1 = vrev32_s8(_pB0);

                int16x8_t _s01 = vmull_s8(_pA0, _pB0);
                int16x8_t _s23 = vmull_s8(_pA1, _pB0);
                int16x8_t _s45 = vmull_s8(_pA2, _pB0);
                int16x8_t _s67 = vmull_s8(_pA3, _pB0);
                int16x8_t _s89 = vmull_s8(_pA0, _pB1);
                int16x8_t _sab = vmull_s8(_pA1, _pB1);
                int16x8_t _scd = vmull_s8(_pA2, _pB1);
                int16x8_t _sef = vmull_s8(_pA3, _pB1);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s23));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s23));
                _sum4 = vaddw_s16(_sum4, vget_low_s16(_s45));
                _sum5 = vaddw_s16(_sum5, vget_high_s16(_s45));
                _sum6 = vaddw_s16(_sum6, vget_low_s16(_s67));
                _sum7 = vaddw_s16(_sum7, vget_high_s16(_s67));
                _sum8 = vaddw_s16(_sum8, vget_low_s16(_s89));
                _sum9 = vaddw_s16(_sum9, vget_high_s16(_s89));
                _suma = vaddw_s16(_suma, vget_low_s16(_sab));
                _sumb = vaddw_s16(_sumb, vget_high_s16(_sab));
                _sumc = vaddw_s16(_sumc, vget_low_s16(_scd));
                _sumd = vaddw_s16(_sumd, vget_high_s16(_scd));
                _sume = vaddw_s16(_sume, vget_low_s16(_sef));
                _sumf = vaddw_s16(_sumf, vget_high_s16(_sef));
#endif // __ARM_FEATURE_DOTPROD

                pA += 8;
                pB += 8;
            }

            //             if (k_end)
            //             {
            // #if __ARM_FEATURE_DOTPROD
            //                 // from
            //                 //      a0 b0 c0 d0
            //                 //      a1 b1 c1 d1
            //                 //      a2 b2 c2 d2
            //                 //      a3 b3 c3 d3
            //                 //      e0 f0 g0 h0
            //                 //      e1 f1 g1 h1
            //                 //      e2 f2 g2 h2
            //                 //      e3 f3 g3 h3
            //                 //      a4 b4 c4 d4
            //                 //      a5 b5 c5 d5
            //                 //      a6 b6 c6 d6
            //                 //      a7 b7 c7 d7
            //                 //      e4 f4 g4 h4
            //                 //      e5 f5 g5 h5
            //                 //      e6 f6 g6 h6
            //                 //      e7 f7 g7 h7
            //                 if (out_elempack == 8)
            //                 {
            //                     vst1q_s32(outptr0, _sum0);
            //                     vst1q_s32(outptr0 + 4, _sum4);
            //                     vst1q_s32(outptr0 + 8, _sum1);
            //                     vst1q_s32(outptr0 + 12, _sum5);
            //                     vst1q_s32(outptr0 + 16, _sum2);
            //                     vst1q_s32(outptr0 + 20, _sum6);
            //                     vst1q_s32(outptr0 + 24, _sum3);
            //                     vst1q_s32(outptr0 + 28, _sum7);
            //                     vst1q_s32(outptr0 + 32, _sum8);
            //                     vst1q_s32(outptr0 + 36, _sumc);
            //                     vst1q_s32(outptr0 + 40, _sum9);
            //                     vst1q_s32(outptr0 + 44, _sumd);
            //                     vst1q_s32(outptr0 + 48, _suma);
            //                     vst1q_s32(outptr0 + 52, _sume);
            //                     vst1q_s32(outptr0 + 56, _sumb);
            //                     vst1q_s32(outptr0 + 60, _sumf);
            //                     outptr0 += 64;
            //                 }
            //                 if (out_elempack == 4)
            //                 {
            //                     vst1q_s32(outptr0, _sum0);
            //                     vst1q_s32(outptr0 + 4, _sum1);
            //                     vst1q_s32(outptr0 + 8, _sum2);
            //                     vst1q_s32(outptr0 + 12, _sum3);
            //                     vst1q_s32(outptr0 + 16, _sum8);
            //                     vst1q_s32(outptr0 + 20, _sum9);
            //                     vst1q_s32(outptr0 + 24, _suma);
            //                     vst1q_s32(outptr0 + 28, _sumb);
            //                     vst1q_s32(outptr0 + out_hstep * 4, _sum4);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum5);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 8, _sum6);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 12, _sum7);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 16, _sumc);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 20, _sumd);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 24, _sume);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 28, _sumf);
            //                     outptr0 += 32;
            //                 }
            //                 if (out_elempack == 1)
            //                 {
            //                     // to
            //                     //      a0 a1 a2 a3
            //                     //      a4 a5 a6 a7
            //                     //      b0 b1 b2 b3
            //                     //      b4 b5 b6 b7
            //                     //      c0 c1 c2 c3
            //                     //      c4 c5 c6 c7
            //                     //      d0 d1 d2 d3
            //                     //      d4 d5 d6 d7
            //                     //      e0 e1 e2 e3
            //                     //      e4 e5 e6 e7
            //                     //      f0 f1 f2 f3
            //                     //      f4 f5 f6 f7
            //                     //      g0 g1 g2 g3
            //                     //      g4 g5 g6 g7
            //                     //      h0 h1 h2 h3
            //                     //      h4 h5 h6 h7
            //                     {
            //                         int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
            //                         int32x4x2_t _t1 = vzipq_s32(_sum2, _sum3);
            //                         int32x4x2_t _t2 = vzipq_s32(_sum8, _sum9);
            //                         int32x4x2_t _t3 = vzipq_s32(_suma, _sumb);
            //                         int32x4x2_t _t4 = vzipq_s32(_sum4, _sum5);
            //                         int32x4x2_t _t5 = vzipq_s32(_sum6, _sum7);
            //                         int32x4x2_t _t6 = vzipq_s32(_sumc, _sumd);
            //                         int32x4x2_t _t7 = vzipq_s32(_sume, _sumf);
            //                         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
            //                         _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
            //                         _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
            //                         _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
            //                         _sum4 = vcombine_s32(vget_low_s32(_t0.val[1]), vget_low_s32(_t1.val[1]));
            //                         _sum5 = vcombine_s32(vget_low_s32(_t2.val[1]), vget_low_s32(_t3.val[1]));
            //                         _sum6 = vcombine_s32(vget_high_s32(_t0.val[1]), vget_high_s32(_t1.val[1]));
            //                         _sum7 = vcombine_s32(vget_high_s32(_t2.val[1]), vget_high_s32(_t3.val[1]));
            //                         _sum8 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
            //                         _sum9 = vcombine_s32(vget_low_s32(_t6.val[0]), vget_low_s32(_t7.val[0]));
            //                         _suma = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
            //                         _sumb = vcombine_s32(vget_high_s32(_t6.val[0]), vget_high_s32(_t7.val[0]));
            //                         _sumc = vcombine_s32(vget_low_s32(_t4.val[1]), vget_low_s32(_t5.val[1]));
            //                         _sumd = vcombine_s32(vget_low_s32(_t6.val[1]), vget_low_s32(_t7.val[1]));
            //                         _sume = vcombine_s32(vget_high_s32(_t4.val[1]), vget_high_s32(_t5.val[1]));
            //                         _sumf = vcombine_s32(vget_high_s32(_t6.val[1]), vget_high_s32(_t7.val[1]));
            //                     }
            //
            //                     vst1q_s32(outptr0, _sum0);
            //                     vst1q_s32(outptr0 + 4, _sum1);
            //                     vst1q_s32(outptr0 + out_hstep, _sum2);
            //                     vst1q_s32(outptr0 + out_hstep + 4, _sum3);
            //                     vst1q_s32(outptr0 + out_hstep * 2, _sum4);
            //                     vst1q_s32(outptr0 + out_hstep * 2 + 4, _sum5);
            //                     vst1q_s32(outptr0 + out_hstep * 3, _sum6);
            //                     vst1q_s32(outptr0 + out_hstep * 3 + 4, _sum7);
            //                     vst1q_s32(outptr0 + out_hstep * 4, _sum8);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum9);
            //                     vst1q_s32(outptr0 + out_hstep * 5, _suma);
            //                     vst1q_s32(outptr0 + out_hstep * 5 + 4, _sumb);
            //                     vst1q_s32(outptr0 + out_hstep * 6, _sumc);
            //                     vst1q_s32(outptr0 + out_hstep * 6 + 4, _sumd);
            //                     vst1q_s32(outptr0 + out_hstep * 7, _sume);
            //                     vst1q_s32(outptr0 + out_hstep * 7 + 4, _sumf);
            //                     outptr0 += 8;
            //                 }
            // #else  // __ARM_FEATURE_DOTPROD
            //
            //                 // from
            //                 //      a0 b1 c2 d3
            //                 //      e4 f5 g6 h7
            //                 //      e0 f1 g2 h3
            //                 //      a4 b5 c6 d7
            //                 //      c0 d1 a2 b3
            //                 //      g4 h5 e6 f7
            //                 //      g0 h1 e2 f3
            //                 //      c4 d5 a6 b7
            //                 //      a3 b2 c1 d0
            //                 //      e7 f6 g5 h4
            //                 //      e3 f2 g1 h0
            //                 //      a7 b6 c5 d4
            //                 //      c3 d2 a1 b0
            //                 //      g7 h6 e5 f4
            //                 //      g3 h2 e1 f0
            //                 //      c7 d6 a5 b4
            //                 if (out_elempack == 8)
            //                 {
            //                     // to
            //                     //      a0 b0 c0 d0
            //                     //      e0 f0 g0 h0
            //                     //      a1 b1 c1 d1
            //                     //      e1 f1 g1 h1
            //                     //      a2 b2 c2 d2
            //                     //      e2 f2 g2 h2
            //                     //      a3 b3 c3 d3
            //                     //      e3 f3 g3 h3
            //                     //      a4 b4 c4 d4
            //                     //      e4 f4 g4 h4
            //                     //      a5 b5 c5 d5
            //                     //      e5 f5 g5 h5
            //                     //      a6 b6 c6 d6
            //                     //      e6 f6 g6 h6
            //                     //      a7 b7 c7 d7
            //                     //      e7 f7 g7 h7
            //                     {
            //                         _sum8 = vrev64q_s32(_sum8);
            //                         _sum9 = vrev64q_s32(_sum9);
            //                         _suma = vrev64q_s32(_suma);
            //                         _sumb = vrev64q_s32(_sumb);
            //                         _sumc = vrev64q_s32(_sumc);
            //                         _sumd = vrev64q_s32(_sumd);
            //                         _sume = vrev64q_s32(_sume);
            //                         _sumf = vrev64q_s32(_sumf);
            //                         _sum8 = vextq_s32(_sum8, _sum8, 2);
            //                         _sum9 = vextq_s32(_sum9, _sum9, 2);
            //                         _suma = vextq_s32(_suma, _suma, 2);
            //                         _sumb = vextq_s32(_sumb, _sumb, 2);
            //                         _sumc = vextq_s32(_sumc, _sumc, 2);
            //                         _sumd = vextq_s32(_sumd, _sumd, 2);
            //                         _sume = vextq_s32(_sume, _sume, 2);
            //                         _sumf = vextq_s32(_sumf, _sumf, 2);
            //                         int32x4x2_t _t0 = vzipq_s32(_sum0, _sumc);
            //                         int32x4x2_t _t1 = vzipq_s32(_sum4, _sum8);
            //                         int32x4x2_t _t2 = vzipq_s32(_sum2, _sume);
            //                         int32x4x2_t _t3 = vzipq_s32(_sum6, _suma);
            //                         int32x4x2_t _t4 = vzipq_s32(_sum3, _sumf);
            //                         int32x4x2_t _t5 = vzipq_s32(_sum7, _sumb);
            //                         int32x4x2_t _t6 = vzipq_s32(_sum1, _sumd);
            //                         int32x4x2_t _t7 = vzipq_s32(_sum5, _sum9);
            //                         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
            //                         _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
            //                         _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
            //                         _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
            //                         _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
            //                         _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
            //                         _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
            //                         _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
            //                         _sum8 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
            //                         _sum9 = vcombine_s32(vget_low_s32(_t6.val[0]), vget_low_s32(_t7.val[0]));
            //                         _suma = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
            //                         _sumb = vcombine_s32(vget_high_s32(_t6.val[0]), vget_high_s32(_t7.val[0]));
            //                         _sumc = vcombine_s32(vget_low_s32(_t5.val[1]), vget_low_s32(_t4.val[1]));
            //                         _sumd = vcombine_s32(vget_low_s32(_t7.val[1]), vget_low_s32(_t6.val[1]));
            //                         _sume = vcombine_s32(vget_high_s32(_t5.val[1]), vget_high_s32(_t4.val[1]));
            //                         _sumf = vcombine_s32(vget_high_s32(_t7.val[1]), vget_high_s32(_t6.val[1]));
            //                         _sum2 = vrev64q_s32(_sum2);
            //                         _sum3 = vrev64q_s32(_sum3);
            //                         _sum6 = vrev64q_s32(_sum6);
            //                         _sum7 = vrev64q_s32(_sum7);
            //                         _suma = vrev64q_s32(_suma);
            //                         _sumb = vrev64q_s32(_sumb);
            //                         _sume = vrev64q_s32(_sume);
            //                         _sumf = vrev64q_s32(_sumf);
            //                     }
            //
            //                     vst1q_s32(outptr0, _sum0);
            //                     vst1q_s32(outptr0 + 4, _sum1);
            //                     vst1q_s32(outptr0 + 8, _sum2);
            //                     vst1q_s32(outptr0 + 12, _sum3);
            //                     vst1q_s32(outptr0 + 16, _sum4);
            //                     vst1q_s32(outptr0 + 20, _sum5);
            //                     vst1q_s32(outptr0 + 24, _sum6);
            //                     vst1q_s32(outptr0 + 28, _sum7);
            //                     vst1q_s32(outptr0 + 32, _sum8);
            //                     vst1q_s32(outptr0 + 36, _sum9);
            //                     vst1q_s32(outptr0 + 40, _suma);
            //                     vst1q_s32(outptr0 + 44, _sumb);
            //                     vst1q_s32(outptr0 + 48, _sumc);
            //                     vst1q_s32(outptr0 + 52, _sumd);
            //                     vst1q_s32(outptr0 + 56, _sume);
            //                     vst1q_s32(outptr0 + 60, _sumf);
            //                     outptr0 += 64;
            //                 }
            //                 if (out_elempack == 4)
            //                 {
            //                     // to
            //                     //      a0 b0 c0 d0
            //                     //      a1 b1 c1 d1
            //                     //      a2 b2 c2 d2
            //                     //      a3 b3 c3 d3
            //                     //      a4 b4 c4 d4
            //                     //      a5 b5 c5 d5
            //                     //      a6 b6 c6 d6
            //                     //      a7 b7 c7 d7
            //                     //      e0 f0 g0 h0
            //                     //      e1 f1 g1 h1
            //                     //      e2 f2 g2 h2
            //                     //      e3 f3 g3 h3
            //                     //      e4 f4 g4 h4
            //                     //      e5 f5 g5 h5
            //                     //      e6 f6 g6 h6
            //                     //      e7 f7 g7 h7
            //                     {
            //                         _sum8 = vrev64q_s32(_sum8);
            //                         _sum9 = vrev64q_s32(_sum9);
            //                         _suma = vrev64q_s32(_suma);
            //                         _sumb = vrev64q_s32(_sumb);
            //                         _sumc = vrev64q_s32(_sumc);
            //                         _sumd = vrev64q_s32(_sumd);
            //                         _sume = vrev64q_s32(_sume);
            //                         _sumf = vrev64q_s32(_sumf);
            //                         _sum8 = vextq_s32(_sum8, _sum8, 2);
            //                         _sum9 = vextq_s32(_sum9, _sum9, 2);
            //                         _suma = vextq_s32(_suma, _suma, 2);
            //                         _sumb = vextq_s32(_sumb, _sumb, 2);
            //                         _sumc = vextq_s32(_sumc, _sumc, 2);
            //                         _sumd = vextq_s32(_sumd, _sumd, 2);
            //                         _sume = vextq_s32(_sume, _sume, 2);
            //                         _sumf = vextq_s32(_sumf, _sumf, 2);
            //                         int32x4x2_t _t0 = vzipq_s32(_sum0, _sumc);
            //                         int32x4x2_t _t1 = vzipq_s32(_sum4, _sum8);
            //                         int32x4x2_t _t2 = vzipq_s32(_sum2, _sume);
            //                         int32x4x2_t _t3 = vzipq_s32(_sum6, _suma);
            //                         int32x4x2_t _t4 = vzipq_s32(_sum3, _sumf);
            //                         int32x4x2_t _t5 = vzipq_s32(_sum7, _sumb);
            //                         int32x4x2_t _t6 = vzipq_s32(_sum1, _sumd);
            //                         int32x4x2_t _t7 = vzipq_s32(_sum5, _sum9);
            //                         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
            //                         _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
            //                         _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
            //                         _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
            //                         _sum4 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
            //                         _sum5 = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
            //                         _sum6 = vcombine_s32(vget_low_s32(_t5.val[1]), vget_low_s32(_t4.val[1]));
            //                         _sum7 = vcombine_s32(vget_high_s32(_t5.val[1]), vget_high_s32(_t4.val[1]));
            //                         _sum8 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
            //                         _sum9 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
            //                         _suma = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
            //                         _sumb = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
            //                         _sumc = vcombine_s32(vget_low_s32(_t6.val[0]), vget_low_s32(_t7.val[0]));
            //                         _sumd = vcombine_s32(vget_high_s32(_t6.val[0]), vget_high_s32(_t7.val[0]));
            //                         _sume = vcombine_s32(vget_low_s32(_t7.val[1]), vget_low_s32(_t6.val[1]));
            //                         _sumf = vcombine_s32(vget_high_s32(_t7.val[1]), vget_high_s32(_t6.val[1]));
            //                         _sum1 = vrev64q_s32(_sum1);
            //                         _sum3 = vrev64q_s32(_sum3);
            //                         _sum5 = vrev64q_s32(_sum5);
            //                         _sum7 = vrev64q_s32(_sum7);
            //                         _sum9 = vrev64q_s32(_sum9);
            //                         _sumb = vrev64q_s32(_sumb);
            //                         _sumd = vrev64q_s32(_sumd);
            //                         _sumf = vrev64q_s32(_sumf);
            //                     }
            //
            //                     vst1q_s32(outptr0, _sum0);
            //                     vst1q_s32(outptr0 + 4, _sum1);
            //                     vst1q_s32(outptr0 + 8, _sum2);
            //                     vst1q_s32(outptr0 + 12, _sum3);
            //                     vst1q_s32(outptr0 + 16, _sum4);
            //                     vst1q_s32(outptr0 + 20, _sum5);
            //                     vst1q_s32(outptr0 + 24, _sum6);
            //                     vst1q_s32(outptr0 + 28, _sum7);
            //                     vst1q_s32(outptr0 + out_hstep * 4, _sum8);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum9);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 8, _suma);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 12, _sumb);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 16, _sumc);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 20, _sumd);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 24, _sume);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 28, _sumf);
            //                     outptr0 += 32;
            //                 }
            //                 if (out_elempack == 1)
            //                 {
            //                     // to
            //                     //      a0 a1 a2 a3
            //                     //      a4 a5 a6 a7
            //                     //      b0 b1 b2 b3
            //                     //      b4 b5 b6 b7
            //                     //      c0 c1 c2 c3
            //                     //      c4 c5 c6 c7
            //                     //      d0 d1 d2 d3
            //                     //      d4 d5 d6 d7
            //                     //      e0 e1 e2 e3
            //                     //      e4 e5 e6 e7
            //                     //      f0 f1 f2 f3
            //                     //      f4 f5 f6 f7
            //                     //      g0 g1 g2 g3
            //                     //      g4 g5 g6 g7
            //                     //      h0 h1 h2 h3
            //                     //      h4 h5 h6 h7
            //                     {
            //                         _sum4 = vextq_s32(_sum4, _sum4, 2);
            //                         _sum5 = vextq_s32(_sum5, _sum5, 2);
            //                         _sum6 = vextq_s32(_sum6, _sum6, 2);
            //                         _sum7 = vextq_s32(_sum7, _sum7, 2);
            //                         _sumc = vextq_s32(_sumc, _sumc, 2);
            //                         _sumd = vextq_s32(_sumd, _sumd, 2);
            //                         _sume = vextq_s32(_sume, _sume, 2);
            //                         _sumf = vextq_s32(_sumf, _sumf, 2);
            //                         int32x4x2_t _t0 = vzipq_s32(_sum0, _sumc);
            //                         int32x4x2_t _t1 = vzipq_s32(_sum4, _sum8);
            //                         int32x4x2_t _t2 = vzipq_s32(_sum3, _sumf);
            //                         int32x4x2_t _t3 = vzipq_s32(_sum7, _sumb);
            //                         int32x4x2_t _t4 = vzipq_s32(_sum2, _sume);
            //                         int32x4x2_t _t5 = vzipq_s32(_sum6, _suma);
            //                         int32x4x2_t _t6 = vzipq_s32(_sum1, _sumd);
            //                         int32x4x2_t _t7 = vzipq_s32(_sum5, _sum9);
            //                         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
            //                         _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
            //                         _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
            //                         _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
            //                         _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
            //                         _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
            //                         _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
            //                         _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
            //                         _sum8 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
            //                         _sum9 = vcombine_s32(vget_low_s32(_t6.val[0]), vget_low_s32(_t7.val[0]));
            //                         _suma = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
            //                         _sumb = vcombine_s32(vget_high_s32(_t6.val[0]), vget_high_s32(_t7.val[0]));
            //                         _sumc = vcombine_s32(vget_low_s32(_t5.val[1]), vget_low_s32(_t4.val[1]));
            //                         _sumd = vcombine_s32(vget_low_s32(_t7.val[1]), vget_low_s32(_t6.val[1]));
            //                         _sume = vcombine_s32(vget_high_s32(_t5.val[1]), vget_high_s32(_t4.val[1]));
            //                         _sumf = vcombine_s32(vget_high_s32(_t7.val[1]), vget_high_s32(_t6.val[1]));
            //                         _sum2 = vrev64q_s32(_sum2);
            //                         _sum3 = vrev64q_s32(_sum3);
            //                         _sum6 = vrev64q_s32(_sum6);
            //                         _sum7 = vrev64q_s32(_sum7);
            //                         _suma = vrev64q_s32(_suma);
            //                         _sumb = vrev64q_s32(_sumb);
            //                         _sume = vrev64q_s32(_sume);
            //                         _sumf = vrev64q_s32(_sumf);
            //                     }
            //
            //                     vst1q_s32(outptr0, _sum0);
            //                     vst1q_s32(outptr0 + 4, _sum1);
            //                     vst1q_s32(outptr0 + out_hstep, _sum2);
            //                     vst1q_s32(outptr0 + out_hstep + 4, _sum3);
            //                     vst1q_s32(outptr0 + out_hstep * 2, _sum4);
            //                     vst1q_s32(outptr0 + out_hstep * 2 + 4, _sum5);
            //                     vst1q_s32(outptr0 + out_hstep * 3, _sum6);
            //                     vst1q_s32(outptr0 + out_hstep * 3 + 4, _sum7);
            //                     vst1q_s32(outptr0 + out_hstep * 4, _sum8);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum9);
            //                     vst1q_s32(outptr0 + out_hstep * 5, _suma);
            //                     vst1q_s32(outptr0 + out_hstep * 5 + 4, _sumb);
            //                     vst1q_s32(outptr0 + out_hstep * 6, _sumc);
            //                     vst1q_s32(outptr0 + out_hstep * 6 + 4, _sumd);
            //                     vst1q_s32(outptr0 + out_hstep * 7, _sume);
            //                     vst1q_s32(outptr0 + out_hstep * 7 + 4, _sumf);
            //                     outptr0 += 8;
            //                 }
            // #endif // __ARM_FEATURE_DOTPROD
            //             }
            //             else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
                vst1q_s32(outptr + 16, _sum4);
                vst1q_s32(outptr + 20, _sum5);
                vst1q_s32(outptr + 24, _sum6);
                vst1q_s32(outptr + 28, _sum7);
                vst1q_s32(outptr + 32, _sum8);
                vst1q_s32(outptr + 36, _sum9);
                vst1q_s32(outptr + 40, _suma);
                vst1q_s32(outptr + 44, _sumb);
                vst1q_s32(outptr + 48, _sumc);
                vst1q_s32(outptr + 52, _sumd);
                vst1q_s32(outptr + 56, _sume);
                vst1q_s32(outptr + 60, _sumf);
            }

            outptr += 64;
#endif // NCNN_GNU_INLINE_ASM
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            asm volatile(
                "cmp    %w7, #0                     \n"
                "beq    0f                          \n"

                "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0] \n"
                "sub    %0, %0, #64                 \n"
                "b      1f                          \n"

                "0:                                 \n"
                "eor    v16.16b, v16.16b, v16.16b   \n"
                "eor    v17.16b, v17.16b, v17.16b   \n"
                "eor    v18.16b, v18.16b, v18.16b   \n"
                "eor    v19.16b, v19.16b, v19.16b   \n"
                "eor    v20.16b, v20.16b, v20.16b   \n"
                "eor    v21.16b, v21.16b, v21.16b   \n"
                "eor    v22.16b, v22.16b, v22.16b   \n"
                "eor    v23.16b, v23.16b, v23.16b   \n"

                "1:                                 \n"
#if __ARM_FEATURE_DOTPROD
                "lsr    w4, %w6, #3                 \n" // w4 = max_kk >> 3
                "cmp    w4, #0                      \n"
                "beq    101f                        \n"

#if __ARM_FEATURE_MATMUL_INT8
                "eor    v24.16b, v24.16b, v24.16b   \n"
                "eor    v25.16b, v25.16b, v25.16b   \n"
                "eor    v26.16b, v26.16b, v26.16b   \n"
                "eor    v27.16b, v27.16b, v27.16b   \n"
                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v29.16b, v29.16b, v29.16b   \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"
                "eor    v31.16b, v31.16b, v31.16b   \n"
#endif // __ARM_FEATURE_MATMUL_INT8

                "2:                                 \n"
                "ld1    {v0.16b, v1.16b, v2.16b, v3.16b}, [%1], #64 \n"
                "ld1    {v4.16b, v5.16b}, [%2], #32 \n"

#if __ARM_FEATURE_MATMUL_INT8
                "smmla  v24.4s, v0.16b, v4.16b      \n"
                "smmla  v25.4s, v1.16b, v4.16b      \n"
                "smmla  v26.4s, v0.16b, v5.16b      \n"
                "smmla  v27.4s, v1.16b, v5.16b      \n"
                "subs   w4, w4, #1                  \n"
                "smmla  v28.4s, v2.16b, v4.16b      \n"
                "smmla  v29.4s, v3.16b, v4.16b      \n"
                "smmla  v30.4s, v2.16b, v5.16b      \n"
                "smmla  v31.4s, v3.16b, v5.16b      \n"
#else  // __ARM_FEATURE_MATMUL_INT8
                "sdot   v16.4s, v0.16b, v4.4b[0]    \n"
                "sdot   v17.4s, v0.16b, v4.4b[1]    \n"
                "sdot   v18.4s, v0.16b, v4.4b[2]    \n"
                "sdot   v19.4s, v0.16b, v4.4b[3]    \n"
                "sdot   v20.4s, v1.16b, v4.4b[0]    \n"
                "sdot   v21.4s, v1.16b, v4.4b[1]    \n"
                "sdot   v22.4s, v1.16b, v4.4b[2]    \n"
                "sdot   v23.4s, v1.16b, v4.4b[3]    \n"
                "subs   w4, w4, #1                  \n"
                "sdot   v16.4s, v2.16b, v5.4b[0]    \n"
                "sdot   v17.4s, v2.16b, v5.4b[1]    \n"
                "sdot   v18.4s, v2.16b, v5.4b[2]    \n"
                "sdot   v19.4s, v2.16b, v5.4b[3]    \n"
                "sdot   v20.4s, v3.16b, v5.4b[0]    \n"
                "sdot   v21.4s, v3.16b, v5.4b[1]    \n"
                "sdot   v22.4s, v3.16b, v5.4b[2]    \n"
                "sdot   v23.4s, v3.16b, v5.4b[3]    \n"
#endif // __ARM_FEATURE_MATMUL_INT8
                "bne    2b                          \n"

#if __ARM_FEATURE_MATMUL_INT8
                "uzp1   v0.4s, v24.4s, v25.4s       \n"
                "uzp2   v1.4s, v24.4s, v25.4s       \n"
                "uzp1   v2.4s, v26.4s, v27.4s       \n"
                "uzp2   v3.4s, v26.4s, v27.4s       \n"
                "uzp1   v4.4s, v28.4s, v29.4s       \n"
                "uzp2   v5.4s, v28.4s, v29.4s       \n"
                "uzp1   v6.4s, v30.4s, v31.4s       \n"
                "uzp2   v7.4s, v30.4s, v31.4s       \n"

                "add    v16.4s, v16.4s, v0.4s       \n"
                "add    v17.4s, v17.4s, v1.4s       \n"
                "add    v18.4s, v18.4s, v2.4s       \n"
                "add    v19.4s, v19.4s, v3.4s       \n"
                "add    v20.4s, v20.4s, v4.4s       \n"
                "add    v21.4s, v21.4s, v5.4s       \n"
                "add    v22.4s, v22.4s, v6.4s       \n"
                "add    v23.4s, v23.4s, v7.4s       \n"
#endif // __ARM_FEATURE_MATMUL_INT8

                "101:                               \n"
                "and    w4, %w6, #4                 \n" // w4 = remain = max_kk & 4
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                // kk += 4 part
                "ld1    {v0.16b, v1.16b}, [%1], #32 \n"
                "ld1    {v2.16b}, [%2], #16         \n"
                "sdot   v16.4s, v0.16b, v2.4b[0]    \n"
                "sdot   v17.4s, v0.16b, v2.4b[1]    \n"
                "sdot   v18.4s, v0.16b, v2.4b[2]    \n"
                "sdot   v19.4s, v0.16b, v2.4b[3]    \n"
                "sdot   v20.4s, v1.16b, v2.4b[0]    \n"
                "sdot   v21.4s, v1.16b, v2.4b[1]    \n"
                "sdot   v22.4s, v1.16b, v2.4b[2]    \n"
                "sdot   v23.4s, v1.16b, v2.4b[3]    \n"
#else  // __ARM_FEATURE_DOTPROD
                "lsr    w4, %w6, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                "2:                                 \n"
                "ld1    {v0.16b, v1.16b}, [%1], #32 \n"
                "ld1    {v4.16b}, [%2], #16         \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "rev64  v2.4s, v0.4s                \n"
                "smull  v10.8h, v2.8b, v4.8b        \n"
                "ext    v5.16b, v4.16b, v4.16b, #8  \n"
                "smull2 v9.8h, v0.16b, v5.16b       \n"
                "rev64  v6.8h, v4.8h                \n"
                "smull2 v11.8h, v2.16b, v5.16b      \n"
                "ext    v7.16b, v6.16b, v6.16b, #8  \n"
                "smull  v12.8h, v0.8b, v6.8b        \n"
                "smull  v14.8h, v2.8b, v6.8b        \n"
                "rev64  v3.4s, v1.4s                \n"
                "smull2 v13.8h, v0.16b, v7.16b      \n"
                "smull2 v15.8h, v2.16b, v7.16b      \n"
                "smlal  v8.8h, v1.8b, v5.8b         \n"
                "smlal  v10.8h, v3.8b, v5.8b        \n"
                "smlal2 v9.8h, v1.16b, v4.16b       \n"
                "smlal2 v11.8h, v3.16b, v4.16b      \n"
                "smlal  v12.8h, v1.8b, v7.8b        \n"
                "smlal  v14.8h, v3.8b, v7.8b        \n"
                "smlal2 v13.8h, v1.16b, v6.16b      \n"
                "smlal2 v15.8h, v3.16b, v6.16b      \n"
                "subs   w4, w4, #1                  \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v19.4s, v11.8h              \n"
                "sadalp v20.4s, v12.8h              \n"
                "sadalp v22.4s, v14.8h              \n"
                "sadalp v21.4s, v13.8h              \n"
                "sadalp v23.4s, v15.8h              \n"
                "bne    2b                          \n"
#endif // __ARM_FEATURE_DOTPROD

                "3:                                 \n"
                "and    w4, %w6, #2                 \n" // w4 = remain = max_kk & 2
                "cmp    w4, #0                      \n"
                "beq    4f                          \n"

                // kk += 2 part
#if __ARM_FEATURE_DOTPROD
                "ld1    {v0.16b}, [%1], #16         \n"
                "ld1    {v1.8b}, [%2], #8           \n"
                "dup    v4.8h, v1.h[0]              \n"
                "dup    v5.8h, v1.h[1]              \n"
                "dup    v6.8h, v1.h[2]              \n"
                "dup    v7.8h, v1.h[3]              \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull  v9.8h, v0.8b, v5.8b         \n"
                "smull  v10.8h, v0.8b, v6.8b        \n"
                "smull  v11.8h, v0.8b, v7.8b        \n"
                "smull2 v12.8h, v0.16b, v4.16b      \n"
                "smull2 v13.8h, v0.16b, v5.16b      \n"
                "smull2 v14.8h, v0.16b, v6.16b      \n"
                "smull2 v15.8h, v0.16b, v7.16b      \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
                "sadalp v20.4s, v12.8h              \n"
                "sadalp v21.4s, v13.8h              \n"
                "sadalp v22.4s, v14.8h              \n"
                "sadalp v23.4s, v15.8h              \n"
#else  // __ARM_FEATURE_DOTPROD
                "ld1    {v0.16b}, [%1], #16         \n"
                "ld1r   {v2.2d}, [%2]               \n"
                "add    %2, %2, #8                  \n"
                "rev64  v1.4s, v0.4s                \n"
                "rev64  v3.8h, v2.8h                \n"
                "smull  v8.8h, v0.8b, v2.8b         \n"
                "smull2 v9.8h, v0.16b, v2.16b       \n"
                "smull  v10.8h, v1.8b, v2.8b        \n"
                "smull2 v11.8h, v1.16b, v2.16b      \n"
                "smull  v12.8h, v0.8b, v3.8b        \n"
                "smull2 v13.8h, v0.16b, v3.16b      \n"
                "smull  v14.8h, v1.8b, v3.8b        \n"
                "smull2 v15.8h, v1.16b, v3.16b      \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
                "sadalp v20.4s, v12.8h              \n"
                "sadalp v21.4s, v13.8h              \n"
                "sadalp v22.4s, v14.8h              \n"
                "sadalp v23.4s, v15.8h              \n"
#endif // __ARM_FEATURE_DOTPROD

                "4:                                 \n"
                "and    w4, %w6, #1                 \n" // w4 = remain = max_kk & 1
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                // kk += 1 part
#if __ARM_FEATURE_DOTPROD
                "ld1    {v0.8b}, [%1], #8           \n"
                "ld1    {v1.8b}, [%2]               \n"
                "add    %2, %2, #4                  \n"
                "dup    v8.8b, v1.b[0]              \n"
                "dup    v9.8b, v1.b[1]              \n"
                "dup    v10.8b, v1.b[2]             \n"
                "dup    v11.8b, v1.b[3]             \n"
                "smull  v8.8h, v0.8b, v8.8b         \n"
                "smull  v9.8h, v0.8b, v9.8b         \n"
                "smull  v10.8h, v0.8b, v10.8b       \n"
                "smull  v11.8h, v0.8b, v11.8b       \n"
                "saddw  v16.4s, v16.4s, v8.4h       \n"
                "saddw  v17.4s, v17.4s, v9.4h       \n"
                "saddw  v18.4s, v18.4s, v10.4h      \n"
                "saddw  v19.4s, v19.4s, v11.4h      \n"
                "saddw2 v20.4s, v20.4s, v8.8h       \n"
                "saddw2 v21.4s, v21.4s, v9.8h       \n"
                "saddw2 v22.4s, v22.4s, v10.8h      \n"
                "saddw2 v23.4s, v23.4s, v11.8h      \n"
#else  // __ARM_FEATURE_DOTPROD
                "ld1    {v0.8b}, [%1], #8           \n"
                "ld1r   {v4.2s}, [%2]               \n"
                "add    %2, %2, #4                  \n"
                "rev32  v1.4h, v0.4h                \n"
                "rev64  v5.8b, v4.8b                \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull  v9.8h, v1.8b, v4.8b         \n"
                "smull  v10.8h, v0.8b, v5.8b        \n"
                "smull  v11.8h, v1.8b, v5.8b        \n"
                "saddw  v16.4s, v16.4s, v8.4h       \n"
                "saddw2 v17.4s, v17.4s, v8.8h       \n"
                "saddw  v18.4s, v18.4s, v9.4h       \n"
                "saddw2 v19.4s, v19.4s, v9.8h       \n"
                "saddw  v20.4s, v20.4s, v10.4h      \n"
                "saddw2 v21.4s, v21.4s, v10.8h      \n"
                "saddw  v22.4s, v22.4s, v11.4h      \n"
                "saddw2 v23.4s, v23.4s, v11.8h      \n"
#endif // __ARM_FEATURE_DOTPROD

                "5:                                 \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB)      // %2
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "r"(max_kk), // %6
                "r"(k)       // %7
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // __aarch64__
            asm volatile(
                "cmp        %7, #0              \n"
                "beq        0f                  \n"

                "vldm       %0!, {d16-d23}      \n"
                "vldm       %0, {d24-d31}       \n"
                "sub        %0, %0, #64         \n"
                "b          1f                  \n"

                "0:                             \n"
                "veor       q8, q8              \n"
                "veor       q9, q9              \n"
                "veor       q10, q10            \n"
                "veor       q11, q11            \n"
                "veor       q12, q12            \n"
                "veor       q13, q13            \n"
                "veor       q14, q14            \n"
                "veor       q15, q15            \n"

                "1:                             \n"
                "lsr        r4, %6, #2          \n" // r4 = max_kk >> 2
                "cmp        r4, #0              \n"
                "beq        3f                  \n"

                ".align 4                       \n"
                "2:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.s8    {d0-d3}, [%1 :64]!  \n"
                "pld        [%2, #128]          \n"
                "vld1.s8    {d4-d5}, [%2]!      \n"
                "vmull.s8   q4, d0, d4          \n"
                "vrev64.32  q3, q0              \n"
                "vmull.s8   q5, d1, d4          \n"
                "vmull.s8   q6, d6, d4          \n"
                "vmull.s8   q7, d7, d4          \n"
                "vrev64.32  q0, q1              \n"
                "vmlal.s8   q4, d2, d5          \n"
                "vmlal.s8   q5, d3, d5          \n"
                "vmlal.s8   q6, d0, d5          \n"
                "vmlal.s8   q7, d1, d5          \n"
                "vrev64.16  q2, q2              \n"
                "vpadal.s16 q8, q4              \n"
                "vrev64.32  q1, q3              \n"
                "vpadal.s16 q9, q5              \n"
                "vmull.s8   q4, d6, d4          \n"
                "vpadal.s16 q10, q6             \n"
                "vmull.s8   q5, d7, d4          \n"
                "vpadal.s16 q11, q7             \n"
                "vmull.s8   q6, d2, d4          \n"
                "vmull.s8   q7, d3, d4          \n"
                "vrev64.32  q3, q0              \n"
                "vmlal.s8   q4, d0, d5          \n"
                "vmlal.s8   q5, d1, d5          \n"
                "vmlal.s8   q6, d6, d5          \n"
                "vmlal.s8   q7, d7, d5          \n"
                "subs       r4, r4, #1          \n"
                "vpadal.s16 q14, q4             \n"
                "vpadal.s16 q15, q5             \n"
                "vpadal.s16 q12, q6             \n"
                "vpadal.s16 q13, q7             \n"
                "bne        2b                  \n"

                "3:                             \n"
                "and        r4, %6, #2          \n" // r4 = remain = max_kk & 2
                "cmp        r4, #0              \n"
                "beq        4f                  \n"

                // kk += 2 part
                "vld1.s8    {d0-d1}, [%1 :64]!  \n"
                "vld1.s8    {d4}, [%2]!         \n"
                "vrev64.32  q1, q0              \n"
                "vrev64.16  d5, d4              \n"
                "vmull.s8   q4, d0, d4          \n"
                "vmull.s8   q5, d1, d4          \n"
                "vmull.s8   q6, d2, d4          \n"
                "vmull.s8   q7, d3, d4          \n"
                "vpadal.s16 q8, q4              \n"
                "vpadal.s16 q9, q5              \n"
                "vpadal.s16 q10, q6             \n"
                "vpadal.s16 q11, q7             \n"
                "vmull.s8   q4, d0, d5          \n"
                "vmull.s8   q5, d1, d5          \n"
                "vmull.s8   q6, d2, d5          \n"
                "vmull.s8   q7, d3, d5          \n"
                "vpadal.s16 q12, q4             \n"
                "vpadal.s16 q13, q5             \n"
                "vpadal.s16 q14, q6             \n"
                "vpadal.s16 q15, q7             \n"

                "4:                             \n"
                "and        r4, %6, #1          \n" // r4 = remain = max_kk & 1
                "cmp        r4, #0              \n"
                "beq        5f                  \n"

                // kk += 1 part
                "vld1.s8    {d0}, [%1 :64]!     \n"
                "vld1.s32   {d2[]}, [%2]!       \n"
                "vrev64.16  d1, d0              \n"
                "vrev64.8   d3, d2              \n"
                "vext.s8    d1, d1, #4          \n"
                "vmull.s8   q4, d0, d2          \n"
                "vmull.s8   q5, d1, d2          \n"
                "vmull.s8   q6, d0, d3          \n"
                "vmull.s8   q7, d1, d3          \n"
                "vaddw.s16  q8, d8              \n"
                "vaddw.s16  q9, d9              \n"
                "vaddw.s16  q10, d10            \n"
                "vaddw.s16  q11, d11            \n"
                "vaddw.s16  q12, d12            \n"
                "vaddw.s16  q13, d13            \n"
                "vaddw.s16  q14, d14            \n"
                "vaddw.s16  q15, d15            \n"

                "5:                             \n"
                "vstm       %0!, {d16-d23}      \n"
                "vstm       %0!, {d24-d31}      \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB)      // %2
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "r"(max_kk), // %6
                "r"(k)       // %7
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;
            int32x4_t _sum4;
            int32x4_t _sum5;
            int32x4_t _sum6;
            int32x4_t _sum7;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
                _sum4 = vdupq_n_s32(0);
                _sum5 = vdupq_n_s32(0);
                _sum6 = vdupq_n_s32(0);
                _sum7 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
                _sum4 = vld1q_s32(outptr + 16);
                _sum5 = vld1q_s32(outptr + 20);
                _sum6 = vld1q_s32(outptr + 24);
                _sum7 = vld1q_s32(outptr + 28);
            }

            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _s0 = vdupq_n_s32(0);
                int32x4_t _s1 = vdupq_n_s32(0);
                int32x4_t _s2 = vdupq_n_s32(0);
                int32x4_t _s3 = vdupq_n_s32(0);
                int32x4_t _s4 = vdupq_n_s32(0);
                int32x4_t _s5 = vdupq_n_s32(0);
                int32x4_t _s6 = vdupq_n_s32(0);
                int32x4_t _s7 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA1 = vld1q_s8(pA + 16);
                    int8x16_t _pA2 = vld1q_s8(pA + 32);
                    int8x16_t _pA3 = vld1q_s8(pA + 48);

                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);

#if __ARM_FEATURE_MATMUL_INT8
                    // aaaaaaaa bbbbbbbb ..... hhhhhhhh
                    // 00000000 11111111 22222222 33333333

                    _s0 = vmmlaq_s32(_s0, _pA0, _pB0);
                    _s1 = vmmlaq_s32(_s1, _pA1, _pB0);
                    _s2 = vmmlaq_s32(_s2, _pA0, _pB1);
                    _s3 = vmmlaq_s32(_s3, _pA1, _pB1);
                    _s4 = vmmlaq_s32(_s4, _pA2, _pB0);
                    _s5 = vmmlaq_s32(_s5, _pA3, _pB0);
                    _s6 = vmmlaq_s32(_s6, _pA2, _pB1);
                    _s7 = vmmlaq_s32(_s7, _pA3, _pB1);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_laneq_s32(_sum0, _pA0, _pB0, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _pA0, _pB0, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _pA0, _pB0, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _pA0, _pB0, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _pA1, _pB0, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _pA1, _pB0, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _pA1, _pB0, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _pA1, _pB0, 3);

                    _sum0 = vdotq_laneq_s32(_sum0, _pA2, _pB1, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _pA2, _pB1, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _pA2, _pB1, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _pA2, _pB1, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _pA3, _pB1, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _pA3, _pB1, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _pA3, _pB1, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _pA3, _pB1, 3);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 64;
                    pB += 32;
                }
#if __ARM_FEATURE_MATMUL_INT8
                int32x4x2_t _ss0 = vuzpq_s32(_s0, _s1);
                int32x4x2_t _ss1 = vuzpq_s32(_s2, _s3);
                int32x4x2_t _ss2 = vuzpq_s32(_s4, _s5);
                int32x4x2_t _ss3 = vuzpq_s32(_s6, _s7);
                _sum0 = vaddq_s32(_sum0, _ss0.val[0]);
                _sum1 = vaddq_s32(_sum1, _ss0.val[1]);
                _sum2 = vaddq_s32(_sum2, _ss1.val[0]);
                _sum3 = vaddq_s32(_sum3, _ss1.val[1]);
                _sum4 = vaddq_s32(_sum4, _ss2.val[0]);
                _sum5 = vaddq_s32(_sum5, _ss2.val[1]);
                _sum6 = vaddq_s32(_sum6, _ss3.val[0]);
                _sum7 = vaddq_s32(_sum7, _ss3.val[1]);
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA1 = vld1q_s8(pA + 16);
                int8x16_t _pB = vld1q_s8(pB);

                // aaaa bbbb cccc dddd   eeee ffff gggg hhhh

                // 0000 1111 2222 3333

                _sum0 = vdotq_laneq_s32(_sum0, _pA0, _pB, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA0, _pB, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA0, _pB, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA0, _pB, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _pA1, _pB, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _pA1, _pB, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _pA1, _pB, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _pA1, _pB, 3);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA2 = vld1q_s8(pA + 16);
                int8x16_t _pB02 = vld1q_s8(pB);

                // aabbccdd eeffgghh

                // ccddaabb gghheeff

                int8x16_t _pA1 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA0)));
                int8x16_t _pA3 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA2)));

                // 00112233 44556677

                // 33221100 77665544

                int8x16_t _pB13 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB02)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB02));
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB02));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB02));
                int16x8_t _s3 = vmull_s8(vget_high_s8(_pA1), vget_low_s8(_pB02));
                int16x8_t _s4 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB13));
                int16x8_t _s5 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB13));
                int16x8_t _s6 = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB13));
                int16x8_t _s7 = vmull_s8(vget_high_s8(_pA1), vget_low_s8(_pB13));

                _s0 = vmlal_s8(_s0, vget_low_s8(_pA2), vget_high_s8(_pB02));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA2), vget_high_s8(_pB02));
                _s2 = vmlal_s8(_s2, vget_low_s8(_pA3), vget_high_s8(_pB02));
                _s3 = vmlal_s8(_s3, vget_high_s8(_pA3), vget_high_s8(_pB02));
                _s4 = vmlal_s8(_s4, vget_low_s8(_pA2), vget_high_s8(_pB13));
                _s5 = vmlal_s8(_s5, vget_high_s8(_pA2), vget_high_s8(_pB13));
                _s6 = vmlal_s8(_s6, vget_low_s8(_pA3), vget_high_s8(_pB13));
                _s7 = vmlal_s8(_s7, vget_high_s8(_pA3), vget_high_s8(_pB13));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
#endif // __ARM_FEATURE_DOTPROD

                pA += 32;
                pB += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                // aabbccdd eeffgghh

                // 00112233
                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 0)));
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 1)));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 2)));
                int16x8_t _s3 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 3)));
                int16x8_t _s4 = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 0)));
                int16x8_t _s5 = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 1)));
                int16x8_t _s6 = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 2)));
                int16x8_t _s7 = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 3)));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x8_t _pB0 = vld1_s8(pB);

                // aabbccdd eeffgghh

                // ccddaabb gghheeff

                int8x16_t _pA1 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA0)));

                // 00112233

                // 33221100

                int8x8_t _pB1 = vreinterpret_s8_s16(vrev64_s16(vreinterpret_s16_s8(_pB0)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), _pB0);
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), _pB0);
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA1), _pB0);
                int16x8_t _s3 = vmull_s8(vget_high_s8(_pA1), _pB0);
                int16x8_t _s4 = vmull_s8(vget_low_s8(_pA0), _pB1);
                int16x8_t _s5 = vmull_s8(vget_high_s8(_pA0), _pB1);
                int16x8_t _s6 = vmull_s8(vget_low_s8(_pA1), _pB1);
                int16x8_t _s7 = vmull_s8(vget_high_s8(_pA1), _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
#endif // __ARM_FEATURE_DOTPROD

                pA += 16;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vld1_s8(pA);
                // int8x8_t _pB0 = vreinterpret_s32_s8(vld1_dup_s32(pB));

                // abcdefgh

                // 0123

                int16x8_t _s01 = vmull_s8(_pA0, vdup_n_s8(pB[0]));
                int16x8_t _s23 = vmull_s8(_pA0, vdup_n_s8(pB[1]));
                int16x8_t _s45 = vmull_s8(_pA0, vdup_n_s8(pB[2]));
                int16x8_t _s67 = vmull_s8(_pA0, vdup_n_s8(pB[3]));
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_low_s16(_s23));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s45));
                _sum3 = vaddw_s16(_sum3, vget_low_s16(_s67));
                _sum4 = vaddw_s16(_sum4, vget_high_s16(_s01));
                _sum5 = vaddw_s16(_sum5, vget_high_s16(_s23));
                _sum6 = vaddw_s16(_sum6, vget_high_s16(_s45));
                _sum7 = vaddw_s16(_sum7, vget_high_s16(_s67));
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vld1_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));
                // int8x8_t _pB0 = vld1_s8(pB);
                // _pB0 = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_pB0), vreinterpret_s32_s8(_pB0)).val[0]);

                // abcdefgh  ->  cdabghef
                int8x8_t _pA1 = vreinterpret_s8_s16(vrev32_s16(vreinterpret_s16_s8(_pA0)));

                // 01230123  ->  32103210
                int8x8_t _pB1 = vrev64_s8(_pB0);

                int16x8_t _s01 = vmull_s8(_pA0, _pB0);
                int16x8_t _s23 = vmull_s8(_pA1, _pB0);
                int16x8_t _s45 = vmull_s8(_pA0, _pB1);
                int16x8_t _s67 = vmull_s8(_pA1, _pB1);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s23));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s23));
                _sum4 = vaddw_s16(_sum4, vget_low_s16(_s45));
                _sum5 = vaddw_s16(_sum5, vget_high_s16(_s45));
                _sum6 = vaddw_s16(_sum6, vget_low_s16(_s67));
                _sum7 = vaddw_s16(_sum7, vget_high_s16(_s67));
#endif // __ARM_FEATURE_DOTPROD

                pA += 8;
                pB += 4;
            }

            //             if (k_end)
            //             {
            // #if __ARM_FEATURE_DOTPROD
            //                 // from
            //                 //      a0 b0 c0 d0
            //                 //      a1 b1 c1 d1
            //                 //      a2 b2 c2 d2
            //                 //      a3 b3 c3 d3
            //                 //      e0 f0 g0 h0
            //                 //      e1 f1 g1 h1
            //                 //      e2 f2 g2 h2
            //                 //      e3 f3 g3 h3
            //                 if (out_elempack == 8)
            //                 {
            //                     vst1q_s32(outptr0, _sum0);
            //                     vst1q_s32(outptr0 + 4, _sum4);
            //                     vst1q_s32(outptr0 + 8, _sum1);
            //                     vst1q_s32(outptr0 + 12, _sum5);
            //                     vst1q_s32(outptr0 + 16, _sum2);
            //                     vst1q_s32(outptr0 + 20, _sum6);
            //                     vst1q_s32(outptr0 + 24, _sum3);
            //                     vst1q_s32(outptr0 + 28, _sum7);
            //                     outptr0 += 32;
            //                 }
            //                 if (out_elempack == 4)
            //                 {
            //                     vst1q_s32(outptr0, _sum0);
            //                     vst1q_s32(outptr0 + 4, _sum1);
            //                     vst1q_s32(outptr0 + 8, _sum2);
            //                     vst1q_s32(outptr0 + 12, _sum3);
            //                     vst1q_s32(outptr0 + out_hstep * 4, _sum4);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum5);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 8, _sum6);
            //                     vst1q_s32(outptr0 + out_hstep * 4 + 12, _sum7);
            //                     outptr0 += 16;
            //                 }
            //                 if (out_elempack == 1)
            //                 {
            //                     // to
            //                     //      a0 a1 a2 a3
            //                     //      b0 b1 b2 b3
            //                     //      c0 c1 c2 c3
            //                     //      d0 d1 d2 d3
            //                     //      e0 e1 e2 e3
            //                     //      f0 f1 f2 f3
            //                     //      g0 g1 g2 g3
            //                     //      h0 h1 h2 h3
            //                     {
            //                         int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
            //                         int32x4x2_t _t1 = vzipq_s32(_sum2, _sum3);
            //                         int32x4x2_t _t2 = vzipq_s32(_sum4, _sum5);
            //                         int32x4x2_t _t3 = vzipq_s32(_sum6, _sum7);
            //                         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
            //                         _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
            //                         _sum2 = vcombine_s32(vget_low_s32(_t0.val[1]), vget_low_s32(_t1.val[1]));
            //                         _sum3 = vcombine_s32(vget_high_s32(_t0.val[1]), vget_high_s32(_t1.val[1]));
            //                         _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
            //                         _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
            //                         _sum6 = vcombine_s32(vget_low_s32(_t2.val[1]), vget_low_s32(_t3.val[1]));
            //                         _sum7 = vcombine_s32(vget_high_s32(_t2.val[1]), vget_high_s32(_t3.val[1]));
            //                     }
            //
            //                     vst1q_s32(outptr0, _sum0);
            //                     vst1q_s32(outptr0 + out_hstep, _sum1);
            //                     vst1q_s32(outptr0 + out_hstep * 2, _sum2);
            //                     vst1q_s32(outptr0 + out_hstep * 3, _sum3);
            //                     vst1q_s32(outptr0 + out_hstep * 4, _sum4);
            //                     vst1q_s32(outptr0 + out_hstep * 5, _sum5);
            //                     vst1q_s32(outptr0 + out_hstep * 6, _sum6);
            //                     vst1q_s32(outptr0 + out_hstep * 7, _sum7);
            //                     outptr0 += 4;
            //                 }
            // #else  // __ARM_FEATURE_DOTPROD
            //
            //                 // from
            //                 //      a0 b1 c2 d3
            //                 //      e0 f1 g2 h3
            //                 //      c0 d1 a2 b3
            //                 //      g0 h1 e2 f3
            //                 //      a3 b2 c1 d0
            //                 //      e3 f2 g1 h0
            //                 //      c3 d2 a1 b0
            //                 //      g3 h2 e1 f0
            //                 // if (out_elempack == 8)
            //                 // {
            //                 //     // to
            //                 //     //      a0 b0 c0 d0
            //                 //     //      e0 f0 g0 h0
            //                 //     //      a1 b1 c1 d1
            //                 //     //      e1 f1 g1 h1
            //                 //     //      a2 b2 c2 d2
            //                 //     //      e2 f2 g2 h2
            //                 //     //      a3 b3 c3 d3
            //                 //     //      e3 f3 g3 h3
            //                 //     {
            //                 //         _sum4 = vrev64q_s32(_sum4);
            //                 //         _sum5 = vrev64q_s32(_sum5);
            //                 //         _sum6 = vrev64q_s32(_sum6);
            //                 //         _sum7 = vrev64q_s32(_sum7);
            //                 //         _sum4 = vextq_s32(_sum4, _sum4, 2);
            //                 //         _sum5 = vextq_s32(_sum5, _sum5, 2);
            //                 //         _sum6 = vextq_s32(_sum6, _sum6, 2);
            //                 //         _sum7 = vextq_s32(_sum7, _sum7, 2);
            //                 //         int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
            //                 //         int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
            //                 //         int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
            //                 //         int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
            //                 //         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
            //                 //         _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
            //                 //         _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
            //                 //         _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
            //                 //         _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
            //                 //         _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
            //                 //         _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
            //                 //         _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
            //                 //         _sum2 = vrev64q_s32(_sum2);
            //                 //         _sum3 = vrev64q_s32(_sum3);
            //                 //         _sum6 = vrev64q_s32(_sum6);
            //                 //         _sum7 = vrev64q_s32(_sum7);
            //                 //     }
            //                 //
            //                 //     vst1q_s32(outptr0, _sum0);
            //                 //     vst1q_s32(outptr0 + 4, _sum1);
            //                 //     vst1q_s32(outptr0 + 8, _sum2);
            //                 //     vst1q_s32(outptr0 + 12, _sum3);
            //                 //     vst1q_s32(outptr0 + 16, _sum4);
            //                 //     vst1q_s32(outptr0 + 20, _sum5);
            //                 //     vst1q_s32(outptr0 + 24, _sum6);
            //                 //     vst1q_s32(outptr0 + 28, _sum7);
            //                 //     outptr0 += 32;
            //                 // }
            //                 // if (out_elempack == 4)
            //                 // {
            //                 //     // to
            //                 //     //      a0 b0 c0 d0
            //                 //     //      a1 b1 c1 d1
            //                 //     //      a2 b2 c2 d2
            //                 //     //      a3 b3 c3 d3
            //                 //     //      e0 f0 g0 h0
            //                 //     //      e1 f1 g1 h1
            //                 //     //      e2 f2 g2 h2
            //                 //     //      e3 f3 g3 h3
            //                 //     {
            //                 //         _sum4 = vrev64q_s32(_sum4);
            //                 //         _sum5 = vrev64q_s32(_sum5);
            //                 //         _sum6 = vrev64q_s32(_sum6);
            //                 //         _sum7 = vrev64q_s32(_sum7);
            //                 //         _sum4 = vextq_s32(_sum4, _sum4, 2);
            //                 //         _sum5 = vextq_s32(_sum5, _sum5, 2);
            //                 //         _sum6 = vextq_s32(_sum6, _sum6, 2);
            //                 //         _sum7 = vextq_s32(_sum7, _sum7, 2);
            //                 //         int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
            //                 //         int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
            //                 //         int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
            //                 //         int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
            //                 //         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
            //                 //         _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
            //                 //         _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
            //                 //         _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
            //                 //         _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
            //                 //         _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
            //                 //         _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
            //                 //         _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
            //                 //         _sum1 = vrev64q_s32(_sum1);
            //                 //         _sum3 = vrev64q_s32(_sum3);
            //                 //         _sum5 = vrev64q_s32(_sum5);
            //                 //         _sum7 = vrev64q_s32(_sum7);
            //                 //     }
            //                 //
            //                 //     vst1q_s32(outptr0, _sum0);
            //                 //     vst1q_s32(outptr0 + 4, _sum1);
            //                 //     vst1q_s32(outptr0 + 8, _sum2);
            //                 //     vst1q_s32(outptr0 + 12, _sum3);
            //                 //     // vst1q_s32(outptr0 + out_hstep * 4, _sum4);
            //                 //     // vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum5);
            //                 //     // vst1q_s32(outptr0 + out_hstep * 4 + 8, _sum6);
            //                 //     // vst1q_s32(outptr0 + out_hstep * 4 + 12, _sum7);
            //                 //     // outptr0 += 16;
            //                 //     vst1q_s32(outptr0 + 16, _sum4);//TODO
            //                 //     vst1q_s32(outptr0 + 20, _sum5);
            //                 //     vst1q_s32(outptr0 + 24, _sum6);
            //                 //     vst1q_s32(outptr0 + 28, _sum7);
            //                 //     outptr0 += 32;
            //                 // }
            //                 // if (out_elempack == 1)
            //                 // {
            //                 //     // to
            //                 //     //      a0 a1 a2 a3
            //                 //     //      b0 b1 b2 b3
            //                 //     //      c0 c1 c2 c3
            //                 //     //      d0 d1 d2 d3
            //                 //     //      e0 e1 e2 e3
            //                 //     //      f0 f1 f2 f3
            //                 //     //      g0 g1 g2 g3
            //                 //     //      h0 h1 h2 h3
            //                 //     {
            //                 //         _sum2 = vextq_s32(_sum2, _sum2, 2);
            //                 //         _sum3 = vextq_s32(_sum3, _sum3, 2);
            //                 //         _sum6 = vextq_s32(_sum6, _sum6, 2);
            //                 //         _sum7 = vextq_s32(_sum7, _sum7, 2);
            //                 //         int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
            //                 //         int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
            //                 //         int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
            //                 //         int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
            //                 //         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
            //                 //         _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
            //                 //         _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
            //                 //         _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
            //                 //         _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
            //                 //         _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
            //                 //         _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
            //                 //         _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
            //                 //         _sum1 = vrev64q_s32(_sum1);
            //                 //         _sum3 = vrev64q_s32(_sum3);
            //                 //         _sum5 = vrev64q_s32(_sum5);
            //                 //         _sum7 = vrev64q_s32(_sum7);
            //                 //     }
            //                 //
            //                 //     vst1q_s32(outptr0, _sum0);
            //                 //     vst1q_s32(outptr0 + out_hstep, _sum1);
            //                 //     vst1q_s32(outptr0 + out_hstep * 2, _sum2);
            //                 //     vst1q_s32(outptr0 + out_hstep * 3, _sum3);
            //                 //     vst1q_s32(outptr0 + out_hstep * 4, _sum4);
            //                 //     vst1q_s32(outptr0 + out_hstep * 5, _sum5);
            //                 //     vst1q_s32(outptr0 + out_hstep * 6, _sum6);
            //                 //     vst1q_s32(outptr0 + out_hstep * 7, _sum7);
            //                 //     outptr0 += 4;
            //                 //     vst1q_s32(outptr0 + out_hstep, _sum1);
            //                 //     vst1q_s32(outptr0 + out_hstep * 2, _sum2);
            //                 //     vst1q_s32(outptr0 + out_hstep * 3, _sum3);
            //                 //     vst1q_s32(outptr0 + out_hstep * 4, _sum4);
            //                 //     vst1q_s32(outptr0 + out_hstep * 5, _sum5);
            //                 //     vst1q_s32(outptr0 + out_hstep * 6, _sum6);
            //                 //     vst1q_s32(outptr0 + out_hstep * 7, _sum7);
            //                 //     outptr0 += 4;
            //                 // }
            //
            //                 vst1q_s32(outptr0, _sum0);
            //                 vst1q_s32(outptr0 + 4, _sum1);
            //                 vst1q_s32(outptr0 + 8, _sum2);
            //                 vst1q_s32(outptr0 + 12, _sum3);
            //                 vst1q_s32(outptr0 + 16, _sum4);
            //                 vst1q_s32(outptr0 + 20, _sum5);
            //                 vst1q_s32(outptr0 + 24, _sum6);
            //                 vst1q_s32(outptr0 + 28, _sum7);
            //                 outptr0 += 32;
            // #endif // __ARM_FEATURE_DOTPROD
            //             }
            //             else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
                vst1q_s32(outptr + 16, _sum4);
                vst1q_s32(outptr + 20, _sum5);
                vst1q_s32(outptr + 24, _sum6);
                vst1q_s32(outptr + 28, _sum7);
            }

            outptr += 32;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
            }

            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _s0 = vdupq_n_s32(0);
                int32x4_t _s1 = vdupq_n_s32(0);
                int32x4_t _s2 = vdupq_n_s32(0);
                int32x4_t _s3 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA1 = vld1q_s8(pA + 16);
                    int8x16_t _pA2 = vld1q_s8(pA + 32);
                    int8x16_t _pA3 = vld1q_s8(pA + 48);

                    int8x16_t _pB = vld1q_s8(pB);

#if __ARM_FEATURE_MATMUL_INT8
                    // aaaaaaaa bbbbbbbb ..... hhhhhhhh
                    // 00000000 11111111

                    _s0 = vmmlaq_s32(_s0, _pA0, _pB);
                    _s1 = vmmlaq_s32(_s1, _pA1, _pB);
                    _s2 = vmmlaq_s32(_s2, _pA2, _pB);
                    _s3 = vmmlaq_s32(_s3, _pA3, _pB);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_laneq_s32(_sum0, _pA0, _pB, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _pA0, _pB, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _pA1, _pB, 0);
                    _sum3 = vdotq_laneq_s32(_sum3, _pA1, _pB, 1);

                    _sum0 = vdotq_laneq_s32(_sum0, _pA2, _pB, 2);
                    _sum1 = vdotq_laneq_s32(_sum1, _pA2, _pB, 3);
                    _sum2 = vdotq_laneq_s32(_sum2, _pA3, _pB, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _pA3, _pB, 3);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 64;
                    pB += 16;
                }
#if __ARM_FEATURE_MATMUL_INT8
                int32x4x2_t _ss0 = vuzpq_s32(_s0, _s1);
                int32x4x2_t _ss1 = vuzpq_s32(_s2, _s3);
                _sum0 = vaddq_s32(_sum0, _ss0.val[0]);
                _sum1 = vaddq_s32(_sum1, _ss0.val[1]);
                _sum2 = vaddq_s32(_sum2, _ss1.val[0]);
                _sum3 = vaddq_s32(_sum3, _ss1.val[1]);
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA1 = vld1q_s8(pA + 16);
                int8x8_t _pB = vld1_s8(pB);

                // aaaa bbbb cccc dddd eeee ffff gggg hhhh

                // 0000 1111

                _sum0 = vdotq_lane_s32(_sum0, _pA0, _pB, 0);
                _sum1 = vdotq_lane_s32(_sum1, _pA0, _pB, 1);
                _sum2 = vdotq_lane_s32(_sum2, _pA1, _pB, 0);
                _sum3 = vdotq_lane_s32(_sum3, _pA1, _pB, 1);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA2 = vld1q_s8(pA + 16);
                int8x8_t _pB = vld1_s8(pB);

                // aabbccdd eeffgghh   aabbccdd eeffgghh

                // 00112233 -> 00110011 22332233

                // 11001100 33223322

                int32x2x2_t _pBB = vzip_s32(vreinterpret_s32_s8(_pB), vreinterpret_s32_s8(_pB));
                int8x16_t _pB02 = vreinterpretq_s8_s32(vcombine_s32(_pBB.val[0], _pBB.val[1]));

                int8x16_t _pB13 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB02)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB02));
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB02));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB13));
                int16x8_t _s3 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB13));
                _s0 = vmlal_s8(_s0, vget_low_s8(_pA2), vget_high_s8(_pB02));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA2), vget_high_s8(_pB02));
                _s2 = vmlal_s8(_s2, vget_low_s8(_pA2), vget_high_s8(_pB13));
                _s3 = vmlal_s8(_s3, vget_high_s8(_pA2), vget_high_s8(_pB13));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
#endif // __ARM_FEATURE_DOTPROD

                pA += 32;
                pB += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int16x4_t _pB = vreinterpret_s16_s32(vld1_dup_s32((const int*)pB));

                int16x4x2_t _pB01 = vuzp_s16(_pB, _pB);
                int8x8_t _pB0 = vreinterpret_s8_s16(_pB01.val[0]);
                int8x8_t _pB1 = vreinterpret_s8_s16(_pB01.val[1]);

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), _pB0);
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA), _pB1);
                int16x8_t _s2 = vmull_s8(vget_high_s8(_pA), _pB0);
                int16x8_t _s3 = vmull_s8(vget_high_s8(_pA), _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));

                // aabbccdd eeffgghh

                // 00110011
                // 11001100

                int8x8_t _pB1 = vreinterpret_s8_s16(vrev64_s16(vreinterpret_s16_s8(_pB0)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), _pB0);
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA), _pB0);
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA), _pB1);
                int16x8_t _s3 = vmull_s8(vget_high_s8(_pA), _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
#endif // __ARM_FEATURE_DOTPROD

                pA += 16;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));

                int8x8x2_t _pB01 = vuzp_s8(_pB, _pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB01.val[0]);
                int16x8_t _s1 = vmull_s8(_pA, _pB01.val[1]);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_low_s16(_s1));
                _sum2 = vaddw_s16(_sum2, vget_high_s16(_s0));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));

                // abcdefgh

                // 01010101
                // 10101010
                int8x8_t _pB1 = vext_s8(_pB0, _pB0, 1);

                int16x8_t _s0 = vmull_s8(_pA, _pB0);
                int16x8_t _s1 = vmull_s8(_pA, _pB1);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));
#endif // __ARM_FEATURE_DOTPROD

                pA += 8;
                pB += 2;
            }

            //             if (k_end)
            //             {
            // #if __ARM_FEATURE_DOTPROD
            //                 // from
            //                 //      a0 b0 c0 d0
            //                 //      a1 b1 c1 d1
            //                 //      e0 f0 g0 h0
            //                 //      e1 f1 g1 h1
            //                 if (out_elempack == 8)
            //                 {
            //                     vst1q_s32(outptr0, _sum0);
            //                     vst1q_s32(outptr0 + 4, _sum2);
            //                     vst1q_s32(outptr0 + 8, _sum1);
            //                     vst1q_s32(outptr0 + 12, _sum3);
            //                     outptr0 += 16;
            //                 }
            //                 // if (out_elempack == 4)
            //                 // {
            //                 //     vst1q_s32(outptr0, _sum0);
            //                 //     vst1q_s32(outptr0 + 4, _sum1);
            //                 //     vst1q_s32(outptr0 + out_hstep * 4, _sum2);
            //                 //     vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum3);
            //                 //     outptr0 += 8;
            //                 // }
            //                 // if (out_elempack == 1)
            //                 // {
            //                 //     // to
            //                 //     //      a0 a1 b0 b1
            //                 //     //      c0 c1 d0 d1
            //                 //     //      e0 e1 f0 f1
            //                 //     //      g0 g1 h0 h1
            //                 //     {
            //                 //         int32x4x2_t _sum02 = vzipq_s32(_sum0, _sum1);
            //                 //         int32x4x2_t _sum13 = vzipq_s32(_sum2, _sum3);
            //                 //         _sum0 = _sum02.val[0];
            //                 //         _sum1 = _sum02.val[1];
            //                 //         _sum2 = _sum13.val[0];
            //                 //         _sum3 = _sum13.val[1];
            //                 //     }
            //                 //
            //                 //     vst1_s32(outptr0, vget_low_s32(_sum0));
            //                 //     vst1_s32(outptr0 + out_hstep, vget_high_s32(_sum0));
            //                 //     vst1_s32(outptr0 + out_hstep * 2, vget_low_s32(_sum1));
            //                 //     vst1_s32(outptr0 + out_hstep * 3, vget_high_s32(_sum1));
            //                 //     vst1_s32(outptr0 + out_hstep * 4, vget_low_s32(_sum2));
            //                 //     vst1_s32(outptr0 + out_hstep * 5, vget_high_s32(_sum2));
            //                 //     vst1_s32(outptr0 + out_hstep * 6, vget_low_s32(_sum3));
            //                 //     vst1_s32(outptr0 + out_hstep * 7, vget_high_s32(_sum3));
            //                 //     outptr0 += 2;
            //                 // }
            // #else  // __ARM_FEATURE_DOTPROD
            //
            //                 // from
            //                 //      a0 b1 c0 d1
            //                 //      e0 f1 g0 h1
            //                 //      a1 b0 c1 d0
            //                 //      e1 f0 g1 h0
            //                 // if (out_elempack == 8)
            //                 // {
            //                 //     // to
            //                 //     //      a0 b0 c0 d0
            //                 //     //      e0 f0 g0 h0
            //                 //     //      a1 b1 c1 d1
            //                 //     //      e1 f1 g1 h1
            //                 //     {
            //                 //         _sum2 = vrev64q_s32(_sum2);
            //                 //         _sum3 = vrev64q_s32(_sum3);
            //                 //         int32x4x2_t _t0 = vzipq_s32(_sum0, _sum2);
            //                 //         int32x4x2_t _t1 = vzipq_s32(_sum1, _sum3);
            //                 //         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
            //                 //         _sum1 = vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t1.val[1]));
            //                 //         _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
            //                 //         _sum3 = vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t1.val[1]));
            //                 //         _sum2 = vrev64q_s32(_sum2);
            //                 //         _sum3 = vrev64q_s32(_sum3);
            //                 //     }
            //                 //
            //                 //     vst1q_s32(outptr0, _sum0);
            //                 //     vst1q_s32(outptr0 + 4, _sum1);
            //                 //     vst1q_s32(outptr0 + 8, _sum2);
            //                 //     vst1q_s32(outptr0 + 12, _sum3);
            //                 //     outptr0 += 16;
            //                 // }
            //                 // if (out_elempack == 4)
            //                 // {
            //                 //     // to
            //                 //     //      a0 b0 c0 d0
            //                 //     //      a1 b1 c1 d1
            //                 //     //      e0 f0 g0 h0
            //                 //     //      e1 f1 g1 h1
            //                 //     {
            //                 //         _sum2 = vrev64q_s32(_sum2);
            //                 //         _sum3 = vrev64q_s32(_sum3);
            //                 //         int32x4x2_t _t0 = vzipq_s32(_sum0, _sum2);
            //                 //         int32x4x2_t _t1 = vzipq_s32(_sum1, _sum3);
            //                 //         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
            //                 //         _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
            //                 //         _sum2 = vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t1.val[1]));
            //                 //         _sum3 = vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t1.val[1]));
            //                 //         _sum1 = vrev64q_s32(_sum1);
            //                 //         _sum3 = vrev64q_s32(_sum3);
            //                 //     }
            //                 //
            //                 //     vst1q_s32(outptr0, _sum0);
            //                 //     vst1q_s32(outptr0 + 4, _sum1);
            //                 //     // vst1q_s32(outptr0 + out_hstep * 4, _sum2);
            //                 //     // vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum3);
            //                 //     // outptr0 += 8;
            //                 //     vst1q_s32(outptr0 + 8, _sum2);
            //                 //     vst1q_s32(outptr0 + 12, _sum3);
            //                 //     outptr0 += 16;
            //                 // }
            //                 // if (out_elempack == 1)
            //                 // {
            //                 //     // to
            //                 //     //      a0 a1 c0 c1
            //                 //     //      b0 b1 d0 d1
            //                 //     //      e0 e1 g0 g1
            //                 //     //      f0 f1 h0 h1
            //                 //     {
            //                 //         int32x4x2_t _t0 = vzipq_s32(_sum0, _sum2);
            //                 //         int32x4x2_t _t1 = vzipq_s32(_sum1, _sum3);
            //                 //         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
            //                 //         _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
            //                 //         _sum2 = vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t1.val[1]));
            //                 //         _sum3 = vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t1.val[1]));
            //                 //         _sum1 = vrev64q_s32(_sum1);
            //                 //         _sum3 = vrev64q_s32(_sum3);
            //                 //     }
            //                 //
            //                 //     vst1_s32(outptr0, vget_low_s32(_sum0));
            //                 //     vst1_s32(outptr0 + out_hstep, vget_low_s32(_sum1));
            //                 //     vst1_s32(outptr0 + out_hstep * 2, vget_high_s32(_sum0));
            //                 //     vst1_s32(outptr0 + out_hstep * 3, vget_high_s32(_sum1));
            //                 //     vst1_s32(outptr0 + out_hstep * 4, vget_low_s32(_sum2));
            //                 //     vst1_s32(outptr0 + out_hstep * 5, vget_low_s32(_sum3));
            //                 //     vst1_s32(outptr0 + out_hstep * 6, vget_high_s32(_sum2));
            //                 //     vst1_s32(outptr0 + out_hstep * 7, vget_high_s32(_sum3));
            //                 //     outptr0 += 2;
            //                 // }
            //                 vst1q_s32(outptr0, _sum0);
            //                 vst1q_s32(outptr0 + 4, _sum1);
            //                 vst1q_s32(outptr0 + 8, _sum2);
            //                 vst1q_s32(outptr0 + 12, _sum3);
            //                 outptr0 += 16;
            // #endif // __ARM_FEATURE_DOTPROD
            //             }
            //             else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            int32x4_t _sum0;
            int32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
            }

            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _s0 = vdupq_n_s32(0);
                int32x4_t _s1 = vdupq_n_s32(0);
                int32x4_t _s2 = vdupq_n_s32(0);
                int32x4_t _s3 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA1 = vld1q_s8(pA + 16);
                    int8x16_t _pA2 = vld1q_s8(pA + 32);
                    int8x16_t _pA3 = vld1q_s8(pA + 48);

                    int8x8_t _pB = vld1_s8(pB);

#if __ARM_FEATURE_MATMUL_INT8
                    // aaaaaaaa bbbbbbbb ..... hhhhhhhh
                    // 00000000
                    int8x16_t _pBB = vcombine_s8(_pB, _pB);

                    _s0 = vdotq_s32(_s0, _pA0, _pBB);
                    _s1 = vdotq_s32(_s1, _pA1, _pBB);
                    _s2 = vdotq_s32(_s2, _pA2, _pBB);
                    _s3 = vdotq_s32(_s3, _pA3, _pBB);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_lane_s32(_sum0, _pA0, _pB, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _pA1, _pB, 0);
                    _sum0 = vdotq_lane_s32(_sum0, _pA2, _pB, 1);
                    _sum1 = vdotq_lane_s32(_sum1, _pA3, _pB, 1);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 64;
                    pB += 8;
                }
#if __ARM_FEATURE_MATMUL_INT8
                _sum0 = vaddq_s32(_sum0, vpaddq_s32(_s0, _s1));
                _sum1 = vaddq_s32(_sum1, vpaddq_s32(_s2, _s3));
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA1 = vld1q_s8(pA + 16);

                int8x8_t _pB = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));

                // aaaa bbbb cccc dddd eeee ffff gggg hhhh

                // 0000 0000

                _sum0 = vdotq_lane_s32(_sum0, _pA0, _pB, 0);
                _sum1 = vdotq_lane_s32(_sum1, _pA1, _pB, 0);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA2 = vld1q_s8(pA + 16);
                int8x8_t _pB0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                int8x8_t _pB1 = vreinterpret_s8_s16(vld1_dup_s16((const short*)(pB + 2)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), _pB0);
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), _pB0);
                _s0 = vmlal_s8(_s0, vget_low_s8(_pA2), _pB1);
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA2), _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
#endif // __ARM_FEATURE_DOTPROD

                pA += 32;
                pB += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), _pB);
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA), _pB);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);

                pA += 16;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vld1_dup_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                pA += 8;
                pB += 1;
            }

            // if (k_end)
            // {
            //     // if (out_elempack == 8)
            //     // {
            //     //     vst1q_s32(outptr0, _sum0);
            //     //     vst1q_s32(outptr0 + 4, _sum1);
            //     //     outptr0 += 8;
            //     // }
            //     if (out_elempack == 4)
            //     {
            //         vst1q_s32(outptr0, _sum0);
            //         // vst1q_s32(outptr0 + out_hstep * 4, _sum1);
            //         // outptr0 += 4;
            //         vst1q_s32(outptr0 + 4, _sum1);
            //         outptr0 += 8;
            //     }
            //     // if (out_elempack == 1)
            //     // {
            //     //     outptr0[0] = vgetq_lane_s32(_sum0, 0);
            //     //     outptr0[out_hstep] = vgetq_lane_s32(_sum0, 1);
            //     //     outptr0[out_hstep * 2] = vgetq_lane_s32(_sum0, 2);
            //     //     outptr0[out_hstep * 3] = vgetq_lane_s32(_sum0, 3);
            //     //     outptr0[out_hstep * 4] = vgetq_lane_s32(_sum1, 0);
            //     //     outptr0[out_hstep * 5] = vgetq_lane_s32(_sum1, 1);
            //     //     outptr0[out_hstep * 6] = vgetq_lane_s32(_sum1, 2);
            //     //     outptr0[out_hstep * 7] = vgetq_lane_s32(_sum1, 3);
            //     //     outptr0++;
            //     // }
            // }
            // else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

#if NCNN_GNU_INLINE_ASM
            asm volatile(
                "cmp    %w7, #0                     \n"
                "beq    0f                          \n"

                "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0] \n"
                "sub    %0, %0, #64                 \n"
                "b      1f                          \n"

                "0:                                 \n"
                "eor    v16.16b, v16.16b, v16.16b   \n"
                "eor    v17.16b, v17.16b, v17.16b   \n"
                "eor    v18.16b, v18.16b, v18.16b   \n"
                "eor    v19.16b, v19.16b, v19.16b   \n"
                "eor    v20.16b, v20.16b, v20.16b   \n"
                "eor    v21.16b, v21.16b, v21.16b   \n"
                "eor    v22.16b, v22.16b, v22.16b   \n"
                "eor    v23.16b, v23.16b, v23.16b   \n"

                "1:                                 \n"
#if __ARM_FEATURE_DOTPROD
                "lsr    w4, %w6, #3                 \n" // w4 = max_kk >> 3
                "cmp    w4, #0                      \n"
                "beq    101f                        \n"

#if __ARM_FEATURE_MATMUL_INT8
                "eor    v24.16b, v24.16b, v24.16b   \n"
                "eor    v25.16b, v25.16b, v25.16b   \n"
                "eor    v26.16b, v26.16b, v26.16b   \n"
                "eor    v27.16b, v27.16b, v27.16b   \n"
                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v29.16b, v29.16b, v29.16b   \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"
                "eor    v31.16b, v31.16b, v31.16b   \n"
#endif // __ARM_FEATURE_MATMUL_INT8

                "2:                                 \n"
                "ld1    {v0.16b, v1.16b}, [%1], #32 \n"
                "ld1    {v2.16b, v3.16b, v4.16b, v5.16b}, [%2], #64 \n"

#if __ARM_FEATURE_MATMUL_INT8
                "smmla  v24.4s, v0.16b, v2.16b      \n"
                "smmla  v25.4s, v1.16b, v2.16b      \n"
                "smmla  v26.4s, v0.16b, v3.16b      \n"
                "smmla  v27.4s, v1.16b, v3.16b      \n"
                "subs   w4, w4, #1                  \n"
                "smmla  v28.4s, v0.16b, v4.16b      \n"
                "smmla  v29.4s, v1.16b, v4.16b      \n"
                "smmla  v30.4s, v0.16b, v5.16b      \n"
                "smmla  v31.4s, v1.16b, v5.16b      \n"
#else  // __ARM_FEATURE_MATMUL_INT8
                "sdot   v16.4s, v0.16b, v2.4b[0]    \n"
                "sdot   v17.4s, v0.16b, v2.4b[1]    \n"
                "sdot   v18.4s, v0.16b, v2.4b[2]    \n"
                "sdot   v19.4s, v0.16b, v2.4b[3]    \n"
                "sdot   v20.4s, v0.16b, v3.4b[0]    \n"
                "sdot   v21.4s, v0.16b, v3.4b[1]    \n"
                "sdot   v22.4s, v0.16b, v3.4b[2]    \n"
                "sdot   v23.4s, v0.16b, v3.4b[3]    \n"
                "subs   w4, w4, #1                  \n"
                "sdot   v16.4s, v1.16b, v4.4b[0]    \n"
                "sdot   v17.4s, v1.16b, v4.4b[1]    \n"
                "sdot   v18.4s, v1.16b, v4.4b[2]    \n"
                "sdot   v19.4s, v1.16b, v4.4b[3]    \n"
                "sdot   v20.4s, v1.16b, v5.4b[0]    \n"
                "sdot   v21.4s, v1.16b, v5.4b[1]    \n"
                "sdot   v22.4s, v1.16b, v5.4b[2]    \n"
                "sdot   v23.4s, v1.16b, v5.4b[3]    \n"
#endif // __ARM_FEATURE_MATMUL_INT8
                "bne    2b                          \n"

#if __ARM_FEATURE_MATMUL_INT8
                "uzp1   v0.4s, v24.4s, v25.4s       \n"
                "uzp2   v1.4s, v24.4s, v25.4s       \n"
                "uzp1   v2.4s, v26.4s, v27.4s       \n"
                "uzp2   v3.4s, v26.4s, v27.4s       \n"
                "uzp1   v4.4s, v28.4s, v29.4s       \n"
                "uzp2   v5.4s, v28.4s, v29.4s       \n"
                "uzp1   v6.4s, v30.4s, v31.4s       \n"
                "uzp2   v7.4s, v30.4s, v31.4s       \n"

                "add    v16.4s, v16.4s, v0.4s       \n"
                "add    v17.4s, v17.4s, v1.4s       \n"
                "add    v18.4s, v18.4s, v2.4s       \n"
                "add    v19.4s, v19.4s, v3.4s       \n"
                "add    v20.4s, v20.4s, v4.4s       \n"
                "add    v21.4s, v21.4s, v5.4s       \n"
                "add    v22.4s, v22.4s, v6.4s       \n"
                "add    v23.4s, v23.4s, v7.4s       \n"
#endif // __ARM_FEATURE_MATMUL_INT8

                "101:                               \n"
                "and    w4, %w6, #4                 \n" // w4 = remain = max_kk & 4
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                // kk += 4 part
                "ld1    {v0.16b}, [%1], #16         \n"
                "ld1    {v2.16b, v3.16b}, [%2], #32 \n"
                "sdot   v16.4s, v0.16b, v2.4b[0]    \n"
                "sdot   v17.4s, v0.16b, v2.4b[1]    \n"
                "sdot   v18.4s, v0.16b, v2.4b[2]    \n"
                "sdot   v19.4s, v0.16b, v2.4b[3]    \n"
                "sdot   v20.4s, v0.16b, v3.4b[0]    \n"
                "sdot   v21.4s, v0.16b, v3.4b[1]    \n"
                "sdot   v22.4s, v0.16b, v3.4b[2]    \n"
                "sdot   v23.4s, v0.16b, v3.4b[3]    \n"
#else  // __ARM_FEATURE_DOTPROD
                "lsr    w4, %w6, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                "2:                                 \n"
                "ld1    {v0.16b}, [%1], #16         \n"
                "ld1    {v4.16b, v5.16b}, [%2], #32 \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull2 v9.8h, v0.16b, v5.16b       \n"
                "rev64  v2.4s, v0.4s                \n"
                "smull  v10.8h, v2.8b, v4.8b        \n"
                "smull2 v11.8h, v2.16b, v5.16b      \n"
                "rev64  v6.8h, v4.8h                \n"
                "smull  v12.8h, v0.8b, v6.8b        \n"
                "smull  v14.8h, v2.8b, v6.8b        \n"
                "rev64  v7.8h, v5.8h                \n"
                "smull2 v13.8h, v0.16b, v7.16b      \n"
                "smull2 v15.8h, v2.16b, v7.16b      \n"
                "ext    v1.16b, v0.16b, v0.16b, #8  \n"
                "ext    v3.16b, v2.16b, v2.16b, #8  \n"
                "smlal  v8.8h, v1.8b, v5.8b         \n"
                "smlal2 v9.8h, v1.16b, v4.16b       \n"
                "smlal  v10.8h, v3.8b, v5.8b        \n"
                "smlal2 v11.8h, v3.16b, v4.16b      \n"
                "smlal  v12.8h, v1.8b, v7.8b        \n"
                "smlal  v14.8h, v3.8b, v7.8b        \n"
                "smlal2 v13.8h, v1.16b, v6.16b      \n"
                "smlal2 v15.8h, v3.16b, v6.16b      \n"
                "subs   w4, w4, #1                  \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
                "sadalp v20.4s, v12.8h              \n"
                "sadalp v22.4s, v14.8h              \n"
                "sadalp v21.4s, v13.8h              \n"
                "sadalp v23.4s, v15.8h              \n"
                "bne    2b                          \n"
#endif // __ARM_FEATURE_DOTPROD

                "3:                                 \n"
                "and    w4, %w6, #2                 \n" // w4 = remain = max_kk & 2
                "cmp    w4, #0                      \n"
                "beq    4f                          \n"

                // kk += 2 part
#if __ARM_FEATURE_DOTPROD
                "ld1    {v0.8b}, [%1], #8           \n"
                "ld1    {v1.16b}, [%2], #16         \n"
                "dup    v4.8h, v1.h[0]              \n"
                "dup    v5.8h, v1.h[1]              \n"
                "dup    v6.8h, v1.h[2]              \n"
                "dup    v7.8h, v1.h[3]              \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull  v9.8h, v0.8b, v5.8b         \n"
                "smull  v10.8h, v0.8b, v6.8b        \n"
                "smull  v11.8h, v0.8b, v7.8b        \n"
                "dup    v4.8h, v1.h[4]              \n"
                "dup    v5.8h, v1.h[5]              \n"
                "dup    v6.8h, v1.h[6]              \n"
                "dup    v7.8h, v1.h[7]              \n"
                "smull  v12.8h, v0.8b, v4.8b        \n"
                "smull  v13.8h, v0.8b, v5.8b        \n"
                "smull  v14.8h, v0.8b, v6.8b        \n"
                "smull  v15.8h, v0.8b, v7.8b        \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
                "sadalp v20.4s, v12.8h              \n"
                "sadalp v21.4s, v13.8h              \n"
                "sadalp v22.4s, v14.8h              \n"
                "sadalp v23.4s, v15.8h              \n"
#else  // __ARM_FEATURE_DOTPROD
                "ld1r   {v0.2d}, [%1]               \n"
                "add    %1, %1, #8                  \n"
                "ld1    {v2.16b}, [%2], #16         \n"
                "rev64  v1.4s, v0.4s                \n"
                "rev64  v3.8h, v2.8h                \n"
                "smull  v8.8h, v0.8b, v2.8b         \n"
                "smull2 v9.8h, v0.16b, v2.16b       \n"
                "smull  v10.8h, v1.8b, v2.8b        \n"
                "smull2 v11.8h, v1.16b, v2.16b      \n"
                "smull  v12.8h, v0.8b, v3.8b        \n"
                "smull2 v13.8h, v0.16b, v3.16b      \n"
                "smull  v14.8h, v1.8b, v3.8b        \n"
                "smull2 v15.8h, v1.16b, v3.16b      \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
                "sadalp v20.4s, v12.8h              \n"
                "sadalp v21.4s, v13.8h              \n"
                "sadalp v22.4s, v14.8h              \n"
                "sadalp v23.4s, v15.8h              \n"
#endif // __ARM_FEATURE_DOTPROD

                "4:                                 \n"
                "and    w4, %w6, #1                 \n" // w4 = remain = max_kk & 1
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                // kk += 1 part
#if __ARM_FEATURE_DOTPROD
                "ld1r   {v0.2s}, [%1]               \n"
                "ld1    {v1.8b}, [%2], #8           \n"
                "add    %1, %1, #4                  \n"
                "dup    v8.8h, v1.h[0]              \n"
                "dup    v9.8h, v1.h[1]              \n"
                "dup    v10.8h, v1.h[2]             \n"
                "dup    v11.8h, v1.h[3]             \n"
                "uzp1   v2.8b, v8.8b, v9.8b         \n"
                "uzp2   v3.8b, v8.8b, v9.8b         \n"
                "uzp1   v4.8b, v10.8b, v11.8b       \n"
                "uzp2   v5.8b, v10.8b, v11.8b       \n"
                "smull  v8.8h, v0.8b, v2.8b         \n"
                "smull  v9.8h, v0.8b, v3.8b         \n"
                "smull  v10.8h, v0.8b, v4.8b        \n"
                "smull  v11.8h, v0.8b, v5.8b        \n"
                "saddw  v16.4s, v16.4s, v8.4h       \n"
                "saddw  v17.4s, v17.4s, v9.4h       \n"
                "saddw2 v18.4s, v18.4s, v8.8h       \n"
                "saddw2 v19.4s, v19.4s, v9.8h       \n"
                "saddw  v20.4s, v20.4s, v10.4h      \n"
                "saddw  v21.4s, v21.4s, v11.4h      \n"
                "saddw2 v22.4s, v22.4s, v10.8h      \n"
                "saddw2 v23.4s, v23.4s, v11.8h      \n"
#else  // __ARM_FEATURE_DOTPROD
                "ld1r   {v0.2s}, [%1]               \n"
                "ld1    {v2.8b}, [%2], #8           \n"
                "add    %1, %1, #4                  \n"
                "ext    v1.8b, v0.8b, v0.8b, #2     \n"
                "rev32  v3.8b, v2.8b                \n"
                "smull  v8.8h, v0.8b, v2.8b         \n"
                "smull  v9.8h, v1.8b, v2.8b         \n"
                "smull  v10.8h, v0.8b, v3.8b        \n"
                "smull  v11.8h, v1.8b, v3.8b        \n"
                "saddw  v16.4s, v16.4s, v8.4h       \n"
                "saddw2 v17.4s, v17.4s, v8.8h       \n"
                "saddw  v18.4s, v18.4s, v9.4h       \n"
                "saddw2 v19.4s, v19.4s, v9.8h       \n"
                "saddw  v20.4s, v20.4s, v10.4h      \n"
                "saddw2 v21.4s, v21.4s, v10.8h      \n"
                "saddw  v22.4s, v22.4s, v11.4h      \n"
                "saddw2 v23.4s, v23.4s, v11.8h      \n"
#endif // __ARM_FEATURE_DOTPROD

                "5:                                 \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB)      // %2
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "r"(max_kk), // %6
                "r"(k)       // %7
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else // NCNN_GNU_INLINE_ASM
            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;
            int32x4_t _sum4;
            int32x4_t _sum5;
            int32x4_t _sum6;
            int32x4_t _sum7;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
                _sum4 = vdupq_n_s32(0);
                _sum5 = vdupq_n_s32(0);
                _sum6 = vdupq_n_s32(0);
                _sum7 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
                _sum4 = vld1q_s32(outptr + 16);
                _sum5 = vld1q_s32(outptr + 20);
                _sum6 = vld1q_s32(outptr + 24);
                _sum7 = vld1q_s32(outptr + 28);
            }

            int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
            {
                int32x4_t _sum00 = vdupq_n_s32(0);
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum10 = vdupq_n_s32(0);
                int32x4_t _sum11 = vdupq_n_s32(0);
                int32x4_t _sum20 = vdupq_n_s32(0);
                int32x4_t _sum21 = vdupq_n_s32(0);
                int32x4_t _sum30 = vdupq_n_s32(0);
                int32x4_t _sum31 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA1 = vld1q_s8(pA + 16);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);
                    int8x16_t _pB2 = vld1q_s8(pB + 32);
                    int8x16_t _pB3 = vld1q_s8(pB + 48);

                    // aaaaaaaa bbbbbbbb cccccccc dddddddd

                    // 00000000 11111111 22222222 33333333
                    // 44444444 55555555 66666666 77777777

                    _sum00 = vmmlaq_s32(_sum00, _pA0, _pB0);
                    _sum01 = vmmlaq_s32(_sum01, _pA1, _pB0);
                    _sum10 = vmmlaq_s32(_sum10, _pA0, _pB1);
                    _sum11 = vmmlaq_s32(_sum11, _pA1, _pB1);
                    _sum20 = vmmlaq_s32(_sum20, _pA0, _pB2);
                    _sum21 = vmmlaq_s32(_sum21, _pA1, _pB2);
                    _sum30 = vmmlaq_s32(_sum30, _pA0, _pB3);
                    _sum31 = vmmlaq_s32(_sum31, _pA1, _pB3);

                    // a0 a1 b0 b1
                    // c0 c1 d0 d1
                    // a2 a3 b2 b3
                    // c2 c3 d2 d3
                    // a4 a5 b4 b5
                    // c4 c5 d4 d5
                    // a6 a7 b6 b7
                    // c6 c7 d6 d7

                    pA += 32;
                    pB += 64;
                }
                int32x4x2_t _ss0 = vuzpq_s32(_sum00, _sum01);
                int32x4x2_t _ss1 = vuzpq_s32(_sum10, _sum11);
                int32x4x2_t _ss2 = vuzpq_s32(_sum20, _sum21);
                int32x4x2_t _ss3 = vuzpq_s32(_sum30, _sum31);
                _sum0 = vaddq_s32(_sum0, _ss0.val[0]);
                _sum1 = vaddq_s32(_sum1, _ss0.val[1]);
                _sum2 = vaddq_s32(_sum2, _ss1.val[0]);
                _sum3 = vaddq_s32(_sum3, _ss1.val[1]);
                _sum4 = vaddq_s32(_sum4, _ss2.val[0]);
                _sum5 = vaddq_s32(_sum5, _ss2.val[1]);
                _sum6 = vaddq_s32(_sum6, _ss3.val[0]);
                _sum7 = vaddq_s32(_sum7, _ss3.val[1]);
            }
#elif __ARM_FEATURE_DOTPROD
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA1 = vld1q_s8(pA + 16);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB1 = vld1q_s8(pB + 16);
                int8x16_t _pB2 = vld1q_s8(pB + 32);
                int8x16_t _pB3 = vld1q_s8(pB + 48);

                _sum0 = vdotq_laneq_s32(_sum0, _pA0, _pB0, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA0, _pB0, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA0, _pB0, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA0, _pB0, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _pA0, _pB1, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _pA0, _pB1, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _pA0, _pB1, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _pA0, _pB1, 3);

                _sum0 = vdotq_laneq_s32(_sum0, _pA1, _pB2, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA1, _pB2, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA1, _pB2, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA1, _pB2, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _pA1, _pB3, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _pA1, _pB3, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _pA1, _pB3, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _pA1, _pB3, 3);

                pA += 32;
                pB += 64;
            }
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB1 = vld1q_s8(pB + 16);

                _sum0 = vdotq_laneq_s32(_sum0, _pA, _pB0, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA, _pB0, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA, _pB0, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA, _pB0, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _pA, _pB1, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _pA, _pB1, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _pA, _pB1, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _pA, _pB1, 3);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA02 = vld1q_s8(pA);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB2 = vld1q_s8(pB + 16);

                int8x16_t _pA13 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA02)));

                int8x16_t _pB1 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB0)));
                int8x16_t _pB3 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB2)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA02), vget_low_s8(_pB0));
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA02), vget_high_s8(_pB0));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA13), vget_low_s8(_pB0));
                int16x8_t _s3 = vmull_s8(vget_low_s8(_pA13), vget_high_s8(_pB0));
                int16x8_t _s4 = vmull_s8(vget_low_s8(_pA02), vget_low_s8(_pB1));
                int16x8_t _s5 = vmull_s8(vget_low_s8(_pA02), vget_high_s8(_pB1));
                int16x8_t _s6 = vmull_s8(vget_low_s8(_pA13), vget_low_s8(_pB1));
                int16x8_t _s7 = vmull_s8(vget_low_s8(_pA13), vget_high_s8(_pB1));

                _s0 = vmlal_s8(_s0, vget_high_s8(_pA02), vget_low_s8(_pB2));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA02), vget_high_s8(_pB2));
                _s2 = vmlal_s8(_s2, vget_high_s8(_pA13), vget_low_s8(_pB2));
                _s3 = vmlal_s8(_s3, vget_high_s8(_pA13), vget_high_s8(_pB2));
                _s4 = vmlal_s8(_s4, vget_high_s8(_pA02), vget_low_s8(_pB3));
                _s5 = vmlal_s8(_s5, vget_high_s8(_pA02), vget_high_s8(_pB3));
                _s6 = vmlal_s8(_s6, vget_high_s8(_pA13), vget_low_s8(_pB3));
                _s7 = vmlal_s8(_s7, vget_high_s8(_pA13), vget_high_s8(_pB3));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
#endif // __ARM_FEATURE_DOTPROD

                pA += 16;
                pB += 32;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vld1_s8(pA);
                int8x16_t _pB01 = vld1q_s8(pB);

                // aabbccdd

                // 00112233 44556677

                int16x8_t _s0 = vmull_s8(_pA0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB01)), 0)));
                int16x8_t _s1 = vmull_s8(_pA0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB01)), 1)));
                int16x8_t _s2 = vmull_s8(_pA0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB01)), 2)));
                int16x8_t _s3 = vmull_s8(_pA0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB01)), 3)));
                int16x8_t _s4 = vmull_s8(_pA0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB01)), 0)));
                int16x8_t _s5 = vmull_s8(_pA0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB01)), 1)));
                int16x8_t _s6 = vmull_s8(_pA0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB01)), 2)));
                int16x8_t _s7 = vmull_s8(_pA0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB01)), 3)));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vld1_s8(pA);
                int8x16_t _pB0 = vld1q_s8(pB);

                // aabbccdd
                // ccddaabb

                int8x8_t _pA1 = vreinterpret_s8_s32(vrev64_s32(vreinterpret_s32_s8(_pA0)));

                // 00112233 44556677
                // 33221100 77665544

                int8x16_t _pB1 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB0)));

                int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                int16x8_t _s1 = vmull_s8(_pA0, vget_high_s8(_pB0));
                int16x8_t _s2 = vmull_s8(_pA1, vget_low_s8(_pB0));
                int16x8_t _s3 = vmull_s8(_pA1, vget_high_s8(_pB0));
                int16x8_t _s4 = vmull_s8(_pA0, vget_low_s8(_pB1));
                int16x8_t _s5 = vmull_s8(_pA0, vget_high_s8(_pB1));
                int16x8_t _s6 = vmull_s8(_pA1, vget_low_s8(_pB1));
                int16x8_t _s7 = vmull_s8(_pA1, vget_high_s8(_pB1));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
#endif // __ARM_FEATURE_DOTPROD

                pA += 8;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pAA = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB = vld1_s8(pB);

                // abcdabcd
                // 01234567  ->  01010101 23232323 45454545 67676767
                int8x8_t _pB0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 0));
                int8x8_t _pB2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 1));
                int8x8_t _pB4 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 2));
                int8x8_t _pB6 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 3));

                int8x8x2_t _pB0123 = vuzp_s8(_pB0, _pB2);
                int8x8x2_t _pB4567 = vuzp_s8(_pB4, _pB6);

                int16x8_t _s02 = vmull_s8(_pAA, _pB0123.val[0]);
                int16x8_t _s13 = vmull_s8(_pAA, _pB0123.val[1]);
                int16x8_t _s46 = vmull_s8(_pAA, _pB4567.val[0]);
                int16x8_t _s57 = vmull_s8(_pAA, _pB4567.val[1]);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s02));
                _sum1 = vaddw_s16(_sum1, vget_low_s16(_s13));
                _sum2 = vaddw_s16(_sum2, vget_high_s16(_s02));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s13));
                _sum4 = vaddw_s16(_sum4, vget_low_s16(_s46));
                _sum5 = vaddw_s16(_sum5, vget_low_s16(_s57));
                _sum6 = vaddw_s16(_sum6, vget_high_s16(_s46));
                _sum7 = vaddw_s16(_sum7, vget_high_s16(_s57));
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB0 = vld1_s8(pB);

                // abcd abcd
                // cdab cdab

                int8x8_t _pA1 = vext_s8(_pA0, _pA0, 2);

                // 0123 4567
                // 3210 7654

                int8x8_t _pB1 = vrev32_s8(_pB0);

                int16x8_t _s01 = vmull_s8(_pA0, _pB0);
                int16x8_t _s23 = vmull_s8(_pA1, _pB0);
                int16x8_t _s45 = vmull_s8(_pA0, _pB1);
                int16x8_t _s67 = vmull_s8(_pA1, _pB1);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s23));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s23));
                _sum4 = vaddw_s16(_sum4, vget_low_s16(_s45));
                _sum5 = vaddw_s16(_sum5, vget_high_s16(_s45));
                _sum6 = vaddw_s16(_sum6, vget_low_s16(_s67));
                _sum7 = vaddw_s16(_sum7, vget_high_s16(_s67));
#endif // __ARM_FEATURE_DOTPROD

                pA += 4;
                pB += 8;
            }

            //             if (k_end)
            //             {
            // #if __ARM_FEATURE_DOTPROD
            //                 // from
            //                 //      a0 b0 c0 d0
            //                 //      a1 b1 c1 d1
            //                 //      a2 b2 c2 d2
            //                 //      a3 b3 c3 d3
            //                 //      a4 b4 c4 d4
            //                 //      a5 b5 c5 d5
            //                 //      a6 b6 c6 d6
            //                 //      a7 b7 c7 d7
            //                 if (out_elempack == 4)
            //                 {
            //                     vst1q_s32(outptr0, _sum0);
            //                     vst1q_s32(outptr0 + 4, _sum1);
            //                     vst1q_s32(outptr0 + 8, _sum2);
            //                     vst1q_s32(outptr0 + 12, _sum3);
            //                     vst1q_s32(outptr0 + 16, _sum4);
            //                     vst1q_s32(outptr0 + 20, _sum5);
            //                     vst1q_s32(outptr0 + 24, _sum6);
            //                     vst1q_s32(outptr0 + 28, _sum7);
            //                     outptr0 += 32;
            //                 }
            //                 // if (out_elempack == 1)
            //                 // {
            //                 //     // to
            //                 //     //      a0 a1 a2 a3
            //                 //     //      a4 a5 a6 a7
            //                 //     //      b0 b1 b2 b3
            //                 //     //      b4 b5 b6 b7
            //                 //     //      c0 c1 c2 c3
            //                 //     //      c4 c5 c6 c7
            //                 //     //      d0 d1 d2 d3
            //                 //     //      d4 d5 d6 d7
            //                 //     {
            //                 //         int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
            //                 //         int32x4x2_t _t1 = vzipq_s32(_sum2, _sum3);
            //                 //         int32x4x2_t _t2 = vzipq_s32(_sum4, _sum5);
            //                 //         int32x4x2_t _t3 = vzipq_s32(_sum6, _sum7);
            //                 //         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
            //                 //         _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
            //                 //         _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
            //                 //         _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
            //                 //         _sum4 = vcombine_s32(vget_low_s32(_t0.val[1]), vget_low_s32(_t1.val[1]));
            //                 //         _sum5 = vcombine_s32(vget_low_s32(_t2.val[1]), vget_low_s32(_t3.val[1]));
            //                 //         _sum6 = vcombine_s32(vget_high_s32(_t0.val[1]), vget_high_s32(_t1.val[1]));
            //                 //         _sum7 = vcombine_s32(vget_high_s32(_t2.val[1]), vget_high_s32(_t3.val[1]));
            //                 //     }
            //                 //
            //                 //     vst1q_s32(outptr0, _sum0);
            //                 //     vst1q_s32(outptr0 + 4, _sum1);
            //                 //     vst1q_s32(outptr0 + out_hstep, _sum2);
            //                 //     vst1q_s32(outptr0 + out_hstep + 4, _sum3);
            //                 //     vst1q_s32(outptr0 + out_hstep * 2, _sum4);
            //                 //     vst1q_s32(outptr0 + out_hstep * 2 + 4, _sum5);
            //                 //     vst1q_s32(outptr0 + out_hstep * 3, _sum6);
            //                 //     vst1q_s32(outptr0 + out_hstep * 3 + 4, _sum7);
            //                 //     outptr0 += 8;
            //                 // }
            // #else  // __ARM_FEATURE_DOTPROD
            //
            //                 // from
            //                 //      a0 b1 c2 d3
            //                 //      a4 b5 c6 d7
            //                 //      c0 d1 a2 b3
            //                 //      c4 d5 a6 b7
            //                 //      a3 b2 c1 d0
            //                 //      a7 b6 c5 d4
            //                 //      c3 d2 a1 b0
            //                 //      c7 d6 a5 b4
            //                 if (out_elempack == 4)
            //                 {
            //                     // to
            //                     //      a0 b0 c0 d0
            //                     //      a1 b1 c1 d1
            //                     //      a2 b2 c2 d2
            //                     //      a3 b3 c3 d3
            //                     //      a4 b4 c4 d4
            //                     //      a5 b5 c5 d5
            //                     //      a6 b6 c6 d6
            //                     //      a7 b7 c7 d7
            //                     {
            //                         _sum4 = vrev64q_s32(_sum4);
            //                         _sum5 = vrev64q_s32(_sum5);
            //                         _sum6 = vrev64q_s32(_sum6);
            //                         _sum7 = vrev64q_s32(_sum7);
            //                         _sum4 = vextq_s32(_sum4, _sum4, 2);
            //                         _sum5 = vextq_s32(_sum5, _sum5, 2);
            //                         _sum6 = vextq_s32(_sum6, _sum6, 2);
            //                         _sum7 = vextq_s32(_sum7, _sum7, 2);
            //                         int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
            //                         int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
            //                         int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
            //                         int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
            //                         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
            //                         _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
            //                         _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
            //                         _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
            //                         _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
            //                         _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
            //                         _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
            //                         _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
            //                         _sum1 = vrev64q_s32(_sum1);
            //                         _sum3 = vrev64q_s32(_sum3);
            //                         _sum5 = vrev64q_s32(_sum5);
            //                         _sum7 = vrev64q_s32(_sum7);
            //                     }
            //
            //                     vst1q_s32(outptr0, _sum0);
            //                     vst1q_s32(outptr0 + 4, _sum1);
            //                     vst1q_s32(outptr0 + 8, _sum2);
            //                     vst1q_s32(outptr0 + 12, _sum3);
            //                     vst1q_s32(outptr0 + 16, _sum4);
            //                     vst1q_s32(outptr0 + 20, _sum5);
            //                     vst1q_s32(outptr0 + 24, _sum6);
            //                     vst1q_s32(outptr0 + 28, _sum7);
            //                     outptr0 += 32;
            //                 }
            //                 // if (out_elempack == 1)
            //                 // {
            //                 //     // to
            //                 //     //      a0 a1 a2 a3
            //                 //     //      a4 a5 a6 a7
            //                 //     //      b0 b1 b2 b3
            //                 //     //      b4 b5 b6 b7
            //                 //     //      c0 c1 c2 c3
            //                 //     //      c4 c5 c6 c7
            //                 //     //      d0 d1 d2 d3
            //                 //     //      d4 d5 d6 d7
            //                 //     {
            //                 //         _sum2 = vextq_s32(_sum2, _sum2, 2);
            //                 //         _sum3 = vextq_s32(_sum3, _sum3, 2);
            //                 //         _sum6 = vextq_s32(_sum6, _sum6, 2);
            //                 //         _sum7 = vextq_s32(_sum7, _sum7, 2);
            //                 //         int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
            //                 //         int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
            //                 //         int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
            //                 //         int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
            //                 //         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
            //                 //         _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
            //                 //         _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
            //                 //         _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
            //                 //         _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
            //                 //         _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
            //                 //         _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
            //                 //         _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
            //                 //         _sum2 = vrev64q_s32(_sum2);
            //                 //         _sum3 = vrev64q_s32(_sum3);
            //                 //         _sum6 = vrev64q_s32(_sum6);
            //                 //         _sum7 = vrev64q_s32(_sum7);
            //                 //     }
            //                 //
            //                 //     vst1q_s32(outptr0, _sum0);
            //                 //     vst1q_s32(outptr0 + 4, _sum1);
            //                 //     vst1q_s32(outptr0 + out_hstep, _sum2);
            //                 //     vst1q_s32(outptr0 + out_hstep + 4, _sum3);
            //                 //     vst1q_s32(outptr0 + out_hstep * 2, _sum4);
            //                 //     vst1q_s32(outptr0 + out_hstep * 2 + 4, _sum5);
            //                 //     vst1q_s32(outptr0 + out_hstep * 3, _sum6);
            //                 //     vst1q_s32(outptr0 + out_hstep * 3 + 4, _sum7);
            //                 //     outptr0 += 8;
            //                 // }
            // #endif // __ARM_FEATURE_DOTPROD
            //             }
            //             else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
                vst1q_s32(outptr + 16, _sum4);
                vst1q_s32(outptr + 20, _sum5);
                vst1q_s32(outptr + 24, _sum6);
                vst1q_s32(outptr + 28, _sum7);
            }

            outptr += 32;
#endif // NCNN_GNU_INLINE_ASM
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            asm volatile(
                "cmp    %w7, #0                     \n"
                "beq    0f                          \n"

                "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0] \n"
                "b      1f                          \n"

                "0:                                 \n"
                "eor    v16.16b, v16.16b, v16.16b   \n"
                "eor    v17.16b, v17.16b, v17.16b   \n"
                "eor    v18.16b, v18.16b, v18.16b   \n"
                "eor    v19.16b, v19.16b, v19.16b   \n"

                "1:                                 \n"
#if __ARM_FEATURE_DOTPROD
                "lsr    w4, %w6, #3                 \n" // w4 = max_kk >> 3
                "cmp    w4, #0                      \n"
                "beq    101f                        \n"

#if __ARM_FEATURE_MATMUL_INT8
                "eor    v24.16b, v24.16b, v24.16b   \n"
                "eor    v25.16b, v25.16b, v25.16b   \n"
                "eor    v26.16b, v26.16b, v26.16b   \n"
                "eor    v27.16b, v27.16b, v27.16b   \n"
#endif // __ARM_FEATURE_MATMUL_INT8

                "2:                                 \n"
                "ld1    {v0.16b, v1.16b}, [%1], #32 \n"
                "ld1    {v4.16b, v5.16b}, [%2], #32 \n"

#if __ARM_FEATURE_MATMUL_INT8
                "smmla  v24.4s, v0.16b, v4.16b      \n"
                "smmla  v25.4s, v1.16b, v4.16b      \n"
                "subs   w4, w4, #1                  \n"
                "smmla  v26.4s, v0.16b, v5.16b      \n"
                "smmla  v27.4s, v1.16b, v5.16b      \n"
#else  // __ARM_FEATURE_MATMUL_INT8
                "sdot   v16.4s, v0.16b, v4.4b[0]    \n"
                "sdot   v17.4s, v0.16b, v4.4b[1]    \n"
                "sdot   v18.4s, v0.16b, v4.4b[2]    \n"
                "sdot   v19.4s, v0.16b, v4.4b[3]    \n"
                "subs   w4, w4, #1                  \n"
                "sdot   v16.4s, v1.16b, v5.4b[0]    \n"
                "sdot   v17.4s, v1.16b, v5.4b[1]    \n"
                "sdot   v18.4s, v1.16b, v5.4b[2]    \n"
                "sdot   v19.4s, v1.16b, v5.4b[3]    \n"
#endif // __ARM_FEATURE_MATMUL_INT8
                "bne    2b                          \n"

#if __ARM_FEATURE_MATMUL_INT8
                "uzp1   v0.4s, v24.4s, v25.4s       \n"
                "uzp2   v1.4s, v24.4s, v25.4s       \n"
                "uzp1   v2.4s, v26.4s, v27.4s       \n"
                "uzp2   v3.4s, v26.4s, v27.4s       \n"

                "add    v16.4s, v16.4s, v0.4s       \n"
                "add    v17.4s, v17.4s, v1.4s       \n"
                "add    v18.4s, v18.4s, v2.4s       \n"
                "add    v19.4s, v19.4s, v3.4s       \n"
#endif // __ARM_FEATURE_MATMUL_INT8

                "101:                               \n"
                "and    w4, %w6, #4                 \n" // w4 = remain = max_kk & 4
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                // kk += 4 part
                "ld1    {v0.16b}, [%1], #16         \n"
                "ld1    {v2.16b}, [%2], #16         \n"
                "sdot   v16.4s, v0.16b, v2.4b[0]    \n"
                "sdot   v17.4s, v0.16b, v2.4b[1]    \n"
                "sdot   v18.4s, v0.16b, v2.4b[2]    \n"
                "sdot   v19.4s, v0.16b, v2.4b[3]    \n"
#else  // __ARM_FEATURE_DOTPROD
                "lsr    w4, %w6, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                "2:                                 \n"
                "ld1    {v0.16b}, [%1], #16         \n"
                "ld1    {v4.16b}, [%2], #16         \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "rev64  v1.4s, v0.4s                \n"
                "smull  v9.8h, v1.8b, v4.8b         \n"
                "rev64  v5.8h, v4.8h                \n"
                "smull  v10.8h, v0.8b, v5.8b        \n"
                "smull  v11.8h, v1.8b, v5.8b        \n"
                "smlal2 v8.8h, v0.16b, v4.16b       \n"
                "smlal2 v9.8h, v1.16b, v4.16b       \n"
                "smlal2 v10.8h, v0.16b, v5.16b      \n"
                "smlal2 v11.8h, v1.16b, v5.16b      \n"
                "subs   w4, w4, #1                  \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
                "bne    2b                          \n"
#endif // __ARM_FEATURE_DOTPROD

                "3:                                 \n"
                "and    w4, %w6, #2                 \n" // w4 = remain = max_kk & 2
                "cmp    w4, #0                      \n"
                "beq    4f                          \n"

                // kk += 2 part
#if __ARM_FEATURE_DOTPROD
                "ld1    {v0.8b}, [%1], #8           \n"
                "ld1    {v1.8b}, [%2], #8           \n"
                "dup    v4.4h, v1.h[0]              \n"
                "dup    v5.4h, v1.h[1]              \n"
                "dup    v6.4h, v1.h[2]              \n"
                "dup    v7.4h, v1.h[3]              \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull  v9.8h, v0.8b, v5.8b         \n"
                "smull  v10.8h, v0.8b, v6.8b        \n"
                "smull  v11.8h, v0.8b, v7.8b        \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
#else  // __ARM_FEATURE_DOTPROD
                "ld1    {v0.8b}, [%1], #8           \n"
                "ld1    {v2.8b}, [%2], #8           \n"
                "ext    v1.8b, v0.8b, v0.8b, #4     \n"
                "rev64  v3.4h, v2.4h                \n"
                "smull  v8.8h, v0.8b, v2.8b         \n"
                "smull  v9.8h, v1.8b, v2.8b         \n"
                "smull  v10.8h, v0.8b, v3.8b        \n"
                "smull  v11.8h, v1.8b, v3.8b        \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
#endif // __ARM_FEATURE_DOTPROD

                "4:                                 \n"
                "and    w4, %w6, #1                 \n" // w4 = remain = max_kk & 1
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                // kk += 1 part
#if __ARM_FEATURE_DOTPROD
                "ld1r   {v0.2s}, [%1]               \n"
                "ld1r   {v1.2s}, [%2]               \n"
                "add    %1, %1, #4                  \n"
                "add    %2, %2, #4                  \n"
                "zip1   v1.8b, v1.8b, v1.8b         \n"
                "zip1   v2.4h, v1.4h, v1.4h         \n"
                "zip2   v3.4h, v1.4h, v1.4h         \n"
                "smull  v8.8h, v0.8b, v2.8b         \n"
                "smull  v9.8h, v0.8b, v3.8b         \n"
                "saddw  v16.4s, v16.4s, v8.4h       \n"
                "saddw2 v17.4s, v17.4s, v8.8h       \n"
                "saddw  v18.4s, v18.4s, v9.4h       \n"
                "saddw2 v19.4s, v19.4s, v9.8h       \n"
#else  // __ARM_FEATURE_DOTPROD
                "ld1    {v0.8b}, [%1]               \n"
                "ld1r   {v4.2s}, [%2]               \n"
                "add    %1, %1, #4                  \n"
                "add    %2, %2, #4                  \n"
                "rev32  v1.4h, v0.4h                \n"
                "zip1   v0.2s, v0.2s, v1.2s         \n"
                "rev32  v5.8b, v4.8b                \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull  v9.8h, v0.8b, v5.8b         \n"
                "saddw  v16.4s, v16.4s, v8.4h       \n"
                "saddw2 v17.4s, v17.4s, v8.8h       \n"
                "saddw  v18.4s, v18.4s, v9.4h       \n"
                "saddw2 v19.4s, v19.4s, v9.8h       \n"
#endif // __ARM_FEATURE_DOTPROD

                "5:                                 \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB)      // %2
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "r"(max_kk), // %6
                "r"(k)       // %7
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // __aarch64__
            asm volatile(
                "cmp        %7, #0              \n"
                "beq        0f                  \n"

                "vldm       %0, {d16-d23}       \n"
                "b          1f                  \n"

                "0:                             \n"
                "veor       q8, q8              \n"
                "veor       q9, q9              \n"
                "veor       q10, q10            \n"
                "veor       q11, q11            \n"

                "1:                             \n"
                "lsr        r4, %6, #2          \n" // r4 = max_kk >> 2
                "cmp        r4, #0              \n"
                "beq        3f                  \n"

                ".align 4                       \n"
                "2:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.s8    {d0-d1}, [%1 :64]!  \n"
                "pld        [%2, #128]          \n"
                "vld1.s8    {d4-d5}, [%2]!      \n"
                "vrev64.32  q1, q0              \n"
                "vmull.s8   q4, d0, d4          \n"
                "vrev64.16  q3, q2              \n"
                "vmull.s8   q5, d2, d4          \n"
                "vmull.s8   q6, d0, d6          \n"
                "vmull.s8   q7, d2, d6          \n"
                "vmlal.s8   q4, d1, d5          \n"
                "vmlal.s8   q5, d3, d5          \n"
                "vmlal.s8   q6, d1, d7          \n"
                "vmlal.s8   q7, d3, d7          \n"
                "subs       r4, r4, #1          \n"
                "vpadal.s16 q8, q4              \n"
                "vpadal.s16 q9, q5              \n"
                "vpadal.s16 q10, q6             \n"
                "vpadal.s16 q11, q7             \n"
                "bne        2b                  \n"

                "3:                             \n"
                "and        r4, %6, #2          \n" // r4 = remain = max_kk & 2
                "cmp        r4, #0              \n"
                "beq        4f                  \n"

                // kk += 2 part
                "vld1.s8    {d0}, [%1 :64]!     \n"
                "vld1.s8    {d4}, [%2]!         \n"
                "vext.8     d1, d0, d0, #4      \n"
                "vrev64.16  d5, d4              \n"
                "vmull.s8   q4, d0, d4          \n"
                "vmull.s8   q5, d1, d4          \n"
                "vmull.s8   q6, d0, d5          \n"
                "vmull.s8   q7, d1, d5          \n"
                "vpadal.s16 q8, q4              \n"
                "vpadal.s16 q9, q5              \n"
                "vpadal.s16 q10, q6             \n"
                "vpadal.s16 q11, q7             \n"

                "4:                             \n"
                "and        r4, %6, #1          \n" // r4 = remain = max_kk & 1
                "cmp        r4, #0              \n"
                "beq        5f                  \n"

                // kk += 1 part
                "vld1.s32   {d0[0]}, [%1]!      \n"
                "vld1.s32   {d2[]}, [%2]!       \n"
                "vrev32.16  d1, d0              \n"
                "vrev32.s8  d3, d2              \n"
                "vzip.32    d0, d1              \n"
                "vmull.s8   q4, d0, d2          \n"
                "vmull.s8   q5, d0, d3          \n"
                "vaddw.s16  q8, d8              \n"
                "vaddw.s16  q9, d9              \n"
                "vaddw.s16  q10, d10            \n"
                "vaddw.s16  q11, d11            \n"

                "5:                             \n"
                "vstm       %0!, {d16-d23}      \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB)      // %2
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "r"(max_kk), // %6
                "r"(k)       // %7
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
            }

            int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
            {
                int32x4_t _sum00 = vdupq_n_s32(0);
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum10 = vdupq_n_s32(0);
                int32x4_t _sum11 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA1 = vld1q_s8(pA + 16);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);

                    // aaaaaaaa bbbbbbbb cccccccc dddddddd

                    // 00000000 11111111 22222222 33333333

                    _sum00 = vmmlaq_s32(_sum00, _pA0, _pB0);
                    _sum01 = vmmlaq_s32(_sum01, _pA1, _pB0);
                    _sum10 = vmmlaq_s32(_sum10, _pA0, _pB1);
                    _sum11 = vmmlaq_s32(_sum11, _pA1, _pB1);

                    // a0 a1 b0 b1
                    // c0 c1 d0 d1
                    // a2 a3 b2 b3
                    // c2 c3 d2 d3

                    pA += 32;
                    pB += 32;
                }
                int32x4x2_t _ss0 = vuzpq_s32(_sum00, _sum01);
                int32x4x2_t _ss1 = vuzpq_s32(_sum10, _sum11);
                _sum0 = vaddq_s32(_sum0, _ss0.val[0]);
                _sum1 = vaddq_s32(_sum1, _ss0.val[1]);
                _sum2 = vaddq_s32(_sum2, _ss1.val[0]);
                _sum3 = vaddq_s32(_sum3, _ss1.val[1]);
            }
#elif __ARM_FEATURE_DOTPROD
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA1 = vld1q_s8(pA + 16);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB1 = vld1q_s8(pB + 16);

                _sum0 = vdotq_laneq_s32(_sum0, _pA0, _pB0, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA0, _pB0, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA0, _pB0, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA0, _pB0, 3);

                _sum0 = vdotq_laneq_s32(_sum0, _pA1, _pB1, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA1, _pB1, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA1, _pB1, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA1, _pB1, 3);

                pA += 32;
                pB += 32;
            }
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x16_t _pB = vld1q_s8(pB);

                _sum0 = vdotq_laneq_s32(_sum0, _pA, _pB, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA, _pB, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA, _pB, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA, _pB, 3);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA02 = vld1q_s8(pA);
                int8x16_t _pB02 = vld1q_s8(pB);

                // aabbccdd eeffgghh
                // ccddaabb gghheeff

                int8x16_t _pA13 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA02)));

                // 00112233 44556677
                // 33221100 77665544

                int8x16_t _pB13 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB02)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA02), vget_low_s8(_pB02));
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA13), vget_low_s8(_pB02));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA02), vget_low_s8(_pB13));
                int16x8_t _s3 = vmull_s8(vget_low_s8(_pA13), vget_low_s8(_pB13));

                _s0 = vmlal_s8(_s0, vget_high_s8(_pA02), vget_high_s8(_pB02));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA13), vget_high_s8(_pB02));
                _s2 = vmlal_s8(_s2, vget_high_s8(_pA02), vget_high_s8(_pB13));
                _s3 = vmlal_s8(_s3, vget_high_s8(_pA13), vget_high_s8(_pB13));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
#endif // __ARM_FEATURE_DOTPROD

                pA += 16;
                pB += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 0)));
                int16x8_t _s1 = vmull_s8(_pA, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 1)));
                int16x8_t _s2 = vmull_s8(_pA, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 2)));
                int16x8_t _s3 = vmull_s8(_pA, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 3)));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vld1_s8(pA);
                int8x8_t _pB0 = vld1_s8(pB);

                // aabbccdd
                // ccddaabb

                int8x8_t _pA1 = vext_s8(_pA0, _pA0, 4);

                // 00112233
                // 33221100

                int8x8_t _pB1 = vreinterpret_s8_s16(vrev64_s16(vreinterpret_s16_s8(_pB0)));

                int16x8_t _s0 = vmull_s8(_pA0, _pB0);
                int16x8_t _s1 = vmull_s8(_pA1, _pB0);
                int16x8_t _s2 = vmull_s8(_pA0, _pB1);
                int16x8_t _s3 = vmull_s8(_pA1, _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
#endif // __ARM_FEATURE_DOTPROD

                pA += 8;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));

                _pB = vzip_s8(_pB, _pB).val[0];
                int16x4x2_t _pB0123 = vzip_s16(vreinterpret_s16_s8(_pB), vreinterpret_s16_s8(_pB));

                int16x8_t _s01 = vmull_s8(_pA, vreinterpret_s8_s16(_pB0123.val[0]));
                int16x8_t _s23 = vmull_s8(_pA, vreinterpret_s8_s16(_pB0123.val[1]));
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s23));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s23));
#else  // __ARM_FEATURE_DOTPROD

                int8x8_t _pA0 = vld1_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));

                // abcd.... -> cdab.... -> abcdcdab
                int8x8_t _pA1 = vreinterpret_s8_s16(vrev32_s16(vreinterpret_s16_s8(_pA0)));
                int8x8_t _pA01 = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_pA0), vreinterpret_s32_s8(_pA1)).val[0]);

                // 01230123 -> 32103210
                int8x8_t _pB1 = vrev32_s8(_pB0);

                int16x8_t _s01 = vmull_s8(_pA01, _pB0);
                int16x8_t _s23 = vmull_s8(_pA01, _pB1);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s23));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s23));
#endif // __ARM_FEATURE_DOTPROD

                pA += 4;
                pB += 4;
            }

            //             if (k_end)
            //             {
            // #if __ARM_FEATURE_DOTPROD
            //                 // from
            //                 //      a0 b0 c0 d0
            //                 //      a1 b1 c1 d1
            //                 //      a2 b2 c2 d2
            //                 //      a3 b3 c3 d3
            //                 if (out_elempack == 4)
            //                 {
            //                     vst1q_s32(outptr0, _sum0);
            //                     vst1q_s32(outptr0 + 4, _sum1);
            //                     vst1q_s32(outptr0 + 8, _sum2);
            //                     vst1q_s32(outptr0 + 12, _sum3);
            //                     outptr0 += 16;
            //                 }
            //                 // if (out_elempack == 1)
            //                 // {
            //                 //     // to
            //                 //     //      a0 a1 a2 a3
            //                 //     //      b0 b1 b2 b3
            //                 //     //      c0 c1 c2 c3
            //                 //     //      d0 d1 d2 d3
            //                 //     {
            //                 //         int32x4x2_t _r01 = vzipq_s32(_sum0, _sum1);
            //                 //         int32x4x2_t _r23 = vzipq_s32(_sum2, _sum3);
            //                 //         _sum0 = vcombine_s32(vget_low_s32(_r01.val[0]), vget_low_s32(_r23.val[0]));
            //                 //         _sum1 = vcombine_s32(vget_high_s32(_r01.val[0]), vget_high_s32(_r23.val[0]));
            //                 //         _sum2 = vcombine_s32(vget_low_s32(_r01.val[1]), vget_low_s32(_r23.val[1]));
            //                 //         _sum3 = vcombine_s32(vget_high_s32(_r01.val[1]), vget_high_s32(_r23.val[1]));
            //                 //     }
            //                 //
            //                 //     vst1q_s32(outptr0, _sum0);
            //                 //     vst1q_s32(outptr0 + out_hstep, _sum1);
            //                 //     vst1q_s32(outptr0 + out_hstep * 2, _sum2);
            //                 //     vst1q_s32(outptr0 + out_hstep * 3, _sum3);
            //                 //     outptr0 += 4;
            //                 // }
            // #else  // __ARM_FEATURE_DOTPROD
            //
            //                 // from
            //                 //      a0 b1 c2 d3
            //                 //      c0 d1 a2 b3
            //                 //      a3 b2 c1 d0
            //                 //      c3 d2 a1 b0
            //                 // if (out_elempack == 4)
            //                 // {
            //                 //     // to
            //                 //     //      a0 b0 c0 d0
            //                 //     //      a1 b1 c1 d1
            //                 //     //      a2 b2 c2 d2
            //                 //     //      a3 b3 c3 d3
            //                 //     {
            //                 //         _sum2 = vrev64q_s32(_sum2);
            //                 //         _sum3 = vrev64q_s32(_sum3);
            //                 //         _sum2 = vextq_s32(_sum2, _sum2, 2);
            //                 //         _sum3 = vextq_s32(_sum3, _sum3, 2);
            //                 //         int32x4x2_t _t0 = vzipq_s32(_sum0, _sum3);
            //                 //         int32x4x2_t _t1 = vzipq_s32(_sum1, _sum2);
            //                 //         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
            //                 //         _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
            //                 //         _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
            //                 //         _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
            //                 //         _sum1 = vrev64q_s32(_sum1);
            //                 //         _sum3 = vrev64q_s32(_sum3);
            //                 //     }
            //                 //
            //                 //     vst1q_s32(outptr0, _sum0);
            //                 //     vst1q_s32(outptr0 + 4, _sum1);
            //                 //     vst1q_s32(outptr0 + 8, _sum2);
            //                 //     vst1q_s32(outptr0 + 12, _sum3);
            //                 //     outptr0 += 16;
            //                 // }
            //                 // if (out_elempack == 1)
            //                 // {
            //                 //     // to
            //                 //     //      a0 a1 a2 a3
            //                 //     //      b0 b1 b2 b3
            //                 //     //      c0 c1 c2 c3
            //                 //     //      d0 d1 d2 d3
            //                 //     {
            //                 //         _sum1 = vextq_s32(_sum1, _sum1, 2);
            //                 //         _sum3 = vextq_s32(_sum3, _sum3, 2);
            //                 //         int32x4x2_t _t0 = vzipq_s32(_sum0, _sum3);
            //                 //         int32x4x2_t _t1 = vzipq_s32(_sum1, _sum2);
            //                 //         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
            //                 //         _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
            //                 //         _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
            //                 //         _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
            //                 //         _sum1 = vrev64q_s32(_sum1);
            //                 //         _sum3 = vrev64q_s32(_sum3);
            //                 //     }
            //                 //
            //                 //     vst1q_s32(outptr0, _sum0);
            //                 //     vst1q_s32(outptr0 + out_hstep, _sum1);
            //                 //     vst1q_s32(outptr0 + out_hstep * 2, _sum2);
            //                 //     vst1q_s32(outptr0 + out_hstep * 3, _sum3);
            //                 //     outptr0 += 4;
            //                 // }
            //                 vst1q_s32(outptr0, _sum0);
            //                 vst1q_s32(outptr0 + 4, _sum1);
            //                 vst1q_s32(outptr0 + 8, _sum2);
            //                 vst1q_s32(outptr0 + 12, _sum3);
            //                 outptr0 += 16;
            // #endif // __ARM_FEATURE_DOTPROD
            //             }
            //             else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
            }

            outptr += 16;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            int32x4_t _sum0;
            int32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
            }

            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _sum00 = vdupq_n_s32(0);
                int32x4_t _sum01 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA1 = vld1q_s8(pA + 16);
                    int8x16_t _pB = vld1q_s8(pB);

#if __ARM_FEATURE_MATMUL_INT8
                    // aaaaaaaa bbbbbbbb cccccccc dddddddd

                    // 00000000 11111111

                    _sum00 = vmmlaq_s32(_sum00, _pA0, _pB);
                    _sum01 = vmmlaq_s32(_sum01, _pA1, _pB);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_laneq_s32(_sum0, _pA0, _pB, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _pA0, _pB, 1);
                    _sum0 = vdotq_laneq_s32(_sum0, _pA1, _pB, 2);
                    _sum1 = vdotq_laneq_s32(_sum1, _pA1, _pB, 3);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 32;
                    pB += 16;
                }
#if __ARM_FEATURE_MATMUL_INT8
                int32x4x2_t _ss = vuzpq_s32(_sum00, _sum01);
                _sum0 = vaddq_s32(_sum0, _ss.val[0]);
                _sum1 = vaddq_s32(_sum1, _ss.val[1]);
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                _sum0 = vdotq_lane_s32(_sum0, _pA, _pB, 0);
                _sum1 = vdotq_lane_s32(_sum1, _pA, _pB, 1);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                // aabbccdd eeffgghh

                // 00112233 -> 00110011 22332233
                // 11001100 33223322

                int32x2x2_t _pBB = vzip_s32(vreinterpret_s32_s8(_pB), vreinterpret_s32_s8(_pB));
                int8x16_t _pB02 = vreinterpretq_s8_s32(vcombine_s32(_pBB.val[0], _pBB.val[1]));

                int8x16_t _pB13 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB02)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), vget_low_s8(_pB02));
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA), vget_low_s8(_pB13));
                _s0 = vmlal_s8(_s0, vget_high_s8(_pA), vget_high_s8(_pB02));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA), vget_high_s8(_pB13));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
#endif // __ARM_FEATURE_DOTPROD

                pA += 16;
                pB += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vld1_s8(pB);
                // aabbccdd
                // 0011....
                int16x8_t _s0 = vmull_s8(_pA, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 0)));
                int16x8_t _s1 = vmull_s8(_pA, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 1)));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));

                // aabbccdd

                // 00110011
                // 11001100
                int8x8_t _pB1 = vext_s8(_pB0, _pB0, 2);

                int16x8_t _s0 = vmull_s8(_pA, _pB0);
                int16x8_t _s1 = vmull_s8(_pA, _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
#endif // __ARM_FEATURE_DOTPROD

                pA += 8;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));

                // abcdabcd

                // 01010101 -> 00001111
                _pB = vuzp_s8(_pB, vext_s8(_pB, _pB, 1)).val[0];

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));

                // abcd abcd

                // 0101 0101 -> 0101 1010

                int8x8_t _pB1 = vext_s8(_pB0, _pB0, 1);
                int8x8_t _pB = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_pB0), vreinterpret_s32_s8(_pB1)).val[0]);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
#endif // __ARM_FEATURE_DOTPROD

                pA += 4;
                pB += 2;
            }

            //             if (k_end)
            //             {
            // #if __ARM_FEATURE_DOTPROD
            //                 // from
            //                 //      a0 b0 c0 d0
            //                 //      a1 b1 c1 d1
            //                 if (out_elempack == 4)
            //                 {
            //                     vst1q_s32(outptr0, _sum0);
            //                     vst1q_s32(outptr0 + 4, _sum1);
            //                     outptr0 += 8;
            //                 }
            //                 // if (out_elempack == 1)
            //                 // {
            //                 //     // to
            //                 //     //      a0 a1 b0 b1
            //                 //     //      c0 c1 d0 d1
            //                 //     {
            //                 //         int32x4x2_t _sum01 = vzipq_s32(_sum0, _sum1);
            //                 //         _sum0 = _sum01.val[0];
            //                 //         _sum1 = _sum01.val[1];
            //                 //     }
            //                 //
            //                 //     vst1_s32(outptr0, vget_low_s32(_sum0));
            //                 //     vst1_s32(outptr0 + out_hstep, vget_high_s32(_sum0));
            //                 //     vst1_s32(outptr0 + out_hstep * 2, vget_low_s32(_sum1));
            //                 //     vst1_s32(outptr0 + out_hstep * 3, vget_high_s32(_sum1));
            //                 //     outptr0 += 2;
            //                 // }
            // #else  // __ARM_FEATURE_DOTPROD
            //
            //                 // from
            //                 //      a0 b1 c0 d1
            //                 //      a1 b0 c1 d0
            //                 // if (out_elempack == 4)
            //                 // {
            //                 //     // to
            //                 //     //      a0 b0 c0 d0
            //                 //     //      a1 b1 c1 d1
            //                 //     {
            //                 //         _sum1 = vrev64q_s32(_sum1);
            //                 //         int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
            //                 //         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
            //                 //         _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
            //                 //         _sum1 = vrev64q_s32(_sum1);
            //                 //     }
            //                 //
            //                 //     vst1q_s32(outptr0, _sum0);
            //                 //     vst1q_s32(outptr0 + 4, _sum1);
            //                 //     outptr0 += 8;
            //                 // }
            //                 // if (out_elempack == 1)
            //                 // {
            //                 //     // to
            //                 //     //      a0 a1 c0 c1
            //                 //     //      b0 b1 d0 d1
            //                 //     {
            //                 //         int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
            //                 //         _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
            //                 //         _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
            //                 //         _sum1 = vrev64q_s32(_sum1);
            //                 //     }
            //                 //
            //                 //     vst1_s32(outptr0, vget_low_s32(_sum0));
            //                 //     vst1_s32(outptr0 + out_hstep, vget_low_s32(_sum1));
            //                 //     vst1_s32(outptr0 + out_hstep * 2, vget_high_s32(_sum0));
            //                 //     vst1_s32(outptr0 + out_hstep * 3, vget_high_s32(_sum1));
            //                 //     outptr0 += 2;
            //                 // }
            //                 vst1q_s32(outptr0, _sum0);
            //                 vst1q_s32(outptr0 + 4, _sum1);
            //                 outptr0 += 8;
            // #endif // __ARM_FEATURE_DOTPROD
            //             }
            //             else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            int32x4_t _sum0;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
            }

            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum23 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA1 = vld1q_s8(pA + 16);
                    int8x8_t _pB = vld1_s8(pB);

#if __ARM_FEATURE_MATMUL_INT8
                    // aaaaaaaa bbbbbbbb cccccccc dddddddd

                    // 00000000

                    int8x16_t _pBB = vcombine_s8(_pB, _pB);

                    _sum01 = vdotq_s32(_sum01, _pA0, _pBB);
                    _sum23 = vdotq_s32(_sum23, _pA1, _pBB);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_lane_s32(_sum0, _pA0, _pB, 0);
                    _sum0 = vdotq_lane_s32(_sum0, _pA1, _pB, 1);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 32;
                    pB += 8;
                }
#if __ARM_FEATURE_MATMUL_INT8
                _sum0 = vaddq_s32(_sum0, vpaddq_s32(_sum01, _sum23));
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                _sum0 = vdotq_lane_s32(_sum0, _pA, _pB, 0);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                int8x8_t _pB1 = vreinterpret_s8_s16(vld1_dup_s16((const short*)(pB + 2)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), _pB0);
                _s0 = vmlal_s8(_s0, vget_high_s8(_pA), _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
#endif // __ARM_FEATURE_DOTPROD

                pA += 16;
                pB += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vpadalq_s16(_sum0, _s0);

                pA += 8;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB = vld1_dup_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));

                pA += 4;
                pB += 1;
            }

            // if (k_end)
            // {
            //     if (out_elempack == 4)
            //     {
            //         vst1q_s32(outptr0, _sum0);
            //         outptr0 += 4;
            //     }
            //     // if (out_elempack == 1)
            //     // {
            //     //     outptr0[0] = vgetq_lane_s32(_sum0, 0);
            //     //     outptr0[out_hstep] = vgetq_lane_s32(_sum0, 1);
            //     //     outptr0[out_hstep * 2] = vgetq_lane_s32(_sum0, 2);
            //     //     outptr0[out_hstep * 3] = vgetq_lane_s32(_sum0, 3);
            //     //     outptr0++;
            //     // }
            // }
            // else
            {
                vst1q_s32(outptr, _sum0);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum23 = vdupq_n_s32(0);
                int32x4_t _sum45 = vdupq_n_s32(0);
                int32x4_t _sum67 = vdupq_n_s32(0);
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x2_t _sum00 = vdup_n_s32(0);
                int32x2_t _sum01 = vdup_n_s32(0);
                int32x2_t _sum10 = vdup_n_s32(0);
                int32x2_t _sum11 = vdup_n_s32(0);
                int32x2_t _sum20 = vdup_n_s32(0);
                int32x2_t _sum21 = vdup_n_s32(0);
                int32x2_t _sum30 = vdup_n_s32(0);
                int32x2_t _sum31 = vdup_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA = vld1q_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);
                    int8x16_t _pB2 = vld1q_s8(pB + 32);
                    int8x16_t _pB3 = vld1q_s8(pB + 48);

#if __ARM_FEATURE_MATMUL_INT8
                    _sum01 = vmmlaq_s32(_sum01, _pA, _pB0);
                    _sum23 = vmmlaq_s32(_sum23, _pA, _pB1);
                    _sum45 = vmmlaq_s32(_sum45, _pA, _pB2);
                    _sum67 = vmmlaq_s32(_sum67, _pA, _pB3);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum00 = vdot_laneq_s32(_sum00, vget_low_s8(_pA), _pB0, 0);
                    _sum01 = vdot_laneq_s32(_sum01, vget_low_s8(_pA), _pB0, 1);
                    _sum10 = vdot_laneq_s32(_sum10, vget_low_s8(_pA), _pB0, 2);
                    _sum11 = vdot_laneq_s32(_sum11, vget_low_s8(_pA), _pB0, 3);
                    _sum20 = vdot_laneq_s32(_sum20, vget_low_s8(_pA), _pB1, 0);
                    _sum21 = vdot_laneq_s32(_sum21, vget_low_s8(_pA), _pB1, 1);
                    _sum30 = vdot_laneq_s32(_sum30, vget_low_s8(_pA), _pB1, 2);
                    _sum31 = vdot_laneq_s32(_sum31, vget_low_s8(_pA), _pB1, 3);
                    _sum00 = vdot_laneq_s32(_sum00, vget_high_s8(_pA), _pB2, 0);
                    _sum01 = vdot_laneq_s32(_sum01, vget_high_s8(_pA), _pB2, 1);
                    _sum10 = vdot_laneq_s32(_sum10, vget_high_s8(_pA), _pB2, 2);
                    _sum11 = vdot_laneq_s32(_sum11, vget_high_s8(_pA), _pB2, 3);
                    _sum20 = vdot_laneq_s32(_sum20, vget_high_s8(_pA), _pB3, 0);
                    _sum21 = vdot_laneq_s32(_sum21, vget_high_s8(_pA), _pB3, 1);
                    _sum30 = vdot_laneq_s32(_sum30, vget_high_s8(_pA), _pB3, 2);
                    _sum31 = vdot_laneq_s32(_sum31, vget_high_s8(_pA), _pB3, 3);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 16;
                    pB += 64;
                }
#if __ARM_FEATURE_MATMUL_INT8
                _sum0 = vaddq_s32(_sum0, vcombine_s32(vget_low_s32(_sum01), vget_low_s32(_sum23)));
                _sum1 = vaddq_s32(_sum1, vcombine_s32(vget_low_s32(_sum45), vget_low_s32(_sum67)));
                _sum2 = vaddq_s32(_sum2, vcombine_s32(vget_high_s32(_sum01), vget_high_s32(_sum23)));
                _sum3 = vaddq_s32(_sum3, vcombine_s32(vget_high_s32(_sum45), vget_high_s32(_sum67)));
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x2x2_t _sum0x = vzip_s32(_sum00, _sum01);
                int32x2x2_t _sum1x = vzip_s32(_sum10, _sum11);
                int32x2x2_t _sum2x = vzip_s32(_sum20, _sum21);
                int32x2x2_t _sum3x = vzip_s32(_sum30, _sum31);
                _sum0 = vaddq_s32(_sum0, vcombine_s32(_sum0x.val[0], _sum1x.val[0]));
                _sum1 = vaddq_s32(_sum1, vcombine_s32(_sum2x.val[0], _sum3x.val[0]));
                _sum2 = vaddq_s32(_sum2, vcombine_s32(_sum0x.val[1], _sum1x.val[1]));
                _sum3 = vaddq_s32(_sum3, vcombine_s32(_sum2x.val[1], _sum3x.val[1]));
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#endif // __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_DOTPROD
                int32x2_t _sum00 = vdup_n_s32(0);
                int32x2_t _sum01 = vdup_n_s32(0);
                int32x2_t _sum10 = vdup_n_s32(0);
                int32x2_t _sum11 = vdup_n_s32(0);
                int32x2_t _sum20 = vdup_n_s32(0);
                int32x2_t _sum21 = vdup_n_s32(0);
                int32x2_t _sum30 = vdup_n_s32(0);
                int32x2_t _sum31 = vdup_n_s32(0);
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);

#if __ARM_FEATURE_DOTPROD
                    _sum00 = vdot_laneq_s32(_sum00, _pA, _pB0, 0);
                    _sum01 = vdot_laneq_s32(_sum01, _pA, _pB0, 1);
                    _sum10 = vdot_laneq_s32(_sum10, _pA, _pB0, 2);
                    _sum11 = vdot_laneq_s32(_sum11, _pA, _pB0, 3);
                    _sum20 = vdot_laneq_s32(_sum20, _pA, _pB1, 0);
                    _sum21 = vdot_laneq_s32(_sum21, _pA, _pB1, 1);
                    _sum30 = vdot_laneq_s32(_sum30, _pA, _pB1, 2);
                    _sum31 = vdot_laneq_s32(_sum31, _pA, _pB1, 3);
#else  // __ARM_FEATURE_DOTPROD
                    int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 0));
                    int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 1));
                    int8x8_t _pA2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 2));
                    int8x8_t _pA3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 3));

                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                    int16x8_t _s1 = vmull_s8(_pA0, vget_high_s8(_pB0));
                    int16x8_t _s2 = vmull_s8(_pA1, vget_low_s8(_pB0));
                    int16x8_t _s3 = vmull_s8(_pA1, vget_high_s8(_pB0));
                    _s0 = vmlal_s8(_s0, _pA2, vget_low_s8(_pB1));
                    _s1 = vmlal_s8(_s1, _pA2, vget_high_s8(_pB1));
                    _s2 = vmlal_s8(_s2, _pA3, vget_low_s8(_pB1));
                    _s3 = vmlal_s8(_s3, _pA3, vget_high_s8(_pB1));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);
#endif // __ARM_FEATURE_DOTPROD

                    pA += 8;
                    pB += 32;
                }
#if __ARM_FEATURE_DOTPROD
                int32x2x2_t _sum0x = vzip_s32(_sum00, _sum01);
                int32x2x2_t _sum1x = vzip_s32(_sum10, _sum11);
                int32x2x2_t _sum2x = vzip_s32(_sum20, _sum21);
                int32x2x2_t _sum3x = vzip_s32(_sum30, _sum31);
                _sum0 = vaddq_s32(_sum0, vcombine_s32(_sum0x.val[0], _sum1x.val[0]));
                _sum1 = vaddq_s32(_sum1, vcombine_s32(_sum2x.val[0], _sum3x.val[0]));
                _sum2 = vaddq_s32(_sum2, vcombine_s32(_sum0x.val[1], _sum1x.val[1]));
                _sum3 = vaddq_s32(_sum3, vcombine_s32(_sum2x.val[1], _sum3x.val[1]));
#endif // __ARM_FEATURE_DOTPROD
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int16x4_t _pA = vreinterpret_s16_s32(vld1_dup_s32((const int*)pA));
                int8x16_t _pB = vld1q_s8(pB);

                int16x4x2_t _pA01 = vuzp_s16(_pA, _pA);
                int8x8_t _pA0 = vreinterpret_s8_s16(_pA01.val[0]);
                int8x8_t _pA1 = vreinterpret_s8_s16(_pA01.val[1]);

                int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB));
                int16x8_t _s1 = vmull_s8(_pA0, vget_high_s8(_pB));
                int16x8_t _s2 = vmull_s8(_pA1, vget_low_s8(_pB));
                int16x8_t _s3 = vmull_s8(_pA1, vget_high_s8(_pB));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);

                pA += 4;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vreinterpret_s8_s16(vld1_dup_s16((const short*)pA));
                int8x8_t _pB = vld1_s8(pB);

                int8x8x2_t _pA01 = vuzp_s8(_pA, _pA);

                int16x8_t _s0 = vmull_s8(_pA01.val[0], _pB);
                int16x8_t _s1 = vmull_s8(_pA01.val[1], _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));

                pA += 2;
                pB += 8;
            }

            // if (k_end)
            // {
            //     // if (out_elempack == 1)
            //     {
            //         vst1q_s32(outptr0, _sum0);
            //         vst1q_s32(outptr0 + 4, _sum1);
            //         // vst1q_s32(outptr0 + out_hstep, _sum2);
            //         // vst1q_s32(outptr0 + out_hstep + 4, _sum3);
            //         // outptr0 += 8;
            //         vst1q_s32(outptr0 + 8, _sum2);
            //         vst1q_s32(outptr0 + 12, _sum3);// TODO
            //         outptr0 += 16;
            //     }
            // }
            // else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
            }

            outptr += 16;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            int32x4_t _sum0;
            int32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum23 = vdupq_n_s32(0);
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x2_t _sum00 = vdup_n_s32(0);
                int32x2_t _sum01 = vdup_n_s32(0);
                int32x2_t _sum10 = vdup_n_s32(0);
                int32x2_t _sum11 = vdup_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA = vld1q_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);

#if __ARM_FEATURE_MATMUL_INT8
                    _sum01 = vmmlaq_s32(_sum01, _pA, _pB0);
                    _sum23 = vmmlaq_s32(_sum23, _pA, _pB1);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum00 = vdot_laneq_s32(_sum00, vget_low_s8(_pA), _pB0, 0);
                    _sum01 = vdot_laneq_s32(_sum01, vget_low_s8(_pA), _pB0, 1);
                    _sum10 = vdot_laneq_s32(_sum10, vget_low_s8(_pA), _pB0, 2);
                    _sum11 = vdot_laneq_s32(_sum11, vget_low_s8(_pA), _pB0, 3);
                    _sum00 = vdot_laneq_s32(_sum00, vget_high_s8(_pA), _pB1, 0);
                    _sum01 = vdot_laneq_s32(_sum01, vget_high_s8(_pA), _pB1, 1);
                    _sum10 = vdot_laneq_s32(_sum10, vget_high_s8(_pA), _pB1, 2);
                    _sum11 = vdot_laneq_s32(_sum11, vget_high_s8(_pA), _pB1, 3);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 16;
                    pB += 32;
                }
#if __ARM_FEATURE_MATMUL_INT8
                _sum0 = vaddq_s32(_sum0, vcombine_s32(vget_low_s32(_sum01), vget_low_s32(_sum23)));
                _sum1 = vaddq_s32(_sum1, vcombine_s32(vget_high_s32(_sum01), vget_high_s32(_sum23)));
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x2x2_t _sum0x = vzip_s32(_sum00, _sum01);
                int32x2x2_t _sum1x = vzip_s32(_sum10, _sum11);
                _sum0 = vaddq_s32(_sum0, vcombine_s32(_sum0x.val[0], _sum1x.val[0]));
                _sum1 = vaddq_s32(_sum1, vcombine_s32(_sum0x.val[1], _sum1x.val[1]));
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#endif // __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_DOTPROD
                int32x2_t _sum00 = vdup_n_s32(0);
                int32x2_t _sum01 = vdup_n_s32(0);
                int32x2_t _sum10 = vdup_n_s32(0);
                int32x2_t _sum11 = vdup_n_s32(0);
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB = vld1q_s8(pB);

#if __ARM_FEATURE_DOTPROD
                    _sum00 = vdot_laneq_s32(_sum00, _pA, _pB, 0);
                    _sum01 = vdot_laneq_s32(_sum01, _pA, _pB, 1);
                    _sum10 = vdot_laneq_s32(_sum10, _pA, _pB, 2);
                    _sum11 = vdot_laneq_s32(_sum11, _pA, _pB, 3);
#else  // __ARM_FEATURE_DOTPROD
                    int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 0));
                    int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 1));
                    int8x8_t _pA2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 2));
                    int8x8_t _pA3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 3));

                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB));
                    int16x8_t _s1 = vmull_s8(_pA1, vget_low_s8(_pB));
                    _s0 = vmlal_s8(_s0, _pA2, vget_high_s8(_pB));
                    _s1 = vmlal_s8(_s1, _pA3, vget_high_s8(_pB));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
#endif // __ARM_FEATURE_DOTPROD

                    pA += 8;
                    pB += 16;
                }
#if __ARM_FEATURE_DOTPROD
                int32x2x2_t _sum0x = vzip_s32(_sum00, _sum01);
                int32x2x2_t _sum1x = vzip_s32(_sum10, _sum11);
                _sum0 = vaddq_s32(_sum0, vcombine_s32(_sum0x.val[0], _sum1x.val[0]));
                _sum1 = vaddq_s32(_sum1, vcombine_s32(_sum0x.val[1], _sum1x.val[1]));
#endif // __ARM_FEATURE_DOTPROD
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int16x4_t _pA = vreinterpret_s16_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pA)), 0));
                int8x8_t _pB = vld1_s8(pB);

                int16x4x2_t _pA01 = vuzp_s16(_pA, _pA);
                int8x8_t _pA0 = vreinterpret_s8_s16(_pA01.val[0]);
                int8x8_t _pA1 = vreinterpret_s8_s16(_pA01.val[1]);

                int16x8_t _s0 = vmull_s8(_pA0, _pB);
                int16x8_t _s1 = vmull_s8(_pA1, _pB);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);

                pA += 4;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vreinterpret_s8_s16(vld1_dup_s16((const short*)pA));
                int8x8_t _pB = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pB)), 0));

                _pA = vzip_s8(_pA, _pA).val[0];
                _pA = vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_pA), vreinterpret_s16_s8(_pA)).val[0]);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                pA += 2;
                pB += 4;
            }

            // if (k_end)
            // {
            //     // if (out_elempack == 1)
            //     {
            //         vst1q_s32(outptr0, _sum0);
            //         // vst1q_s32(outptr0 + out_hstep, _sum1);
            //         // outptr0 += 4;
            //         vst1q_s32(outptr0 + 4, _sum1);//TODO
            //         outptr0 += 8;
            //     }
            // }
            // else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __ARM_NEON
#if 1
            int32x4_t _sum;

            if (k == 0)
            {
                _sum = vdupq_n_s32(0);
            }
            else
            {
                _sum = vld1q_s32(outptr);
            }

            const signed char* pA = pAT;
            int kk = 0;

#if __ARM_FEATURE_DOTPROD
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x16_t _pA = vld1q_s8(pA);
                int8x16_t _pB = vld1q_s8(pB);

#if __ARM_FEATURE_MATMUL_INT8
                _sum = vmmlaq_s32(_sum, _pA, _pB);
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x4x2_t _pAA = vzipq_s32(vreinterpretq_s32_s8(_pA), vreinterpretq_s32_s8(_pA));
                int8x16_t _pA01 = vreinterpretq_s8_s32(_pAA.val[0]);
                int8x16_t _pA23 = vreinterpretq_s8_s32(_pAA.val[1]);
                int8x16_t _pB01 = vcombine_s8(vget_low_s8(_pB), vget_low_s8(_pB));
                int8x16_t _pB23 = vcombine_s8(vget_high_s8(_pB), vget_high_s8(_pB));

                _sum = vdotq_s32(_sum, _pA01, _pB01);
                _sum = vdotq_s32(_sum, _pA23, _pB23);
#endif // __ARM_FEATURE_MATMUL_INT8

                pA += 16;
                pB += 16;
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

#if __ARM_FEATURE_DOTPROD
                int32x2x2_t _pAA = vzip_s32(vreinterpret_s32_s8(_pA), vreinterpret_s32_s8(_pA));
                int8x16_t _pA01 = vreinterpretq_s8_s32(vcombine_s32(_pAA.val[0], _pAA.val[1]));

                int8x16_t _pB01 = vcombine_s8(_pB, _pB);

                _sum = vdotq_s32(_sum, _pA01, _pB01);
#else  // __ARM_FEATURE_DOTPROD
                int16x4x2_t _pA01 = vzip_s16(vreinterpret_s16_s8(_pA), vreinterpret_s16_s8(_pA));
                int32x2x2_t _pB01 = vzip_s32(vreinterpret_s32_s8(_pB), vreinterpret_s32_s8(_pB));

                int16x8_t _s0 = vmull_s8(vreinterpret_s8_s16(_pA01.val[0]), vreinterpret_s8_s32(_pB01.val[0]));
                _s0 = vmlal_s8(_s0, vreinterpret_s8_s16(_pA01.val[1]), vreinterpret_s8_s32(_pB01.val[1]));
                _sum = vpadalq_s16(_sum, _s0);
#endif // __ARM_FEATURE_DOTPROD

                pA += 8;
                pB += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                _pA = vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_pA), vreinterpret_s16_s8(_pA)).val[0]);
                _pB = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_pB), vreinterpret_s32_s8(_pB)).val[0]);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum = vpadalq_s16(_sum, _s0);

                // A0 A1 A2 A3
                // B0 B1 B2 B3

                // A0 A1 A0 A1 A2 A3 A2 A3
                // B0 B1 B2 B3 B0 B1 B2 B3

                pA += 4;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vreinterpret_s8_s16(vld1_dup_s16((const short*)pA));
                int8x8_t _pB = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vld1_s8(pB)), 0));

                _pA = vzip_s8(_pA, _pA).val[0];

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum = vaddw_s16(_sum, vget_low_s16(_s0));

                // A0 A1 A0 A1
                // B0 B1 B0 B1

                // A0 A0 A1 A1

                pA += 2;
                pB += 2;
            }

            // if (k_end)
            // {
            //     // if (out_elempack == 1)
            //     {
            //         vst1_s32(outptr0, vget_low_s32(_sum));
            //         // vst1_s32(outptr0 + out_hstep, vget_high_s32(_sum));
            //         // outptr0 += 2;
            //         vst1_s32(outptr0 + 2, vget_high_s32(_sum));// TODO
            //         outptr0 += 4;
            //     }
            // }
            // else
            {
                vst1q_s32(outptr, _sum);
            }
            outptr += 4;
#else

            int sum00;
            int sum10;
            int sum01;
            int sum11;

            if (k == 0)
            {
                sum00 = 0;
                sum10 = 0;
                sum01 = 0;
                sum11 = 0;
            }
            else
            {
                sum00 = outptr[0];
                sum10 = outptr[1];
                sum01 = outptr[2];
                sum11 = outptr[3];
            }

            const signed char* pA = pAT;
            int kk = 0;

            for (; kk + 1 < max_kk; kk += 2)
            {
                sum00 += pA[0] * pB[0];
                sum00 += pA[1] * pB[1];

                sum10 += pA[2] * pB[0];
                sum10 += pA[3] * pB[1];

                sum01 += pA[0] * pB[2];
                sum01 += pA[1] * pB[3];

                sum11 += pA[2] * pB[2];
                sum11 += pA[3] * pB[3];

                pA += 4;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum10 += pA[1] * pB[0];
                sum01 += pA[0] * pB[1];
                sum11 += pA[1] * pB[1];

                pA += 2;
                pB += 2;
            }

            {
                outptr[0] = sum00;
                outptr[1] = sum10;
                outptr[2] = sum01;
                outptr[3] = sum11;
            }
            outptr += 4;
#endif

#else // __ARM_NEON
            int sum00;
            int sum10;
            int sum01;
            int sum11;

            if (k == 0)
            {
                sum00 = 0;
                sum10 = 0;
                sum01 = 0;
                sum11 = 0;
            }
            else
            {
                sum00 = outptr[0];
                sum10 = outptr[1];
                sum01 = outptr[2];
                sum11 = outptr[3];
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_SIMD32 && NCNN_GNU_INLINE_ASM
            for (; kk + 1 < max_kk; kk += 2)
            {
                // fomit-frame-pointer implied in optimized flag spare one register
                // let us stay away from error: ‘asm’ operand has impossible constraints   --- nihui
#if __OPTIMIZE__
                asm volatile(
                    "ldr    r2, [%0], #4    \n" // int8x4_t _pA = *((int8x4_t*)pA); pA += 4;
                    "ldr    r4, [%1], #4    \n" // int8x4_t _pB = *((int8x4_t*)pB); pB += 4;
                    "ror    r3, r2, #8      \n" // int8x4_t _pA_r8 = __ror(_pA, 8);
                    "ror    r5, r4, #8      \n" // int8x4_t _pB_r8 = __ror(_pB, 8);
                    "sxtb16 r2, r2          \n" // int16x2_t _pA0 = __sxtb16(_pA);
                    "sxtb16 r4, r4          \n" // int16x2_t _pA1 = __sxtb16(_pA_r8);
                    "sxtb16 r3, r3          \n" // int16x2_t _pB0 = __sxtb16(_pB);
                    "sxtb16 r5, r5          \n" // int16x2_t _pB1 = __sxtb16(_pB_r8);
                    "smlad  %2, r2, r4, %2  \n" // sum00 = __smlad(_pA0, _pB0, sum00);
                    "smlad  %3, r3, r4, %3  \n" // sum10 = __smlad(_pA1, _pB0, sum10);
                    "smlad  %4, r2, r5, %4  \n" // sum01 = __smlad(_pA0, _pB1, sum01);
                    "smlad  %5, r3, r5, %5  \n" // sum11 = __smlad(_pA1, _pB1, sum11);
                    : "=r"(pA),
                    "=r"(pB),
                    "=r"(sum00),
                    "=r"(sum10),
                    "=r"(sum01),
                    "=r"(sum11)
                    : "0"(pA),
                    "1"(pB),
                    "2"(sum00),
                    "3"(sum10),
                    "4"(sum01),
                    "5"(sum11)
                    : "memory", "r2", "r3", "r4", "r5");
#else
                int _pA0 = *((int*)pA);
                int _pB0 = *((int*)pB);
                int _pA1;
                int _pB1;
                asm volatile("ror %0, %1, #8"
                             : "=r"(_pA1)
                             : "r"(_pA0)
                             :);
                asm volatile("ror %0, %1, #8"
                             : "=r"(_pB1)
                             : "r"(_pB0)
                             :);
                asm volatile("sxtb16 %0, %0"
                             : "=r"(_pA0)
                             : "0"(_pA0)
                             :);
                asm volatile("sxtb16 %0, %0"
                             : "=r"(_pA1)
                             : "0"(_pA1)
                             :);
                asm volatile("sxtb16 %0, %0"
                             : "=r"(_pB0)
                             : "0"(_pB0)
                             :);
                asm volatile("sxtb16 %0, %0"
                             : "=r"(_pB1)
                             : "0"(_pB1)
                             :);
                asm volatile("smlad %0, %2, %3, %0"
                             : "=r"(sum00)
                             : "0"(sum00), "r"(_pA0), "r"(_pB0)
                             :);
                asm volatile("smlad %0, %2, %3, %0"
                             : "=r"(sum10)
                             : "0"(sum10), "r"(_pA1), "r"(_pB0)
                             :);
                asm volatile("smlad %0, %2, %3, %0"
                             : "=r"(sum01)
                             : "0"(sum01), "r"(_pA0), "r"(_pB1)
                             :);
                asm volatile("smlad %0, %2, %3, %0"
                             : "=r"(sum11)
                             : "0"(sum11), "r"(_pA1), "r"(_pB1)
                             :);
                pA += 4;
                pB += 4;
#endif
            }
#endif // __ARM_FEATURE_SIMD32 && NCNN_GNU_INLINE_ASM
            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum10 += pA[1] * pB[0];
                sum01 += pA[0] * pB[1];
                sum11 += pA[1] * pB[1];

                pA += 2;
                pB += 2;
            }

            // if (k_end)
            // {
            //     // if (out_elempack == 1)
            //     {
            //         outptr0[0] = sum00;
            //         outptr0[1] = sum01;
            //         // outptr0[out_hstep] = sum10;
            //         // outptr0[out_hstep + 1] = sum11;
            //         // outptr0 += 2;
            //         outptr0[2] = sum10;// TODO
            //         outptr0[3] = sum11;
            //         outptr0 += 4;
            //     }
            // }
            // else
            {
                outptr[0] = sum00;
                outptr[1] = sum10;
                outptr[2] = sum01;
                outptr[3] = sum11;
            }

            outptr += 4;
#endif // __ARM_NEON
        }
        for (; jj < max_jj; jj += 1)
        {
#if __ARM_NEON
            int32x2_t _sum;

            if (k == 0)
            {
                _sum = vdup_n_s32(0);
            }
            else
            {
                _sum = vld1_s32(outptr);
            }
#else  // __ARM_NEON
            int sum0;
            int sum1;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }
#endif // __ARM_NEON

            const signed char* pA = pAT;
            int kk = 0;
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA = vld1q_s8(pA);
                    int8x8_t _pB = vld1_s8(pB);

                    int8x16_t _pBB = vcombine_s8(_pB, _pB);

                    _sum0 = vdotq_s32(_sum0, _pA, _pBB);

                    pA += 16;
                    pB += 8;
                }
                int32x2_t _ss = vpadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                _sum = vadd_s32(_sum, _ss);
            }
#else  // __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                _sum = vdot_lane_s32(_sum, vget_low_s8(_pA), _pB, 0);
                _sum = vdot_lane_s32(_sum, vget_high_s8(_pA), _pB, 1);

                pA += 16;
                pB += 8;
            }
#endif // __ARM_FEATURE_MATMUL_INT8
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));

                _sum = vdot_s32(_sum, _pA, _pB);

                pA += 8;
                pB += 4;
            }
#else  // __ARM_FEATURE_DOTPROD
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x8_t _pB = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pB)), 0));

                    _pB = vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_pB), vreinterpret_s16_s8(_pB)).val[0]);

                    int16x8_t _s0 = vmull_s8(_pA, _pB);
                    _sum0 = vpadalq_s16(_sum0, _s0);

                    pA += 8;
                    pB += 4;
                }
                int32x2_t _ss = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                _sum = vadd_s32(_sum, _ss);
            }
#endif // __ARM_FEATURE_DOTPROD
            int sum0 = vget_lane_s32(_sum, 0);
            int sum1 = vget_lane_s32(_sum, 1);
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum1 += pA[2] * pB[0];
                sum1 += pA[3] * pB[1];
                pA += 4;
                pB += 2;
            }
#endif // __ARM_NEON
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];
                pA += 2;
                pB += 1;
            }

            // if (k_end)
            // {
            //     // if (out_elempack == 1)
            //     {
            //         outptr0[0] = sum0;
            //         // outptr0[out_hstep] = sum1;
            //         // outptr0++;
            //         outptr0[1] = sum1;//TODO
            //         outptr0 += 2;
            //     }
            // }
            // else
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
        const signed char* pB = pBT;

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            int32x4_t _sum0;
            int32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _sum00 = vdupq_n_s32(0);
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum10 = vdupq_n_s32(0);
                int32x4_t _sum11 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);
                    int8x16_t _pB2 = vld1q_s8(pB + 32);
                    int8x16_t _pB3 = vld1q_s8(pB + 48);

#if __ARM_FEATURE_MATMUL_INT8
                    int8x16_t _pAA = vcombine_s8(_pA, _pA);
                    _sum00 = vdotq_s32(_sum00, _pAA, _pB0);
                    _sum01 = vdotq_s32(_sum01, _pAA, _pB1);
                    _sum10 = vdotq_s32(_sum10, _pAA, _pB2);
                    _sum11 = vdotq_s32(_sum11, _pAA, _pB3);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_lane_s32(_sum0, _pB0, _pA, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _pB1, _pA, 0);
                    _sum0 = vdotq_lane_s32(_sum0, _pB2, _pA, 1);
                    _sum1 = vdotq_lane_s32(_sum1, _pB3, _pA, 1);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 8;
                    pB += 64;
                }
#if __ARM_FEATURE_MATMUL_INT8
                _sum0 = vaddq_s32(_sum0, vpaddq_s32(_sum00, _sum01));
                _sum1 = vaddq_s32(_sum1, vpaddq_s32(_sum10, _sum11));
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#else  // __ARM_FEATURE_DOTPROD
            {
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                int32x4_t _sum4 = vdupq_n_s32(0);
                int32x4_t _sum5 = vdupq_n_s32(0);
                int32x4_t _sum6 = vdupq_n_s32(0);
                int32x4_t _sum7 = vdupq_n_s32(0);
                for (; kk + 15 < max_kk; kk += 16)
                {
                    // TODO
                    // __builtin_prefetch(pA + 16);
                    // __builtin_prefetch(pB + 128);
                    int8x16_t _pA = vld1q_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);
                    int8x16_t _pB2 = vld1q_s8(pB + 32);
                    int8x16_t _pB3 = vld1q_s8(pB + 48);
                    int8x16_t _pB4 = vld1q_s8(pB + 64);
                    int8x16_t _pB5 = vld1q_s8(pB + 80);
                    int8x16_t _pB6 = vld1q_s8(pB + 96);
                    int8x16_t _pB7 = vld1q_s8(pB + 112);

                    int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pA)), 0));
                    int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pA)), 1));
                    int8x8_t _pA2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pA)), 2));
                    int8x8_t _pA3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pA)), 3));
                    int8x8_t _pA4 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pA)), 0));
                    int8x8_t _pA5 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pA)), 1));
                    int8x8_t _pA6 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pA)), 2));
                    int8x8_t _pA7 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pA)), 3));
                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                    int16x8_t _s1 = vmull_s8(_pA0, vget_high_s8(_pB0));
                    int16x8_t _s2 = vmull_s8(_pA2, vget_low_s8(_pB2));
                    int16x8_t _s3 = vmull_s8(_pA2, vget_high_s8(_pB2));
                    int16x8_t _s4 = vmull_s8(_pA4, vget_low_s8(_pB4));
                    int16x8_t _s5 = vmull_s8(_pA4, vget_high_s8(_pB4));
                    int16x8_t _s6 = vmull_s8(_pA6, vget_low_s8(_pB6));
                    int16x8_t _s7 = vmull_s8(_pA6, vget_high_s8(_pB6));
                    _s0 = vmlal_s8(_s0, _pA1, vget_low_s8(_pB1));
                    _s1 = vmlal_s8(_s1, _pA1, vget_high_s8(_pB1));
                    _s2 = vmlal_s8(_s2, _pA3, vget_low_s8(_pB3));
                    _s3 = vmlal_s8(_s3, _pA3, vget_high_s8(_pB3));
                    _s4 = vmlal_s8(_s4, _pA5, vget_low_s8(_pB5));
                    _s5 = vmlal_s8(_s5, _pA5, vget_high_s8(_pB5));
                    _s6 = vmlal_s8(_s6, _pA7, vget_low_s8(_pB7));
                    _s7 = vmlal_s8(_s7, _pA7, vget_high_s8(_pB7));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);
                    _sum4 = vpadalq_s16(_sum4, _s4);
                    _sum5 = vpadalq_s16(_sum5, _s5);
                    _sum6 = vpadalq_s16(_sum6, _s6);
                    _sum7 = vpadalq_s16(_sum7, _s7);

                    pA += 16;
                    pB += 128;
                }
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);
                    int8x16_t _pB2 = vld1q_s8(pB + 32);
                    int8x16_t _pB3 = vld1q_s8(pB + 48);

                    int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 0));
                    int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 1));
                    int8x8_t _pA2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 2));
                    int8x8_t _pA3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 3));
                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                    int16x8_t _s1 = vmull_s8(_pA0, vget_high_s8(_pB0));
                    int16x8_t _s2 = vmull_s8(_pA2, vget_low_s8(_pB2));
                    int16x8_t _s3 = vmull_s8(_pA2, vget_high_s8(_pB2));
                    _s0 = vmlal_s8(_s0, _pA1, vget_low_s8(_pB1));
                    _s1 = vmlal_s8(_s1, _pA1, vget_high_s8(_pB1));
                    _s2 = vmlal_s8(_s2, _pA3, vget_low_s8(_pB3));
                    _s3 = vmlal_s8(_s3, _pA3, vget_high_s8(_pB3));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);

                    pA += 8;
                    pB += 64;
                }
                _sum0 = vaddq_s32(_sum0, _sum2);
                _sum1 = vaddq_s32(_sum1, _sum3);
                _sum0 = vaddq_s32(_sum0, _sum4);
                _sum1 = vaddq_s32(_sum1, _sum5);
                _sum0 = vaddq_s32(_sum0, _sum6);
                _sum1 = vaddq_s32(_sum1, _sum7);
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x8_t _pA = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pA)), 0));
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB1 = vld1q_s8(pB + 16);

#if __ARM_FEATURE_DOTPROD
                _sum0 = vdotq_lane_s32(_sum0, _pB0, _pA, 0);
                _sum1 = vdotq_lane_s32(_sum1, _pB1, _pA, 0);
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 0));
                int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 1));
                int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                int16x8_t _s1 = vmull_s8(_pA0, vget_high_s8(_pB0));
                _s0 = vmlal_s8(_s0, _pA1, vget_low_s8(_pB1));
                _s1 = vmlal_s8(_s1, _pA1, vget_high_s8(_pB1));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
#endif // __ARM_FEATURE_DOTPROD

                pA += 4;
                pB += 32;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8_t _pA = vreinterpret_s8_s16(vld1_dup_s16((const short*)pA));
                int8x16_t _pB = vld1q_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, vget_low_s8(_pB));
                int16x8_t _s1 = vmull_s8(_pA, vget_high_s8(_pB));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);

                pA += 2;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vld1_dup_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                pA += 1;
                pB += 8;
            }

            // if (k_end)
            // {
            //     // if (out_elempack == 1)
            //     {
            //         vst1q_s32(outptr0, _sum0);
            //         vst1q_s32(outptr0 + 4, _sum1);
            //         outptr0 += 8;
            //     }
            // }
            // else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            int32x4_t _sum0;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _sum00 = vdupq_n_s32(0);
                int32x4_t _sum01 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);

#if __ARM_FEATURE_MATMUL_INT8
                    int8x16_t _pAA = vcombine_s8(_pA, _pA);
                    _sum00 = vdotq_s32(_sum00, _pAA, _pB0);
                    _sum01 = vdotq_s32(_sum01, _pAA, _pB1);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_lane_s32(_sum0, _pB0, _pA, 0);
                    _sum0 = vdotq_lane_s32(_sum0, _pB1, _pA, 1);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 8;
                    pB += 32;
                }
#if __ARM_FEATURE_MATMUL_INT8
                _sum0 = vaddq_s32(_sum0, vpaddq_s32(_sum00, _sum01));
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#else  // __ARM_FEATURE_DOTPROD
            {
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                for (; kk + 15 < max_kk; kk += 16)
                {
                    // TODO
                    // __builtin_prefetch(pA + 16);
                    // __builtin_prefetch(pB + 64);
                    int8x16_t _pA = vld1q_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);
                    int8x16_t _pB2 = vld1q_s8(pB + 32);
                    int8x16_t _pB3 = vld1q_s8(pB + 48);

                    int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pA)), 0));
                    int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pA)), 1));
                    int8x8_t _pA2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pA)), 2));
                    int8x8_t _pA3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pA)), 3));
                    int8x8_t _pA4 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pA)), 0));
                    int8x8_t _pA5 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pA)), 1));
                    int8x8_t _pA6 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pA)), 2));
                    int8x8_t _pA7 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pA)), 3));
                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                    int16x8_t _s1 = vmull_s8(_pA2, vget_low_s8(_pB1));
                    int16x8_t _s2 = vmull_s8(_pA4, vget_low_s8(_pB2));
                    int16x8_t _s3 = vmull_s8(_pA6, vget_low_s8(_pB3));
                    _s0 = vmlal_s8(_s0, _pA1, vget_high_s8(_pB0));
                    _s1 = vmlal_s8(_s1, _pA3, vget_high_s8(_pB1));
                    _s2 = vmlal_s8(_s2, _pA5, vget_high_s8(_pB2));
                    _s3 = vmlal_s8(_s3, _pA7, vget_high_s8(_pB3));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);

                    pA += 16;
                    pB += 64;
                }
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);

                    int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 0));
                    int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 1));
                    int8x8_t _pA2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 2));
                    int8x8_t _pA3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 3));
                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                    int16x8_t _s1 = vmull_s8(_pA2, vget_low_s8(_pB1));
                    _s0 = vmlal_s8(_s0, _pA1, vget_high_s8(_pB0));
                    _s1 = vmlal_s8(_s1, _pA3, vget_high_s8(_pB1));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);

                    pA += 8;
                    pB += 32;
                }
                _sum0 = vaddq_s32(_sum0, _sum1);
                _sum0 = vaddq_s32(_sum0, _sum2);
                _sum0 = vaddq_s32(_sum0, _sum3);
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x16_t _pB = vld1q_s8(pB);

#if __ARM_FEATURE_DOTPROD
                _sum0 = vdotq_lane_s32(_sum0, _pB, _pA, 0);
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 0));
                int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 1));
                int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB));
                _s0 = vmlal_s8(_s0, _pA1, vget_high_s8(_pB));
                _sum0 = vpadalq_s16(_sum0, _s0);
#endif // __ARM_FEATURE_DOTPROD

                pA += 4;
                pB += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8_t _pA = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vld1_s8(pA)), 0));
                int8x8_t _pB = vld1_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vpadalq_s16(_sum0, _s0);

                pA += 2;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vld1_dup_s8(pA);
                int8x8_t _pB = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pB)), 0));

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));

                pA += 1;
                pB += 4;
            }

            // if (k_end)
            // {
            //     // if (out_elempack == 1)
            //     {
            //         vst1q_s32(outptr0, _sum0);
            //         outptr0 += 4;
            //     }
            // }
            // else
            {
                vst1q_s32(outptr, _sum0);
            }

            outptr += 4;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __ARM_NEON
            int32x2_t _sum;

            if (k == 0)
            {
                _sum = vdup_n_s32(0);
            }
            else
            {
                _sum = vld1_s32(outptr);
            }
#else  // __ARM_NEON
            int sum0;
            int sum1;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }
#endif // __ARM_NEON

            const signed char* pA = pAT;
            int kk = 0;
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB = vld1q_s8(pB);

                    int8x16_t _pAA = vcombine_s8(_pA, _pA);

                    _sum0 = vdotq_s32(_sum0, _pAA, _pB);

                    pA += 8;
                    pB += 16;
                }
                int32x2_t _ss = vpadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                _sum = vadd_s32(_sum, _ss);
            }
#else  // __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x16_t _pB = vld1q_s8(pB);

                _sum = vdot_lane_s32(_sum, vget_low_s8(_pB), _pA, 0);
                _sum = vdot_lane_s32(_sum, vget_high_s8(_pB), _pA, 1);

                pA += 8;
                pB += 16;
            }
#endif // __ARM_FEATURE_MATMUL_INT8
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x8_t _pA = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB = vld1_s8(pB);

                _sum = vdot_s32(_sum, _pA, _pB);

                pA += 4;
                pB += 8;
            }
#else  // __ARM_FEATURE_DOTPROD
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                for (; kk + 15 < max_kk; kk += 16)
                {
                    int8x16_t _pA = vld1q_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);

                    int16x8x2_t _pAA = vzipq_s16(vreinterpretq_s16_s8(_pA), vreinterpretq_s16_s8(_pA));

                    int8x8_t _pA0 = vreinterpret_s8_s16(vget_low_s16(_pAA.val[0]));
                    int8x8_t _pA1 = vreinterpret_s8_s16(vget_high_s16(_pAA.val[0]));
                    int8x8_t _pA2 = vreinterpret_s8_s16(vget_low_s16(_pAA.val[1]));
                    int8x8_t _pA3 = vreinterpret_s8_s16(vget_high_s16(_pAA.val[1]));

                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                    int16x8_t _s1 = vmull_s8(_pA2, vget_low_s8(_pB1));
                    _s0 = vmlal_s8(_s0, _pA1, vget_high_s8(_pB0));
                    _s1 = vmlal_s8(_s1, _pA3, vget_high_s8(_pB1));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);

                    pA += 16;
                    pB += 32;
                }
                _sum0 = vaddq_s32(_sum0, _sum1);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB = vld1q_s8(pB);

                    int16x4x2_t _pAA = vzip_s16(vreinterpret_s16_s8(_pA), vreinterpret_s16_s8(_pA));

                    int8x8_t _pA0 = vreinterpret_s8_s16(_pAA.val[0]);
                    int8x8_t _pA1 = vreinterpret_s8_s16(_pAA.val[1]);

                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB));
                    _s0 = vmlal_s8(_s0, _pA1, vget_high_s8(_pB));
                    _sum0 = vpadalq_s16(_sum0, _s0);

                    pA += 8;
                    pB += 16;
                }
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int8x8_t _pA = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pA)), 0));
                    int8x8_t _pB = vld1_s8(pB);

                    _pA = vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_pA), vreinterpret_s16_s8(_pA)).val[0]);

                    int16x8_t _s0 = vmull_s8(_pA, _pB);
                    _sum0 = vpadalq_s16(_sum0, _s0);

                    pA += 4;
                    pB += 8;
                }
                int32x2_t _ss = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                _sum = vadd_s32(_sum, _ss);
            }
#endif // __ARM_FEATURE_DOTPROD
            int sum0 = vget_lane_s32(_sum, 0);
            int sum1 = vget_lane_s32(_sum, 1);
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum1 += pA[0] * pB[2];
                sum1 += pA[1] * pB[3];
                pA += 2;
                pB += 4;
            }
#endif // __ARM_NEON
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];
                pA += 1;
                pB += 2;
            }

            // if (k_end)
            // {
            //     // if (out_elempack == 1)
            //     {
            //         outptr0[0] = sum0;
            //         outptr0[1] = sum1;
            //         outptr0 += 2;
            //     }
            // }
            // else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            int sum;

            if (k == 0)
            {
                sum = 0;
            }
            else
            {
                sum = outptr[0];
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __ARM_NEON
            int32x4_t _sum = vdupq_n_s32(0);
            int32x4_t _sum1 = vdupq_n_s32(0);
            for (; kk + 15 < max_kk; kk += 16)
            {
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA1 = vld1q_s8(pA + 16);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB1 = vld1q_s8(pB + 16);

#if __ARM_FEATURE_DOTPROD
                _sum = vdotq_s32(_sum, _pA0, _pB0);
                _sum1 = vdotq_s32(_sum1, _pA1, _pB1);
#else  // __ARM_FEATURE_DOTPROD
                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB0));
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB1));
                _s0 = vmlal_s8(_s0, vget_high_s8(_pA0), vget_high_s8(_pB0));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA1), vget_high_s8(_pB1));
                _sum = vpadalq_s16(_sum, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
#endif // __ARM_FEATURE_DOTPROD

                pA += 32;
                pB += 32;
            }
            _sum = vaddq_s32(_sum, _sum1);
            for (; kk + 15 < max_kk; kk += 16)
            {
                int8x16_t _pA = vld1q_s8(pA);
                int8x16_t _pB = vld1q_s8(pB);

#if __ARM_FEATURE_DOTPROD
                _sum = vdotq_s32(_sum, _pA, _pB);
#else  // __ARM_FEATURE_DOTPROD
                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), vget_low_s8(_pB));
                _s0 = vmlal_s8(_s0, vget_high_s8(_pA), vget_high_s8(_pB));
                _sum = vpadalq_s16(_sum, _s0);
#endif // __ARM_FEATURE_DOTPROD

                pA += 16;
                pB += 16;
            }
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum = vpadalq_s16(_sum, _s0);

                pA += 8;
                pB += 8;
            }
#if __aarch64__
            sum += vaddvq_s32(_sum);
#else
            int32x2_t _ss = vadd_s32(vget_low_s32(_sum), vget_high_s32(_sum));
            _ss = vpadd_s32(_ss, _ss);
            sum += vget_lane_s32(_ss, 0);
#endif
#endif // __ARM_NEON
            for (; kk < max_kk; kk += 1)
            {
                sum += pA[0] * pB[0];
                pA += 1;
                pB += 1;
            }

            // if (k_end)
            // {
            //     // if (out_elempack == 1)
            //     {
            //         outptr0[0] = sum;
            //         outptr0++;
            //     }
            // }
            // else
            {
                outptr[0] = sum;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void get_optimal_tile_mnk_int8(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / (2 * sizeof(signed char) + sizeof(int)));

    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(8, tile_size / 8 * 8);

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(signed char) / TILE_K);

            TILE_M = std::max(8, tile_size / 8 * 8);
            TILE_N = std::max(4, tile_size / 4 * 4);
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        int nn_M = (M + TILE_M - 1) / TILE_M;
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
    }

    if (nT > 1)
    {
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
    }

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
    }

    if (constant_TILE_N > 0)
    {
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
    }

    if (constant_TILE_K > 0)
    {
        TILE_K = (constant_TILE_K + 7) / 8 * 8;
    }
}
