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

#include "testutil.h"

static int test_multiheadattention(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int embed_dim, int num_heads, int attn_mask)
{
    const int qdim = q.w;
    const int kdim = k.w;
    const int vdim = v.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, kdim);
    pd.set(4, vdim);
    pd.set(5, attn_mask);

    std::vector<ncnn::Mat> weights(8);
    weights[0] = RandomMat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomMat(embed_dim * kdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomMat(embed_dim * vdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomMat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    if (attn_mask)
    {
        as.push_back(RandomMat(k.h, q.h));
    }

    float epsilon = 0.005;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention failed q=(%d %d) k=(%d %d) v=(%d %d) embed_dim=%d num_heads=%d kdim=%d vdim=%d attn_mask=%d\n", q.w, q.h, k.w, k.h, v.w, v.h, embed_dim, num_heads, kdim, vdim, attn_mask);
    }

    return ret;
}

static int test_multiheadattention_samekv(const ncnn::Mat& q, const ncnn::Mat& kv, int embed_dim, int num_heads)
{
    const int qdim = q.w;
    const int kvdim = kv.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, kvdim);
    pd.set(4, kvdim);

    std::vector<ncnn::Mat> weights(8);
    weights[0] = RandomMat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomMat(embed_dim * kvdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomMat(embed_dim * kvdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomMat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);

    std::vector<ncnn::Mat> as(2);
    as[0] = q;
    as[1] = kv;

    float epsilon = 0.005;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_samekv failed q=(%d %d) kv=(%d %d) embed_dim=%d num_heads=%d kvdim=%d\n", q.w, q.h, kv.w, kv.h, embed_dim, num_heads, kvdim);
    }

    return ret;
}

static int test_multiheadattention_sameqkv(const ncnn::Mat& a, int embed_dim, int num_heads)
{
    const int qdim = a.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, qdim);
    pd.set(4, qdim);
    pd.set(6, 0.7f / sqrtf(embed_dim / num_heads));

    std::vector<ncnn::Mat> weights(8);
    weights[0] = RandomMat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomMat(embed_dim * qdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomMat(embed_dim * qdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomMat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);

    std::vector<ncnn::Mat> as(1);
    as[0] = a;

    float epsilon = 0.005;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_sameqkv failed a=(%d %d) embed_dim=%d num_heads=%d\n", a.w, a.h, embed_dim, num_heads);
    }

    return ret;
}

static int test_multiheadattention_0()
{
    return 0
           || test_multiheadattention(RandomMat(62, 66), RandomMat(32, 66), RandomMat(20, 66), 62, 2, 0)
           || test_multiheadattention(RandomMat(26, 64), RandomMat(32, 64), RandomMat(18, 64), 26, 2, 1)
           || test_multiheadattention(RandomMat(64, 128), RandomMat(64, 128), RandomMat(64, 128), 64, 4, 0)
           || test_multiheadattention(RandomMat(48, 127), RandomMat(64, 127), RandomMat(64, 127), 64, 16, 1)
           || test_multiheadattention(RandomMat(16, 128), RandomMat(44, 128), RandomMat(55, 128), 16, 2, 0)
           || test_multiheadattention(RandomMat(12, 128), RandomMat(44, 127), RandomMat(55, 127), 16, 4, 1)
           || test_multiheadattention(RandomMat(12, 17), RandomMat(28, 127), RandomMat(32, 127), 12, 3, 0)
           || test_multiheadattention(RandomMat(12, 17), RandomMat(28, 32), RandomMat(11, 32), 12, 3, 1);
}

static int test_multiheadattention_1()
{
    return 0
           || test_multiheadattention_samekv(RandomMat(64, 128), RandomMat(64, 128), 64, 4)
           || test_multiheadattention_samekv(RandomMat(48, 127), RandomMat(64, 127), 64, 16)
           || test_multiheadattention_samekv(RandomMat(16, 128), RandomMat(44, 128), 16, 2)
           || test_multiheadattention_samekv(RandomMat(12, 128), RandomMat(22, 127), 16, 4)
           || test_multiheadattention_samekv(RandomMat(12, 17), RandomMat(28, 127), 12, 3)
           || test_multiheadattention_samekv(RandomMat(12, 17), RandomMat(11, 32), 12, 3);
}

static int test_multiheadattention_2()
{
    return 0
           || test_multiheadattention_sameqkv(RandomMat(64, 128), 64, 4)
           || test_multiheadattention_sameqkv(RandomMat(48, 127), 64, 8);
}

#if NCNN_INT8
static int test_multiheadattention_int8(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int embed_dim, int num_heads, int attn_mask)
{
    attn_mask = 0; //TODO

    const int qdim = q.w;
    const int kdim = k.w;
    const int vdim = v.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, kdim);
    pd.set(4, vdim);
    pd.set(5, attn_mask);
    pd.set(6, 1.f); // TODO
    pd.set(18, 2);  // int8_scale_term

    std::vector<ncnn::Mat> weights(12);
    weights[0] = RandomS8Mat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomS8Mat(embed_dim * kdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomS8Mat(embed_dim * vdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomS8Mat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);
    weights[8] = RandomMat(1, 100.f, 200.f);
    weights[9] = RandomMat(1, 100.f, 200.f);
    weights[10] = RandomMat(1, 100.f, 200.f);
    weights[11] = RandomMat(1, 100.f, 200.f);

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    if (attn_mask)
    {
        as.push_back(RandomMat(k.h, q.h));
    }

    float epsilon = 0.005;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_int8 failed q=(%d %d) k=(%d %d) v=(%d %d) embed_dim=%d num_heads=%d kdim=%d vdim=%d attn_mask=%d\n", q.w, q.h, k.w, k.h, v.w, v.h, embed_dim, num_heads, kdim, vdim, attn_mask);
    }
    else
        fprintf(stderr, "test_multiheadattention_int8 pass q=(%d %d) k=(%d %d) v=(%d %d) embed_dim=%d num_heads=%d kdim=%d vdim=%d attn_mask=%d\n", q.w, q.h, k.w, k.h, v.w, v.h, embed_dim, num_heads, kdim, vdim, attn_mask);

    return ret;
}

static int test_multiheadattention_int8_samekv(const ncnn::Mat& q, const ncnn::Mat& kv, int embed_dim, int num_heads)
{
    const int qdim = q.w;
    const int kvdim = kv.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, kvdim);
    pd.set(4, kvdim);
    pd.set(6, 1.f); // TODO
    pd.set(18, 2);  // int8_scale_term

    std::vector<ncnn::Mat> weights(12);
    weights[0] = RandomS8Mat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomS8Mat(embed_dim * kvdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomS8Mat(embed_dim * kvdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomS8Mat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);
    weights[8] = RandomMat(1, 100.f, 200.f);
    weights[9] = RandomMat(1, 100.f, 200.f);
    weights[10] = RandomMat(1, 100.f, 200.f);
    weights[11] = RandomMat(1, 100.f, 200.f);

    std::vector<ncnn::Mat> as(2);
    as[0] = q;
    as[1] = kv;

    float epsilon = 0.005;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_int8_samekv failed q=(%d %d) kv=(%d %d) embed_dim=%d num_heads=%d kvdim=%d\n", q.w, q.h, kv.w, kv.h, embed_dim, num_heads, kvdim);
    }

    return ret;
}

static int test_multiheadattention_int8_sameqkv(const ncnn::Mat& a, int embed_dim, int num_heads)
{
    const int qdim = a.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, qdim);
    pd.set(4, qdim);
    // pd.set(6, 0.7f / sqrtf(embed_dim / num_heads));
    pd.set(6, 1.f); // TODO
    pd.set(18, 2);  // int8_scale_term

    std::vector<ncnn::Mat> weights(12);
    weights[0] = RandomS8Mat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomS8Mat(embed_dim * qdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomS8Mat(embed_dim * qdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomS8Mat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);
    weights[8] = RandomMat(1, 100.f, 200.f);
    weights[9] = RandomMat(1, 100.f, 200.f);
    weights[10] = RandomMat(1, 100.f, 200.f);
    weights[11] = RandomMat(1, 100.f, 200.f);

    std::vector<ncnn::Mat> as(1);
    as[0] = a;

    float epsilon = 0.005;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_int8_sameqkv failed a=(%d %d) embed_dim=%d num_heads=%d\n", a.w, a.h, embed_dim, num_heads);
    }

    return ret;
}

static int test_multiheadattention_3()
{
    return 0
           || test_multiheadattention_int8(RandomMat(62, 66), RandomMat(32, 66), RandomMat(20, 66), 62, 2, 0)
           || test_multiheadattention_int8(RandomMat(26, 64), RandomMat(32, 64), RandomMat(18, 64), 26, 2, 1)
           || test_multiheadattention_int8(RandomMat(64, 128), RandomMat(64, 128), RandomMat(64, 128), 64, 4, 0)
           || test_multiheadattention_int8(RandomMat(48, 127), RandomMat(64, 127), RandomMat(64, 127), 64, 16, 1)
           || test_multiheadattention_int8(RandomMat(16, 128), RandomMat(44, 128), RandomMat(55, 128), 16, 2, 0)
           || test_multiheadattention_int8(RandomMat(12, 128), RandomMat(44, 127), RandomMat(55, 127), 16, 4, 1)
           || test_multiheadattention_int8(RandomMat(12, 17), RandomMat(28, 127), RandomMat(32, 127), 12, 3, 0)
           || test_multiheadattention_int8(RandomMat(12, 17), RandomMat(28, 32), RandomMat(11, 32), 12, 3, 1);
}

static int test_multiheadattention_4()
{
    return 0
           || test_multiheadattention_int8_samekv(RandomMat(64, 128), RandomMat(64, 128), 64, 4)
           || test_multiheadattention_int8_samekv(RandomMat(48, 127), RandomMat(64, 127), 64, 16)
           || test_multiheadattention_int8_samekv(RandomMat(16, 128), RandomMat(44, 128), 16, 2)
           || test_multiheadattention_int8_samekv(RandomMat(12, 128), RandomMat(22, 127), 16, 4)
           || test_multiheadattention_int8_samekv(RandomMat(12, 17), RandomMat(28, 127), 12, 3)
           || test_multiheadattention_int8_samekv(RandomMat(12, 17), RandomMat(11, 32), 12, 3);
}

static int test_multiheadattention_5()
{
    return 0
           || test_multiheadattention_int8_sameqkv(RandomMat(64, 128), 64, 4)
           || test_multiheadattention_int8_sameqkv(RandomMat(48, 127), 64, 8);
}
#endif

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return 0
           // || test_multiheadattention_0()
           // || test_multiheadattention_1()
           // || test_multiheadattention_2()
           || test_multiheadattention_3()
           || test_multiheadattention_4()
           || test_multiheadattention_5();
#else
    return 0
           || test_multiheadattention_0()
           || test_multiheadattention_1()
           || test_multiheadattention_2();
#endif
}
