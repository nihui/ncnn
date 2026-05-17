// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void sdpa_qk_tile_fp32(const float* query, const float* key, float* score, int i, int max_ii, int n_start, int max_jj, int embed_dim, float scale)
{
    const int BLOCK_N = 256;

    int ii = 0;
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const float* qptr0 = query + (i + ii) * embed_dim;
        const float* qptr1 = query + (i + ii + 1) * embed_dim;
        const float* qptr2 = query + (i + ii + 2) * embed_dim;
        const float* qptr3 = query + (i + ii + 3) * embed_dim;
        const float* qptr4 = query + (i + ii + 4) * embed_dim;
        const float* qptr5 = query + (i + ii + 5) * embed_dim;
        const float* qptr6 = query + (i + ii + 6) * embed_dim;
        const float* qptr7 = query + (i + ii + 7) * embed_dim;
        const float* qptr8 = query + (i + ii + 8) * embed_dim;
        const float* qptr9 = query + (i + ii + 9) * embed_dim;
        const float* qptra = query + (i + ii + 10) * embed_dim;
        const float* qptrb = query + (i + ii + 11) * embed_dim;
        const float* qptrc = query + (i + ii + 12) * embed_dim;
        const float* qptrd = query + (i + ii + 13) * embed_dim;
        const float* qptre = query + (i + ii + 14) * embed_dim;
        const float* qptrf = query + (i + ii + 15) * embed_dim;

        float* sptr0 = score + ii * BLOCK_N;
        float* sptr1 = sptr0 + BLOCK_N;
        float* sptr2 = sptr1 + BLOCK_N;
        float* sptr3 = sptr2 + BLOCK_N;
        float* sptr4 = sptr3 + BLOCK_N;
        float* sptr5 = sptr4 + BLOCK_N;
        float* sptr6 = sptr5 + BLOCK_N;
        float* sptr7 = sptr6 + BLOCK_N;
        float* sptr8 = sptr7 + BLOCK_N;
        float* sptr9 = sptr8 + BLOCK_N;
        float* sptra = sptr9 + BLOCK_N;
        float* sptrb = sptra + BLOCK_N;
        float* sptrc = sptrb + BLOCK_N;
        float* sptrd = sptrc + BLOCK_N;
        float* sptre = sptrd + BLOCK_N;
        float* sptrf = sptre + BLOCK_N;

        for (int j = 0; j < max_jj; j++)
        {
            const float* kptr = key + (n_start + j) * embed_dim;

            int k = 0;
            __m512 _sum0 = _mm512_setzero_ps();
            __m512 _sum1 = _mm512_setzero_ps();
            __m512 _sum2 = _mm512_setzero_ps();
            __m512 _sum3 = _mm512_setzero_ps();
            __m512 _sum4 = _mm512_setzero_ps();
            __m512 _sum5 = _mm512_setzero_ps();
            __m512 _sum6 = _mm512_setzero_ps();
            __m512 _sum7 = _mm512_setzero_ps();
            __m512 _sum8 = _mm512_setzero_ps();
            __m512 _sum9 = _mm512_setzero_ps();
            __m512 _suma = _mm512_setzero_ps();
            __m512 _sumb = _mm512_setzero_ps();
            __m512 _sumc = _mm512_setzero_ps();
            __m512 _sumd = _mm512_setzero_ps();
            __m512 _sume = _mm512_setzero_ps();
            __m512 _sumf = _mm512_setzero_ps();

            for (; k + 15 < embed_dim; k += 16)
            {
                __m512 _k = _mm512_loadu_ps(kptr + k);
                _sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr0 + k), _k, _sum0);
                _sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr1 + k), _k, _sum1);
                _sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr2 + k), _k, _sum2);
                _sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr3 + k), _k, _sum3);
                _sum4 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr4 + k), _k, _sum4);
                _sum5 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr5 + k), _k, _sum5);
                _sum6 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr6 + k), _k, _sum6);
                _sum7 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr7 + k), _k, _sum7);
                _sum8 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr8 + k), _k, _sum8);
                _sum9 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr9 + k), _k, _sum9);
                _suma = _mm512_fmadd_ps(_mm512_loadu_ps(qptra + k), _k, _suma);
                _sumb = _mm512_fmadd_ps(_mm512_loadu_ps(qptrb + k), _k, _sumb);
                _sumc = _mm512_fmadd_ps(_mm512_loadu_ps(qptrc + k), _k, _sumc);
                _sumd = _mm512_fmadd_ps(_mm512_loadu_ps(qptrd + k), _k, _sumd);
                _sume = _mm512_fmadd_ps(_mm512_loadu_ps(qptre + k), _k, _sume);
                _sumf = _mm512_fmadd_ps(_mm512_loadu_ps(qptrf + k), _k, _sumf);
            }

            float sum0 = _mm512_comp_reduce_add_ps(_sum0);
            float sum1 = _mm512_comp_reduce_add_ps(_sum1);
            float sum2 = _mm512_comp_reduce_add_ps(_sum2);
            float sum3 = _mm512_comp_reduce_add_ps(_sum3);
            float sum4 = _mm512_comp_reduce_add_ps(_sum4);
            float sum5 = _mm512_comp_reduce_add_ps(_sum5);
            float sum6 = _mm512_comp_reduce_add_ps(_sum6);
            float sum7 = _mm512_comp_reduce_add_ps(_sum7);
            float sum8 = _mm512_comp_reduce_add_ps(_sum8);
            float sum9 = _mm512_comp_reduce_add_ps(_sum9);
            float suma = _mm512_comp_reduce_add_ps(_suma);
            float sumb = _mm512_comp_reduce_add_ps(_sumb);
            float sumc = _mm512_comp_reduce_add_ps(_sumc);
            float sumd = _mm512_comp_reduce_add_ps(_sumd);
            float sume = _mm512_comp_reduce_add_ps(_sume);
            float sumf = _mm512_comp_reduce_add_ps(_sumf);

            for (; k < embed_dim; k++)
            {
                const float v = kptr[k];
                sum0 += qptr0[k] * v;
                sum1 += qptr1[k] * v;
                sum2 += qptr2[k] * v;
                sum3 += qptr3[k] * v;
                sum4 += qptr4[k] * v;
                sum5 += qptr5[k] * v;
                sum6 += qptr6[k] * v;
                sum7 += qptr7[k] * v;
                sum8 += qptr8[k] * v;
                sum9 += qptr9[k] * v;
                suma += qptra[k] * v;
                sumb += qptrb[k] * v;
                sumc += qptrc[k] * v;
                sumd += qptrd[k] * v;
                sume += qptre[k] * v;
                sumf += qptrf[k] * v;
            }

            sptr0[j] = sum0 * scale;
            sptr1[j] = sum1 * scale;
            sptr2[j] = sum2 * scale;
            sptr3[j] = sum3 * scale;
            sptr4[j] = sum4 * scale;
            sptr5[j] = sum5 * scale;
            sptr6[j] = sum6 * scale;
            sptr7[j] = sum7 * scale;
            sptr8[j] = sum8 * scale;
            sptr9[j] = sum9 * scale;
            sptra[j] = suma * scale;
            sptrb[j] = sumb * scale;
            sptrc[j] = sumc * scale;
            sptrd[j] = sumd * scale;
            sptre[j] = sume * scale;
            sptrf[j] = sumf * scale;
        }
    }
#endif

    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* qptr0 = query + (i + ii) * embed_dim;
        const float* qptr1 = query + (i + ii + 1) * embed_dim;
        const float* qptr2 = query + (i + ii + 2) * embed_dim;
        const float* qptr3 = query + (i + ii + 3) * embed_dim;
        const float* qptr4 = query + (i + ii + 4) * embed_dim;
        const float* qptr5 = query + (i + ii + 5) * embed_dim;
        const float* qptr6 = query + (i + ii + 6) * embed_dim;
        const float* qptr7 = query + (i + ii + 7) * embed_dim;

        float* sptr0 = score + ii * BLOCK_N;
        float* sptr1 = sptr0 + BLOCK_N;
        float* sptr2 = sptr1 + BLOCK_N;
        float* sptr3 = sptr2 + BLOCK_N;
        float* sptr4 = sptr3 + BLOCK_N;
        float* sptr5 = sptr4 + BLOCK_N;
        float* sptr6 = sptr5 + BLOCK_N;
        float* sptr7 = sptr6 + BLOCK_N;

        for (int j = 0; j < max_jj; j++)
        {
            const float* kptr = key + (n_start + j) * embed_dim;

            int k = 0;
            float sum0 = 0.f;
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            float sum4 = 0.f;
            float sum5 = 0.f;
            float sum6 = 0.f;
            float sum7 = 0.f;

#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _sum0 = _mm512_setzero_ps();
            __m512 _sum1 = _mm512_setzero_ps();
            __m512 _sum2 = _mm512_setzero_ps();
            __m512 _sum3 = _mm512_setzero_ps();
            __m512 _sum4 = _mm512_setzero_ps();
            __m512 _sum5 = _mm512_setzero_ps();
            __m512 _sum6 = _mm512_setzero_ps();
            __m512 _sum7 = _mm512_setzero_ps();
            for (; k + 15 < embed_dim; k += 16)
            {
                __m512 _k = _mm512_loadu_ps(kptr + k);
                _sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr0 + k), _k, _sum0);
                _sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr1 + k), _k, _sum1);
                _sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr2 + k), _k, _sum2);
                _sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr3 + k), _k, _sum3);
                _sum4 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr4 + k), _k, _sum4);
                _sum5 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr5 + k), _k, _sum5);
                _sum6 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr6 + k), _k, _sum6);
                _sum7 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr7 + k), _k, _sum7);
            }
            sum0 += _mm512_comp_reduce_add_ps(_sum0);
            sum1 += _mm512_comp_reduce_add_ps(_sum1);
            sum2 += _mm512_comp_reduce_add_ps(_sum2);
            sum3 += _mm512_comp_reduce_add_ps(_sum3);
            sum4 += _mm512_comp_reduce_add_ps(_sum4);
            sum5 += _mm512_comp_reduce_add_ps(_sum5);
            sum6 += _mm512_comp_reduce_add_ps(_sum6);
            sum7 += _mm512_comp_reduce_add_ps(_sum7);
#endif // __AVX512F__
            {
                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();
                __m256 _sum2 = _mm256_setzero_ps();
                __m256 _sum3 = _mm256_setzero_ps();
                __m256 _sum4 = _mm256_setzero_ps();
                __m256 _sum5 = _mm256_setzero_ps();
                __m256 _sum6 = _mm256_setzero_ps();
                __m256 _sum7 = _mm256_setzero_ps();
                for (; k + 7 < embed_dim; k += 8)
                {
                    __m256 _k = _mm256_loadu_ps(kptr + k);
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr0 + k), _k, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr1 + k), _k, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr2 + k), _k, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr3 + k), _k, _sum3);
                    _sum4 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr4 + k), _k, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr5 + k), _k, _sum5);
                    _sum6 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr6 + k), _k, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr7 + k), _k, _sum7);
                }
                sum0 += _mm256_reduce_add_ps(_sum0);
                sum1 += _mm256_reduce_add_ps(_sum1);
                sum2 += _mm256_reduce_add_ps(_sum2);
                sum3 += _mm256_reduce_add_ps(_sum3);
                sum4 += _mm256_reduce_add_ps(_sum4);
                sum5 += _mm256_reduce_add_ps(_sum5);
                sum6 += _mm256_reduce_add_ps(_sum6);
                sum7 += _mm256_reduce_add_ps(_sum7);
            }
#endif // __AVX__
            {
                __m128 _sum0 = _mm_setzero_ps();
                __m128 _sum1 = _mm_setzero_ps();
                __m128 _sum2 = _mm_setzero_ps();
                __m128 _sum3 = _mm_setzero_ps();
                __m128 _sum4 = _mm_setzero_ps();
                __m128 _sum5 = _mm_setzero_ps();
                __m128 _sum6 = _mm_setzero_ps();
                __m128 _sum7 = _mm_setzero_ps();
                for (; k + 3 < embed_dim; k += 4)
                {
                    __m128 _k = _mm_loadu_ps(kptr + k);
                    _sum0 = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr0 + k), _k, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr1 + k), _k, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr2 + k), _k, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr3 + k), _k, _sum3);
                    _sum4 = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr4 + k), _k, _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr5 + k), _k, _sum5);
                    _sum6 = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr6 + k), _k, _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr7 + k), _k, _sum7);
                }
                sum0 += _mm_reduce_add_ps(_sum0);
                sum1 += _mm_reduce_add_ps(_sum1);
                sum2 += _mm_reduce_add_ps(_sum2);
                sum3 += _mm_reduce_add_ps(_sum3);
                sum4 += _mm_reduce_add_ps(_sum4);
                sum5 += _mm_reduce_add_ps(_sum5);
                sum6 += _mm_reduce_add_ps(_sum6);
                sum7 += _mm_reduce_add_ps(_sum7);
            }
#endif // __SSE2__
            for (; k < embed_dim; k++)
            {
                const float v = kptr[k];
                sum0 += qptr0[k] * v;
                sum1 += qptr1[k] * v;
                sum2 += qptr2[k] * v;
                sum3 += qptr3[k] * v;
                sum4 += qptr4[k] * v;
                sum5 += qptr5[k] * v;
                sum6 += qptr6[k] * v;
                sum7 += qptr7[k] * v;
            }

            sptr0[j] = sum0 * scale;
            sptr1[j] = sum1 * scale;
            sptr2[j] = sum2 * scale;
            sptr3[j] = sum3 * scale;
            sptr4[j] = sum4 * scale;
            sptr5[j] = sum5 * scale;
            sptr6[j] = sum6 * scale;
            sptr7[j] = sum7 * scale;
        }
    }

    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* qptr0 = query + (i + ii) * embed_dim;
        const float* qptr1 = query + (i + ii + 1) * embed_dim;
        const float* qptr2 = query + (i + ii + 2) * embed_dim;
        const float* qptr3 = query + (i + ii + 3) * embed_dim;

        float* sptr0 = score + ii * BLOCK_N;
        float* sptr1 = sptr0 + BLOCK_N;
        float* sptr2 = sptr1 + BLOCK_N;
        float* sptr3 = sptr2 + BLOCK_N;

        for (int j = 0; j < max_jj; j++)
        {
            const float* kptr = key + (n_start + j) * embed_dim;

            int k = 0;
            float sum0 = 0.f;
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;

#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _sum0 = _mm512_setzero_ps();
            __m512 _sum1 = _mm512_setzero_ps();
            __m512 _sum2 = _mm512_setzero_ps();
            __m512 _sum3 = _mm512_setzero_ps();
            for (; k + 15 < embed_dim; k += 16)
            {
                __m512 _k = _mm512_loadu_ps(kptr + k);
                _sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr0 + k), _k, _sum0);
                _sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr1 + k), _k, _sum1);
                _sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr2 + k), _k, _sum2);
                _sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr3 + k), _k, _sum3);
            }
            sum0 += _mm512_comp_reduce_add_ps(_sum0);
            sum1 += _mm512_comp_reduce_add_ps(_sum1);
            sum2 += _mm512_comp_reduce_add_ps(_sum2);
            sum3 += _mm512_comp_reduce_add_ps(_sum3);
#endif // __AVX512F__
            {
                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();
                __m256 _sum2 = _mm256_setzero_ps();
                __m256 _sum3 = _mm256_setzero_ps();
                for (; k + 7 < embed_dim; k += 8)
                {
                    __m256 _k = _mm256_loadu_ps(kptr + k);
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr0 + k), _k, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr1 + k), _k, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr2 + k), _k, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr3 + k), _k, _sum3);
                }
                sum0 += _mm256_reduce_add_ps(_sum0);
                sum1 += _mm256_reduce_add_ps(_sum1);
                sum2 += _mm256_reduce_add_ps(_sum2);
                sum3 += _mm256_reduce_add_ps(_sum3);
            }
#endif // __AVX__
            {
                __m128 _sum0 = _mm_setzero_ps();
                __m128 _sum1 = _mm_setzero_ps();
                __m128 _sum2 = _mm_setzero_ps();
                __m128 _sum3 = _mm_setzero_ps();
                for (; k + 3 < embed_dim; k += 4)
                {
                    __m128 _k = _mm_loadu_ps(kptr + k);
                    _sum0 = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr0 + k), _k, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr1 + k), _k, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr2 + k), _k, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr3 + k), _k, _sum3);
                }
                sum0 += _mm_reduce_add_ps(_sum0);
                sum1 += _mm_reduce_add_ps(_sum1);
                sum2 += _mm_reduce_add_ps(_sum2);
                sum3 += _mm_reduce_add_ps(_sum3);
            }
#endif // __SSE2__
            for (; k < embed_dim; k++)
            {
                const float v = kptr[k];
                sum0 += qptr0[k] * v;
                sum1 += qptr1[k] * v;
                sum2 += qptr2[k] * v;
                sum3 += qptr3[k] * v;
            }

            sptr0[j] = sum0 * scale;
            sptr1[j] = sum1 * scale;
            sptr2[j] = sum2 * scale;
            sptr3[j] = sum3 * scale;
        }
    }

    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* qptr0 = query + (i + ii) * embed_dim;
        const float* qptr1 = query + (i + ii + 1) * embed_dim;

        float* sptr0 = score + ii * BLOCK_N;
        float* sptr1 = sptr0 + BLOCK_N;

        for (int j = 0; j < max_jj; j++)
        {
            const float* kptr = key + (n_start + j) * embed_dim;

            int k = 0;
            float sum0 = 0.f;
            float sum1 = 0.f;

#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _sum0 = _mm512_setzero_ps();
            __m512 _sum1 = _mm512_setzero_ps();
            for (; k + 15 < embed_dim; k += 16)
            {
                __m512 _k = _mm512_loadu_ps(kptr + k);
                _sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr0 + k), _k, _sum0);
                _sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr1 + k), _k, _sum1);
            }
            sum0 += _mm512_comp_reduce_add_ps(_sum0);
            sum1 += _mm512_comp_reduce_add_ps(_sum1);
#endif // __AVX512F__
            {
                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();
                for (; k + 7 < embed_dim; k += 8)
                {
                    __m256 _k = _mm256_loadu_ps(kptr + k);
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr0 + k), _k, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr1 + k), _k, _sum1);
                }
                sum0 += _mm256_reduce_add_ps(_sum0);
                sum1 += _mm256_reduce_add_ps(_sum1);
            }
#endif // __AVX__
            {
                __m128 _sum0 = _mm_setzero_ps();
                __m128 _sum1 = _mm_setzero_ps();
                for (; k + 3 < embed_dim; k += 4)
                {
                    __m128 _k = _mm_loadu_ps(kptr + k);
                    _sum0 = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr0 + k), _k, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr1 + k), _k, _sum1);
                }
                sum0 += _mm_reduce_add_ps(_sum0);
                sum1 += _mm_reduce_add_ps(_sum1);
            }
#endif // __SSE2__
            for (; k < embed_dim; k++)
            {
                const float v = kptr[k];
                sum0 += qptr0[k] * v;
                sum1 += qptr1[k] * v;
            }

            sptr0[j] = sum0 * scale;
            sptr1[j] = sum1 * scale;
        }
    }

    for (; ii < max_ii; ii++)
    {
        const float* qptr = query + (i + ii) * embed_dim;
        float* sptr = score + ii * BLOCK_N;

        for (int j = 0; j < max_jj; j++)
        {
            const float* kptr = key + (n_start + j) * embed_dim;
            float sum0 = 0.f;

            int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _sum0 = _mm512_setzero_ps();
            for (; k + 15 < embed_dim; k += 16)
            {
                __m512 _q = _mm512_loadu_ps(qptr + k);
                __m512 _k = _mm512_loadu_ps(kptr + k);
                _sum0 = _mm512_fmadd_ps(_q, _k, _sum0);
            }
            sum0 += _mm512_comp_reduce_add_ps(_sum0);
#endif // __AVX512F__
            {
                __m256 _sum0 = _mm256_setzero_ps();
                for (; k + 7 < embed_dim; k += 8)
                {
                    __m256 _q = _mm256_loadu_ps(qptr + k);
                    __m256 _k = _mm256_loadu_ps(kptr + k);
                    _sum0 = _mm256_comp_fmadd_ps(_q, _k, _sum0);
                }
                sum0 += _mm256_reduce_add_ps(_sum0);
            }
#endif // __AVX__
            {
                __m128 _sum0 = _mm_setzero_ps();
                for (; k + 3 < embed_dim; k += 4)
                {
                    __m128 _q = _mm_loadu_ps(qptr + k);
                    __m128 _k = _mm_loadu_ps(kptr + k);
                    _sum0 = _mm_comp_fmadd_ps(_q, _k, _sum0);
                }
                sum0 += _mm_reduce_add_ps(_sum0);
            }
#endif // __SSE2__
            for (; k < embed_dim; k++)
            {
                sum0 += qptr[k] * kptr[k];
            }

            sptr[j] = sum0 * scale;
        }
    }
}

static inline size_t sdpa_packed_key_tile_stride_fp32(int tile_n, int embed_dim)
{
    return (size_t)tile_n * embed_dim;
}

static inline size_t sdpa_packed_value_tile_stride_fp32(int tile_n, int out_embed_dim)
{
    return (size_t)tile_n * out_embed_dim;
}

// packed_key tile layout:
//   j blocks are stored as 16 / 8 / 4 / 1 rows, selected by available ISA.
//   Inside each j block, data is k-major and lane-contiguous:
//     [k0: key_j0 key_j1 ...], [k1: key_j0 key_j1 ...], ...
//
// packed_value tile layout:
//   k blocks are stored as 16 / 8 / 4 / 1 columns, selected by available ISA.
//   Inside each k block, data is j-major and k-lane-contiguous:
//     [value_j0_k0..k15], [value_j1_k0..k15], ...
//
// score tile layout is j-major with query lanes contiguous.
// outT layout is output-dim-major with query lanes contiguous.

#if __AVX512F__
static void sdpa_pack_query_tile16_fp32(const float* query, float* queryT, int i, int embed_dim)
{
    const float* qptr0 = query + (i + 0) * embed_dim;
    const float* qptr1 = query + (i + 1) * embed_dim;
    const float* qptr2 = query + (i + 2) * embed_dim;
    const float* qptr3 = query + (i + 3) * embed_dim;
    const float* qptr4 = query + (i + 4) * embed_dim;
    const float* qptr5 = query + (i + 5) * embed_dim;
    const float* qptr6 = query + (i + 6) * embed_dim;
    const float* qptr7 = query + (i + 7) * embed_dim;
    const float* qptr8 = query + (i + 8) * embed_dim;
    const float* qptr9 = query + (i + 9) * embed_dim;
    const float* qptra = query + (i + 10) * embed_dim;
    const float* qptrb = query + (i + 11) * embed_dim;
    const float* qptrc = query + (i + 12) * embed_dim;
    const float* qptrd = query + (i + 13) * embed_dim;
    const float* qptre = query + (i + 14) * embed_dim;
    const float* qptrf = query + (i + 15) * embed_dim;

    int k = 0;
    for (; k + 15 < embed_dim; k += 16)
    {
        __m512 _r0 = _mm512_loadu_ps(qptr0 + k);
        __m512 _r1 = _mm512_loadu_ps(qptr1 + k);
        __m512 _r2 = _mm512_loadu_ps(qptr2 + k);
        __m512 _r3 = _mm512_loadu_ps(qptr3 + k);
        __m512 _r4 = _mm512_loadu_ps(qptr4 + k);
        __m512 _r5 = _mm512_loadu_ps(qptr5 + k);
        __m512 _r6 = _mm512_loadu_ps(qptr6 + k);
        __m512 _r7 = _mm512_loadu_ps(qptr7 + k);
        __m512 _r8 = _mm512_loadu_ps(qptr8 + k);
        __m512 _r9 = _mm512_loadu_ps(qptr9 + k);
        __m512 _ra = _mm512_loadu_ps(qptra + k);
        __m512 _rb = _mm512_loadu_ps(qptrb + k);
        __m512 _rc = _mm512_loadu_ps(qptrc + k);
        __m512 _rd = _mm512_loadu_ps(qptrd + k);
        __m512 _re = _mm512_loadu_ps(qptre + k);
        __m512 _rf = _mm512_loadu_ps(qptrf + k);

        transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);

        float* outptr = queryT + k * 16;
        _mm512_storeu_ps(outptr, _r0);
        _mm512_storeu_ps(outptr + 16, _r1);
        _mm512_storeu_ps(outptr + 32, _r2);
        _mm512_storeu_ps(outptr + 48, _r3);
        _mm512_storeu_ps(outptr + 64, _r4);
        _mm512_storeu_ps(outptr + 80, _r5);
        _mm512_storeu_ps(outptr + 96, _r6);
        _mm512_storeu_ps(outptr + 112, _r7);
        _mm512_storeu_ps(outptr + 128, _r8);
        _mm512_storeu_ps(outptr + 144, _r9);
        _mm512_storeu_ps(outptr + 160, _ra);
        _mm512_storeu_ps(outptr + 176, _rb);
        _mm512_storeu_ps(outptr + 192, _rc);
        _mm512_storeu_ps(outptr + 208, _rd);
        _mm512_storeu_ps(outptr + 224, _re);
        _mm512_storeu_ps(outptr + 240, _rf);
    }

    for (; k < embed_dim; k++)
    {
        float* outptr = queryT + k * 16;
        outptr[0] = qptr0[k];
        outptr[1] = qptr1[k];
        outptr[2] = qptr2[k];
        outptr[3] = qptr3[k];
        outptr[4] = qptr4[k];
        outptr[5] = qptr5[k];
        outptr[6] = qptr6[k];
        outptr[7] = qptr7[k];
        outptr[8] = qptr8[k];
        outptr[9] = qptr9[k];
        outptr[10] = qptra[k];
        outptr[11] = qptrb[k];
        outptr[12] = qptrc[k];
        outptr[13] = qptrd[k];
        outptr[14] = qptre[k];
        outptr[15] = qptrf[k];
    }
}

#endif // __AVX512F__

static void sdpa_store_out_tile_fp32(float* outptr, const float* outT, int out_embed_dim, int packn, const float* scale)
{
#if __AVX512F__
    if (packn == 16)
    {
        __m512 _scale = _mm512_loadu_ps(scale);

        float* outptr0 = outptr;
        float* outptr1 = outptr0 + out_embed_dim;
        float* outptr2 = outptr1 + out_embed_dim;
        float* outptr3 = outptr2 + out_embed_dim;
        float* outptr4 = outptr3 + out_embed_dim;
        float* outptr5 = outptr4 + out_embed_dim;
        float* outptr6 = outptr5 + out_embed_dim;
        float* outptr7 = outptr6 + out_embed_dim;
        float* outptr8 = outptr7 + out_embed_dim;
        float* outptr9 = outptr8 + out_embed_dim;
        float* outptra = outptr9 + out_embed_dim;
        float* outptrb = outptra + out_embed_dim;
        float* outptrc = outptrb + out_embed_dim;
        float* outptrd = outptrc + out_embed_dim;
        float* outptre = outptrd + out_embed_dim;
        float* outptrf = outptre + out_embed_dim;

        const float* pp = outT;
        int k = 0;
        for (; k + 15 < out_embed_dim; k += 16)
        {
            __m512 _r0 = _mm512_mul_ps(_mm512_loadu_ps(pp), _scale);
            __m512 _r1 = _mm512_mul_ps(_mm512_loadu_ps(pp + 16), _scale);
            __m512 _r2 = _mm512_mul_ps(_mm512_loadu_ps(pp + 16 * 2), _scale);
            __m512 _r3 = _mm512_mul_ps(_mm512_loadu_ps(pp + 16 * 3), _scale);
            __m512 _r4 = _mm512_mul_ps(_mm512_loadu_ps(pp + 16 * 4), _scale);
            __m512 _r5 = _mm512_mul_ps(_mm512_loadu_ps(pp + 16 * 5), _scale);
            __m512 _r6 = _mm512_mul_ps(_mm512_loadu_ps(pp + 16 * 6), _scale);
            __m512 _r7 = _mm512_mul_ps(_mm512_loadu_ps(pp + 16 * 7), _scale);
            __m512 _r8 = _mm512_mul_ps(_mm512_loadu_ps(pp + 16 * 8), _scale);
            __m512 _r9 = _mm512_mul_ps(_mm512_loadu_ps(pp + 16 * 9), _scale);
            __m512 _ra = _mm512_mul_ps(_mm512_loadu_ps(pp + 16 * 10), _scale);
            __m512 _rb = _mm512_mul_ps(_mm512_loadu_ps(pp + 16 * 11), _scale);
            __m512 _rc = _mm512_mul_ps(_mm512_loadu_ps(pp + 16 * 12), _scale);
            __m512 _rd = _mm512_mul_ps(_mm512_loadu_ps(pp + 16 * 13), _scale);
            __m512 _re = _mm512_mul_ps(_mm512_loadu_ps(pp + 16 * 14), _scale);
            __m512 _rf = _mm512_mul_ps(_mm512_loadu_ps(pp + 16 * 15), _scale);

            transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);

            _mm512_storeu_ps(outptr0 + k, _r0);
            _mm512_storeu_ps(outptr1 + k, _r1);
            _mm512_storeu_ps(outptr2 + k, _r2);
            _mm512_storeu_ps(outptr3 + k, _r3);
            _mm512_storeu_ps(outptr4 + k, _r4);
            _mm512_storeu_ps(outptr5 + k, _r5);
            _mm512_storeu_ps(outptr6 + k, _r6);
            _mm512_storeu_ps(outptr7 + k, _r7);
            _mm512_storeu_ps(outptr8 + k, _r8);
            _mm512_storeu_ps(outptr9 + k, _r9);
            _mm512_storeu_ps(outptra + k, _ra);
            _mm512_storeu_ps(outptrb + k, _rb);
            _mm512_storeu_ps(outptrc + k, _rc);
            _mm512_storeu_ps(outptrd + k, _rd);
            _mm512_storeu_ps(outptre + k, _re);
            _mm512_storeu_ps(outptrf + k, _rf);

            pp += 256;
        }

        for (; k < out_embed_dim; k++)
        {
            outptr0[k] = pp[0] * scale[0];
            outptr1[k] = pp[1] * scale[1];
            outptr2[k] = pp[2] * scale[2];
            outptr3[k] = pp[3] * scale[3];
            outptr4[k] = pp[4] * scale[4];
            outptr5[k] = pp[5] * scale[5];
            outptr6[k] = pp[6] * scale[6];
            outptr7[k] = pp[7] * scale[7];
            outptr8[k] = pp[8] * scale[8];
            outptr9[k] = pp[9] * scale[9];
            outptra[k] = pp[10] * scale[10];
            outptrb[k] = pp[11] * scale[11];
            outptrc[k] = pp[12] * scale[12];
            outptrd[k] = pp[13] * scale[13];
            outptre[k] = pp[14] * scale[14];
            outptrf[k] = pp[15] * scale[15];
            pp += 16;
        }

        return;
    }
#endif // __AVX512F__

#if __AVX__
    if (packn == 8)
    {
        const __m256 _scale = _mm256_loadu_ps(scale);

        float* outptr0 = outptr;
        float* outptr1 = outptr0 + out_embed_dim;
        float* outptr2 = outptr1 + out_embed_dim;
        float* outptr3 = outptr2 + out_embed_dim;
        float* outptr4 = outptr3 + out_embed_dim;
        float* outptr5 = outptr4 + out_embed_dim;
        float* outptr6 = outptr5 + out_embed_dim;
        float* outptr7 = outptr6 + out_embed_dim;

        const float* pp = outT;
        int k = 0;
#if __AVX512F__
        for (; k + 15 < out_embed_dim; k += 16)
        {
            __m256 _r0 = _mm256_mul_ps(_mm256_loadu_ps(pp), _scale);
            __m256 _r1 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8), _scale);
            __m256 _r2 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 2), _scale);
            __m256 _r3 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 3), _scale);
            __m256 _r4 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 4), _scale);
            __m256 _r5 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 5), _scale);
            __m256 _r6 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 6), _scale);
            __m256 _r7 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 7), _scale);
            __m256 _r8 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 8), _scale);
            __m256 _r9 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 9), _scale);
            __m256 _ra = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 10), _scale);
            __m256 _rb = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 11), _scale);
            __m256 _rc = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 12), _scale);
            __m256 _rd = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 13), _scale);
            __m256 _re = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 14), _scale);
            __m256 _rf = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 15), _scale);

            transpose8x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);

            _mm512_storeu_ps(outptr0 + k, combine8x2_ps(_r0, _r1));
            _mm512_storeu_ps(outptr1 + k, combine8x2_ps(_r2, _r3));
            _mm512_storeu_ps(outptr2 + k, combine8x2_ps(_r4, _r5));
            _mm512_storeu_ps(outptr3 + k, combine8x2_ps(_r6, _r7));
            _mm512_storeu_ps(outptr4 + k, combine8x2_ps(_r8, _r9));
            _mm512_storeu_ps(outptr5 + k, combine8x2_ps(_ra, _rb));
            _mm512_storeu_ps(outptr6 + k, combine8x2_ps(_rc, _rd));
            _mm512_storeu_ps(outptr7 + k, combine8x2_ps(_re, _rf));

            pp += 128;
        }
#endif // __AVX512F__
        for (; k + 7 < out_embed_dim; k += 8)
        {
            __m256 _r0 = _mm256_mul_ps(_mm256_loadu_ps(pp), _scale);
            __m256 _r1 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8), _scale);
            __m256 _r2 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 2), _scale);
            __m256 _r3 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 3), _scale);
            __m256 _r4 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 4), _scale);
            __m256 _r5 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 5), _scale);
            __m256 _r6 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 6), _scale);
            __m256 _r7 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8 * 7), _scale);

            transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

            _mm256_storeu_ps(outptr0 + k, _r0);
            _mm256_storeu_ps(outptr1 + k, _r1);
            _mm256_storeu_ps(outptr2 + k, _r2);
            _mm256_storeu_ps(outptr3 + k, _r3);
            _mm256_storeu_ps(outptr4 + k, _r4);
            _mm256_storeu_ps(outptr5 + k, _r5);
            _mm256_storeu_ps(outptr6 + k, _r6);
            _mm256_storeu_ps(outptr7 + k, _r7);

            pp += 64;
        }

        for (; k < out_embed_dim; k++)
        {
            outptr0[k] = pp[0] * scale[0];
            outptr1[k] = pp[1] * scale[1];
            outptr2[k] = pp[2] * scale[2];
            outptr3[k] = pp[3] * scale[3];
            outptr4[k] = pp[4] * scale[4];
            outptr5[k] = pp[5] * scale[5];
            outptr6[k] = pp[6] * scale[6];
            outptr7[k] = pp[7] * scale[7];
            pp += 8;
        }

        return;
    }
#endif // __AVX__

#if __SSE2__
    if (packn == 4)
    {
        const __m128 _scale = _mm_loadu_ps(scale);

        float* outptr0 = outptr;
        float* outptr1 = outptr0 + out_embed_dim;
        float* outptr2 = outptr1 + out_embed_dim;
        float* outptr3 = outptr2 + out_embed_dim;

        const float* pp = outT;
        int k = 0;
#if __AVX512F__
        for (; k + 15 < out_embed_dim; k += 16)
        {
            __m128 _r0 = _mm_mul_ps(_mm_loadu_ps(pp), _scale);
            __m128 _r1 = _mm_mul_ps(_mm_loadu_ps(pp + 4), _scale);
            __m128 _r2 = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 2), _scale);
            __m128 _r3 = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 3), _scale);
            __m128 _r4 = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 4), _scale);
            __m128 _r5 = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 5), _scale);
            __m128 _r6 = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 6), _scale);
            __m128 _r7 = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 7), _scale);
            __m128 _r8 = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 8), _scale);
            __m128 _r9 = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 9), _scale);
            __m128 _ra = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 10), _scale);
            __m128 _rb = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 11), _scale);
            __m128 _rc = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 12), _scale);
            __m128 _rd = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 13), _scale);
            __m128 _re = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 14), _scale);
            __m128 _rf = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 15), _scale);

            _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
            _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
            _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);
            _MM_TRANSPOSE4_PS(_rc, _rd, _re, _rf);

            _mm512_storeu_ps(outptr0 + k, combine4x4_ps(_r0, _r4, _r8, _rc));
            _mm512_storeu_ps(outptr1 + k, combine4x4_ps(_r1, _r5, _r9, _rd));
            _mm512_storeu_ps(outptr2 + k, combine4x4_ps(_r2, _r6, _ra, _re));
            _mm512_storeu_ps(outptr3 + k, combine4x4_ps(_r3, _r7, _rb, _rf));

            pp += 64;
        }
#endif // __AVX512F__
#if __AVX__
        for (; k + 7 < out_embed_dim; k += 8)
        {
            __m128 _r0 = _mm_mul_ps(_mm_loadu_ps(pp), _scale);
            __m128 _r1 = _mm_mul_ps(_mm_loadu_ps(pp + 4), _scale);
            __m128 _r2 = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 2), _scale);
            __m128 _r3 = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 3), _scale);
            __m128 _r4 = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 4), _scale);
            __m128 _r5 = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 5), _scale);
            __m128 _r6 = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 6), _scale);
            __m128 _r7 = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 7), _scale);

            _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
            _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);

            _mm256_storeu_ps(outptr0 + k, combine4x2_ps(_r0, _r4));
            _mm256_storeu_ps(outptr1 + k, combine4x2_ps(_r1, _r5));
            _mm256_storeu_ps(outptr2 + k, combine4x2_ps(_r2, _r6));
            _mm256_storeu_ps(outptr3 + k, combine4x2_ps(_r3, _r7));

            pp += 32;
        }
#endif // __AVX__
        for (; k + 3 < out_embed_dim; k += 4)
        {
            __m128 _r0 = _mm_mul_ps(_mm_loadu_ps(pp), _scale);
            __m128 _r1 = _mm_mul_ps(_mm_loadu_ps(pp + 4), _scale);
            __m128 _r2 = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 2), _scale);
            __m128 _r3 = _mm_mul_ps(_mm_loadu_ps(pp + 4 * 3), _scale);

            _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

            _mm_storeu_ps(outptr0 + k, _r0);
            _mm_storeu_ps(outptr1 + k, _r1);
            _mm_storeu_ps(outptr2 + k, _r2);
            _mm_storeu_ps(outptr3 + k, _r3);

            pp += 16;
        }

        for (; k < out_embed_dim; k++)
        {
            outptr0[k] = pp[0] * scale[0];
            outptr1[k] = pp[1] * scale[1];
            outptr2[k] = pp[2] * scale[2];
            outptr3[k] = pp[3] * scale[3];
            pp += 4;
        }

        return;
    }
#endif // __SSE2__

    const float scale0 = scale[0];
    int k = 0;
#if __AVX512F__
    {
        const __m512 _scale = _mm512_set1_ps(scale0);
        for (; k + 15 < out_embed_dim; k += 16)
        {
            _mm512_storeu_ps(outptr + k, _mm512_mul_ps(_mm512_loadu_ps(outT + k), _scale));
        }
    }
#endif // __AVX512F__
#if __AVX__
    {
        const __m256 _scale = _mm256_set1_ps(scale0);
        for (; k + 7 < out_embed_dim; k += 8)
        {
            _mm256_storeu_ps(outptr + k, _mm256_mul_ps(_mm256_loadu_ps(outT + k), _scale));
        }
    }
#endif // __AVX__
#if __SSE2__
    {
        const __m128 _scale = _mm_set1_ps(scale0);
        for (; k + 3 < out_embed_dim; k += 4)
        {
            _mm_storeu_ps(outptr + k, _mm_mul_ps(_mm_loadu_ps(outT + k), _scale));
        }
    }
#endif // __SSE2__
    for (; k < out_embed_dim; k++)
    {
        outptr[k] = outT[k] * scale0;
    }
}

#if __AVX__
static void sdpa_pack_query_tile8_fp32(const float* query, float* queryT, int i, int embed_dim)
{
    const float* qptr0 = query + (i + 0) * embed_dim;
    const float* qptr1 = query + (i + 1) * embed_dim;
    const float* qptr2 = query + (i + 2) * embed_dim;
    const float* qptr3 = query + (i + 3) * embed_dim;
    const float* qptr4 = query + (i + 4) * embed_dim;
    const float* qptr5 = query + (i + 5) * embed_dim;
    const float* qptr6 = query + (i + 6) * embed_dim;
    const float* qptr7 = query + (i + 7) * embed_dim;

    int k = 0;
    for (; k + 7 < embed_dim; k += 8)
    {
        __m256 _r0 = _mm256_loadu_ps(qptr0 + k);
        __m256 _r1 = _mm256_loadu_ps(qptr1 + k);
        __m256 _r2 = _mm256_loadu_ps(qptr2 + k);
        __m256 _r3 = _mm256_loadu_ps(qptr3 + k);
        __m256 _r4 = _mm256_loadu_ps(qptr4 + k);
        __m256 _r5 = _mm256_loadu_ps(qptr5 + k);
        __m256 _r6 = _mm256_loadu_ps(qptr6 + k);
        __m256 _r7 = _mm256_loadu_ps(qptr7 + k);

        transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

        float* outptr = queryT + k * 8;
        _mm256_storeu_ps(outptr, _r0);
        _mm256_storeu_ps(outptr + 8, _r1);
        _mm256_storeu_ps(outptr + 16, _r2);
        _mm256_storeu_ps(outptr + 24, _r3);
        _mm256_storeu_ps(outptr + 32, _r4);
        _mm256_storeu_ps(outptr + 40, _r5);
        _mm256_storeu_ps(outptr + 48, _r6);
        _mm256_storeu_ps(outptr + 56, _r7);
    }

    for (; k < embed_dim; k++)
    {
        float* outptr = queryT + k * 8;
        outptr[0] = qptr0[k];
        outptr[1] = qptr1[k];
        outptr[2] = qptr2[k];
        outptr[3] = qptr3[k];
        outptr[4] = qptr4[k];
        outptr[5] = qptr5[k];
        outptr[6] = qptr6[k];
        outptr[7] = qptr7[k];
    }
}
#endif // __AVX__

#if __SSE2__
static void sdpa_pack_query_tile4_fp32(const float* query, float* queryT, int i, int embed_dim)
{
    const float* qptr0 = query + (i + 0) * embed_dim;
    const float* qptr1 = query + (i + 1) * embed_dim;
    const float* qptr2 = query + (i + 2) * embed_dim;
    const float* qptr3 = query + (i + 3) * embed_dim;

    int k = 0;
    for (; k + 3 < embed_dim; k += 4)
    {
        __m128 _r0 = _mm_loadu_ps(qptr0 + k);
        __m128 _r1 = _mm_loadu_ps(qptr1 + k);
        __m128 _r2 = _mm_loadu_ps(qptr2 + k);
        __m128 _r3 = _mm_loadu_ps(qptr3 + k);

        _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

        float* outptr = queryT + k * 4;
        _mm_storeu_ps(outptr, _r0);
        _mm_storeu_ps(outptr + 4, _r1);
        _mm_storeu_ps(outptr + 8, _r2);
        _mm_storeu_ps(outptr + 12, _r3);
    }

    for (; k < embed_dim; k++)
    {
        float* outptr = queryT + k * 4;
        outptr[0] = qptr0[k];
        outptr[1] = qptr1[k];
        outptr[2] = qptr2[k];
        outptr[3] = qptr3[k];
    }
}
#endif // __SSE2__


#if __AVX__
static __m256 sdpa_qk_tile8_packed_key_fp32(const float* queryT, const float* packed_key_tile, float* scoreT, int max_jj, int embed_dim, float scale)
{
#if __AVX512F__
    const __m512 _scale16 = _mm512_set1_ps(scale);
#endif // __AVX512F__
    const __m256 _scale = _mm256_set1_ps(scale);
    __m256 _max = _mm256_set1_ps(-FLT_MAX);

    const float* pK = packed_key_tile;
    int j = 0;
#if __AVX512F__
    for (; j + 15 < max_jj; j += 16)
    {
        __m512 _sum0 = _mm512_setzero_ps();
        __m512 _sum1 = _mm512_setzero_ps();
        __m512 _sum2 = _mm512_setzero_ps();
        __m512 _sum3 = _mm512_setzero_ps();
        __m512 _sum4 = _mm512_setzero_ps();
        __m512 _sum5 = _mm512_setzero_ps();
        __m512 _sum6 = _mm512_setzero_ps();
        __m512 _sum7 = _mm512_setzero_ps();

        const float* pQ = queryT;
        for (int k = 0; k < embed_dim; k++)
        {
            __m512 _k = _mm512_loadu_ps(pK);

            _sum0 = _mm512_fmadd_ps(_k, _mm512_set1_ps(pQ[0]), _sum0);
            _sum1 = _mm512_fmadd_ps(_k, _mm512_set1_ps(pQ[1]), _sum1);
            _sum2 = _mm512_fmadd_ps(_k, _mm512_set1_ps(pQ[2]), _sum2);
            _sum3 = _mm512_fmadd_ps(_k, _mm512_set1_ps(pQ[3]), _sum3);
            _sum4 = _mm512_fmadd_ps(_k, _mm512_set1_ps(pQ[4]), _sum4);
            _sum5 = _mm512_fmadd_ps(_k, _mm512_set1_ps(pQ[5]), _sum5);
            _sum6 = _mm512_fmadd_ps(_k, _mm512_set1_ps(pQ[6]), _sum6);
            _sum7 = _mm512_fmadd_ps(_k, _mm512_set1_ps(pQ[7]), _sum7);

            pQ += 8;
            pK += 16;
        }

        _sum0 = _mm512_mul_ps(_sum0, _scale16);
        _sum1 = _mm512_mul_ps(_sum1, _scale16);
        _sum2 = _mm512_mul_ps(_sum2, _scale16);
        _sum3 = _mm512_mul_ps(_sum3, _scale16);
        _sum4 = _mm512_mul_ps(_sum4, _scale16);
        _sum5 = _mm512_mul_ps(_sum5, _scale16);
        _sum6 = _mm512_mul_ps(_sum6, _scale16);
        _sum7 = _mm512_mul_ps(_sum7, _scale16);

        transpose16x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

        float* outptr = scoreT + j * 8;
        __m256 _score = _mm512_extractf32x8_ps(_sum0, 0);
        _max = _mm256_max_ps(_max, _score);
        _mm256_storeu_ps(outptr, _score);
        _score = _mm512_extractf32x8_ps(_sum0, 1);
        _max = _mm256_max_ps(_max, _score);
        _mm256_storeu_ps(outptr + 8, _score);
        _score = _mm512_extractf32x8_ps(_sum1, 0);
        _max = _mm256_max_ps(_max, _score);
        _mm256_storeu_ps(outptr + 8 * 2, _score);
        _score = _mm512_extractf32x8_ps(_sum1, 1);
        _max = _mm256_max_ps(_max, _score);
        _mm256_storeu_ps(outptr + 8 * 3, _score);
        _score = _mm512_extractf32x8_ps(_sum2, 0);
        _max = _mm256_max_ps(_max, _score);
        _mm256_storeu_ps(outptr + 8 * 4, _score);
        _score = _mm512_extractf32x8_ps(_sum2, 1);
        _max = _mm256_max_ps(_max, _score);
        _mm256_storeu_ps(outptr + 8 * 5, _score);
        _score = _mm512_extractf32x8_ps(_sum3, 0);
        _max = _mm256_max_ps(_max, _score);
        _mm256_storeu_ps(outptr + 8 * 6, _score);
        _score = _mm512_extractf32x8_ps(_sum3, 1);
        _max = _mm256_max_ps(_max, _score);
        _mm256_storeu_ps(outptr + 8 * 7, _score);
        _score = _mm512_extractf32x8_ps(_sum4, 0);
        _max = _mm256_max_ps(_max, _score);
        _mm256_storeu_ps(outptr + 8 * 8, _score);
        _score = _mm512_extractf32x8_ps(_sum4, 1);
        _max = _mm256_max_ps(_max, _score);
        _mm256_storeu_ps(outptr + 8 * 9, _score);
        _score = _mm512_extractf32x8_ps(_sum5, 0);
        _max = _mm256_max_ps(_max, _score);
        _mm256_storeu_ps(outptr + 8 * 10, _score);
        _score = _mm512_extractf32x8_ps(_sum5, 1);
        _max = _mm256_max_ps(_max, _score);
        _mm256_storeu_ps(outptr + 8 * 11, _score);
        _score = _mm512_extractf32x8_ps(_sum6, 0);
        _max = _mm256_max_ps(_max, _score);
        _mm256_storeu_ps(outptr + 8 * 12, _score);
        _score = _mm512_extractf32x8_ps(_sum6, 1);
        _max = _mm256_max_ps(_max, _score);
        _mm256_storeu_ps(outptr + 8 * 13, _score);
        _score = _mm512_extractf32x8_ps(_sum7, 0);
        _max = _mm256_max_ps(_max, _score);
        _mm256_storeu_ps(outptr + 8 * 14, _score);
        _score = _mm512_extractf32x8_ps(_sum7, 1);
        _max = _mm256_max_ps(_max, _score);
        _mm256_storeu_ps(outptr + 8 * 15, _score);
    }
#endif // __AVX512F__

    for (; j + 7 < max_jj; j += 8)
    {
        __m256 _sum0 = _mm256_setzero_ps();
        __m256 _sum1 = _mm256_setzero_ps();
        __m256 _sum2 = _mm256_setzero_ps();
        __m256 _sum3 = _mm256_setzero_ps();
        __m256 _sum4 = _mm256_setzero_ps();
        __m256 _sum5 = _mm256_setzero_ps();
        __m256 _sum6 = _mm256_setzero_ps();
        __m256 _sum7 = _mm256_setzero_ps();

        const float* pQ = queryT;
        for (int k = 0; k < embed_dim; k++)
        {
            __m256 _k = _mm256_loadu_ps(pK);

            _sum0 = _mm256_comp_fmadd_ps(_k, _mm256_set1_ps(pQ[0]), _sum0);
            _sum1 = _mm256_comp_fmadd_ps(_k, _mm256_set1_ps(pQ[1]), _sum1);
            _sum2 = _mm256_comp_fmadd_ps(_k, _mm256_set1_ps(pQ[2]), _sum2);
            _sum3 = _mm256_comp_fmadd_ps(_k, _mm256_set1_ps(pQ[3]), _sum3);
            _sum4 = _mm256_comp_fmadd_ps(_k, _mm256_set1_ps(pQ[4]), _sum4);
            _sum5 = _mm256_comp_fmadd_ps(_k, _mm256_set1_ps(pQ[5]), _sum5);
            _sum6 = _mm256_comp_fmadd_ps(_k, _mm256_set1_ps(pQ[6]), _sum6);
            _sum7 = _mm256_comp_fmadd_ps(_k, _mm256_set1_ps(pQ[7]), _sum7);

            pQ += 8;
            pK += 8;
        }

        _sum0 = _mm256_mul_ps(_sum0, _scale);
        _sum1 = _mm256_mul_ps(_sum1, _scale);
        _sum2 = _mm256_mul_ps(_sum2, _scale);
        _sum3 = _mm256_mul_ps(_sum3, _scale);
        _sum4 = _mm256_mul_ps(_sum4, _scale);
        _sum5 = _mm256_mul_ps(_sum5, _scale);
        _sum6 = _mm256_mul_ps(_sum6, _scale);
        _sum7 = _mm256_mul_ps(_sum7, _scale);

        transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

        float* outptr = scoreT + j * 8;
        _max = _mm256_max_ps(_max, _sum0);
        _max = _mm256_max_ps(_max, _sum1);
        _max = _mm256_max_ps(_max, _sum2);
        _max = _mm256_max_ps(_max, _sum3);
        _max = _mm256_max_ps(_max, _sum4);
        _max = _mm256_max_ps(_max, _sum5);
        _max = _mm256_max_ps(_max, _sum6);
        _max = _mm256_max_ps(_max, _sum7);
        _mm256_storeu_ps(outptr, _sum0);
        _mm256_storeu_ps(outptr + 8, _sum1);
        _mm256_storeu_ps(outptr + 8 * 2, _sum2);
        _mm256_storeu_ps(outptr + 8 * 3, _sum3);
        _mm256_storeu_ps(outptr + 8 * 4, _sum4);
        _mm256_storeu_ps(outptr + 8 * 5, _sum5);
        _mm256_storeu_ps(outptr + 8 * 6, _sum6);
        _mm256_storeu_ps(outptr + 8 * 7, _sum7);
    }

    for (; j + 3 < max_jj; j += 4)
    {
        __m256 _sum0 = _mm256_setzero_ps();
        __m256 _sum1 = _mm256_setzero_ps();
        __m256 _sum2 = _mm256_setzero_ps();
        __m256 _sum3 = _mm256_setzero_ps();

        const float* pQ = queryT;
        for (int k = 0; k < embed_dim; k++)
        {
            __m256 _q = _mm256_loadu_ps(pQ);

            _sum0 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[0]), _sum0);
            _sum1 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[1]), _sum1);
            _sum2 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[2]), _sum2);
            _sum3 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[3]), _sum3);

            pQ += 8;
            pK += 4;
        }

        float* outptr = scoreT + j * 8;
        _sum0 = _mm256_mul_ps(_sum0, _scale);
        _sum1 = _mm256_mul_ps(_sum1, _scale);
        _sum2 = _mm256_mul_ps(_sum2, _scale);
        _sum3 = _mm256_mul_ps(_sum3, _scale);
        _max = _mm256_max_ps(_max, _sum0);
        _max = _mm256_max_ps(_max, _sum1);
        _max = _mm256_max_ps(_max, _sum2);
        _max = _mm256_max_ps(_max, _sum3);
        _mm256_storeu_ps(outptr, _sum0);
        _mm256_storeu_ps(outptr + 8, _sum1);
        _mm256_storeu_ps(outptr + 8 * 2, _sum2);
        _mm256_storeu_ps(outptr + 8 * 3, _sum3);
    }

    for (; j < max_jj; j++)
    {
        const float* pQ = queryT;
        __m256 _sum0 = _mm256_setzero_ps();

        for (int k = 0; k < embed_dim; k++)
        {
            _sum0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pQ), _mm256_set1_ps(pK[k]), _sum0);
            pQ += 8;
        }

        pK += embed_dim;
        _sum0 = _mm256_mul_ps(_sum0, _scale);
        _max = _mm256_max_ps(_max, _sum0);
        _mm256_storeu_ps(scoreT + j * 8, _sum0);
    }

    return _max;
}
#endif // __AVX__

#if __AVX512F__
static void sdpa_pv_tile16_packed_value_fp32(float* outT, const float* packed_value_tile, const float* scoreT, int max_jj, int out_embed_dim)
{
    const float* vptr = packed_value_tile;
    float* pO = outT;

    int k = 0;
    for (; k + 15 < out_embed_dim; k += 16)
    {
        __m512 _out0 = _mm512_loadu_ps(pO);
        __m512 _out1 = _mm512_loadu_ps(pO + 16);
        __m512 _out2 = _mm512_loadu_ps(pO + 16 * 2);
        __m512 _out3 = _mm512_loadu_ps(pO + 16 * 3);
        __m512 _out4 = _mm512_loadu_ps(pO + 16 * 4);
        __m512 _out5 = _mm512_loadu_ps(pO + 16 * 5);
        __m512 _out6 = _mm512_loadu_ps(pO + 16 * 6);
        __m512 _out7 = _mm512_loadu_ps(pO + 16 * 7);
        __m512 _out8 = _mm512_loadu_ps(pO + 16 * 8);
        __m512 _out9 = _mm512_loadu_ps(pO + 16 * 9);
        __m512 _outa = _mm512_loadu_ps(pO + 16 * 10);
        __m512 _outb = _mm512_loadu_ps(pO + 16 * 11);
        __m512 _outc = _mm512_loadu_ps(pO + 16 * 12);
        __m512 _outd = _mm512_loadu_ps(pO + 16 * 13);
        __m512 _oute = _mm512_loadu_ps(pO + 16 * 14);
        __m512 _outf = _mm512_loadu_ps(pO + 16 * 15);

        const float* sptr = scoreT;
        const float* vptr0 = vptr;
        for (int j = 0; j < max_jj; j++)
        {
            __m512 _p = _mm512_loadu_ps(sptr);
            _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[0]), _out0);
            _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[1]), _out1);
            _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[2]), _out2);
            _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[3]), _out3);
            _out4 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[4]), _out4);
            _out5 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[5]), _out5);
            _out6 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[6]), _out6);
            _out7 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[7]), _out7);
            _out8 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[8]), _out8);
            _out9 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[9]), _out9);
            _outa = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[10]), _outa);
            _outb = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[11]), _outb);
            _outc = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[12]), _outc);
            _outd = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[13]), _outd);
            _oute = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[14]), _oute);
            _outf = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[15]), _outf);
            sptr += 16;
            vptr0 += 16;
        }

        _mm512_storeu_ps(pO, _out0);
        _mm512_storeu_ps(pO + 16, _out1);
        _mm512_storeu_ps(pO + 16 * 2, _out2);
        _mm512_storeu_ps(pO + 16 * 3, _out3);
        _mm512_storeu_ps(pO + 16 * 4, _out4);
        _mm512_storeu_ps(pO + 16 * 5, _out5);
        _mm512_storeu_ps(pO + 16 * 6, _out6);
        _mm512_storeu_ps(pO + 16 * 7, _out7);
        _mm512_storeu_ps(pO + 16 * 8, _out8);
        _mm512_storeu_ps(pO + 16 * 9, _out9);
        _mm512_storeu_ps(pO + 16 * 10, _outa);
        _mm512_storeu_ps(pO + 16 * 11, _outb);
        _mm512_storeu_ps(pO + 16 * 12, _outc);
        _mm512_storeu_ps(pO + 16 * 13, _outd);
        _mm512_storeu_ps(pO + 16 * 14, _oute);
        _mm512_storeu_ps(pO + 16 * 15, _outf);

        vptr += (size_t)max_jj * 16;
        pO += 256;
    }
    for (; k + 7 < out_embed_dim; k += 8)
    {
        __m512 _out0 = _mm512_loadu_ps(pO);
        __m512 _out1 = _mm512_loadu_ps(pO + 16);
        __m512 _out2 = _mm512_loadu_ps(pO + 16 * 2);
        __m512 _out3 = _mm512_loadu_ps(pO + 16 * 3);
        __m512 _out4 = _mm512_loadu_ps(pO + 16 * 4);
        __m512 _out5 = _mm512_loadu_ps(pO + 16 * 5);
        __m512 _out6 = _mm512_loadu_ps(pO + 16 * 6);
        __m512 _out7 = _mm512_loadu_ps(pO + 16 * 7);

        const float* sptr = scoreT;
        const float* vptr0 = vptr;
        for (int j = 0; j < max_jj; j++)
        {
            __m512 _p = _mm512_loadu_ps(sptr);
            _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[0]), _out0);
            _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[1]), _out1);
            _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[2]), _out2);
            _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[3]), _out3);
            _out4 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[4]), _out4);
            _out5 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[5]), _out5);
            _out6 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[6]), _out6);
            _out7 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[7]), _out7);
            sptr += 16;
            vptr0 += 8;
        }

        _mm512_storeu_ps(pO, _out0);
        _mm512_storeu_ps(pO + 16, _out1);
        _mm512_storeu_ps(pO + 16 * 2, _out2);
        _mm512_storeu_ps(pO + 16 * 3, _out3);
        _mm512_storeu_ps(pO + 16 * 4, _out4);
        _mm512_storeu_ps(pO + 16 * 5, _out5);
        _mm512_storeu_ps(pO + 16 * 6, _out6);
        _mm512_storeu_ps(pO + 16 * 7, _out7);

        vptr += (size_t)max_jj * 8;
        pO += 128;
    }
    for (; k + 3 < out_embed_dim; k += 4)
    {
        __m512 _out0 = _mm512_loadu_ps(pO);
        __m512 _out1 = _mm512_loadu_ps(pO + 16);
        __m512 _out2 = _mm512_loadu_ps(pO + 16 * 2);
        __m512 _out3 = _mm512_loadu_ps(pO + 16 * 3);

        const float* sptr = scoreT;
        const float* vptr0 = vptr;
        for (int j = 0; j < max_jj; j++)
        {
            __m512 _p = _mm512_loadu_ps(sptr);
            _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[0]), _out0);
            _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[1]), _out1);
            _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[2]), _out2);
            _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[3]), _out3);
            sptr += 16;
            vptr0 += 4;
        }

        _mm512_storeu_ps(pO, _out0);
        _mm512_storeu_ps(pO + 16, _out1);
        _mm512_storeu_ps(pO + 16 * 2, _out2);
        _mm512_storeu_ps(pO + 16 * 3, _out3);

        vptr += (size_t)max_jj * 4;
        pO += 64;
    }
    for (; k < out_embed_dim; k++)
    {
        __m512 _out = _mm512_loadu_ps(pO);
        const float* sptr = scoreT;
        const float* vptr0 = vptr;
        for (int j = 0; j < max_jj; j++)
        {
            _out = _mm512_fmadd_ps(_mm512_loadu_ps(sptr), _mm512_set1_ps(*vptr0), _out);
            sptr += 16;
            vptr0++;
        }

        _mm512_storeu_ps(pO, _out);
        vptr += max_jj;
        pO += 16;
    }
}
#endif // __AVX512F__

#if __AVX__
static void sdpa_pv_tile8_packed_value_fp32(float* outT, const float* packed_value_tile, const float* scoreT, int max_jj, int out_embed_dim)
{
    const float* vptr = packed_value_tile;
    float* pO = outT;

    int k = 0;
#if __AVX512F__
    for (; k + 15 < out_embed_dim; k += 16)
    {
        __m256 _out0 = _mm256_loadu_ps(pO);
        __m256 _out1 = _mm256_loadu_ps(pO + 8);
        __m256 _out2 = _mm256_loadu_ps(pO + 8 * 2);
        __m256 _out3 = _mm256_loadu_ps(pO + 8 * 3);
        __m256 _out4 = _mm256_loadu_ps(pO + 8 * 4);
        __m256 _out5 = _mm256_loadu_ps(pO + 8 * 5);
        __m256 _out6 = _mm256_loadu_ps(pO + 8 * 6);
        __m256 _out7 = _mm256_loadu_ps(pO + 8 * 7);
        __m256 _out8 = _mm256_loadu_ps(pO + 8 * 8);
        __m256 _out9 = _mm256_loadu_ps(pO + 8 * 9);
        __m256 _outa = _mm256_loadu_ps(pO + 8 * 10);
        __m256 _outb = _mm256_loadu_ps(pO + 8 * 11);
        __m256 _outc = _mm256_loadu_ps(pO + 8 * 12);
        __m256 _outd = _mm256_loadu_ps(pO + 8 * 13);
        __m256 _oute = _mm256_loadu_ps(pO + 8 * 14);
        __m256 _outf = _mm256_loadu_ps(pO + 8 * 15);

        const float* sptr = scoreT;
        const float* vptr0 = vptr;
        for (int j = 0; j < max_jj; j++)
        {
            __m256 _p = _mm256_loadu_ps(sptr);
            _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[0]), _out0);
            _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[1]), _out1);
            _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[2]), _out2);
            _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[3]), _out3);
            _out4 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[4]), _out4);
            _out5 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[5]), _out5);
            _out6 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[6]), _out6);
            _out7 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[7]), _out7);
            _out8 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[8]), _out8);
            _out9 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[9]), _out9);
            _outa = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[10]), _outa);
            _outb = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[11]), _outb);
            _outc = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[12]), _outc);
            _outd = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[13]), _outd);
            _oute = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[14]), _oute);
            _outf = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[15]), _outf);
            sptr += 8;
            vptr0 += 16;
        }

        _mm256_storeu_ps(pO, _out0);
        _mm256_storeu_ps(pO + 8, _out1);
        _mm256_storeu_ps(pO + 8 * 2, _out2);
        _mm256_storeu_ps(pO + 8 * 3, _out3);
        _mm256_storeu_ps(pO + 8 * 4, _out4);
        _mm256_storeu_ps(pO + 8 * 5, _out5);
        _mm256_storeu_ps(pO + 8 * 6, _out6);
        _mm256_storeu_ps(pO + 8 * 7, _out7);
        _mm256_storeu_ps(pO + 8 * 8, _out8);
        _mm256_storeu_ps(pO + 8 * 9, _out9);
        _mm256_storeu_ps(pO + 8 * 10, _outa);
        _mm256_storeu_ps(pO + 8 * 11, _outb);
        _mm256_storeu_ps(pO + 8 * 12, _outc);
        _mm256_storeu_ps(pO + 8 * 13, _outd);
        _mm256_storeu_ps(pO + 8 * 14, _oute);
        _mm256_storeu_ps(pO + 8 * 15, _outf);

        vptr += (size_t)max_jj * 16;
        pO += 128;
    }
#endif // __AVX512F__
    for (; k + 7 < out_embed_dim; k += 8)
    {
        __m256 _out0 = _mm256_loadu_ps(pO);
        __m256 _out1 = _mm256_loadu_ps(pO + 8);
        __m256 _out2 = _mm256_loadu_ps(pO + 8 * 2);
        __m256 _out3 = _mm256_loadu_ps(pO + 8 * 3);
        __m256 _out4 = _mm256_loadu_ps(pO + 8 * 4);
        __m256 _out5 = _mm256_loadu_ps(pO + 8 * 5);
        __m256 _out6 = _mm256_loadu_ps(pO + 8 * 6);
        __m256 _out7 = _mm256_loadu_ps(pO + 8 * 7);

        const float* sptr = scoreT;
        const float* vptr0 = vptr;
        for (int j = 0; j < max_jj; j++)
        {
            __m256 _p = _mm256_loadu_ps(sptr);
            _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[0]), _out0);
            _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[1]), _out1);
            _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[2]), _out2);
            _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[3]), _out3);
            _out4 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[4]), _out4);
            _out5 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[5]), _out5);
            _out6 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[6]), _out6);
            _out7 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[7]), _out7);
            sptr += 8;
            vptr0 += 8;
        }

        _mm256_storeu_ps(pO, _out0);
        _mm256_storeu_ps(pO + 8, _out1);
        _mm256_storeu_ps(pO + 8 * 2, _out2);
        _mm256_storeu_ps(pO + 8 * 3, _out3);
        _mm256_storeu_ps(pO + 8 * 4, _out4);
        _mm256_storeu_ps(pO + 8 * 5, _out5);
        _mm256_storeu_ps(pO + 8 * 6, _out6);
        _mm256_storeu_ps(pO + 8 * 7, _out7);

        vptr += (size_t)max_jj * 8;
        pO += 64;
    }
    for (; k + 3 < out_embed_dim; k += 4)
    {
        __m256 _out0 = _mm256_loadu_ps(pO);
        __m256 _out1 = _mm256_loadu_ps(pO + 8);
        __m256 _out2 = _mm256_loadu_ps(pO + 8 * 2);
        __m256 _out3 = _mm256_loadu_ps(pO + 8 * 3);

        const float* sptr = scoreT;
        const float* vptr0 = vptr;
        for (int j = 0; j < max_jj; j++)
        {
            __m256 _p = _mm256_loadu_ps(sptr);
            _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[0]), _out0);
            _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[1]), _out1);
            _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[2]), _out2);
            _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[3]), _out3);
            sptr += 8;
            vptr0 += 4;
        }

        _mm256_storeu_ps(pO, _out0);
        _mm256_storeu_ps(pO + 8, _out1);
        _mm256_storeu_ps(pO + 8 * 2, _out2);
        _mm256_storeu_ps(pO + 8 * 3, _out3);

        vptr += (size_t)max_jj * 4;
        pO += 32;
    }
    for (; k < out_embed_dim; k++)
    {
        __m256 _out = _mm256_loadu_ps(pO);
        const float* sptr = scoreT;
        const float* vptr0 = vptr;
        for (int j = 0; j < max_jj; j++)
        {
            _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(sptr), _mm256_set1_ps(*vptr0), _out);
            sptr += 8;
            vptr0++;
        }

        _mm256_storeu_ps(pO, _out);
        vptr += max_jj;
        pO += 8;
    }
}
#endif // __AVX__

#if __SSE2__
static void sdpa_pv_tile4_packed_value_fp32(float* outT, const float* packed_value_tile, const float* scoreT, int max_jj, int out_embed_dim)
{
    const float* vptr = packed_value_tile;
    float* pO = outT;

    int k = 0;
#if __AVX512F__
    for (; k + 15 < out_embed_dim; k += 16)
    {
        __m128 _out0 = _mm_loadu_ps(pO);
        __m128 _out1 = _mm_loadu_ps(pO + 4);
        __m128 _out2 = _mm_loadu_ps(pO + 4 * 2);
        __m128 _out3 = _mm_loadu_ps(pO + 4 * 3);
        __m128 _out4 = _mm_loadu_ps(pO + 4 * 4);
        __m128 _out5 = _mm_loadu_ps(pO + 4 * 5);
        __m128 _out6 = _mm_loadu_ps(pO + 4 * 6);
        __m128 _out7 = _mm_loadu_ps(pO + 4 * 7);
        __m128 _out8 = _mm_loadu_ps(pO + 4 * 8);
        __m128 _out9 = _mm_loadu_ps(pO + 4 * 9);
        __m128 _outa = _mm_loadu_ps(pO + 4 * 10);
        __m128 _outb = _mm_loadu_ps(pO + 4 * 11);
        __m128 _outc = _mm_loadu_ps(pO + 4 * 12);
        __m128 _outd = _mm_loadu_ps(pO + 4 * 13);
        __m128 _oute = _mm_loadu_ps(pO + 4 * 14);
        __m128 _outf = _mm_loadu_ps(pO + 4 * 15);

        const float* sptr = scoreT;
        const float* vptr0 = vptr;
        for (int j = 0; j < max_jj; j++)
        {
            __m128 _p = _mm_loadu_ps(sptr);
            _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[0]), _out0);
            _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[1]), _out1);
            _out2 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[2]), _out2);
            _out3 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[3]), _out3);
            _out4 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[4]), _out4);
            _out5 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[5]), _out5);
            _out6 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[6]), _out6);
            _out7 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[7]), _out7);
            _out8 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[8]), _out8);
            _out9 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[9]), _out9);
            _outa = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[10]), _outa);
            _outb = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[11]), _outb);
            _outc = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[12]), _outc);
            _outd = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[13]), _outd);
            _oute = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[14]), _oute);
            _outf = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[15]), _outf);
            sptr += 4;
            vptr0 += 16;
        }

        _mm_storeu_ps(pO, _out0);
        _mm_storeu_ps(pO + 4, _out1);
        _mm_storeu_ps(pO + 4 * 2, _out2);
        _mm_storeu_ps(pO + 4 * 3, _out3);
        _mm_storeu_ps(pO + 4 * 4, _out4);
        _mm_storeu_ps(pO + 4 * 5, _out5);
        _mm_storeu_ps(pO + 4 * 6, _out6);
        _mm_storeu_ps(pO + 4 * 7, _out7);
        _mm_storeu_ps(pO + 4 * 8, _out8);
        _mm_storeu_ps(pO + 4 * 9, _out9);
        _mm_storeu_ps(pO + 4 * 10, _outa);
        _mm_storeu_ps(pO + 4 * 11, _outb);
        _mm_storeu_ps(pO + 4 * 12, _outc);
        _mm_storeu_ps(pO + 4 * 13, _outd);
        _mm_storeu_ps(pO + 4 * 14, _oute);
        _mm_storeu_ps(pO + 4 * 15, _outf);

        vptr += (size_t)max_jj * 16;
        pO += 64;
    }
#endif // __AVX512F__
#if __AVX__
    for (; k + 7 < out_embed_dim; k += 8)
    {
        __m128 _out0 = _mm_loadu_ps(pO);
        __m128 _out1 = _mm_loadu_ps(pO + 4);
        __m128 _out2 = _mm_loadu_ps(pO + 4 * 2);
        __m128 _out3 = _mm_loadu_ps(pO + 4 * 3);
        __m128 _out4 = _mm_loadu_ps(pO + 4 * 4);
        __m128 _out5 = _mm_loadu_ps(pO + 4 * 5);
        __m128 _out6 = _mm_loadu_ps(pO + 4 * 6);
        __m128 _out7 = _mm_loadu_ps(pO + 4 * 7);

        const float* sptr = scoreT;
        const float* vptr0 = vptr;
        for (int j = 0; j < max_jj; j++)
        {
            __m128 _p = _mm_loadu_ps(sptr);
            _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[0]), _out0);
            _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[1]), _out1);
            _out2 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[2]), _out2);
            _out3 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[3]), _out3);
            _out4 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[4]), _out4);
            _out5 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[5]), _out5);
            _out6 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[6]), _out6);
            _out7 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[7]), _out7);
            sptr += 4;
            vptr0 += 8;
        }

        _mm_storeu_ps(pO, _out0);
        _mm_storeu_ps(pO + 4, _out1);
        _mm_storeu_ps(pO + 4 * 2, _out2);
        _mm_storeu_ps(pO + 4 * 3, _out3);
        _mm_storeu_ps(pO + 4 * 4, _out4);
        _mm_storeu_ps(pO + 4 * 5, _out5);
        _mm_storeu_ps(pO + 4 * 6, _out6);
        _mm_storeu_ps(pO + 4 * 7, _out7);

        vptr += (size_t)max_jj * 8;
        pO += 32;
    }
#endif // __AVX__
    for (; k + 3 < out_embed_dim; k += 4)
    {
        __m128 _out0 = _mm_loadu_ps(pO);
        __m128 _out1 = _mm_loadu_ps(pO + 4);
        __m128 _out2 = _mm_loadu_ps(pO + 4 * 2);
        __m128 _out3 = _mm_loadu_ps(pO + 4 * 3);

        const float* sptr = scoreT;
        const float* vptr0 = vptr;
        for (int j = 0; j < max_jj; j++)
        {
            __m128 _p = _mm_loadu_ps(sptr);
            _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[0]), _out0);
            _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[1]), _out1);
            _out2 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[2]), _out2);
            _out3 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[3]), _out3);
            sptr += 4;
            vptr0 += 4;
        }

        _mm_storeu_ps(pO, _out0);
        _mm_storeu_ps(pO + 4, _out1);
        _mm_storeu_ps(pO + 4 * 2, _out2);
        _mm_storeu_ps(pO + 4 * 3, _out3);

        vptr += (size_t)max_jj * 4;
        pO += 16;
    }
    for (; k < out_embed_dim; k++)
    {
        __m128 _out = _mm_loadu_ps(pO);
        const float* sptr = scoreT;
        const float* vptr0 = vptr;
        for (int j = 0; j < max_jj; j++)
        {
            _out = _mm_comp_fmadd_ps(_mm_loadu_ps(sptr), _mm_set1_ps(*vptr0), _out);
            sptr += 4;
            vptr0++;
        }

        _mm_storeu_ps(pO, _out);
        vptr += max_jj;
        pO += 4;
    }
}
#endif // __SSE2__

static void sdpa_pv_tile1_packed_value_fp32(float* outptr, const float* packed_value_tile, const float* scoreT, int max_jj, int out_embed_dim)
{
    const float* vptr = packed_value_tile;

    int k = 0;
#if __AVX512F__
    for (; k + 15 < out_embed_dim; k += 16)
    {
        __m512 _out = _mm512_loadu_ps(outptr + k);
        const float* vptr0 = vptr;
        for (int j = 0; j < max_jj; j++)
        {
            _out = _mm512_fmadd_ps(_mm512_set1_ps(scoreT[j]), _mm512_loadu_ps(vptr0), _out);
            vptr0 += 16;
        }
        _mm512_storeu_ps(outptr + k, _out);
        vptr += (size_t)max_jj * 16;
    }
#endif // __AVX512F__
#if __AVX__
    for (; k + 7 < out_embed_dim; k += 8)
    {
        __m256 _out = _mm256_loadu_ps(outptr + k);
        const float* vptr0 = vptr;
        for (int j = 0; j < max_jj; j++)
        {
            _out = _mm256_comp_fmadd_ps(_mm256_set1_ps(scoreT[j]), _mm256_loadu_ps(vptr0), _out);
            vptr0 += 8;
        }
        _mm256_storeu_ps(outptr + k, _out);
        vptr += (size_t)max_jj * 8;
    }
#endif // __AVX__
#if __SSE2__
    for (; k + 3 < out_embed_dim; k += 4)
    {
        __m128 _out = _mm_loadu_ps(outptr + k);
        const float* vptr0 = vptr;
        for (int j = 0; j < max_jj; j++)
        {
            _out = _mm_comp_fmadd_ps(_mm_set1_ps(scoreT[j]), _mm_loadu_ps(vptr0), _out);
            vptr0 += 4;
        }
        _mm_storeu_ps(outptr + k, _out);
        vptr += (size_t)max_jj * 4;
    }
#endif // __SSE2__
    for (; k < out_embed_dim; k++)
    {
        float out = outptr[k];
        const float* vptr0 = vptr;
        for (int j = 0; j < max_jj; j++)
        {
            out += scoreT[j] * *vptr0++;
        }
        outptr[k] = out;
        vptr += max_jj;
    }
}

#if __AVX512F__
static void sdpa_pv_tile16_unpacked_value_fp32(float* outT, const float* value, const float* scoreT, int max_jj, int out_embed_dim)
{
    float* pO = outT;

    int k = 0;
    for (; k + 15 < out_embed_dim; k += 16)
    {
        __m512 _out0 = _mm512_loadu_ps(pO);
        __m512 _out1 = _mm512_loadu_ps(pO + 16);
        __m512 _out2 = _mm512_loadu_ps(pO + 16 * 2);
        __m512 _out3 = _mm512_loadu_ps(pO + 16 * 3);
        __m512 _out4 = _mm512_loadu_ps(pO + 16 * 4);
        __m512 _out5 = _mm512_loadu_ps(pO + 16 * 5);
        __m512 _out6 = _mm512_loadu_ps(pO + 16 * 6);
        __m512 _out7 = _mm512_loadu_ps(pO + 16 * 7);
        __m512 _out8 = _mm512_loadu_ps(pO + 16 * 8);
        __m512 _out9 = _mm512_loadu_ps(pO + 16 * 9);
        __m512 _outa = _mm512_loadu_ps(pO + 16 * 10);
        __m512 _outb = _mm512_loadu_ps(pO + 16 * 11);
        __m512 _outc = _mm512_loadu_ps(pO + 16 * 12);
        __m512 _outd = _mm512_loadu_ps(pO + 16 * 13);
        __m512 _oute = _mm512_loadu_ps(pO + 16 * 14);
        __m512 _outf = _mm512_loadu_ps(pO + 16 * 15);

        const float* sptr = scoreT;
        const float* vptr0 = value + k;
        for (int j = 0; j < max_jj; j++)
        {
            __m512 _p = _mm512_loadu_ps(sptr);
            _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[0]), _out0);
            _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[1]), _out1);
            _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[2]), _out2);
            _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[3]), _out3);
            _out4 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[4]), _out4);
            _out5 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[5]), _out5);
            _out6 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[6]), _out6);
            _out7 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[7]), _out7);
            _out8 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[8]), _out8);
            _out9 = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[9]), _out9);
            _outa = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[10]), _outa);
            _outb = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[11]), _outb);
            _outc = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[12]), _outc);
            _outd = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[13]), _outd);
            _oute = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[14]), _oute);
            _outf = _mm512_fmadd_ps(_p, _mm512_set1_ps(vptr0[15]), _outf);
            sptr += 16;
            vptr0 += out_embed_dim;
        }

        _mm512_storeu_ps(pO, _out0);
        _mm512_storeu_ps(pO + 16, _out1);
        _mm512_storeu_ps(pO + 16 * 2, _out2);
        _mm512_storeu_ps(pO + 16 * 3, _out3);
        _mm512_storeu_ps(pO + 16 * 4, _out4);
        _mm512_storeu_ps(pO + 16 * 5, _out5);
        _mm512_storeu_ps(pO + 16 * 6, _out6);
        _mm512_storeu_ps(pO + 16 * 7, _out7);
        _mm512_storeu_ps(pO + 16 * 8, _out8);
        _mm512_storeu_ps(pO + 16 * 9, _out9);
        _mm512_storeu_ps(pO + 16 * 10, _outa);
        _mm512_storeu_ps(pO + 16 * 11, _outb);
        _mm512_storeu_ps(pO + 16 * 12, _outc);
        _mm512_storeu_ps(pO + 16 * 13, _outd);
        _mm512_storeu_ps(pO + 16 * 14, _oute);
        _mm512_storeu_ps(pO + 16 * 15, _outf);
        pO += 256;
    }
    for (; k < out_embed_dim; k++)
    {
        __m512 _out = _mm512_loadu_ps(pO);
        const float* sptr = scoreT;
        const float* vptr0 = value + k;
        for (int j = 0; j < max_jj; j++)
        {
            _out = _mm512_fmadd_ps(_mm512_loadu_ps(sptr), _mm512_set1_ps(*vptr0), _out);
            sptr += 16;
            vptr0 += out_embed_dim;
        }
        _mm512_storeu_ps(pO, _out);
        pO += 16;
    }
}
#endif // __AVX512F__

#if __AVX__
static void sdpa_pv_tile8_unpacked_value_fp32(float* outT, const float* value, const float* scoreT, int max_jj, int out_embed_dim)
{
    float* pO = outT;

    int k = 0;
#if __AVX512F__
    for (; k + 15 < out_embed_dim; k += 16)
    {
        __m256 _out0 = _mm256_loadu_ps(pO);
        __m256 _out1 = _mm256_loadu_ps(pO + 8);
        __m256 _out2 = _mm256_loadu_ps(pO + 8 * 2);
        __m256 _out3 = _mm256_loadu_ps(pO + 8 * 3);
        __m256 _out4 = _mm256_loadu_ps(pO + 8 * 4);
        __m256 _out5 = _mm256_loadu_ps(pO + 8 * 5);
        __m256 _out6 = _mm256_loadu_ps(pO + 8 * 6);
        __m256 _out7 = _mm256_loadu_ps(pO + 8 * 7);
        __m256 _out8 = _mm256_loadu_ps(pO + 8 * 8);
        __m256 _out9 = _mm256_loadu_ps(pO + 8 * 9);
        __m256 _outa = _mm256_loadu_ps(pO + 8 * 10);
        __m256 _outb = _mm256_loadu_ps(pO + 8 * 11);
        __m256 _outc = _mm256_loadu_ps(pO + 8 * 12);
        __m256 _outd = _mm256_loadu_ps(pO + 8 * 13);
        __m256 _oute = _mm256_loadu_ps(pO + 8 * 14);
        __m256 _outf = _mm256_loadu_ps(pO + 8 * 15);

        const float* sptr = scoreT;
        const float* vptr0 = value + k;
        for (int j = 0; j < max_jj; j++)
        {
            __m256 _p = _mm256_loadu_ps(sptr);
            _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[0]), _out0);
            _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[1]), _out1);
            _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[2]), _out2);
            _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[3]), _out3);
            _out4 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[4]), _out4);
            _out5 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[5]), _out5);
            _out6 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[6]), _out6);
            _out7 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[7]), _out7);
            _out8 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[8]), _out8);
            _out9 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[9]), _out9);
            _outa = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[10]), _outa);
            _outb = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[11]), _outb);
            _outc = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[12]), _outc);
            _outd = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[13]), _outd);
            _oute = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[14]), _oute);
            _outf = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[15]), _outf);
            sptr += 8;
            vptr0 += out_embed_dim;
        }

        _mm256_storeu_ps(pO, _out0);
        _mm256_storeu_ps(pO + 8, _out1);
        _mm256_storeu_ps(pO + 8 * 2, _out2);
        _mm256_storeu_ps(pO + 8 * 3, _out3);
        _mm256_storeu_ps(pO + 8 * 4, _out4);
        _mm256_storeu_ps(pO + 8 * 5, _out5);
        _mm256_storeu_ps(pO + 8 * 6, _out6);
        _mm256_storeu_ps(pO + 8 * 7, _out7);
        _mm256_storeu_ps(pO + 8 * 8, _out8);
        _mm256_storeu_ps(pO + 8 * 9, _out9);
        _mm256_storeu_ps(pO + 8 * 10, _outa);
        _mm256_storeu_ps(pO + 8 * 11, _outb);
        _mm256_storeu_ps(pO + 8 * 12, _outc);
        _mm256_storeu_ps(pO + 8 * 13, _outd);
        _mm256_storeu_ps(pO + 8 * 14, _oute);
        _mm256_storeu_ps(pO + 8 * 15, _outf);
        pO += 128;
    }
#endif // __AVX512F__
    for (; k + 7 < out_embed_dim; k += 8)
    {
        __m256 _out0 = _mm256_loadu_ps(pO);
        __m256 _out1 = _mm256_loadu_ps(pO + 8);
        __m256 _out2 = _mm256_loadu_ps(pO + 8 * 2);
        __m256 _out3 = _mm256_loadu_ps(pO + 8 * 3);
        __m256 _out4 = _mm256_loadu_ps(pO + 8 * 4);
        __m256 _out5 = _mm256_loadu_ps(pO + 8 * 5);
        __m256 _out6 = _mm256_loadu_ps(pO + 8 * 6);
        __m256 _out7 = _mm256_loadu_ps(pO + 8 * 7);

        const float* sptr = scoreT;
        const float* vptr0 = value + k;
        for (int j = 0; j < max_jj; j++)
        {
            __m256 _p = _mm256_loadu_ps(sptr);
            _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[0]), _out0);
            _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[1]), _out1);
            _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[2]), _out2);
            _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[3]), _out3);
            _out4 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[4]), _out4);
            _out5 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[5]), _out5);
            _out6 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[6]), _out6);
            _out7 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(vptr0[7]), _out7);
            sptr += 8;
            vptr0 += out_embed_dim;
        }

        _mm256_storeu_ps(pO, _out0);
        _mm256_storeu_ps(pO + 8, _out1);
        _mm256_storeu_ps(pO + 8 * 2, _out2);
        _mm256_storeu_ps(pO + 8 * 3, _out3);
        _mm256_storeu_ps(pO + 8 * 4, _out4);
        _mm256_storeu_ps(pO + 8 * 5, _out5);
        _mm256_storeu_ps(pO + 8 * 6, _out6);
        _mm256_storeu_ps(pO + 8 * 7, _out7);
        pO += 64;
    }
    for (; k < out_embed_dim; k++)
    {
        __m256 _out = _mm256_loadu_ps(pO);
        const float* sptr = scoreT;
        const float* vptr0 = value + k;
        for (int j = 0; j < max_jj; j++)
        {
            _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(sptr), _mm256_set1_ps(*vptr0), _out);
            sptr += 8;
            vptr0 += out_embed_dim;
        }
        _mm256_storeu_ps(pO, _out);
        pO += 8;
    }
}
#endif // __AVX__

#if __SSE2__
static void sdpa_pv_tile4_unpacked_value_fp32(float* outT, const float* value, const float* scoreT, int max_jj, int out_embed_dim)
{
    float* pO = outT;

    int k = 0;
#if __AVX512F__
    for (; k + 15 < out_embed_dim; k += 16)
    {
        __m128 _out0 = _mm_loadu_ps(pO);
        __m128 _out1 = _mm_loadu_ps(pO + 4);
        __m128 _out2 = _mm_loadu_ps(pO + 4 * 2);
        __m128 _out3 = _mm_loadu_ps(pO + 4 * 3);
        __m128 _out4 = _mm_loadu_ps(pO + 4 * 4);
        __m128 _out5 = _mm_loadu_ps(pO + 4 * 5);
        __m128 _out6 = _mm_loadu_ps(pO + 4 * 6);
        __m128 _out7 = _mm_loadu_ps(pO + 4 * 7);
        __m128 _out8 = _mm_loadu_ps(pO + 4 * 8);
        __m128 _out9 = _mm_loadu_ps(pO + 4 * 9);
        __m128 _outa = _mm_loadu_ps(pO + 4 * 10);
        __m128 _outb = _mm_loadu_ps(pO + 4 * 11);
        __m128 _outc = _mm_loadu_ps(pO + 4 * 12);
        __m128 _outd = _mm_loadu_ps(pO + 4 * 13);
        __m128 _oute = _mm_loadu_ps(pO + 4 * 14);
        __m128 _outf = _mm_loadu_ps(pO + 4 * 15);

        const float* sptr = scoreT;
        const float* vptr0 = value + k;
        for (int j = 0; j < max_jj; j++)
        {
            __m128 _p = _mm_loadu_ps(sptr);
            _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[0]), _out0);
            _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[1]), _out1);
            _out2 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[2]), _out2);
            _out3 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[3]), _out3);
            _out4 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[4]), _out4);
            _out5 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[5]), _out5);
            _out6 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[6]), _out6);
            _out7 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[7]), _out7);
            _out8 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[8]), _out8);
            _out9 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[9]), _out9);
            _outa = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[10]), _outa);
            _outb = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[11]), _outb);
            _outc = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[12]), _outc);
            _outd = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[13]), _outd);
            _oute = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[14]), _oute);
            _outf = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[15]), _outf);
            sptr += 4;
            vptr0 += out_embed_dim;
        }

        _mm_storeu_ps(pO, _out0);
        _mm_storeu_ps(pO + 4, _out1);
        _mm_storeu_ps(pO + 4 * 2, _out2);
        _mm_storeu_ps(pO + 4 * 3, _out3);
        _mm_storeu_ps(pO + 4 * 4, _out4);
        _mm_storeu_ps(pO + 4 * 5, _out5);
        _mm_storeu_ps(pO + 4 * 6, _out6);
        _mm_storeu_ps(pO + 4 * 7, _out7);
        _mm_storeu_ps(pO + 4 * 8, _out8);
        _mm_storeu_ps(pO + 4 * 9, _out9);
        _mm_storeu_ps(pO + 4 * 10, _outa);
        _mm_storeu_ps(pO + 4 * 11, _outb);
        _mm_storeu_ps(pO + 4 * 12, _outc);
        _mm_storeu_ps(pO + 4 * 13, _outd);
        _mm_storeu_ps(pO + 4 * 14, _oute);
        _mm_storeu_ps(pO + 4 * 15, _outf);
        pO += 64;
    }
#endif // __AVX512F__
#if __AVX__
    for (; k + 7 < out_embed_dim; k += 8)
    {
        __m128 _out0 = _mm_loadu_ps(pO);
        __m128 _out1 = _mm_loadu_ps(pO + 4);
        __m128 _out2 = _mm_loadu_ps(pO + 4 * 2);
        __m128 _out3 = _mm_loadu_ps(pO + 4 * 3);
        __m128 _out4 = _mm_loadu_ps(pO + 4 * 4);
        __m128 _out5 = _mm_loadu_ps(pO + 4 * 5);
        __m128 _out6 = _mm_loadu_ps(pO + 4 * 6);
        __m128 _out7 = _mm_loadu_ps(pO + 4 * 7);

        const float* sptr = scoreT;
        const float* vptr0 = value + k;
        for (int j = 0; j < max_jj; j++)
        {
            __m128 _p = _mm_loadu_ps(sptr);
            _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[0]), _out0);
            _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[1]), _out1);
            _out2 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[2]), _out2);
            _out3 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[3]), _out3);
            _out4 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[4]), _out4);
            _out5 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[5]), _out5);
            _out6 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[6]), _out6);
            _out7 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[7]), _out7);
            sptr += 4;
            vptr0 += out_embed_dim;
        }

        _mm_storeu_ps(pO, _out0);
        _mm_storeu_ps(pO + 4, _out1);
        _mm_storeu_ps(pO + 4 * 2, _out2);
        _mm_storeu_ps(pO + 4 * 3, _out3);
        _mm_storeu_ps(pO + 4 * 4, _out4);
        _mm_storeu_ps(pO + 4 * 5, _out5);
        _mm_storeu_ps(pO + 4 * 6, _out6);
        _mm_storeu_ps(pO + 4 * 7, _out7);
        pO += 32;
    }
#endif // __AVX__
    for (; k + 3 < out_embed_dim; k += 4)
    {
        __m128 _out0 = _mm_loadu_ps(pO);
        __m128 _out1 = _mm_loadu_ps(pO + 4);
        __m128 _out2 = _mm_loadu_ps(pO + 4 * 2);
        __m128 _out3 = _mm_loadu_ps(pO + 4 * 3);

        const float* sptr = scoreT;
        const float* vptr0 = value + k;
        for (int j = 0; j < max_jj; j++)
        {
            __m128 _p = _mm_loadu_ps(sptr);
            _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[0]), _out0);
            _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[1]), _out1);
            _out2 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[2]), _out2);
            _out3 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(vptr0[3]), _out3);
            sptr += 4;
            vptr0 += out_embed_dim;
        }

        _mm_storeu_ps(pO, _out0);
        _mm_storeu_ps(pO + 4, _out1);
        _mm_storeu_ps(pO + 4 * 2, _out2);
        _mm_storeu_ps(pO + 4 * 3, _out3);
        pO += 16;
    }
    for (; k < out_embed_dim; k++)
    {
        __m128 _out = _mm_loadu_ps(pO);
        const float* sptr = scoreT;
        const float* vptr0 = value + k;
        for (int j = 0; j < max_jj; j++)
        {
            _out = _mm_comp_fmadd_ps(_mm_loadu_ps(sptr), _mm_set1_ps(*vptr0), _out);
            sptr += 4;
            vptr0 += out_embed_dim;
        }
        _mm_storeu_ps(pO, _out);
        pO += 4;
    }
}
#endif // __SSE2__

static void sdpa_pv_tile1_unpacked_value_fp32(float* outptr, const float* value, const float* scoreT, int max_jj, int out_embed_dim)
{
    int k = 0;
#if __AVX512F__
    for (; k + 15 < out_embed_dim; k += 16)
    {
        __m512 _out = _mm512_loadu_ps(outptr + k);
        const float* vptr0 = value + k;
        for (int j = 0; j < max_jj; j++)
        {
            _out = _mm512_fmadd_ps(_mm512_set1_ps(scoreT[j]), _mm512_loadu_ps(vptr0), _out);
            vptr0 += out_embed_dim;
        }
        _mm512_storeu_ps(outptr + k, _out);
    }
#endif // __AVX512F__
#if __AVX__
    for (; k + 7 < out_embed_dim; k += 8)
    {
        __m256 _out = _mm256_loadu_ps(outptr + k);
        const float* vptr0 = value + k;
        for (int j = 0; j < max_jj; j++)
        {
            _out = _mm256_comp_fmadd_ps(_mm256_set1_ps(scoreT[j]), _mm256_loadu_ps(vptr0), _out);
            vptr0 += out_embed_dim;
        }
        _mm256_storeu_ps(outptr + k, _out);
    }
#endif // __AVX__
#if __SSE2__
    for (; k + 3 < out_embed_dim; k += 4)
    {
        __m128 _out = _mm_loadu_ps(outptr + k);
        const float* vptr0 = value + k;
        for (int j = 0; j < max_jj; j++)
        {
            _out = _mm_comp_fmadd_ps(_mm_set1_ps(scoreT[j]), _mm_loadu_ps(vptr0), _out);
            vptr0 += out_embed_dim;
        }
        _mm_storeu_ps(outptr + k, _out);
    }
#endif // __SSE2__
    for (; k < out_embed_dim; k++)
    {
        float out = outptr[k];
        const float* vptr0 = value + k;
        for (int j = 0; j < max_jj; j++)
        {
            out += scoreT[j] * *vptr0;
            vptr0 += out_embed_dim;
        }
        outptr[k] = out;
    }
}

static void sdpa_flash_attention_tile_packed_key_fp32(const float* query, const float* packed_key, const float* value, const float* packed_value, float* outptr_or_outT, float* scoreT, float* m_vec, float* l_vec, int i, int max_ii, int n_begin, int n_end, int nstep, int embed_dim, int out_embed_dim, float scale, bool k_end)
{
    const int MAX_EMBED_DIM = 128;
    const int MAX_OUT_EMBED_DIM = 128;
    const size_t packed_key_tile_stride = sdpa_packed_key_tile_stride_fp32(nstep, embed_dim);
    const size_t packed_value_tile_stride = sdpa_packed_value_tile_stride_fp32(nstep, out_embed_dim);

    int ii = 0;

#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const int TILE_M = 16;
        const int tile_i = i + ii;
        float* outptr_tile = outptr_or_outT + ii * out_embed_dim;
        float* m_tile = m_vec + ii;
        float* l_tile = l_vec + ii;

        float queryT[TILE_M * MAX_EMBED_DIM];
        float outT[TILE_M * MAX_OUT_EMBED_DIM];

        sdpa_pack_query_tile16_fp32(query, queryT, tile_i, embed_dim);

        float* pO = outT;
        for (int k = 0; k < out_embed_dim; k++)
        {
            _mm512_storeu_ps(pO, _mm512_setzero_ps());
            pO += 16;
        }

        __m512 _m = _mm512_set1_ps(-FLT_MAX);
        __m512 _l = _mm512_setzero_ps();
        const __m512 _qk_scale = _mm512_set1_ps(scale);

        for (int n_start = n_begin; n_start < n_end; n_start += nstep)
        {
            int max_jj = nstep;
            if (n_start + nstep > n_end)
                max_jj = n_end - n_start;

            const int n_tile = n_start / nstep;
            const float* pK = packed_key + (size_t)n_tile * packed_key_tile_stride;

            int j = 0;
            for (; j + 15 < max_jj; j += 16)
            {
                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();
                __m512 _sum2 = _mm512_setzero_ps();
                __m512 _sum3 = _mm512_setzero_ps();
                __m512 _sum4 = _mm512_setzero_ps();
                __m512 _sum5 = _mm512_setzero_ps();
                __m512 _sum6 = _mm512_setzero_ps();
                __m512 _sum7 = _mm512_setzero_ps();
                __m512 _sum8 = _mm512_setzero_ps();
                __m512 _sum9 = _mm512_setzero_ps();
                __m512 _suma = _mm512_setzero_ps();
                __m512 _sumb = _mm512_setzero_ps();
                __m512 _sumc = _mm512_setzero_ps();
                __m512 _sumd = _mm512_setzero_ps();
                __m512 _sume = _mm512_setzero_ps();
                __m512 _sumf = _mm512_setzero_ps();

                const float* pQ = queryT;
                for (int k = 0; k < embed_dim; k++)
                {
                    __m512 _q = _mm512_loadu_ps(pQ);

                    _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[0]), _sum0);
                    _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[1]), _sum1);
                    _sum2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[2]), _sum2);
                    _sum3 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[3]), _sum3);
                    _sum4 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[4]), _sum4);
                    _sum5 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[5]), _sum5);
                    _sum6 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[6]), _sum6);
                    _sum7 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[7]), _sum7);
                    _sum8 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[8]), _sum8);
                    _sum9 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[9]), _sum9);
                    _suma = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[10]), _suma);
                    _sumb = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[11]), _sumb);
                    _sumc = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[12]), _sumc);
                    _sumd = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[13]), _sumd);
                    _sume = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[14]), _sume);
                    _sumf = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[15]), _sumf);

                    pQ += 16;
                    pK += 16;
                }

                float* outptr = scoreT + j * 16;
                _mm512_storeu_ps(outptr, _mm512_mul_ps(_sum0, _qk_scale));
                _mm512_storeu_ps(outptr + 16, _mm512_mul_ps(_sum1, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 2, _mm512_mul_ps(_sum2, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 3, _mm512_mul_ps(_sum3, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 4, _mm512_mul_ps(_sum4, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 5, _mm512_mul_ps(_sum5, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 6, _mm512_mul_ps(_sum6, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 7, _mm512_mul_ps(_sum7, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 8, _mm512_mul_ps(_sum8, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 9, _mm512_mul_ps(_sum9, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 10, _mm512_mul_ps(_suma, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 11, _mm512_mul_ps(_sumb, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 12, _mm512_mul_ps(_sumc, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 13, _mm512_mul_ps(_sumd, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 14, _mm512_mul_ps(_sume, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 15, _mm512_mul_ps(_sumf, _qk_scale));
            }

            for (; j + 7 < max_jj; j += 8)
            {
                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();
                __m512 _sum2 = _mm512_setzero_ps();
                __m512 _sum3 = _mm512_setzero_ps();
                __m512 _sum4 = _mm512_setzero_ps();
                __m512 _sum5 = _mm512_setzero_ps();
                __m512 _sum6 = _mm512_setzero_ps();
                __m512 _sum7 = _mm512_setzero_ps();

                const float* pQ = queryT;
                for (int k = 0; k < embed_dim; k++)
                {
                    __m512 _q = _mm512_loadu_ps(pQ);

                    _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[0]), _sum0);
                    _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[1]), _sum1);
                    _sum2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[2]), _sum2);
                    _sum3 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[3]), _sum3);
                    _sum4 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[4]), _sum4);
                    _sum5 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[5]), _sum5);
                    _sum6 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[6]), _sum6);
                    _sum7 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[7]), _sum7);

                    pQ += 16;
                    pK += 8;
                }

                float* outptr = scoreT + j * 16;
                _mm512_storeu_ps(outptr, _mm512_mul_ps(_sum0, _qk_scale));
                _mm512_storeu_ps(outptr + 16, _mm512_mul_ps(_sum1, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 2, _mm512_mul_ps(_sum2, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 3, _mm512_mul_ps(_sum3, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 4, _mm512_mul_ps(_sum4, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 5, _mm512_mul_ps(_sum5, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 6, _mm512_mul_ps(_sum6, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 7, _mm512_mul_ps(_sum7, _qk_scale));
            }

            for (; j + 3 < max_jj; j += 4)
            {
                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();
                __m512 _sum2 = _mm512_setzero_ps();
                __m512 _sum3 = _mm512_setzero_ps();

                const float* pQ = queryT;
                for (int k = 0; k < embed_dim; k++)
                {
                    __m512 _q = _mm512_loadu_ps(pQ);

                    _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[0]), _sum0);
                    _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[1]), _sum1);
                    _sum2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[2]), _sum2);
                    _sum3 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[3]), _sum3);

                    pQ += 16;
                    pK += 4;
                }

                float* outptr = scoreT + j * 16;
                _mm512_storeu_ps(outptr, _mm512_mul_ps(_sum0, _qk_scale));
                _mm512_storeu_ps(outptr + 16, _mm512_mul_ps(_sum1, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 2, _mm512_mul_ps(_sum2, _qk_scale));
                _mm512_storeu_ps(outptr + 16 * 3, _mm512_mul_ps(_sum3, _qk_scale));
            }

            for (; j < max_jj; j++)
            {
                const float* pQ = queryT;
                __m512 _sum0 = _mm512_setzero_ps();

                for (int k = 0; k < embed_dim; k++)
                {
                    _sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(pQ), _mm512_set1_ps(pK[k]), _sum0);
                    pQ += 16;
                }

                pK += embed_dim;
                _mm512_storeu_ps(scoreT + j * 16, _mm512_mul_ps(_sum0, _qk_scale));
            }

            __m512 _max = _m;
            float* pS = scoreT;
            for (int j = 0; j < max_jj; j++)
            {
                _max = _mm512_max_ps(_max, _mm512_loadu_ps(pS));
                pS += 16;
            }

            __m512 _correction = exp512_ps(_mm512_sub_ps(_m, _max));

            if (n_start != n_begin)
            {
                pO = outT;
                for (int k = 0; k < out_embed_dim; k++)
                {
                    _mm512_storeu_ps(pO, _mm512_mul_ps(_mm512_loadu_ps(pO), _correction));
                    pO += 16;
                }
            }

            __m512 _sum = _mm512_setzero_ps();
            pS = scoreT;
            for (int j = 0; j < max_jj; j++)
            {
                __m512 _p = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(pS), _max));
                _mm512_storeu_ps(pS, _p);
                _sum = _mm512_add_ps(_sum, _p);
                pS += 16;
            }

            _l = _mm512_fmadd_ps(_l, _correction, _sum);
            _m = _max;

            if (packed_value)
            {
                const int n_tile = n_start / nstep;
                const float* packed_value_tile = packed_value + (size_t)n_tile * packed_value_tile_stride;
                sdpa_pv_tile16_packed_value_fp32(outT, packed_value_tile, scoreT, max_jj, out_embed_dim);
            }
            else
            {
                const float* value_tile = value + (size_t)n_start * out_embed_dim;
                sdpa_pv_tile16_unpacked_value_fp32(outT, value_tile, scoreT, max_jj, out_embed_dim);
            }
        }

        if (k_end)
        {
            float inv_l[16];
            _mm512_storeu_ps(inv_l, _mm512_div_ps(_mm512_set1_ps(1.f), _l));
            sdpa_store_out_tile_fp32(outptr_tile, outT, out_embed_dim, 16, inv_l);
        }
        else
        {
            _mm512_storeu_ps(m_tile, _m);
            _mm512_storeu_ps(l_tile, _l);
            memcpy(outptr_tile, outT, (size_t)out_embed_dim * 16 * sizeof(float));
        }
    }
#endif // __AVX512F__

#if __AVX__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const int TILE_M = 8;
        const int tile_i = i + ii;
        float* outptr_tile = outptr_or_outT + ii * out_embed_dim;
        float* m_tile = m_vec + ii;
        float* l_tile = l_vec + ii;

        float queryT[TILE_M * MAX_EMBED_DIM];
        float outT[TILE_M * MAX_OUT_EMBED_DIM];

        sdpa_pack_query_tile8_fp32(query, queryT, tile_i, embed_dim);

        float* pO = outT;
        for (int k = 0; k < out_embed_dim; k++)
        {
            _mm256_storeu_ps(pO, _mm256_setzero_ps());
            pO += 8;
        }

        __m256 _m = _mm256_set1_ps(-FLT_MAX);
        __m256 _l = _mm256_setzero_ps();

        for (int n_start = n_begin; n_start < n_end; n_start += nstep)
        {
            int max_jj = nstep;
            if (n_start + nstep > n_end)
                max_jj = n_end - n_start;

            const int n_tile = n_start / nstep;
            const float* packed_key_tile = packed_key + (size_t)n_tile * packed_key_tile_stride;
            __m256 _max = _mm256_max_ps(_m, sdpa_qk_tile8_packed_key_fp32(queryT, packed_key_tile, scoreT, max_jj, embed_dim, scale));

            __m256 _correction = exp256_ps(_mm256_sub_ps(_m, _max));

            if (n_start != n_begin)
            {
                pO = outT;
                for (int k = 0; k < out_embed_dim; k++)
                {
                    _mm256_storeu_ps(pO, _mm256_mul_ps(_mm256_loadu_ps(pO), _correction));
                    pO += 8;
                }
            }

            __m256 _sum = _mm256_setzero_ps();
            float* pS = scoreT;
            for (int j = 0; j < max_jj; j++)
            {
                __m256 _p = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(pS), _max));
                _mm256_storeu_ps(pS, _p);
                _sum = _mm256_add_ps(_sum, _p);
                pS += 8;
            }

            _l = _mm256_comp_fmadd_ps(_l, _correction, _sum);
            _m = _max;

            if (packed_value)
            {
                const int n_tile = n_start / nstep;
                const float* packed_value_tile = packed_value + (size_t)n_tile * packed_value_tile_stride;
                sdpa_pv_tile8_packed_value_fp32(outT, packed_value_tile, scoreT, max_jj, out_embed_dim);
            }
            else
            {
                const float* value_tile = value + (size_t)n_start * out_embed_dim;
                sdpa_pv_tile8_unpacked_value_fp32(outT, value_tile, scoreT, max_jj, out_embed_dim);
            }
        }

        if (k_end)
        {
            float inv_l[8];
            _mm256_storeu_ps(inv_l, _mm256_div_ps(_mm256_set1_ps(1.f), _l));
            sdpa_store_out_tile_fp32(outptr_tile, outT, out_embed_dim, 8, inv_l);
        }
        else
        {
            _mm256_storeu_ps(m_tile, _m);
            _mm256_storeu_ps(l_tile, _l);
            memcpy(outptr_tile, outT, (size_t)out_embed_dim * 8 * sizeof(float));
        }
    }
#endif // __AVX__

#if __SSE2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const int TILE_M = 4;
        const int tile_i = i + ii;
        float* outptr_tile = outptr_or_outT + ii * out_embed_dim;
        float* m_tile = m_vec + ii;
        float* l_tile = l_vec + ii;

        float queryT[TILE_M * MAX_EMBED_DIM];
        float outT[TILE_M * MAX_OUT_EMBED_DIM];

        sdpa_pack_query_tile4_fp32(query, queryT, tile_i, embed_dim);

        float* pO = outT;
        for (int k = 0; k < out_embed_dim; k++)
        {
            _mm_storeu_ps(pO, _mm_setzero_ps());
            pO += 4;
        }

        __m128 _m = _mm_set1_ps(-FLT_MAX);
        __m128 _l = _mm_setzero_ps();
#if __AVX512F__
        const __m512 _qk_scale16 = _mm512_set1_ps(scale);
#endif // __AVX512F__
#if __AVX__
        const __m256 _qk_scale8 = _mm256_set1_ps(scale);
#endif // __AVX__
        const __m128 _qk_scale4 = _mm_set1_ps(scale);

        for (int n_start = n_begin; n_start < n_end; n_start += nstep)
        {
            int max_jj = nstep;
            if (n_start + nstep > n_end)
                max_jj = n_end - n_start;

            const int n_tile = n_start / nstep;
            const float* pK = packed_key + (size_t)n_tile * packed_key_tile_stride;

            int j = 0;
#if __AVX512F__
            for (; j + 15 < max_jj; j += 16)
            {
                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();
                __m512 _sum2 = _mm512_setzero_ps();
                __m512 _sum3 = _mm512_setzero_ps();

                const float* pQ = queryT;
                for (int k = 0; k < embed_dim; k++)
                {
                    __m512 _k = _mm512_loadu_ps(pK);

                    _sum0 = _mm512_fmadd_ps(_k, _mm512_set1_ps(pQ[0]), _sum0);
                    _sum1 = _mm512_fmadd_ps(_k, _mm512_set1_ps(pQ[1]), _sum1);
                    _sum2 = _mm512_fmadd_ps(_k, _mm512_set1_ps(pQ[2]), _sum2);
                    _sum3 = _mm512_fmadd_ps(_k, _mm512_set1_ps(pQ[3]), _sum3);

                    pQ += 4;
                    pK += 16;
                }

                _sum0 = _mm512_mul_ps(_sum0, _qk_scale16);
                _sum1 = _mm512_mul_ps(_sum1, _qk_scale16);
                _sum2 = _mm512_mul_ps(_sum2, _qk_scale16);
                _sum3 = _mm512_mul_ps(_sum3, _qk_scale16);

                transpose16x4_ps(_sum0, _sum1, _sum2, _sum3);

                float* outptr = scoreT + j * 4;
                _mm512_storeu_ps(outptr, _sum0);
                _mm512_storeu_ps(outptr + 16, _sum1);
                _mm512_storeu_ps(outptr + 16 * 2, _sum2);
                _mm512_storeu_ps(outptr + 16 * 3, _sum3);
            }
#endif // __AVX512F__

#if __AVX__
            for (; j + 7 < max_jj; j += 8)
            {
                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();
                __m256 _sum2 = _mm256_setzero_ps();
                __m256 _sum3 = _mm256_setzero_ps();

                const float* pQ = queryT;
                for (int k = 0; k < embed_dim; k++)
                {
                    __m256 _k = _mm256_loadu_ps(pK);

                    _sum0 = _mm256_comp_fmadd_ps(_k, _mm256_set1_ps(pQ[0]), _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_k, _mm256_set1_ps(pQ[1]), _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_k, _mm256_set1_ps(pQ[2]), _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_k, _mm256_set1_ps(pQ[3]), _sum3);

                    pQ += 4;
                    pK += 8;
                }

                _sum0 = _mm256_mul_ps(_sum0, _qk_scale8);
                _sum1 = _mm256_mul_ps(_sum1, _qk_scale8);
                _sum2 = _mm256_mul_ps(_sum2, _qk_scale8);
                _sum3 = _mm256_mul_ps(_sum3, _qk_scale8);

                transpose8x4_ps(_sum0, _sum1, _sum2, _sum3);

                float* outptr = scoreT + j * 4;
                _mm256_storeu_ps(outptr, _sum0);
                _mm256_storeu_ps(outptr + 8, _sum1);
                _mm256_storeu_ps(outptr + 8 * 2, _sum2);
                _mm256_storeu_ps(outptr + 8 * 3, _sum3);
            }
#endif // __AVX__

            for (; j + 3 < max_jj; j += 4)
            {
                __m128 _sum0 = _mm_setzero_ps();
                __m128 _sum1 = _mm_setzero_ps();
                __m128 _sum2 = _mm_setzero_ps();
                __m128 _sum3 = _mm_setzero_ps();

                const float* pQ = queryT;
                for (int k = 0; k < embed_dim; k++)
                {
                    __m128 _k = _mm_loadu_ps(pK);

                    _sum0 = _mm_comp_fmadd_ps(_k, _mm_set1_ps(pQ[0]), _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_k, _mm_set1_ps(pQ[1]), _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_k, _mm_set1_ps(pQ[2]), _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_k, _mm_set1_ps(pQ[3]), _sum3);

                    pQ += 4;
                    pK += 4;
                }

                _sum0 = _mm_mul_ps(_sum0, _qk_scale4);
                _sum1 = _mm_mul_ps(_sum1, _qk_scale4);
                _sum2 = _mm_mul_ps(_sum2, _qk_scale4);
                _sum3 = _mm_mul_ps(_sum3, _qk_scale4);

                _MM_TRANSPOSE4_PS(_sum0, _sum1, _sum2, _sum3);

                float* outptr = scoreT + j * 4;
                _mm_storeu_ps(outptr, _sum0);
                _mm_storeu_ps(outptr + 4, _sum1);
                _mm_storeu_ps(outptr + 4 * 2, _sum2);
                _mm_storeu_ps(outptr + 4 * 3, _sum3);
            }

            for (; j < max_jj; j++)
            {
                const float* pQ = queryT;
                __m128 _sum0 = _mm_setzero_ps();

                for (int k = 0; k < embed_dim; k++)
                {
                    _sum0 = _mm_comp_fmadd_ps(_mm_loadu_ps(pQ), _mm_set1_ps(pK[k]), _sum0);
                    pQ += 4;
                }

                pK += embed_dim;
                _mm_storeu_ps(scoreT + j * 4, _mm_mul_ps(_sum0, _qk_scale4));
            }

            __m128 _max = _m;
            float* pS = scoreT;
            for (int j = 0; j < max_jj; j++)
            {
                _max = _mm_max_ps(_max, _mm_loadu_ps(pS));
                pS += 4;
            }

            __m128 _correction = exp_ps(_mm_sub_ps(_m, _max));

            if (n_start != n_begin)
            {
                pO = outT;
                for (int k = 0; k < out_embed_dim; k++)
                {
                    _mm_storeu_ps(pO, _mm_mul_ps(_mm_loadu_ps(pO), _correction));
                    pO += 4;
                }
            }

            __m128 _sum = _mm_setzero_ps();
            pS = scoreT;
            for (int j = 0; j < max_jj; j++)
            {
                __m128 _p = exp_ps(_mm_sub_ps(_mm_loadu_ps(pS), _max));
                _mm_storeu_ps(pS, _p);
                _sum = _mm_add_ps(_sum, _p);
                pS += 4;
            }

            _l = _mm_comp_fmadd_ps(_l, _correction, _sum);
            _m = _max;

            if (packed_value)
            {
                const int n_tile = n_start / nstep;
                const float* packed_value_tile = packed_value + (size_t)n_tile * packed_value_tile_stride;
                sdpa_pv_tile4_packed_value_fp32(outT, packed_value_tile, scoreT, max_jj, out_embed_dim);
            }
            else
            {
                const float* value_tile = value + (size_t)n_start * out_embed_dim;
                sdpa_pv_tile4_unpacked_value_fp32(outT, value_tile, scoreT, max_jj, out_embed_dim);
            }
        }

        if (k_end)
        {
            float inv_l[4];
            _mm_storeu_ps(inv_l, _mm_div_ps(_mm_set1_ps(1.f), _l));
            sdpa_store_out_tile_fp32(outptr_tile, outT, out_embed_dim, 4, inv_l);
        }
        else
        {
            _mm_storeu_ps(m_tile, _m);
            _mm_storeu_ps(l_tile, _l);
            memcpy(outptr_tile, outT, (size_t)out_embed_dim * 4 * sizeof(float));
        }
    }
#endif // __SSE2__

    for (; ii < max_ii; ii++)
    {
        const int tile_i = i + ii;
        float* outptr_tile = outptr_or_outT + ii * out_embed_dim;
        float* m_tile = m_vec + ii;
        float* l_tile = l_vec + ii;
        const float* qptr = query + tile_i * embed_dim;

        for (int k = 0; k < out_embed_dim; k++)
        {
            outptr_tile[k] = 0.f;
        }

        float m = -FLT_MAX;
        float l = 0.f;

        for (int n_start = n_begin; n_start < n_end; n_start += nstep)
        {
            int max_jj = nstep;
            if (n_start + nstep > n_end)
                max_jj = n_end - n_start;

            const int n_tile = n_start / nstep;
            const float* packed_key_tile = packed_key + (size_t)n_tile * packed_key_tile_stride;

            const float* pK = packed_key_tile;
            int j = 0;
#if __AVX512F__
            __m512 _scale16 = _mm512_set1_ps(scale);
            for (; j + 15 < max_jj; j += 16)
            {
                __m512 _sum = _mm512_setzero_ps();
                for (int k = 0; k < embed_dim; k++)
                {
                    _sum = _mm512_fmadd_ps(_mm512_set1_ps(qptr[k]), _mm512_loadu_ps(pK), _sum);
                    pK += 16;
                }
                _mm512_storeu_ps(scoreT + j, _mm512_mul_ps(_sum, _scale16));
            }
#endif // __AVX512F__

#if __AVX__
            __m256 _scale8 = _mm256_set1_ps(scale);
            for (; j + 7 < max_jj; j += 8)
            {
                __m256 _sum = _mm256_setzero_ps();
                for (int k = 0; k < embed_dim; k++)
                {
                    _sum = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr[k]), _mm256_loadu_ps(pK), _sum);
                    pK += 8;
                }
                _mm256_storeu_ps(scoreT + j, _mm256_mul_ps(_sum, _scale8));
            }
#endif // __AVX__

#if __SSE2__
            __m128 _scale4 = _mm_set1_ps(scale);
            for (; j + 3 < max_jj; j += 4)
            {
                __m128 _sum = _mm_setzero_ps();
                for (int k = 0; k < embed_dim; k++)
                {
                    _sum = _mm_comp_fmadd_ps(_mm_set1_ps(qptr[k]), _mm_loadu_ps(pK), _sum);
                    pK += 4;
                }
                _mm_storeu_ps(scoreT + j, _mm_mul_ps(_sum, _scale4));
            }
#endif // __SSE2__

            for (; j < max_jj; j++)
            {
                float sum = 0.f;

                for (int k = 0; k < embed_dim; k++)
                {
                    sum += qptr[k] * pK[k];
                }

                pK += embed_dim;
                scoreT[j] = sum * scale;
            }

            float max = m;
            for (int j = 0; j < max_jj; j++)
            {
                max = std::max(max, scoreT[j]);
            }

            const float correction = expf(m - max);
            if (correction != 1.f)
            {
                for (int k = 0; k < out_embed_dim; k++)
                {
                    outptr_tile[k] *= correction;
                }
            }

            float sum = 0.f;
            for (int j = 0; j < max_jj; j++)
            {
                const float p = expf(scoreT[j] - max);
                scoreT[j] = p;
                sum += p;
            }

            l = l * correction + sum;
            m = max;

            if (packed_value)
            {
                const float* packed_value_tile = packed_value + (size_t)n_tile * packed_value_tile_stride;
                sdpa_pv_tile1_packed_value_fp32(outptr_tile, packed_value_tile, scoreT, max_jj, out_embed_dim);
            }
            else
            {
                const float* value_tile = value + (size_t)n_start * out_embed_dim;
                sdpa_pv_tile1_unpacked_value_fp32(outptr_tile, value_tile, scoreT, max_jj, out_embed_dim);
            }
        }

        if (k_end)
        {
            const float scale0 = 1.f / l;
            for (int k = 0; k < out_embed_dim; k++)
            {
                outptr_tile[k] *= scale0;
            }
        }
        else
        {
            m_tile[0] = m;
            l_tile[0] = l;
        }
    }
}

static void sdpa_pv_tile_fp32(float* outptr, const float* value, const float* score, int max_ii, int n_start, int max_jj, int out_embed_dim)
{
    const int BLOCK_N = 256;

    int ii = 0;
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        float* outptr0 = outptr + ii * out_embed_dim;
        float* outptr1 = outptr0 + out_embed_dim;
        float* outptr2 = outptr1 + out_embed_dim;
        float* outptr3 = outptr2 + out_embed_dim;
        float* outptr4 = outptr3 + out_embed_dim;
        float* outptr5 = outptr4 + out_embed_dim;
        float* outptr6 = outptr5 + out_embed_dim;
        float* outptr7 = outptr6 + out_embed_dim;
        float* outptr8 = outptr7 + out_embed_dim;
        float* outptr9 = outptr8 + out_embed_dim;
        float* outptra = outptr9 + out_embed_dim;
        float* outptrb = outptra + out_embed_dim;
        float* outptrc = outptrb + out_embed_dim;
        float* outptrd = outptrc + out_embed_dim;
        float* outptre = outptrd + out_embed_dim;
        float* outptrf = outptre + out_embed_dim;

        const float* sptr0 = score + ii * BLOCK_N;
        const float* sptr1 = sptr0 + BLOCK_N;
        const float* sptr2 = sptr1 + BLOCK_N;
        const float* sptr3 = sptr2 + BLOCK_N;
        const float* sptr4 = sptr3 + BLOCK_N;
        const float* sptr5 = sptr4 + BLOCK_N;
        const float* sptr6 = sptr5 + BLOCK_N;
        const float* sptr7 = sptr6 + BLOCK_N;
        const float* sptr8 = sptr7 + BLOCK_N;
        const float* sptr9 = sptr8 + BLOCK_N;
        const float* sptra = sptr9 + BLOCK_N;
        const float* sptrb = sptra + BLOCK_N;
        const float* sptrc = sptrb + BLOCK_N;
        const float* sptrd = sptrc + BLOCK_N;
        const float* sptre = sptrd + BLOCK_N;
        const float* sptrf = sptre + BLOCK_N;

        const float* vptr = value + n_start * out_embed_dim;

        int k = 0;
        for (; k + 15 < out_embed_dim; k += 16)
        {
            __m512 _sum0 = _mm512_setzero_ps();
            __m512 _sum1 = _mm512_setzero_ps();
            __m512 _sum2 = _mm512_setzero_ps();
            __m512 _sum3 = _mm512_setzero_ps();
            __m512 _sum4 = _mm512_setzero_ps();
            __m512 _sum5 = _mm512_setzero_ps();
            __m512 _sum6 = _mm512_setzero_ps();
            __m512 _sum7 = _mm512_setzero_ps();
            __m512 _sum8 = _mm512_setzero_ps();
            __m512 _sum9 = _mm512_setzero_ps();
            __m512 _suma = _mm512_setzero_ps();
            __m512 _sumb = _mm512_setzero_ps();
            __m512 _sumc = _mm512_setzero_ps();
            __m512 _sumd = _mm512_setzero_ps();
            __m512 _sume = _mm512_setzero_ps();
            __m512 _sumf = _mm512_setzero_ps();

            for (int j = 0; j < max_jj; j++)
            {
                __m512 _v = _mm512_loadu_ps(vptr + j * out_embed_dim + k);
                _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(sptr0[j]), _v, _sum0);
                _sum1 = _mm512_fmadd_ps(_mm512_set1_ps(sptr1[j]), _v, _sum1);
                _sum2 = _mm512_fmadd_ps(_mm512_set1_ps(sptr2[j]), _v, _sum2);
                _sum3 = _mm512_fmadd_ps(_mm512_set1_ps(sptr3[j]), _v, _sum3);
                _sum4 = _mm512_fmadd_ps(_mm512_set1_ps(sptr4[j]), _v, _sum4);
                _sum5 = _mm512_fmadd_ps(_mm512_set1_ps(sptr5[j]), _v, _sum5);
                _sum6 = _mm512_fmadd_ps(_mm512_set1_ps(sptr6[j]), _v, _sum6);
                _sum7 = _mm512_fmadd_ps(_mm512_set1_ps(sptr7[j]), _v, _sum7);
                _sum8 = _mm512_fmadd_ps(_mm512_set1_ps(sptr8[j]), _v, _sum8);
                _sum9 = _mm512_fmadd_ps(_mm512_set1_ps(sptr9[j]), _v, _sum9);
                _suma = _mm512_fmadd_ps(_mm512_set1_ps(sptra[j]), _v, _suma);
                _sumb = _mm512_fmadd_ps(_mm512_set1_ps(sptrb[j]), _v, _sumb);
                _sumc = _mm512_fmadd_ps(_mm512_set1_ps(sptrc[j]), _v, _sumc);
                _sumd = _mm512_fmadd_ps(_mm512_set1_ps(sptrd[j]), _v, _sumd);
                _sume = _mm512_fmadd_ps(_mm512_set1_ps(sptre[j]), _v, _sume);
                _sumf = _mm512_fmadd_ps(_mm512_set1_ps(sptrf[j]), _v, _sumf);
            }

            _mm512_storeu_ps(outptr0 + k, _sum0);
            _mm512_storeu_ps(outptr1 + k, _sum1);
            _mm512_storeu_ps(outptr2 + k, _sum2);
            _mm512_storeu_ps(outptr3 + k, _sum3);
            _mm512_storeu_ps(outptr4 + k, _sum4);
            _mm512_storeu_ps(outptr5 + k, _sum5);
            _mm512_storeu_ps(outptr6 + k, _sum6);
            _mm512_storeu_ps(outptr7 + k, _sum7);
            _mm512_storeu_ps(outptr8 + k, _sum8);
            _mm512_storeu_ps(outptr9 + k, _sum9);
            _mm512_storeu_ps(outptra + k, _suma);
            _mm512_storeu_ps(outptrb + k, _sumb);
            _mm512_storeu_ps(outptrc + k, _sumc);
            _mm512_storeu_ps(outptrd + k, _sumd);
            _mm512_storeu_ps(outptre + k, _sume);
            _mm512_storeu_ps(outptrf + k, _sumf);
        }
        for (; k < out_embed_dim; k++)
        {
            float sum0 = 0.f;
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            float sum4 = 0.f;
            float sum5 = 0.f;
            float sum6 = 0.f;
            float sum7 = 0.f;
            float sum8 = 0.f;
            float sum9 = 0.f;
            float suma = 0.f;
            float sumb = 0.f;
            float sumc = 0.f;
            float sumd = 0.f;
            float sume = 0.f;
            float sumf = 0.f;

            for (int j = 0; j < max_jj; j++)
            {
                const float v = vptr[j * out_embed_dim + k];
                sum0 += sptr0[j] * v;
                sum1 += sptr1[j] * v;
                sum2 += sptr2[j] * v;
                sum3 += sptr3[j] * v;
                sum4 += sptr4[j] * v;
                sum5 += sptr5[j] * v;
                sum6 += sptr6[j] * v;
                sum7 += sptr7[j] * v;
                sum8 += sptr8[j] * v;
                sum9 += sptr9[j] * v;
                suma += sptra[j] * v;
                sumb += sptrb[j] * v;
                sumc += sptrc[j] * v;
                sumd += sptrd[j] * v;
                sume += sptre[j] * v;
                sumf += sptrf[j] * v;
            }

            outptr0[k] = sum0;
            outptr1[k] = sum1;
            outptr2[k] = sum2;
            outptr3[k] = sum3;
            outptr4[k] = sum4;
            outptr5[k] = sum5;
            outptr6[k] = sum6;
            outptr7[k] = sum7;
            outptr8[k] = sum8;
            outptr9[k] = sum9;
            outptra[k] = suma;
            outptrb[k] = sumb;
            outptrc[k] = sumc;
            outptrd[k] = sumd;
            outptre[k] = sume;
            outptrf[k] = sumf;
        }
    }
#endif

    for (; ii + 7 < max_ii; ii += 8)
    {
        float* outptr0 = outptr + ii * out_embed_dim;
        float* outptr1 = outptr0 + out_embed_dim;
        float* outptr2 = outptr1 + out_embed_dim;
        float* outptr3 = outptr2 + out_embed_dim;
        float* outptr4 = outptr3 + out_embed_dim;
        float* outptr5 = outptr4 + out_embed_dim;
        float* outptr6 = outptr5 + out_embed_dim;
        float* outptr7 = outptr6 + out_embed_dim;

        const float* sptr0 = score + ii * BLOCK_N;
        const float* sptr1 = sptr0 + BLOCK_N;
        const float* sptr2 = sptr1 + BLOCK_N;
        const float* sptr3 = sptr2 + BLOCK_N;
        const float* sptr4 = sptr3 + BLOCK_N;
        const float* sptr5 = sptr4 + BLOCK_N;
        const float* sptr6 = sptr5 + BLOCK_N;
        const float* sptr7 = sptr6 + BLOCK_N;

        const float* vptr = value + n_start * out_embed_dim;

        int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        {
            for (; k + 15 < out_embed_dim; k += 16)
            {
                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();
                __m512 _sum2 = _mm512_setzero_ps();
                __m512 _sum3 = _mm512_setzero_ps();
                __m512 _sum4 = _mm512_setzero_ps();
                __m512 _sum5 = _mm512_setzero_ps();
                __m512 _sum6 = _mm512_setzero_ps();
                __m512 _sum7 = _mm512_setzero_ps();

                for (int j = 0; j < max_jj; j++)
                {
                    __m512 _v = _mm512_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm512_fmadd_ps(_mm512_set1_ps(sptr1[j]), _v, _sum1);
                    _sum2 = _mm512_fmadd_ps(_mm512_set1_ps(sptr2[j]), _v, _sum2);
                    _sum3 = _mm512_fmadd_ps(_mm512_set1_ps(sptr3[j]), _v, _sum3);
                    _sum4 = _mm512_fmadd_ps(_mm512_set1_ps(sptr4[j]), _v, _sum4);
                    _sum5 = _mm512_fmadd_ps(_mm512_set1_ps(sptr5[j]), _v, _sum5);
                    _sum6 = _mm512_fmadd_ps(_mm512_set1_ps(sptr6[j]), _v, _sum6);
                    _sum7 = _mm512_fmadd_ps(_mm512_set1_ps(sptr7[j]), _v, _sum7);
                }

                _mm512_storeu_ps(outptr0 + k, _sum0);
                _mm512_storeu_ps(outptr1 + k, _sum1);
                _mm512_storeu_ps(outptr2 + k, _sum2);
                _mm512_storeu_ps(outptr3 + k, _sum3);
                _mm512_storeu_ps(outptr4 + k, _sum4);
                _mm512_storeu_ps(outptr5 + k, _sum5);
                _mm512_storeu_ps(outptr6 + k, _sum6);
                _mm512_storeu_ps(outptr7 + k, _sum7);
            }
        }
#endif // __AVX512F__
        {
            for (; k + 7 < out_embed_dim; k += 8)
            {
                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();
                __m256 _sum2 = _mm256_setzero_ps();
                __m256 _sum3 = _mm256_setzero_ps();
                __m256 _sum4 = _mm256_setzero_ps();
                __m256 _sum5 = _mm256_setzero_ps();
                __m256 _sum6 = _mm256_setzero_ps();
                __m256 _sum7 = _mm256_setzero_ps();

                for (int j = 0; j < max_jj; j++)
                {
                    __m256 _v = _mm256_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr1[j]), _v, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr2[j]), _v, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr3[j]), _v, _sum3);
                    _sum4 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr4[j]), _v, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr5[j]), _v, _sum5);
                    _sum6 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr6[j]), _v, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr7[j]), _v, _sum7);
                }

                _mm256_storeu_ps(outptr0 + k, _sum0);
                _mm256_storeu_ps(outptr1 + k, _sum1);
                _mm256_storeu_ps(outptr2 + k, _sum2);
                _mm256_storeu_ps(outptr3 + k, _sum3);
                _mm256_storeu_ps(outptr4 + k, _sum4);
                _mm256_storeu_ps(outptr5 + k, _sum5);
                _mm256_storeu_ps(outptr6 + k, _sum6);
                _mm256_storeu_ps(outptr7 + k, _sum7);
            }
        }
#endif // __AVX__
        {
            for (; k + 3 < out_embed_dim; k += 4)
            {
                __m128 _sum0 = _mm_setzero_ps();
                __m128 _sum1 = _mm_setzero_ps();
                __m128 _sum2 = _mm_setzero_ps();
                __m128 _sum3 = _mm_setzero_ps();
                __m128 _sum4 = _mm_setzero_ps();
                __m128 _sum5 = _mm_setzero_ps();
                __m128 _sum6 = _mm_setzero_ps();
                __m128 _sum7 = _mm_setzero_ps();

                for (int j = 0; j < max_jj; j++)
                {
                    __m128 _v = _mm_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr1[j]), _v, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr2[j]), _v, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr3[j]), _v, _sum3);
                    _sum4 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr4[j]), _v, _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr5[j]), _v, _sum5);
                    _sum6 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr6[j]), _v, _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr7[j]), _v, _sum7);
                }

                _mm_storeu_ps(outptr0 + k, _sum0);
                _mm_storeu_ps(outptr1 + k, _sum1);
                _mm_storeu_ps(outptr2 + k, _sum2);
                _mm_storeu_ps(outptr3 + k, _sum3);
                _mm_storeu_ps(outptr4 + k, _sum4);
                _mm_storeu_ps(outptr5 + k, _sum5);
                _mm_storeu_ps(outptr6 + k, _sum6);
                _mm_storeu_ps(outptr7 + k, _sum7);
            }
        }
#endif // __SSE2__
        for (; k < out_embed_dim; k++)
        {
            float sum0 = 0.f;
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            float sum4 = 0.f;
            float sum5 = 0.f;
            float sum6 = 0.f;
            float sum7 = 0.f;

            for (int j = 0; j < max_jj; j++)
            {
                const float v = vptr[j * out_embed_dim + k];
                sum0 += sptr0[j] * v;
                sum1 += sptr1[j] * v;
                sum2 += sptr2[j] * v;
                sum3 += sptr3[j] * v;
                sum4 += sptr4[j] * v;
                sum5 += sptr5[j] * v;
                sum6 += sptr6[j] * v;
                sum7 += sptr7[j] * v;
            }

            outptr0[k] = sum0;
            outptr1[k] = sum1;
            outptr2[k] = sum2;
            outptr3[k] = sum3;
            outptr4[k] = sum4;
            outptr5[k] = sum5;
            outptr6[k] = sum6;
            outptr7[k] = sum7;
        }
    }

    for (; ii + 3 < max_ii; ii += 4)
    {
        float* outptr0 = outptr + ii * out_embed_dim;
        float* outptr1 = outptr0 + out_embed_dim;
        float* outptr2 = outptr1 + out_embed_dim;
        float* outptr3 = outptr2 + out_embed_dim;

        const float* sptr0 = score + ii * BLOCK_N;
        const float* sptr1 = sptr0 + BLOCK_N;
        const float* sptr2 = sptr1 + BLOCK_N;
        const float* sptr3 = sptr2 + BLOCK_N;

        const float* vptr = value + n_start * out_embed_dim;

        int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        {
            for (; k + 15 < out_embed_dim; k += 16)
            {
                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();
                __m512 _sum2 = _mm512_setzero_ps();
                __m512 _sum3 = _mm512_setzero_ps();

                for (int j = 0; j < max_jj; j++)
                {
                    __m512 _v = _mm512_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm512_fmadd_ps(_mm512_set1_ps(sptr1[j]), _v, _sum1);
                    _sum2 = _mm512_fmadd_ps(_mm512_set1_ps(sptr2[j]), _v, _sum2);
                    _sum3 = _mm512_fmadd_ps(_mm512_set1_ps(sptr3[j]), _v, _sum3);
                }

                _mm512_storeu_ps(outptr0 + k, _sum0);
                _mm512_storeu_ps(outptr1 + k, _sum1);
                _mm512_storeu_ps(outptr2 + k, _sum2);
                _mm512_storeu_ps(outptr3 + k, _sum3);
            }
        }
#endif // __AVX512F__
        {
            for (; k + 7 < out_embed_dim; k += 8)
            {
                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();
                __m256 _sum2 = _mm256_setzero_ps();
                __m256 _sum3 = _mm256_setzero_ps();

                for (int j = 0; j < max_jj; j++)
                {
                    __m256 _v = _mm256_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr1[j]), _v, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr2[j]), _v, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr3[j]), _v, _sum3);
                }

                _mm256_storeu_ps(outptr0 + k, _sum0);
                _mm256_storeu_ps(outptr1 + k, _sum1);
                _mm256_storeu_ps(outptr2 + k, _sum2);
                _mm256_storeu_ps(outptr3 + k, _sum3);
            }
        }
#endif // __AVX__
        {
            for (; k + 3 < out_embed_dim; k += 4)
            {
                __m128 _sum0 = _mm_setzero_ps();
                __m128 _sum1 = _mm_setzero_ps();
                __m128 _sum2 = _mm_setzero_ps();
                __m128 _sum3 = _mm_setzero_ps();

                for (int j = 0; j < max_jj; j++)
                {
                    __m128 _v = _mm_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr1[j]), _v, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr2[j]), _v, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr3[j]), _v, _sum3);
                }

                _mm_storeu_ps(outptr0 + k, _sum0);
                _mm_storeu_ps(outptr1 + k, _sum1);
                _mm_storeu_ps(outptr2 + k, _sum2);
                _mm_storeu_ps(outptr3 + k, _sum3);
            }
        }
#endif // __SSE2__
        for (; k < out_embed_dim; k++)
        {
            float sum0 = 0.f;
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;

            for (int j = 0; j < max_jj; j++)
            {
                const float v = vptr[j * out_embed_dim + k];
                sum0 += sptr0[j] * v;
                sum1 += sptr1[j] * v;
                sum2 += sptr2[j] * v;
                sum3 += sptr3[j] * v;
            }

            outptr0[k] = sum0;
            outptr1[k] = sum1;
            outptr2[k] = sum2;
            outptr3[k] = sum3;
        }
    }

    for (; ii + 1 < max_ii; ii += 2)
    {
        float* outptr0 = outptr + ii * out_embed_dim;
        float* outptr1 = outptr0 + out_embed_dim;

        const float* sptr0 = score + ii * BLOCK_N;
        const float* sptr1 = sptr0 + BLOCK_N;

        const float* vptr = value + n_start * out_embed_dim;

        int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        {
            for (; k + 15 < out_embed_dim; k += 16)
            {
                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();

                for (int j = 0; j < max_jj; j++)
                {
                    __m512 _v = _mm512_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm512_fmadd_ps(_mm512_set1_ps(sptr1[j]), _v, _sum1);
                }

                _mm512_storeu_ps(outptr0 + k, _sum0);
                _mm512_storeu_ps(outptr1 + k, _sum1);
            }
        }
#endif // __AVX512F__
        {
            for (; k + 7 < out_embed_dim; k += 8)
            {
                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();

                for (int j = 0; j < max_jj; j++)
                {
                    __m256 _v = _mm256_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr1[j]), _v, _sum1);
                }

                _mm256_storeu_ps(outptr0 + k, _sum0);
                _mm256_storeu_ps(outptr1 + k, _sum1);
            }
        }
#endif // __AVX__
        {
            for (; k + 3 < out_embed_dim; k += 4)
            {
                __m128 _sum0 = _mm_setzero_ps();
                __m128 _sum1 = _mm_setzero_ps();

                for (int j = 0; j < max_jj; j++)
                {
                    __m128 _v = _mm_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr1[j]), _v, _sum1);
                }

                _mm_storeu_ps(outptr0 + k, _sum0);
                _mm_storeu_ps(outptr1 + k, _sum1);
            }
        }
#endif // __SSE2__
        for (; k < out_embed_dim; k++)
        {
            float sum0 = 0.f;
            float sum1 = 0.f;

            for (int j = 0; j < max_jj; j++)
            {
                const float v = vptr[j * out_embed_dim + k];
                sum0 += sptr0[j] * v;
                sum1 += sptr1[j] * v;
            }

            outptr0[k] = sum0;
            outptr1[k] = sum1;
        }
    }

    for (; ii < max_ii; ii++)
    {
        float* outptr0 = outptr + ii * out_embed_dim;
        const float* sptr = score + ii * BLOCK_N;

        const float* vptr = value + n_start * out_embed_dim;

        int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        {
            for (; k + 15 < out_embed_dim; k += 16)
            {
                __m512 _sum0 = _mm512_setzero_ps();

                for (int j = 0; j < max_jj; j++)
                {
                    __m512 _v = _mm512_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(sptr[j]), _v, _sum0);
                }

                _mm512_storeu_ps(outptr0 + k, _sum0);
            }
        }
#endif // __AVX512F__
        {
            for (; k + 7 < out_embed_dim; k += 8)
            {
                __m256 _sum0 = _mm256_setzero_ps();

                for (int j = 0; j < max_jj; j++)
                {
                    __m256 _v = _mm256_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr[j]), _v, _sum0);
                }

                _mm256_storeu_ps(outptr0 + k, _sum0);
            }
        }
#endif // __AVX__
        {
            for (; k + 3 < out_embed_dim; k += 4)
            {
                __m128 _sum0 = _mm_setzero_ps();

                for (int j = 0; j < max_jj; j++)
                {
                    __m128 _v = _mm_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr[j]), _v, _sum0);
                }

                _mm_storeu_ps(outptr0 + k, _sum0);
            }
        }
#endif // __SSE2__
        for (; k < out_embed_dim; k++)
        {
            float sum0 = 0.f;

            for (int j = 0; j < max_jj; j++)
            {
                sum0 += sptr[j] * vptr[j * out_embed_dim + k];
            }

            outptr0[k] = sum0;
        }
    }
}

static void sdpa_pv_tile_accumulate_fp32(float* outptr, const float* value, const float* score, int max_ii, int n_start, int max_jj, int out_embed_dim)
{
    const int BLOCK_N = 256;

    int ii = 0;
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        float* outptr0 = outptr + ii * out_embed_dim;
        float* outptr1 = outptr0 + out_embed_dim;
        float* outptr2 = outptr1 + out_embed_dim;
        float* outptr3 = outptr2 + out_embed_dim;
        float* outptr4 = outptr3 + out_embed_dim;
        float* outptr5 = outptr4 + out_embed_dim;
        float* outptr6 = outptr5 + out_embed_dim;
        float* outptr7 = outptr6 + out_embed_dim;
        float* outptr8 = outptr7 + out_embed_dim;
        float* outptr9 = outptr8 + out_embed_dim;
        float* outptra = outptr9 + out_embed_dim;
        float* outptrb = outptra + out_embed_dim;
        float* outptrc = outptrb + out_embed_dim;
        float* outptrd = outptrc + out_embed_dim;
        float* outptre = outptrd + out_embed_dim;
        float* outptrf = outptre + out_embed_dim;

        const float* sptr0 = score + ii * BLOCK_N;
        const float* sptr1 = sptr0 + BLOCK_N;
        const float* sptr2 = sptr1 + BLOCK_N;
        const float* sptr3 = sptr2 + BLOCK_N;
        const float* sptr4 = sptr3 + BLOCK_N;
        const float* sptr5 = sptr4 + BLOCK_N;
        const float* sptr6 = sptr5 + BLOCK_N;
        const float* sptr7 = sptr6 + BLOCK_N;
        const float* sptr8 = sptr7 + BLOCK_N;
        const float* sptr9 = sptr8 + BLOCK_N;
        const float* sptra = sptr9 + BLOCK_N;
        const float* sptrb = sptra + BLOCK_N;
        const float* sptrc = sptrb + BLOCK_N;
        const float* sptrd = sptrc + BLOCK_N;
        const float* sptre = sptrd + BLOCK_N;
        const float* sptrf = sptre + BLOCK_N;

        const float* vptr = value + n_start * out_embed_dim;

        int k = 0;
        for (; k + 15 < out_embed_dim; k += 16)
        {
            __m512 _sum0 = _mm512_loadu_ps(outptr0 + k);
            __m512 _sum1 = _mm512_loadu_ps(outptr1 + k);
            __m512 _sum2 = _mm512_loadu_ps(outptr2 + k);
            __m512 _sum3 = _mm512_loadu_ps(outptr3 + k);
            __m512 _sum4 = _mm512_loadu_ps(outptr4 + k);
            __m512 _sum5 = _mm512_loadu_ps(outptr5 + k);
            __m512 _sum6 = _mm512_loadu_ps(outptr6 + k);
            __m512 _sum7 = _mm512_loadu_ps(outptr7 + k);
            __m512 _sum8 = _mm512_loadu_ps(outptr8 + k);
            __m512 _sum9 = _mm512_loadu_ps(outptr9 + k);
            __m512 _suma = _mm512_loadu_ps(outptra + k);
            __m512 _sumb = _mm512_loadu_ps(outptrb + k);
            __m512 _sumc = _mm512_loadu_ps(outptrc + k);
            __m512 _sumd = _mm512_loadu_ps(outptrd + k);
            __m512 _sume = _mm512_loadu_ps(outptre + k);
            __m512 _sumf = _mm512_loadu_ps(outptrf + k);

            for (int j = 0; j < max_jj; j++)
            {
                __m512 _v = _mm512_loadu_ps(vptr + j * out_embed_dim + k);
                _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(sptr0[j]), _v, _sum0);
                _sum1 = _mm512_fmadd_ps(_mm512_set1_ps(sptr1[j]), _v, _sum1);
                _sum2 = _mm512_fmadd_ps(_mm512_set1_ps(sptr2[j]), _v, _sum2);
                _sum3 = _mm512_fmadd_ps(_mm512_set1_ps(sptr3[j]), _v, _sum3);
                _sum4 = _mm512_fmadd_ps(_mm512_set1_ps(sptr4[j]), _v, _sum4);
                _sum5 = _mm512_fmadd_ps(_mm512_set1_ps(sptr5[j]), _v, _sum5);
                _sum6 = _mm512_fmadd_ps(_mm512_set1_ps(sptr6[j]), _v, _sum6);
                _sum7 = _mm512_fmadd_ps(_mm512_set1_ps(sptr7[j]), _v, _sum7);
                _sum8 = _mm512_fmadd_ps(_mm512_set1_ps(sptr8[j]), _v, _sum8);
                _sum9 = _mm512_fmadd_ps(_mm512_set1_ps(sptr9[j]), _v, _sum9);
                _suma = _mm512_fmadd_ps(_mm512_set1_ps(sptra[j]), _v, _suma);
                _sumb = _mm512_fmadd_ps(_mm512_set1_ps(sptrb[j]), _v, _sumb);
                _sumc = _mm512_fmadd_ps(_mm512_set1_ps(sptrc[j]), _v, _sumc);
                _sumd = _mm512_fmadd_ps(_mm512_set1_ps(sptrd[j]), _v, _sumd);
                _sume = _mm512_fmadd_ps(_mm512_set1_ps(sptre[j]), _v, _sume);
                _sumf = _mm512_fmadd_ps(_mm512_set1_ps(sptrf[j]), _v, _sumf);
            }

            _mm512_storeu_ps(outptr0 + k, _sum0);
            _mm512_storeu_ps(outptr1 + k, _sum1);
            _mm512_storeu_ps(outptr2 + k, _sum2);
            _mm512_storeu_ps(outptr3 + k, _sum3);
            _mm512_storeu_ps(outptr4 + k, _sum4);
            _mm512_storeu_ps(outptr5 + k, _sum5);
            _mm512_storeu_ps(outptr6 + k, _sum6);
            _mm512_storeu_ps(outptr7 + k, _sum7);
            _mm512_storeu_ps(outptr8 + k, _sum8);
            _mm512_storeu_ps(outptr9 + k, _sum9);
            _mm512_storeu_ps(outptra + k, _suma);
            _mm512_storeu_ps(outptrb + k, _sumb);
            _mm512_storeu_ps(outptrc + k, _sumc);
            _mm512_storeu_ps(outptrd + k, _sumd);
            _mm512_storeu_ps(outptre + k, _sume);
            _mm512_storeu_ps(outptrf + k, _sumf);
        }
        for (; k < out_embed_dim; k++)
        {
            float sum0 = outptr0[k];
            float sum1 = outptr1[k];
            float sum2 = outptr2[k];
            float sum3 = outptr3[k];
            float sum4 = outptr4[k];
            float sum5 = outptr5[k];
            float sum6 = outptr6[k];
            float sum7 = outptr7[k];
            float sum8 = outptr8[k];
            float sum9 = outptr9[k];
            float suma = outptra[k];
            float sumb = outptrb[k];
            float sumc = outptrc[k];
            float sumd = outptrd[k];
            float sume = outptre[k];
            float sumf = outptrf[k];

            for (int j = 0; j < max_jj; j++)
            {
                const float v = vptr[j * out_embed_dim + k];
                sum0 += sptr0[j] * v;
                sum1 += sptr1[j] * v;
                sum2 += sptr2[j] * v;
                sum3 += sptr3[j] * v;
                sum4 += sptr4[j] * v;
                sum5 += sptr5[j] * v;
                sum6 += sptr6[j] * v;
                sum7 += sptr7[j] * v;
                sum8 += sptr8[j] * v;
                sum9 += sptr9[j] * v;
                suma += sptra[j] * v;
                sumb += sptrb[j] * v;
                sumc += sptrc[j] * v;
                sumd += sptrd[j] * v;
                sume += sptre[j] * v;
                sumf += sptrf[j] * v;
            }

            outptr0[k] = sum0;
            outptr1[k] = sum1;
            outptr2[k] = sum2;
            outptr3[k] = sum3;
            outptr4[k] = sum4;
            outptr5[k] = sum5;
            outptr6[k] = sum6;
            outptr7[k] = sum7;
            outptr8[k] = sum8;
            outptr9[k] = sum9;
            outptra[k] = suma;
            outptrb[k] = sumb;
            outptrc[k] = sumc;
            outptrd[k] = sumd;
            outptre[k] = sume;
            outptrf[k] = sumf;
        }
    }
#endif

    for (; ii + 7 < max_ii; ii += 8)
    {
        float* outptr0 = outptr + ii * out_embed_dim;
        float* outptr1 = outptr0 + out_embed_dim;
        float* outptr2 = outptr1 + out_embed_dim;
        float* outptr3 = outptr2 + out_embed_dim;
        float* outptr4 = outptr3 + out_embed_dim;
        float* outptr5 = outptr4 + out_embed_dim;
        float* outptr6 = outptr5 + out_embed_dim;
        float* outptr7 = outptr6 + out_embed_dim;

        const float* sptr0 = score + ii * BLOCK_N;
        const float* sptr1 = sptr0 + BLOCK_N;
        const float* sptr2 = sptr1 + BLOCK_N;
        const float* sptr3 = sptr2 + BLOCK_N;
        const float* sptr4 = sptr3 + BLOCK_N;
        const float* sptr5 = sptr4 + BLOCK_N;
        const float* sptr6 = sptr5 + BLOCK_N;
        const float* sptr7 = sptr6 + BLOCK_N;

        const float* vptr = value + n_start * out_embed_dim;

        int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        {
            for (; k + 15 < out_embed_dim; k += 16)
            {
                __m512 _sum0 = _mm512_loadu_ps(outptr0 + k);
                __m512 _sum1 = _mm512_loadu_ps(outptr1 + k);
                __m512 _sum2 = _mm512_loadu_ps(outptr2 + k);
                __m512 _sum3 = _mm512_loadu_ps(outptr3 + k);
                __m512 _sum4 = _mm512_loadu_ps(outptr4 + k);
                __m512 _sum5 = _mm512_loadu_ps(outptr5 + k);
                __m512 _sum6 = _mm512_loadu_ps(outptr6 + k);
                __m512 _sum7 = _mm512_loadu_ps(outptr7 + k);

                for (int j = 0; j < max_jj; j++)
                {
                    __m512 _v = _mm512_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm512_fmadd_ps(_mm512_set1_ps(sptr1[j]), _v, _sum1);
                    _sum2 = _mm512_fmadd_ps(_mm512_set1_ps(sptr2[j]), _v, _sum2);
                    _sum3 = _mm512_fmadd_ps(_mm512_set1_ps(sptr3[j]), _v, _sum3);
                    _sum4 = _mm512_fmadd_ps(_mm512_set1_ps(sptr4[j]), _v, _sum4);
                    _sum5 = _mm512_fmadd_ps(_mm512_set1_ps(sptr5[j]), _v, _sum5);
                    _sum6 = _mm512_fmadd_ps(_mm512_set1_ps(sptr6[j]), _v, _sum6);
                    _sum7 = _mm512_fmadd_ps(_mm512_set1_ps(sptr7[j]), _v, _sum7);
                }

                _mm512_storeu_ps(outptr0 + k, _sum0);
                _mm512_storeu_ps(outptr1 + k, _sum1);
                _mm512_storeu_ps(outptr2 + k, _sum2);
                _mm512_storeu_ps(outptr3 + k, _sum3);
                _mm512_storeu_ps(outptr4 + k, _sum4);
                _mm512_storeu_ps(outptr5 + k, _sum5);
                _mm512_storeu_ps(outptr6 + k, _sum6);
                _mm512_storeu_ps(outptr7 + k, _sum7);
            }
        }
#endif // __AVX512F__
        {
            for (; k + 7 < out_embed_dim; k += 8)
            {
                __m256 _sum0 = _mm256_loadu_ps(outptr0 + k);
                __m256 _sum1 = _mm256_loadu_ps(outptr1 + k);
                __m256 _sum2 = _mm256_loadu_ps(outptr2 + k);
                __m256 _sum3 = _mm256_loadu_ps(outptr3 + k);
                __m256 _sum4 = _mm256_loadu_ps(outptr4 + k);
                __m256 _sum5 = _mm256_loadu_ps(outptr5 + k);
                __m256 _sum6 = _mm256_loadu_ps(outptr6 + k);
                __m256 _sum7 = _mm256_loadu_ps(outptr7 + k);

                for (int j = 0; j < max_jj; j++)
                {
                    __m256 _v = _mm256_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr1[j]), _v, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr2[j]), _v, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr3[j]), _v, _sum3);
                    _sum4 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr4[j]), _v, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr5[j]), _v, _sum5);
                    _sum6 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr6[j]), _v, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr7[j]), _v, _sum7);
                }

                _mm256_storeu_ps(outptr0 + k, _sum0);
                _mm256_storeu_ps(outptr1 + k, _sum1);
                _mm256_storeu_ps(outptr2 + k, _sum2);
                _mm256_storeu_ps(outptr3 + k, _sum3);
                _mm256_storeu_ps(outptr4 + k, _sum4);
                _mm256_storeu_ps(outptr5 + k, _sum5);
                _mm256_storeu_ps(outptr6 + k, _sum6);
                _mm256_storeu_ps(outptr7 + k, _sum7);
            }
        }
#endif // __AVX__
        {
            for (; k + 3 < out_embed_dim; k += 4)
            {
                __m128 _sum0 = _mm_loadu_ps(outptr0 + k);
                __m128 _sum1 = _mm_loadu_ps(outptr1 + k);
                __m128 _sum2 = _mm_loadu_ps(outptr2 + k);
                __m128 _sum3 = _mm_loadu_ps(outptr3 + k);
                __m128 _sum4 = _mm_loadu_ps(outptr4 + k);
                __m128 _sum5 = _mm_loadu_ps(outptr5 + k);
                __m128 _sum6 = _mm_loadu_ps(outptr6 + k);
                __m128 _sum7 = _mm_loadu_ps(outptr7 + k);

                for (int j = 0; j < max_jj; j++)
                {
                    __m128 _v = _mm_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr1[j]), _v, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr2[j]), _v, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr3[j]), _v, _sum3);
                    _sum4 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr4[j]), _v, _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr5[j]), _v, _sum5);
                    _sum6 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr6[j]), _v, _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr7[j]), _v, _sum7);
                }

                _mm_storeu_ps(outptr0 + k, _sum0);
                _mm_storeu_ps(outptr1 + k, _sum1);
                _mm_storeu_ps(outptr2 + k, _sum2);
                _mm_storeu_ps(outptr3 + k, _sum3);
                _mm_storeu_ps(outptr4 + k, _sum4);
                _mm_storeu_ps(outptr5 + k, _sum5);
                _mm_storeu_ps(outptr6 + k, _sum6);
                _mm_storeu_ps(outptr7 + k, _sum7);
            }
        }
#endif // __SSE2__
        for (; k < out_embed_dim; k++)
        {
            float sum0 = outptr0[k];
            float sum1 = outptr1[k];
            float sum2 = outptr2[k];
            float sum3 = outptr3[k];
            float sum4 = outptr4[k];
            float sum5 = outptr5[k];
            float sum6 = outptr6[k];
            float sum7 = outptr7[k];

            for (int j = 0; j < max_jj; j++)
            {
                const float v = vptr[j * out_embed_dim + k];
                sum0 += sptr0[j] * v;
                sum1 += sptr1[j] * v;
                sum2 += sptr2[j] * v;
                sum3 += sptr3[j] * v;
                sum4 += sptr4[j] * v;
                sum5 += sptr5[j] * v;
                sum6 += sptr6[j] * v;
                sum7 += sptr7[j] * v;
            }

            outptr0[k] = sum0;
            outptr1[k] = sum1;
            outptr2[k] = sum2;
            outptr3[k] = sum3;
            outptr4[k] = sum4;
            outptr5[k] = sum5;
            outptr6[k] = sum6;
            outptr7[k] = sum7;
        }
    }

    for (; ii + 3 < max_ii; ii += 4)
    {
        float* outptr0 = outptr + ii * out_embed_dim;
        float* outptr1 = outptr0 + out_embed_dim;
        float* outptr2 = outptr1 + out_embed_dim;
        float* outptr3 = outptr2 + out_embed_dim;

        const float* sptr0 = score + ii * BLOCK_N;
        const float* sptr1 = sptr0 + BLOCK_N;
        const float* sptr2 = sptr1 + BLOCK_N;
        const float* sptr3 = sptr2 + BLOCK_N;

        const float* vptr = value + n_start * out_embed_dim;

        int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        {
            for (; k + 15 < out_embed_dim; k += 16)
            {
                __m512 _sum0 = _mm512_loadu_ps(outptr0 + k);
                __m512 _sum1 = _mm512_loadu_ps(outptr1 + k);
                __m512 _sum2 = _mm512_loadu_ps(outptr2 + k);
                __m512 _sum3 = _mm512_loadu_ps(outptr3 + k);

                for (int j = 0; j < max_jj; j++)
                {
                    __m512 _v = _mm512_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm512_fmadd_ps(_mm512_set1_ps(sptr1[j]), _v, _sum1);
                    _sum2 = _mm512_fmadd_ps(_mm512_set1_ps(sptr2[j]), _v, _sum2);
                    _sum3 = _mm512_fmadd_ps(_mm512_set1_ps(sptr3[j]), _v, _sum3);
                }

                _mm512_storeu_ps(outptr0 + k, _sum0);
                _mm512_storeu_ps(outptr1 + k, _sum1);
                _mm512_storeu_ps(outptr2 + k, _sum2);
                _mm512_storeu_ps(outptr3 + k, _sum3);
            }
        }
#endif // __AVX512F__
        {
            for (; k + 7 < out_embed_dim; k += 8)
            {
                __m256 _sum0 = _mm256_loadu_ps(outptr0 + k);
                __m256 _sum1 = _mm256_loadu_ps(outptr1 + k);
                __m256 _sum2 = _mm256_loadu_ps(outptr2 + k);
                __m256 _sum3 = _mm256_loadu_ps(outptr3 + k);

                for (int j = 0; j < max_jj; j++)
                {
                    __m256 _v = _mm256_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr1[j]), _v, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr2[j]), _v, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr3[j]), _v, _sum3);
                }

                _mm256_storeu_ps(outptr0 + k, _sum0);
                _mm256_storeu_ps(outptr1 + k, _sum1);
                _mm256_storeu_ps(outptr2 + k, _sum2);
                _mm256_storeu_ps(outptr3 + k, _sum3);
            }
        }
#endif // __AVX__
        {
            for (; k + 3 < out_embed_dim; k += 4)
            {
                __m128 _sum0 = _mm_loadu_ps(outptr0 + k);
                __m128 _sum1 = _mm_loadu_ps(outptr1 + k);
                __m128 _sum2 = _mm_loadu_ps(outptr2 + k);
                __m128 _sum3 = _mm_loadu_ps(outptr3 + k);

                for (int j = 0; j < max_jj; j++)
                {
                    __m128 _v = _mm_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr1[j]), _v, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr2[j]), _v, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr3[j]), _v, _sum3);
                }

                _mm_storeu_ps(outptr0 + k, _sum0);
                _mm_storeu_ps(outptr1 + k, _sum1);
                _mm_storeu_ps(outptr2 + k, _sum2);
                _mm_storeu_ps(outptr3 + k, _sum3);
            }
        }
#endif // __SSE2__
        for (; k < out_embed_dim; k++)
        {
            float sum0 = outptr0[k];
            float sum1 = outptr1[k];
            float sum2 = outptr2[k];
            float sum3 = outptr3[k];

            for (int j = 0; j < max_jj; j++)
            {
                const float v = vptr[j * out_embed_dim + k];
                sum0 += sptr0[j] * v;
                sum1 += sptr1[j] * v;
                sum2 += sptr2[j] * v;
                sum3 += sptr3[j] * v;
            }

            outptr0[k] = sum0;
            outptr1[k] = sum1;
            outptr2[k] = sum2;
            outptr3[k] = sum3;
        }
    }

    for (; ii + 1 < max_ii; ii += 2)
    {
        float* outptr0 = outptr + ii * out_embed_dim;
        float* outptr1 = outptr0 + out_embed_dim;

        const float* sptr0 = score + ii * BLOCK_N;
        const float* sptr1 = sptr0 + BLOCK_N;

        const float* vptr = value + n_start * out_embed_dim;

        int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        {
            for (; k + 15 < out_embed_dim; k += 16)
            {
                __m512 _sum0 = _mm512_loadu_ps(outptr0 + k);
                __m512 _sum1 = _mm512_loadu_ps(outptr1 + k);

                for (int j = 0; j < max_jj; j++)
                {
                    __m512 _v = _mm512_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm512_fmadd_ps(_mm512_set1_ps(sptr1[j]), _v, _sum1);
                }

                _mm512_storeu_ps(outptr0 + k, _sum0);
                _mm512_storeu_ps(outptr1 + k, _sum1);
            }
        }
#endif // __AVX512F__
        {
            for (; k + 7 < out_embed_dim; k += 8)
            {
                __m256 _sum0 = _mm256_loadu_ps(outptr0 + k);
                __m256 _sum1 = _mm256_loadu_ps(outptr1 + k);

                for (int j = 0; j < max_jj; j++)
                {
                    __m256 _v = _mm256_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr1[j]), _v, _sum1);
                }

                _mm256_storeu_ps(outptr0 + k, _sum0);
                _mm256_storeu_ps(outptr1 + k, _sum1);
            }
        }
#endif // __AVX__
        {
            for (; k + 3 < out_embed_dim; k += 4)
            {
                __m128 _sum0 = _mm_loadu_ps(outptr0 + k);
                __m128 _sum1 = _mm_loadu_ps(outptr1 + k);

                for (int j = 0; j < max_jj; j++)
                {
                    __m128 _v = _mm_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr0[j]), _v, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr1[j]), _v, _sum1);
                }

                _mm_storeu_ps(outptr0 + k, _sum0);
                _mm_storeu_ps(outptr1 + k, _sum1);
            }
        }
#endif // __SSE2__
        for (; k < out_embed_dim; k++)
        {
            float sum0 = outptr0[k];
            float sum1 = outptr1[k];

            for (int j = 0; j < max_jj; j++)
            {
                const float v = vptr[j * out_embed_dim + k];
                sum0 += sptr0[j] * v;
                sum1 += sptr1[j] * v;
            }

            outptr0[k] = sum0;
            outptr1[k] = sum1;
        }
    }

    for (; ii < max_ii; ii++)
    {
        float* outptr0 = outptr + ii * out_embed_dim;
        const float* sptr = score + ii * BLOCK_N;

        const float* vptr = value + n_start * out_embed_dim;

        int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        {
            for (; k + 15 < out_embed_dim; k += 16)
            {
                __m512 _sum0 = _mm512_loadu_ps(outptr0 + k);

                for (int j = 0; j < max_jj; j++)
                {
                    __m512 _v = _mm512_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(sptr[j]), _v, _sum0);
                }

                _mm512_storeu_ps(outptr0 + k, _sum0);
            }
        }
#endif // __AVX512F__
        {
            for (; k + 7 < out_embed_dim; k += 8)
            {
                __m256 _sum0 = _mm256_loadu_ps(outptr0 + k);

                for (int j = 0; j < max_jj; j++)
                {
                    __m256 _v = _mm256_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(sptr[j]), _v, _sum0);
                }

                _mm256_storeu_ps(outptr0 + k, _sum0);
            }
        }
#endif // __AVX__
        {
            for (; k + 3 < out_embed_dim; k += 4)
            {
                __m128 _sum0 = _mm_loadu_ps(outptr0 + k);

                for (int j = 0; j < max_jj; j++)
                {
                    __m128 _v = _mm_loadu_ps(vptr + j * out_embed_dim + k);
                    _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(sptr[j]), _v, _sum0);
                }

                _mm_storeu_ps(outptr0 + k, _sum0);
            }
        }
#endif // __SSE2__
        for (; k < out_embed_dim; k++)
        {
            float sum0 = outptr0[k];

            for (int j = 0; j < max_jj; j++)
            {
                sum0 += sptr[j] * vptr[j * out_embed_dim + k];
            }

            outptr0[k] = sum0;
        }
    }
}

static void sdpa_flash_attention_tile_fp32(const float* query, const float* past_key, const float* cur_key, const float* past_value, const float* cur_value, const float* mask, float* outptr, float* score, float* m_vec, float* l_vec, int i, int max_ii, int n_begin, int n_end, int nstep, int past_seqlen, int embed_dim, int out_embed_dim, int mask_stride, float scale, bool k_end)
{
    const int BLOCK_N = 256;

    for (int ii = 0; ii < max_ii; ii++)
    {
        m_vec[ii] = -FLT_MAX;
        l_vec[ii] = 0.f;
    }

    for (int n_start = n_begin; n_start < n_end; n_start += nstep)
    {
        int max_jj = nstep;
        if (n_start + nstep > n_end)
            max_jj = n_end - n_start;

        int past_n = 0;
        if (n_start < past_seqlen)
        {
            past_n = past_seqlen - n_start;
            if (past_n > max_jj)
                past_n = max_jj;
        }

        if (past_n > 0)
            sdpa_qk_tile_fp32(query, past_key, score, i, max_ii, n_start, past_n, embed_dim, scale);

        if (past_n < max_jj)
        {
            const int cur_start = n_start + past_n - past_seqlen;
            const int cur_n = max_jj - past_n;
            sdpa_qk_tile_fp32(query, cur_key, score + past_n, i, max_ii, cur_start, cur_n, embed_dim, scale);
        }

        for (int ii = 0; ii < max_ii; ii++)
        {
            float* sptr = score + ii * BLOCK_N;
            float* outptr0 = outptr + ii * out_embed_dim;
            const float* mptr = mask ? mask + (i + ii) * mask_stride + n_start : 0;

            float max = m_vec[ii];

            int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            {
                __m512 _max = _mm512_set1_ps(max);
                for (; j + 15 < max_jj; j += 16)
                {
                    __m512 _s = _mm512_loadu_ps(sptr + j);
                    if (mptr)
                    {
                        _s = _mm512_add_ps(_s, _mm512_loadu_ps(mptr + j));
                        _mm512_storeu_ps(sptr + j, _s);
                    }
                    _max = _mm512_max_ps(_max, _s);
                }
                max = std::max(max, _mm512_comp_reduce_max_ps(_max));
            }
#endif // __AVX512F__
            {
                __m256 _max = _mm256_set1_ps(max);
                for (; j + 7 < max_jj; j += 8)
                {
                    __m256 _s = _mm256_loadu_ps(sptr + j);
                    if (mptr)
                    {
                        _s = _mm256_add_ps(_s, _mm256_loadu_ps(mptr + j));
                        _mm256_storeu_ps(sptr + j, _s);
                    }
                    _max = _mm256_max_ps(_max, _s);
                }
                max = std::max(max, _mm256_reduce_max_ps(_max));
            }
#endif // __AVX__
            {
                __m128 _max = _mm_set1_ps(max);
                for (; j + 3 < max_jj; j += 4)
                {
                    __m128 _s = _mm_loadu_ps(sptr + j);
                    if (mptr)
                    {
                        _s = _mm_add_ps(_s, _mm_loadu_ps(mptr + j));
                        _mm_storeu_ps(sptr + j, _s);
                    }
                    _max = _mm_max_ps(_max, _s);
                }
                max = std::max(max, _mm_reduce_max_ps(_max));
            }
#endif // __SSE2__
            for (; j < max_jj; j++)
            {
                if (mptr)
                    sptr[j] += mptr[j];
                max = std::max(max, sptr[j]);
            }

            if (l_vec[ii] != 0.f)
            {
                const float correction = expf(m_vec[ii] - max);
                if (correction != 1.f)
                {
                    int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    {
                        __m512 _correction = _mm512_set1_ps(correction);
                        for (; k + 15 < out_embed_dim; k += 16)
                        {
                            _mm512_storeu_ps(outptr0 + k, _mm512_mul_ps(_mm512_loadu_ps(outptr0 + k), _correction));
                        }
                    }
#endif // __AVX512F__
                    {
                        __m256 _correction = _mm256_set1_ps(correction);
                        for (; k + 7 < out_embed_dim; k += 8)
                        {
                            _mm256_storeu_ps(outptr0 + k, _mm256_mul_ps(_mm256_loadu_ps(outptr0 + k), _correction));
                        }
                    }
#endif // __AVX__
                    {
                        __m128 _correction = _mm_set1_ps(correction);
                        for (; k + 3 < out_embed_dim; k += 4)
                        {
                            _mm_storeu_ps(outptr0 + k, _mm_mul_ps(_mm_loadu_ps(outptr0 + k), _correction));
                        }
                    }
#endif // __SSE2__
                    for (; k < out_embed_dim; k++)
                    {
                        outptr0[k] *= correction;
                    }

                    l_vec[ii] *= correction;
                }
            }

            float sum = 0.f;

            j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            {
                __m512 _max0 = _mm512_set1_ps(max);
                __m512 _sum0 = _mm512_setzero_ps();
                for (; j + 15 < max_jj; j += 16)
                {
                    __m512 _p = _mm512_loadu_ps(sptr + j);
                    _p = exp512_ps(_mm512_sub_ps(_p, _max0));
                    _mm512_storeu_ps(sptr + j, _p);
                    _sum0 = _mm512_add_ps(_sum0, _p);
                }
                sum += _mm512_comp_reduce_add_ps(_sum0);
            }
#endif // __AVX512F__
            {
                __m256 _max0 = _mm256_set1_ps(max);
                __m256 _sum0 = _mm256_setzero_ps();
                for (; j + 7 < max_jj; j += 8)
                {
                    __m256 _p = _mm256_loadu_ps(sptr + j);
                    _p = exp256_ps(_mm256_sub_ps(_p, _max0));
                    _mm256_storeu_ps(sptr + j, _p);
                    _sum0 = _mm256_add_ps(_sum0, _p);
                }
                sum += _mm256_reduce_add_ps(_sum0);
            }
#endif // __AVX__
            {
                __m128 _max0 = _mm_set1_ps(max);
                __m128 _sum0 = _mm_setzero_ps();
                for (; j + 3 < max_jj; j += 4)
                {
                    __m128 _p = _mm_loadu_ps(sptr + j);
                    _p = exp_ps(_mm_sub_ps(_p, _max0));
                    _mm_storeu_ps(sptr + j, _p);
                    _sum0 = _mm_add_ps(_sum0, _p);
                }
                sum += _mm_reduce_add_ps(_sum0);
            }
#endif // __SSE2__
            for (; j < max_jj; j++)
            {
                float v = expf(sptr[j] - max);
                sptr[j] = v;
                sum += v;
            }

            l_vec[ii] += sum;
            m_vec[ii] = max;
        }

        if (past_n > 0)
        {
            if (n_start == n_begin)
                sdpa_pv_tile_fp32(outptr, past_value, score, max_ii, n_start, past_n, out_embed_dim);
            else
                sdpa_pv_tile_accumulate_fp32(outptr, past_value, score, max_ii, n_start, past_n, out_embed_dim);
        }

        if (past_n < max_jj)
        {
            const int cur_start = n_start + past_n - past_seqlen;
            const int cur_n = max_jj - past_n;
            if (n_start == n_begin && past_n == 0)
                sdpa_pv_tile_fp32(outptr, cur_value, score + past_n, max_ii, cur_start, cur_n, out_embed_dim);
            else
                sdpa_pv_tile_accumulate_fp32(outptr, cur_value, score + past_n, max_ii, cur_start, cur_n, out_embed_dim);
        }
    }

    if (k_end)
    {
        for (int ii = 0; ii < max_ii; ii++)
        {
            float* outptr0 = outptr + ii * out_embed_dim;
            const float scale0 = 1.f / l_vec[ii];

            int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            {
                __m512 _scale0 = _mm512_set1_ps(scale0);
                for (; k + 15 < out_embed_dim; k += 16)
                {
                    _mm512_storeu_ps(outptr0 + k, _mm512_mul_ps(_mm512_loadu_ps(outptr0 + k), _scale0));
                }
            }
#endif // __AVX512F__
            {
                __m256 _scale0 = _mm256_set1_ps(scale0);
                for (; k + 7 < out_embed_dim; k += 8)
                {
                    _mm256_storeu_ps(outptr0 + k, _mm256_mul_ps(_mm256_loadu_ps(outptr0 + k), _scale0));
                }
            }
#endif // __AVX__
            {
                __m128 _scale0 = _mm_set1_ps(scale0);
                for (; k + 3 < out_embed_dim; k += 4)
                {
                    _mm_storeu_ps(outptr0 + k, _mm_mul_ps(_mm_loadu_ps(outptr0 + k), _scale0));
                }
            }
#endif // __SSE2__
            for (; k < out_embed_dim; k++)
            {
                outptr0[k] *= scale0;
            }
        }
    }
}

static int sdpa_pack_key_tile_fp32(const Mat& cur_key, Mat& packed_key, int tile_n, const Option& opt)
{
    const int embed_dim = cur_key.w;
    const int cur_seqlen = cur_key.h;
    const int num_group = cur_key.c;
    const size_t packed_key_tile_stride = sdpa_packed_key_tile_stride_fp32(tile_n, embed_dim);
    const int num_tiles = (cur_seqlen + tile_n - 1) / tile_n;

    packed_key.create(packed_key_tile_stride, num_tiles, num_group, 4u, opt.workspace_allocator);
    if (packed_key.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int task_id = 0; task_id < num_group * num_tiles; task_id++)
    {
        const int q = task_id / num_tiles;
        const int tile_id = task_id % num_tiles;
        const int tile_base = tile_id * tile_n;

        const Mat cur_key_head = cur_key.channel(q);
        Mat packed_key_head = packed_key.channel(q);
        const float* key_base = (const float*)cur_key_head + (size_t)tile_base * embed_dim;
        float* pp = (float*)packed_key_head + (size_t)tile_id * packed_key_tile_stride;

        const int max_jj = std::min(tile_n, cur_seqlen - tile_base);
        int j = 0;

#if __AVX512F__
        for (; j + 15 < max_jj; j += 16)
        {
            const float* p0 = key_base + (size_t)j * embed_dim;
            const float* p1 = p0 + embed_dim;
            const float* p2 = p1 + embed_dim;
            const float* p3 = p2 + embed_dim;
            const float* p4 = p3 + embed_dim;
            const float* p5 = p4 + embed_dim;
            const float* p6 = p5 + embed_dim;
            const float* p7 = p6 + embed_dim;
            const float* p8 = p7 + embed_dim;
            const float* p9 = p8 + embed_dim;
            const float* pa = p9 + embed_dim;
            const float* pb = pa + embed_dim;
            const float* pc = pb + embed_dim;
            const float* pd = pc + embed_dim;
            const float* pe = pd + embed_dim;
            const float* pf = pe + embed_dim;

            int k = 0;
            for (; k + 15 < embed_dim; k += 16)
            {
                __m512 _r0 = _mm512_loadu_ps(p0);
                __m512 _r1 = _mm512_loadu_ps(p1);
                __m512 _r2 = _mm512_loadu_ps(p2);
                __m512 _r3 = _mm512_loadu_ps(p3);
                __m512 _r4 = _mm512_loadu_ps(p4);
                __m512 _r5 = _mm512_loadu_ps(p5);
                __m512 _r6 = _mm512_loadu_ps(p6);
                __m512 _r7 = _mm512_loadu_ps(p7);
                __m512 _r8 = _mm512_loadu_ps(p8);
                __m512 _r9 = _mm512_loadu_ps(p9);
                __m512 _ra = _mm512_loadu_ps(pa);
                __m512 _rb = _mm512_loadu_ps(pb);
                __m512 _rc = _mm512_loadu_ps(pc);
                __m512 _rd = _mm512_loadu_ps(pd);
                __m512 _re = _mm512_loadu_ps(pe);
                __m512 _rf = _mm512_loadu_ps(pf);

                transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);

                _mm512_storeu_ps(pp, _r0);
                _mm512_storeu_ps(pp + 16, _r1);
                _mm512_storeu_ps(pp + 32, _r2);
                _mm512_storeu_ps(pp + 48, _r3);
                _mm512_storeu_ps(pp + 64, _r4);
                _mm512_storeu_ps(pp + 80, _r5);
                _mm512_storeu_ps(pp + 96, _r6);
                _mm512_storeu_ps(pp + 112, _r7);
                _mm512_storeu_ps(pp + 128, _r8);
                _mm512_storeu_ps(pp + 144, _r9);
                _mm512_storeu_ps(pp + 160, _ra);
                _mm512_storeu_ps(pp + 176, _rb);
                _mm512_storeu_ps(pp + 192, _rc);
                _mm512_storeu_ps(pp + 208, _rd);
                _mm512_storeu_ps(pp + 224, _re);
                _mm512_storeu_ps(pp + 240, _rf);
                pp += 256;

                p0 += 16;
                p1 += 16;
                p2 += 16;
                p3 += 16;
                p4 += 16;
                p5 += 16;
                p6 += 16;
                p7 += 16;
                p8 += 16;
                p9 += 16;
                pa += 16;
                pb += 16;
                pc += 16;
                pd += 16;
                pe += 16;
                pf += 16;
            }
            for (; k < embed_dim; k++)
            {
                pp[0] = *p0++;
                pp[1] = *p1++;
                pp[2] = *p2++;
                pp[3] = *p3++;
                pp[4] = *p4++;
                pp[5] = *p5++;
                pp[6] = *p6++;
                pp[7] = *p7++;
                pp[8] = *p8++;
                pp[9] = *p9++;
                pp[10] = *pa++;
                pp[11] = *pb++;
                pp[12] = *pc++;
                pp[13] = *pd++;
                pp[14] = *pe++;
                pp[15] = *pf++;
                pp += 16;
            }
        }
#endif // __AVX512F__
#if __AVX__
        for (; j + 7 < max_jj; j += 8)
        {
            const float* p0 = key_base + (size_t)j * embed_dim;
            const float* p1 = p0 + embed_dim;
            const float* p2 = p1 + embed_dim;
            const float* p3 = p2 + embed_dim;
            const float* p4 = p3 + embed_dim;
            const float* p5 = p4 + embed_dim;
            const float* p6 = p5 + embed_dim;
            const float* p7 = p6 + embed_dim;

            int k = 0;
            for (; k + 7 < embed_dim; k += 8)
            {
                __m256 _r0 = _mm256_loadu_ps(p0);
                __m256 _r1 = _mm256_loadu_ps(p1);
                __m256 _r2 = _mm256_loadu_ps(p2);
                __m256 _r3 = _mm256_loadu_ps(p3);
                __m256 _r4 = _mm256_loadu_ps(p4);
                __m256 _r5 = _mm256_loadu_ps(p5);
                __m256 _r6 = _mm256_loadu_ps(p6);
                __m256 _r7 = _mm256_loadu_ps(p7);

                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                _mm256_storeu_ps(pp, _r0);
                _mm256_storeu_ps(pp + 8, _r1);
                _mm256_storeu_ps(pp + 16, _r2);
                _mm256_storeu_ps(pp + 24, _r3);
                _mm256_storeu_ps(pp + 32, _r4);
                _mm256_storeu_ps(pp + 40, _r5);
                _mm256_storeu_ps(pp + 48, _r6);
                _mm256_storeu_ps(pp + 56, _r7);
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
            for (; k < embed_dim; k++)
            {
                pp[0] = *p0++;
                pp[1] = *p1++;
                pp[2] = *p2++;
                pp[3] = *p3++;
                pp[4] = *p4++;
                pp[5] = *p5++;
                pp[6] = *p6++;
                pp[7] = *p7++;
                pp += 8;
            }
        }
#endif // __AVX__
#if __SSE2__
        for (; j + 3 < max_jj; j += 4)
        {
            const float* p0 = key_base + (size_t)j * embed_dim;
            const float* p1 = p0 + embed_dim;
            const float* p2 = p1 + embed_dim;
            const float* p3 = p2 + embed_dim;

            int k = 0;
            for (; k + 3 < embed_dim; k += 4)
            {
                __m128 _r0 = _mm_loadu_ps(p0);
                __m128 _r1 = _mm_loadu_ps(p1);
                __m128 _r2 = _mm_loadu_ps(p2);
                __m128 _r3 = _mm_loadu_ps(p3);

                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                _mm_storeu_ps(pp, _r0);
                _mm_storeu_ps(pp + 4, _r1);
                _mm_storeu_ps(pp + 8, _r2);
                _mm_storeu_ps(pp + 12, _r3);
                pp += 16;

                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
            for (; k < embed_dim; k++)
            {
                pp[0] = *p0++;
                pp[1] = *p1++;
                pp[2] = *p2++;
                pp[3] = *p3++;
                pp += 4;
            }
        }
#endif // __SSE2__

        for (; j < max_jj; j++)
        {
            const float* p0 = key_base + (size_t)j * embed_dim;
            memcpy(pp, p0, (size_t)embed_dim * sizeof(float));
            pp += embed_dim;
        }
    }

    return 0;
}

static int sdpa_pack_value_tile_fp32(const Mat& cur_value, Mat& packed_value, int tile_n, const Option& opt)
{
    const int out_embed_dim = cur_value.w;
    const int cur_seqlen = cur_value.h;
    const int num_group = cur_value.c;
    const size_t packed_value_tile_stride = sdpa_packed_value_tile_stride_fp32(tile_n, out_embed_dim);
    const int num_tiles = (cur_seqlen + tile_n - 1) / tile_n;

    packed_value.create(packed_value_tile_stride, num_tiles, num_group, 4u, opt.workspace_allocator);
    if (packed_value.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int task_id = 0; task_id < num_group * num_tiles; task_id++)
    {
        const int q = task_id / num_tiles;
        const int tile_id = task_id % num_tiles;
        const int tile_base = tile_id * tile_n;

        const Mat cur_value_head = cur_value.channel(q);
        Mat packed_value_head = packed_value.channel(q);
        const float* value_base = (const float*)cur_value_head + (size_t)tile_base * out_embed_dim;
        float* pp = (float*)packed_value_head + (size_t)tile_id * packed_value_tile_stride;

        const int max_jj = std::min(tile_n, cur_seqlen - tile_base);

        int k = 0;
#if __AVX512F__
        for (; k + 15 < out_embed_dim; k += 16)
        {
            const float* p0 = value_base + k;
            const float* p1 = p0 + out_embed_dim;
            const float* p2 = p1 + out_embed_dim;
            const float* p3 = p2 + out_embed_dim;
            int j = 0;
            for (; j + 3 < max_jj; j += 4)
            {
                _mm512_storeu_ps(pp, _mm512_loadu_ps(p0));
                _mm512_storeu_ps(pp + 16, _mm512_loadu_ps(p1));
                _mm512_storeu_ps(pp + 32, _mm512_loadu_ps(p2));
                _mm512_storeu_ps(pp + 48, _mm512_loadu_ps(p3));
                pp += 64;
                p0 += (size_t)out_embed_dim * 4;
                p1 += (size_t)out_embed_dim * 4;
                p2 += (size_t)out_embed_dim * 4;
                p3 += (size_t)out_embed_dim * 4;
            }
            for (; j < max_jj; j++)
            {
                _mm512_storeu_ps(pp, _mm512_loadu_ps(p0));
                pp += 16;
                p0 += out_embed_dim;
            }
        }
#endif // __AVX512F__
#if __AVX__
        for (; k + 7 < out_embed_dim; k += 8)
        {
            const float* p0 = value_base + k;
            const float* p1 = p0 + out_embed_dim;
            const float* p2 = p1 + out_embed_dim;
            const float* p3 = p2 + out_embed_dim;
            int j = 0;
            for (; j + 3 < max_jj; j += 4)
            {
                _mm256_storeu_ps(pp, _mm256_loadu_ps(p0));
                _mm256_storeu_ps(pp + 8, _mm256_loadu_ps(p1));
                _mm256_storeu_ps(pp + 16, _mm256_loadu_ps(p2));
                _mm256_storeu_ps(pp + 24, _mm256_loadu_ps(p3));
                pp += 32;
                p0 += (size_t)out_embed_dim * 4;
                p1 += (size_t)out_embed_dim * 4;
                p2 += (size_t)out_embed_dim * 4;
                p3 += (size_t)out_embed_dim * 4;
            }
            for (; j < max_jj; j++)
            {
                _mm256_storeu_ps(pp, _mm256_loadu_ps(p0));
                pp += 8;
                p0 += out_embed_dim;
            }
        }
#endif // __AVX__
#if __SSE2__
        for (; k + 3 < out_embed_dim; k += 4)
        {
            const float* p0 = value_base + k;
            const float* p1 = p0 + out_embed_dim;
            const float* p2 = p1 + out_embed_dim;
            const float* p3 = p2 + out_embed_dim;
            int j = 0;
            for (; j + 3 < max_jj; j += 4)
            {
                _mm_storeu_ps(pp, _mm_loadu_ps(p0));
                _mm_storeu_ps(pp + 4, _mm_loadu_ps(p1));
                _mm_storeu_ps(pp + 8, _mm_loadu_ps(p2));
                _mm_storeu_ps(pp + 12, _mm_loadu_ps(p3));
                pp += 16;
                p0 += (size_t)out_embed_dim * 4;
                p1 += (size_t)out_embed_dim * 4;
                p2 += (size_t)out_embed_dim * 4;
                p3 += (size_t)out_embed_dim * 4;
            }
            for (; j < max_jj; j++)
            {
                _mm_storeu_ps(pp, _mm_loadu_ps(p0));
                pp += 4;
                p0 += out_embed_dim;
            }
        }
#endif // __SSE2__
        for (; k < out_embed_dim; k++)
        {
            const float* p0 = value_base + k;
            for (int j = 0; j < max_jj; j++)
            {
                *pp++ = *p0;
                p0 += out_embed_dim;
            }
        }
    }

    return 0;
}

static void sdpa_flash_attention_reduce_fp32(float* outptr, const Mat& partials_head, int block_m, int out_embed_dim, int num_kv_chunks)
{
    const int MAX_BLOCK_M = 16;

    float m_vec[MAX_BLOCK_M];
    float l_vec[MAX_BLOCK_M];

    for (int i = 0; i < block_m; i++)
    {
        m_vec[i] = -FLT_MAX;
        l_vec[i] = 0.f;

        float* outptr0 = outptr + i * out_embed_dim;

        int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        {
            __m512 _zero = _mm512_setzero_ps();
            for (; k + 15 < out_embed_dim; k += 16)
            {
                _mm512_storeu_ps(outptr0 + k, _zero);
            }
        }
#endif // __AVX512F__
        {
            __m256 _zero = _mm256_setzero_ps();
            for (; k + 7 < out_embed_dim; k += 8)
            {
                _mm256_storeu_ps(outptr0 + k, _zero);
            }
        }
#endif // __AVX__
        {
            __m128 _zero = _mm_setzero_ps();
            for (; k + 3 < out_embed_dim; k += 4)
            {
                _mm_storeu_ps(outptr0 + k, _zero);
            }
        }
#endif // __SSE2__
        for (; k < out_embed_dim; k++)
        {
            outptr0[k] = 0.f;
        }
    }

    for (int c = 0; c < num_kv_chunks; c++)
    {
        const float* partial = partials_head.row(c);
        const float* mptr = partial;
        const float* lptr = partial + MAX_BLOCK_M;
        const float* outptr_chunk = partial + MAX_BLOCK_M * 2;

        for (int i = 0; i < block_m; i++)
        {
            const float l_chunk = lptr[i];
            if (l_chunk != 0.f)
            {
                const float m_chunk = mptr[i];
                const float m = std::max(m_vec[i], m_chunk);
                const float scale0 = expf(m_vec[i] - m);
                const float scale1 = expf(m_chunk - m);

                float* outptr0 = outptr + i * out_embed_dim;
                const float* outptr1 = outptr_chunk + i * out_embed_dim;

                int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                {
                    __m512 _scale0 = _mm512_set1_ps(scale0);
                    __m512 _scale1 = _mm512_set1_ps(scale1);
                    for (; k + 15 < out_embed_dim; k += 16)
                    {
                        __m512 _out0 = _mm512_loadu_ps(outptr0 + k);
                        __m512 _out1 = _mm512_loadu_ps(outptr1 + k);
                        _out0 = _mm512_mul_ps(_out0, _scale0);
                        _out0 = _mm512_fmadd_ps(_out1, _scale1, _out0);
                        _mm512_storeu_ps(outptr0 + k, _out0);
                    }
                }
#endif // __AVX512F__
                {
                    __m256 _scale0 = _mm256_set1_ps(scale0);
                    __m256 _scale1 = _mm256_set1_ps(scale1);
                    for (; k + 7 < out_embed_dim; k += 8)
                    {
                        __m256 _out0 = _mm256_loadu_ps(outptr0 + k);
                        __m256 _out1 = _mm256_loadu_ps(outptr1 + k);
                        _out0 = _mm256_mul_ps(_out0, _scale0);
                        _out0 = _mm256_comp_fmadd_ps(_out1, _scale1, _out0);
                        _mm256_storeu_ps(outptr0 + k, _out0);
                    }
                }
#endif // __AVX__
                {
                    __m128 _scale0 = _mm_set1_ps(scale0);
                    __m128 _scale1 = _mm_set1_ps(scale1);
                    for (; k + 3 < out_embed_dim; k += 4)
                    {
                        __m128 _out0 = _mm_loadu_ps(outptr0 + k);
                        __m128 _out1 = _mm_loadu_ps(outptr1 + k);
                        _out0 = _mm_mul_ps(_out0, _scale0);
                        _out0 = _mm_comp_fmadd_ps(_out1, _scale1, _out0);
                        _mm_storeu_ps(outptr0 + k, _out0);
                    }
                }
#endif // __SSE2__
                for (; k < out_embed_dim; k++)
                {
                    outptr0[k] = outptr0[k] * scale0 + outptr1[k] * scale1;
                }

                l_vec[i] = l_vec[i] * scale0 + l_chunk * scale1;
                m_vec[i] = m;
            }
        }
    }

    for (int i = 0; i < block_m; i++)
    {
        float* outptr0 = outptr + i * out_embed_dim;
        const float scale0 = 1.f / l_vec[i];

        int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        {
            __m512 _scale0 = _mm512_set1_ps(scale0);
            for (; k + 15 < out_embed_dim; k += 16)
            {
                _mm512_storeu_ps(outptr0 + k, _mm512_mul_ps(_mm512_loadu_ps(outptr0 + k), _scale0));
            }
        }
#endif // __AVX512F__
        {
            __m256 _scale0 = _mm256_set1_ps(scale0);
            for (; k + 7 < out_embed_dim; k += 8)
            {
                _mm256_storeu_ps(outptr0 + k, _mm256_mul_ps(_mm256_loadu_ps(outptr0 + k), _scale0));
            }
        }
#endif // __AVX__
        {
            __m128 _scale0 = _mm_set1_ps(scale0);
            for (; k + 3 < out_embed_dim; k += 4)
            {
                _mm_storeu_ps(outptr0 + k, _mm_mul_ps(_mm_loadu_ps(outptr0 + k), _scale0));
            }
        }
#endif // __SSE2__
        for (; k < out_embed_dim; k++)
        {
            outptr0[k] *= scale0;
        }
    }
}

static void sdpa_flash_attention_reduce_outT_fp32(float* outptr, const Mat& partials_head, int block_m, int out_embed_dim, int num_kv_chunks)
{
    const int MAX_BLOCK_M = 16;
    const int MAX_OUT_EMBED_DIM = 128;

    float m_vec[MAX_BLOCK_M];
    float l_vec[MAX_BLOCK_M];
    float outT[MAX_BLOCK_M * MAX_OUT_EMBED_DIM];

    for (int i = 0; i < block_m; i++)
    {
        m_vec[i] = -FLT_MAX;
        l_vec[i] = 0.f;
    }

    for (int i = 0; i < block_m * out_embed_dim; i++)
    {
        outT[i] = 0.f;
    }

    for (int c = 0; c < num_kv_chunks; c++)
    {
        const float* partial = partials_head.row(c);
        const float* mptr = partial;
        const float* lptr = partial + MAX_BLOCK_M;
        const float* outptr_chunk = partial + MAX_BLOCK_M * 2;

        int ii = 0;
#if __AVX512F__
        for (; ii + 15 < block_m; ii += 16)
        {
            __m512 _m0 = _mm512_loadu_ps(m_vec + ii);
            __m512 _l0 = _mm512_loadu_ps(l_vec + ii);
            __m512 _m1 = _mm512_loadu_ps(mptr + ii);
            __m512 _l1 = _mm512_loadu_ps(lptr + ii);

            __m512 _m = _mm512_max_ps(_m0, _m1);
            __m512 _scale0 = exp512_ps(_mm512_sub_ps(_m0, _m));
            __m512 _scale1 = exp512_ps(_mm512_sub_ps(_m1, _m));

            float* outptr0 = outT + ii * out_embed_dim;
            const float* outptr1 = outptr_chunk + ii * out_embed_dim;

            for (int k = 0; k < out_embed_dim; k++)
            {
                __m512 _out0 = _mm512_loadu_ps(outptr0);
                __m512 _out1 = _mm512_loadu_ps(outptr1);
                _out0 = _mm512_mul_ps(_out0, _scale0);
                _out0 = _mm512_fmadd_ps(_out1, _scale1, _out0);
                _mm512_storeu_ps(outptr0, _out0);

                outptr0 += 16;
                outptr1 += 16;
            }

            _l0 = _mm512_fmadd_ps(_l0, _scale0, _mm512_mul_ps(_l1, _scale1));
            _mm512_storeu_ps(m_vec + ii, _m);
            _mm512_storeu_ps(l_vec + ii, _l0);
        }
#endif // __AVX512F__

#if __AVX__
        for (; ii + 7 < block_m; ii += 8)
        {
            __m256 _m0 = _mm256_loadu_ps(m_vec + ii);
            __m256 _l0 = _mm256_loadu_ps(l_vec + ii);
            __m256 _m1 = _mm256_loadu_ps(mptr + ii);
            __m256 _l1 = _mm256_loadu_ps(lptr + ii);

            __m256 _m = _mm256_max_ps(_m0, _m1);
            __m256 _scale0 = exp256_ps(_mm256_sub_ps(_m0, _m));
            __m256 _scale1 = exp256_ps(_mm256_sub_ps(_m1, _m));

            float* outptr0 = outT + ii * out_embed_dim;
            const float* outptr1 = outptr_chunk + ii * out_embed_dim;

            for (int k = 0; k < out_embed_dim; k++)
            {
                __m256 _out0 = _mm256_loadu_ps(outptr0);
                __m256 _out1 = _mm256_loadu_ps(outptr1);
                _out0 = _mm256_mul_ps(_out0, _scale0);
                _out0 = _mm256_comp_fmadd_ps(_out1, _scale1, _out0);
                _mm256_storeu_ps(outptr0, _out0);

                outptr0 += 8;
                outptr1 += 8;
            }

            _l0 = _mm256_comp_fmadd_ps(_l0, _scale0, _mm256_mul_ps(_l1, _scale1));
            _mm256_storeu_ps(m_vec + ii, _m);
            _mm256_storeu_ps(l_vec + ii, _l0);
        }
#endif // __AVX__

#if __SSE2__
        for (; ii + 3 < block_m; ii += 4)
        {
            __m128 _m0 = _mm_loadu_ps(m_vec + ii);
            __m128 _l0 = _mm_loadu_ps(l_vec + ii);
            __m128 _m1 = _mm_loadu_ps(mptr + ii);
            __m128 _l1 = _mm_loadu_ps(lptr + ii);

            __m128 _m = _mm_max_ps(_m0, _m1);
            __m128 _scale0 = exp_ps(_mm_sub_ps(_m0, _m));
            __m128 _scale1 = exp_ps(_mm_sub_ps(_m1, _m));

            float* outptr0 = outT + ii * out_embed_dim;
            const float* outptr1 = outptr_chunk + ii * out_embed_dim;

            for (int k = 0; k < out_embed_dim; k++)
            {
                __m128 _out0 = _mm_loadu_ps(outptr0);
                __m128 _out1 = _mm_loadu_ps(outptr1);
                _out0 = _mm_mul_ps(_out0, _scale0);
                _out0 = _mm_comp_fmadd_ps(_out1, _scale1, _out0);
                _mm_storeu_ps(outptr0, _out0);

                outptr0 += 4;
                outptr1 += 4;
            }

            _l0 = _mm_comp_fmadd_ps(_l0, _scale0, _mm_mul_ps(_l1, _scale1));
            _mm_storeu_ps(m_vec + ii, _m);
            _mm_storeu_ps(l_vec + ii, _l0);
        }
#endif // __SSE2__

        for (; ii < block_m; ii++)
        {
            const float l_chunk = lptr[ii];
            if (l_chunk == 0.f)
                continue;

            const float m_chunk = mptr[ii];
            const float m = std::max(m_vec[ii], m_chunk);
            const float scale0 = expf(m_vec[ii] - m);
            const float scale1 = expf(m_chunk - m);

            float* outptr0 = outT + ii * out_embed_dim;
            const float* outptr1 = outptr_chunk + ii * out_embed_dim;

            for (int k = 0; k < out_embed_dim; k++)
            {
                outptr0[k] = outptr0[k] * scale0 + outptr1[k] * scale1;
            }

            l_vec[ii] = l_vec[ii] * scale0 + l_chunk * scale1;
            m_vec[ii] = m;
        }
    }

    int ii = 0;
#if __AVX512F__
    for (; ii + 15 < block_m; ii += 16)
    {
        __m512 _l = _mm512_loadu_ps(l_vec + ii);
        float inv_l[16];
        _mm512_storeu_ps(inv_l, _mm512_div_ps(_mm512_set1_ps(1.f), _l));
        sdpa_store_out_tile_fp32(outptr + ii * out_embed_dim, outT + ii * out_embed_dim, out_embed_dim, 16, inv_l);
    }
#endif // __AVX512F__

#if __AVX__
    for (; ii + 7 < block_m; ii += 8)
    {
        __m256 _l = _mm256_loadu_ps(l_vec + ii);
        float inv_l[8];
        _mm256_storeu_ps(inv_l, _mm256_div_ps(_mm256_set1_ps(1.f), _l));
        sdpa_store_out_tile_fp32(outptr + ii * out_embed_dim, outT + ii * out_embed_dim, out_embed_dim, 8, inv_l);
    }
#endif // __AVX__

#if __SSE2__
    for (; ii + 3 < block_m; ii += 4)
    {
        __m128 _l = _mm_loadu_ps(l_vec + ii);
        float inv_l[4];
        _mm_storeu_ps(inv_l, _mm_div_ps(_mm_set1_ps(1.f), _l));
        sdpa_store_out_tile_fp32(outptr + ii * out_embed_dim, outT + ii * out_embed_dim, out_embed_dim, 4, inv_l);
    }
#endif // __SSE2__

    for (; ii < block_m; ii++)
    {
        float* outptr0 = outptr + ii * out_embed_dim;
        const float* pO = outT + ii * out_embed_dim;
        const float scale0 = 1.f / l_vec[ii];

        for (int k = 0; k < out_embed_dim; k++)
        {
            outptr0[k] = pO[k] * scale0;
        }
    }
}

static inline Mat sdpa_flash_attention_get_mask_head_fp32(const Mat& attn_mask_blob, int attn_mask, int q)
{
    if (!attn_mask)
        return Mat();

    if (attn_mask_blob.dims == 3)
        return attn_mask_blob.c > 1 ? attn_mask_blob.channel(q) : attn_mask_blob.channel(0);

    return attn_mask_blob;
}

static inline void sdpa_flash_attention_run_unpacked_tile_fp32(const Mat& query, const Mat& cur_key, const Mat& cur_value, const Mat& attn_mask_blob, const Mat& past_key, const Mat& past_value, float* outptr, float* score, float* m_vec, float* l_vec, int q, int g, int i, int max_ii, int n_begin, int n_end, int nstep, int past_seqlen, int embed_dim, int out_embed_dim, int attn_mask, float scale, bool k_end)
{
    const Mat query_head = query.channel(q);
    const Mat past_key_head = past_seqlen > 0 ? past_key.channel(g) : Mat();
    const Mat cur_key_head = cur_key.channel(g);
    const Mat past_value_head = past_seqlen > 0 ? past_value.channel(g) : Mat();
    const Mat cur_value_head = cur_value.channel(g);
    const Mat mask_head = sdpa_flash_attention_get_mask_head_fp32(attn_mask_blob, attn_mask, q);

    const float* query_ptr = query_head;
    const float* past_key_ptr = past_seqlen > 0 ? (const float*)past_key_head : 0;
    const float* cur_key_ptr = cur_key_head;
    const float* past_value_ptr = past_seqlen > 0 ? (const float*)past_value_head : 0;
    const float* cur_value_ptr = cur_value_head;
    const float* mask_ptr = mask_head.empty() ? 0 : (const float*)mask_head;
    const int mask_stride = mask_head.empty() ? 0 : mask_head.w;

    sdpa_flash_attention_tile_fp32(query_ptr, past_key_ptr, cur_key_ptr, past_value_ptr, cur_value_ptr, mask_ptr, outptr, score, m_vec, l_vec, i, max_ii, n_begin, n_end, nstep, past_seqlen, embed_dim, out_embed_dim, mask_stride, scale, k_end);
}

static inline void sdpa_flash_attention_run_packed_key_tile_fp32(const Mat& query, const Mat& packed_key, const Mat& cur_value, const Mat& packed_value, float* outptr_or_outT, float* scoreT, float* m_vec, float* l_vec, int q, int g, int i, int max_ii, int n_begin, int n_end, int nstep, int embed_dim, int out_embed_dim, int use_packed_value, float scale, bool k_end)
{
    const Mat query_head = query.channel(q);
    const Mat packed_key_head = packed_key.channel(g);
    const Mat cur_value_head = cur_value.channel(g);
    const Mat packed_value_head = use_packed_value ? packed_value.channel(g) : Mat();

    const float* query_ptr = query_head;
    const float* packed_key_ptr = packed_key_head;
    const float* value_ptr = cur_value_head;
    const float* packed_value_ptr = use_packed_value ? (const float*)packed_value_head : 0;

    sdpa_flash_attention_tile_packed_key_fp32(query_ptr, packed_key_ptr, value_ptr, packed_value_ptr, outptr_or_outT, scoreT, m_vec, l_vec, i, max_ii, n_begin, n_end, nstep, embed_dim, out_embed_dim, scale, k_end);
}

static int sdpa_concat_kv_cache_fp32(const Mat& cur_key, const Mat& cur_value, const Mat& past_key, const Mat& past_value, Mat& key, Mat& value, const Option& opt)
{
    const int embed_dim = cur_key.w;
    const int out_embed_dim = cur_value.w;
    const int cur_seqlen = cur_key.h;
    const int past_seqlen = past_key.h;
    const int dst_seqlen = past_seqlen + cur_seqlen;
    const int num_group = cur_key.c;

    if (past_seqlen == 0)
    {
        key = cur_key;
        value = cur_value;
        return 0;
    }

    key.create(embed_dim, dst_seqlen, num_group, 4u, opt.blob_allocator);
    value.create(out_embed_dim, dst_seqlen, num_group, 4u, opt.blob_allocator);
    if (key.empty() || value.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_group; q++)
    {
        const Mat past_key_head = past_key.channel(q);
        const Mat cur_key_head = cur_key.channel(q);
        Mat key_head = key.channel(q);

        memcpy(key_head.row(0), past_key_head, embed_dim * past_seqlen * sizeof(float));
        memcpy(key_head.row(past_seqlen), cur_key_head, embed_dim * cur_seqlen * sizeof(float));

        const Mat past_value_head = past_value.channel(q);
        const Mat cur_value_head = cur_value.channel(q);
        Mat value_head = value.channel(q);

        memcpy(value_head.row(0), past_value_head, out_embed_dim * past_seqlen * sizeof(float));
        memcpy(value_head.row(past_seqlen), cur_value_head, out_embed_dim * cur_seqlen * sizeof(float));
    }

    return 0;
}

static void sdpa_flash_attention_get_optimal_tile_mn(int M, int N, int K, int out_K, int& TILE_M, int& TILE_N)
{
    TILE_M = 4;
    if (M == 1)
        TILE_M = 1;
    if (out_K <= 128 && M >= 16)
        TILE_M = 8;
    if (out_K <= 128 && M >= 64)
        TILE_M = 16;

    TILE_N = 256;
    if (M != 1 && K >= 80 && M > 256)
        TILE_N = 128;
    if (N > 0 && TILE_N > N)
        TILE_N = N;
}

static int sdpa_flash_attention_fp32(const Mat& query, const Mat& cur_key, const Mat& cur_value, const Mat& attn_mask_blob, const Mat& past_key, const Mat& past_value, std::vector<Mat>& top_blobs, const Option& opt, int attn_mask, float scale, int kv_cache)
{
    const int MAX_BLOCK_M = 16;
    const int BLOCK_N = 256;

    const int embed_dim = query.w;
    const int src_seqlen = query.h;
    const int num_heads = query.c;
    const int cur_seqlen = cur_key.h;
    const int num_group = cur_key.c;
    const int out_embed_dim = cur_value.w;
    const int past_seqlen = kv_cache ? past_key.h : 0;
    const int dst_seqlen = past_seqlen + cur_seqlen;

    const int num_heads_per_group = num_heads / num_group;
    const float _scale = scale == 0.f ? 1.f / sqrtf((float)embed_dim) : scale;

    int TILE_M;
    int TILE_N;
    sdpa_flash_attention_get_optimal_tile_mn(src_seqlen, dst_seqlen, embed_dim, out_embed_dim, TILE_M, TILE_N);

    const int num_i_tiles = (src_seqlen + TILE_M - 1) / TILE_M;
    const int num_tiles = num_heads * num_i_tiles;

    Mat& top_blob = top_blobs[0];
    top_blob.create(out_embed_dim, src_seqlen, num_heads, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int num_kv_blocks = (dst_seqlen + TILE_N - 1) / TILE_N;

    Mat score_workspace(BLOCK_N * MAX_BLOCK_M, opt.num_threads, 4u, opt.workspace_allocator);
    if (score_workspace.empty())
        return -100;

    Mat packed_key;
    Mat packed_value;
    if (past_seqlen == 0 && src_seqlen >= 32 && embed_dim <= 128 && !attn_mask && out_embed_dim <= 128)
    {
        int ret = sdpa_pack_key_tile_fp32(cur_key, packed_key, TILE_N, opt);
        if (ret != 0)
            return ret;

        const int pack_value = num_heads_per_group > 1;
        if (pack_value)
        {
            ret = sdpa_pack_value_tile_fp32(cur_value, packed_value, TILE_N, opt);
            if (ret != 0)
                return ret;
        }
    }

    const int use_packed_key = !packed_key.empty();
    const int use_packed_value = !packed_value.empty();

    const int num_kv_chunks = num_tiles < opt.num_threads && num_kv_blocks >= 2 ? std::min(opt.num_threads, num_kv_blocks) : 1;

    if (num_kv_chunks == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int task_id = 0; task_id < num_tiles; task_id++)
        {
            const int q = task_id / num_i_tiles;
            const int i_tile = task_id % num_i_tiles;
            const int g = q / num_heads_per_group;
            const int i = i_tile * TILE_M;
            const int max_ii = i + TILE_M < src_seqlen ? TILE_M : src_seqlen - i;

            Mat top_blob_head = top_blob.channel(q);
            float* score = score_workspace.row(get_omp_thread_num());
            float* outptr = top_blob_head.row(i);
            float m_vec[MAX_BLOCK_M];
            float l_vec[MAX_BLOCK_M];

            if (use_packed_key)
            {
                float* scoreT = score;
                sdpa_flash_attention_run_packed_key_tile_fp32(query, packed_key, cur_value, packed_value, outptr, scoreT, m_vec, l_vec, q, g, i, max_ii, 0, dst_seqlen, TILE_N, embed_dim, out_embed_dim, use_packed_value, _scale, true);
            }
            else
            {
                sdpa_flash_attention_run_unpacked_tile_fp32(query, cur_key, cur_value, attn_mask_blob, past_key, past_value, outptr, score, m_vec, l_vec, q, g, i, max_ii, 0, dst_seqlen, TILE_N, past_seqlen, embed_dim, out_embed_dim, attn_mask, _scale, true);
            }
        }
    }
    else
    {
        const int partial_stride = MAX_BLOCK_M * 2 + MAX_BLOCK_M * out_embed_dim;
        Mat partials(partial_stride, num_kv_chunks, num_tiles, 4u, opt.workspace_allocator);
        if (partials.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int task_id = 0; task_id < num_tiles * num_kv_chunks; task_id++)
        {
            const int tile_id = task_id / num_kv_chunks;
            const int chunk_id = task_id % num_kv_chunks;
            const int q = tile_id / num_i_tiles;
            const int i_tile = tile_id % num_i_tiles;
            const int g = q / num_heads_per_group;
            const int i = i_tile * TILE_M;
            const int max_ii = i + TILE_M < src_seqlen ? TILE_M : src_seqlen - i;
            const int n_begin = use_packed_key ? chunk_id * num_kv_blocks / num_kv_chunks * TILE_N : chunk_id * dst_seqlen / num_kv_chunks;
            const int n_end = use_packed_key ? std::min((chunk_id + 1) * num_kv_blocks / num_kv_chunks * TILE_N, dst_seqlen) : (chunk_id + 1) * dst_seqlen / num_kv_chunks;

            float* score = score_workspace.row(get_omp_thread_num());
            float* partial = partials.channel(tile_id).row(chunk_id);
            float* m_vec = partial;
            float* l_vec = partial + MAX_BLOCK_M;

            if (use_packed_key)
            {
                float* scoreT = score;
                float* partial_outT = partial + MAX_BLOCK_M * 2;
                sdpa_flash_attention_run_packed_key_tile_fp32(query, packed_key, cur_value, packed_value, partial_outT, scoreT, m_vec, l_vec, q, g, i, max_ii, n_begin, n_end, TILE_N, embed_dim, out_embed_dim, use_packed_value, _scale, false);
            }
            else
            {
                float* partial_outptr = partial + MAX_BLOCK_M * 2;
                sdpa_flash_attention_run_unpacked_tile_fp32(query, cur_key, cur_value, attn_mask_blob, past_key, past_value, partial_outptr, score, m_vec, l_vec, q, g, i, max_ii, n_begin, n_end, TILE_N, past_seqlen, embed_dim, out_embed_dim, attn_mask, _scale, false);
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int tile_id = 0; tile_id < num_tiles; tile_id++)
        {
            const int q = tile_id / num_i_tiles;
            const int i_tile = tile_id % num_i_tiles;
            const int i = i_tile * TILE_M;
            const int max_ii = i + TILE_M < src_seqlen ? TILE_M : src_seqlen - i;

            Mat top_blob_head = top_blob.channel(q);
            if (use_packed_key && !attn_mask && out_embed_dim <= 128)
            {
                sdpa_flash_attention_reduce_outT_fp32(top_blob_head.row(i), partials.channel(tile_id), max_ii, out_embed_dim, num_kv_chunks);
            }
            else
            {
                sdpa_flash_attention_reduce_fp32(top_blob_head.row(i), partials.channel(tile_id), max_ii, out_embed_dim, num_kv_chunks);
            }
        }
    }

    if (kv_cache)
    {
        Mat key;
        Mat value;
        int ret = sdpa_concat_kv_cache_fp32(cur_key, cur_value, past_key, past_value, key, value, opt);
        if (ret != 0)
            return ret;

        top_blobs[1] = key;
        top_blobs[2] = value;
    }

    return 0;
}
