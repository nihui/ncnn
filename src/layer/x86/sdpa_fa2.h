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

static void sdpa_qk_tile_packed_fp32(const float* query, const float* packed_key, float* score, int i, int max_ii, int n_start, int max_jj, int embed_dim, int dst_seqlen, float scale)
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

        int j = 0;
        __m512 _scale = _mm512_set1_ps(scale);
        for (; j + 15 < max_jj; j += 16)
        {
            const float* kptr = packed_key + n_start + j;

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

            for (int k = 0; k < embed_dim; k++)
            {
                __m512 _k = _mm512_loadu_ps(kptr);
                _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(qptr0[k]), _k, _sum0);
                _sum1 = _mm512_fmadd_ps(_mm512_set1_ps(qptr1[k]), _k, _sum1);
                _sum2 = _mm512_fmadd_ps(_mm512_set1_ps(qptr2[k]), _k, _sum2);
                _sum3 = _mm512_fmadd_ps(_mm512_set1_ps(qptr3[k]), _k, _sum3);
                _sum4 = _mm512_fmadd_ps(_mm512_set1_ps(qptr4[k]), _k, _sum4);
                _sum5 = _mm512_fmadd_ps(_mm512_set1_ps(qptr5[k]), _k, _sum5);
                _sum6 = _mm512_fmadd_ps(_mm512_set1_ps(qptr6[k]), _k, _sum6);
                _sum7 = _mm512_fmadd_ps(_mm512_set1_ps(qptr7[k]), _k, _sum7);
                _sum8 = _mm512_fmadd_ps(_mm512_set1_ps(qptr8[k]), _k, _sum8);
                _sum9 = _mm512_fmadd_ps(_mm512_set1_ps(qptr9[k]), _k, _sum9);
                _suma = _mm512_fmadd_ps(_mm512_set1_ps(qptra[k]), _k, _suma);
                _sumb = _mm512_fmadd_ps(_mm512_set1_ps(qptrb[k]), _k, _sumb);
                _sumc = _mm512_fmadd_ps(_mm512_set1_ps(qptrc[k]), _k, _sumc);
                _sumd = _mm512_fmadd_ps(_mm512_set1_ps(qptrd[k]), _k, _sumd);
                _sume = _mm512_fmadd_ps(_mm512_set1_ps(qptre[k]), _k, _sume);
                _sumf = _mm512_fmadd_ps(_mm512_set1_ps(qptrf[k]), _k, _sumf);
                kptr += dst_seqlen;
            }

            _mm512_storeu_ps(sptr0 + j, _mm512_mul_ps(_sum0, _scale));
            _mm512_storeu_ps(sptr1 + j, _mm512_mul_ps(_sum1, _scale));
            _mm512_storeu_ps(sptr2 + j, _mm512_mul_ps(_sum2, _scale));
            _mm512_storeu_ps(sptr3 + j, _mm512_mul_ps(_sum3, _scale));
            _mm512_storeu_ps(sptr4 + j, _mm512_mul_ps(_sum4, _scale));
            _mm512_storeu_ps(sptr5 + j, _mm512_mul_ps(_sum5, _scale));
            _mm512_storeu_ps(sptr6 + j, _mm512_mul_ps(_sum6, _scale));
            _mm512_storeu_ps(sptr7 + j, _mm512_mul_ps(_sum7, _scale));
            _mm512_storeu_ps(sptr8 + j, _mm512_mul_ps(_sum8, _scale));
            _mm512_storeu_ps(sptr9 + j, _mm512_mul_ps(_sum9, _scale));
            _mm512_storeu_ps(sptra + j, _mm512_mul_ps(_suma, _scale));
            _mm512_storeu_ps(sptrb + j, _mm512_mul_ps(_sumb, _scale));
            _mm512_storeu_ps(sptrc + j, _mm512_mul_ps(_sumc, _scale));
            _mm512_storeu_ps(sptrd + j, _mm512_mul_ps(_sumd, _scale));
            _mm512_storeu_ps(sptre + j, _mm512_mul_ps(_sume, _scale));
            _mm512_storeu_ps(sptrf + j, _mm512_mul_ps(_sumf, _scale));
        }
        for (; j < max_jj; j++)
        {
            const float* kptr = packed_key + n_start + j;

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

            for (int k = 0; k < embed_dim; k++)
            {
                const float v = *kptr;
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
                kptr += dst_seqlen;
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

        int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        {
            __m512 _scale = _mm512_set1_ps(scale);
            for (; j + 15 < max_jj; j += 16)
            {
                const float* kptr = packed_key + n_start + j;

                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();
                __m512 _sum2 = _mm512_setzero_ps();
                __m512 _sum3 = _mm512_setzero_ps();
                __m512 _sum4 = _mm512_setzero_ps();
                __m512 _sum5 = _mm512_setzero_ps();
                __m512 _sum6 = _mm512_setzero_ps();
                __m512 _sum7 = _mm512_setzero_ps();

                for (int k = 0; k < embed_dim; k++)
                {
                    __m512 _k = _mm512_loadu_ps(kptr);
                    _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(qptr0[k]), _k, _sum0);
                    _sum1 = _mm512_fmadd_ps(_mm512_set1_ps(qptr1[k]), _k, _sum1);
                    _sum2 = _mm512_fmadd_ps(_mm512_set1_ps(qptr2[k]), _k, _sum2);
                    _sum3 = _mm512_fmadd_ps(_mm512_set1_ps(qptr3[k]), _k, _sum3);
                    _sum4 = _mm512_fmadd_ps(_mm512_set1_ps(qptr4[k]), _k, _sum4);
                    _sum5 = _mm512_fmadd_ps(_mm512_set1_ps(qptr5[k]), _k, _sum5);
                    _sum6 = _mm512_fmadd_ps(_mm512_set1_ps(qptr6[k]), _k, _sum6);
                    _sum7 = _mm512_fmadd_ps(_mm512_set1_ps(qptr7[k]), _k, _sum7);
                    kptr += dst_seqlen;
                }

                _mm512_storeu_ps(sptr0 + j, _mm512_mul_ps(_sum0, _scale));
                _mm512_storeu_ps(sptr1 + j, _mm512_mul_ps(_sum1, _scale));
                _mm512_storeu_ps(sptr2 + j, _mm512_mul_ps(_sum2, _scale));
                _mm512_storeu_ps(sptr3 + j, _mm512_mul_ps(_sum3, _scale));
                _mm512_storeu_ps(sptr4 + j, _mm512_mul_ps(_sum4, _scale));
                _mm512_storeu_ps(sptr5 + j, _mm512_mul_ps(_sum5, _scale));
                _mm512_storeu_ps(sptr6 + j, _mm512_mul_ps(_sum6, _scale));
                _mm512_storeu_ps(sptr7 + j, _mm512_mul_ps(_sum7, _scale));
            }
        }
#endif // __AVX512F__
        {
            __m256 _scale = _mm256_set1_ps(scale);
            for (; j + 7 < max_jj; j += 8)
            {
                const float* kptr = packed_key + n_start + j;

                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();
                __m256 _sum2 = _mm256_setzero_ps();
                __m256 _sum3 = _mm256_setzero_ps();
                __m256 _sum4 = _mm256_setzero_ps();
                __m256 _sum5 = _mm256_setzero_ps();
                __m256 _sum6 = _mm256_setzero_ps();
                __m256 _sum7 = _mm256_setzero_ps();

                int k = 0;
                for (; k + 3 < embed_dim; k += 4)
                {
                    const float* kptr0 = kptr;
                    const float* kptr1 = kptr0 + dst_seqlen;
                    const float* kptr2 = kptr1 + dst_seqlen;
                    const float* kptr3 = kptr2 + dst_seqlen;

                    __m256 _k0 = _mm256_loadu_ps(kptr0);
                    __m256 _k1 = _mm256_loadu_ps(kptr1);
                    __m256 _k2 = _mm256_loadu_ps(kptr2);
                    __m256 _k3 = _mm256_loadu_ps(kptr3);

                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr0[k]), _k0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr1[k]), _k0, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr2[k]), _k0, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr3[k]), _k0, _sum3);
                    _sum4 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr4[k]), _k0, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr5[k]), _k0, _sum5);
                    _sum6 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr6[k]), _k0, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr7[k]), _k0, _sum7);

                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr0[k + 1]), _k1, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr1[k + 1]), _k1, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr2[k + 1]), _k1, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr3[k + 1]), _k1, _sum3);
                    _sum4 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr4[k + 1]), _k1, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr5[k + 1]), _k1, _sum5);
                    _sum6 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr6[k + 1]), _k1, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr7[k + 1]), _k1, _sum7);

                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr0[k + 2]), _k2, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr1[k + 2]), _k2, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr2[k + 2]), _k2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr3[k + 2]), _k2, _sum3);
                    _sum4 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr4[k + 2]), _k2, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr5[k + 2]), _k2, _sum5);
                    _sum6 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr6[k + 2]), _k2, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr7[k + 2]), _k2, _sum7);

                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr0[k + 3]), _k3, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr1[k + 3]), _k3, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr2[k + 3]), _k3, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr3[k + 3]), _k3, _sum3);
                    _sum4 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr4[k + 3]), _k3, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr5[k + 3]), _k3, _sum5);
                    _sum6 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr6[k + 3]), _k3, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr7[k + 3]), _k3, _sum7);

                    kptr = kptr3 + dst_seqlen;
                }
                for (; k < embed_dim; k++)
                {
                    __m256 _k = _mm256_loadu_ps(kptr);
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr0[k]), _k, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr1[k]), _k, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr2[k]), _k, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr3[k]), _k, _sum3);
                    _sum4 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr4[k]), _k, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr5[k]), _k, _sum5);
                    _sum6 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr6[k]), _k, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr7[k]), _k, _sum7);
                    kptr += dst_seqlen;
                }

                _mm256_storeu_ps(sptr0 + j, _mm256_mul_ps(_sum0, _scale));
                _mm256_storeu_ps(sptr1 + j, _mm256_mul_ps(_sum1, _scale));
                _mm256_storeu_ps(sptr2 + j, _mm256_mul_ps(_sum2, _scale));
                _mm256_storeu_ps(sptr3 + j, _mm256_mul_ps(_sum3, _scale));
                _mm256_storeu_ps(sptr4 + j, _mm256_mul_ps(_sum4, _scale));
                _mm256_storeu_ps(sptr5 + j, _mm256_mul_ps(_sum5, _scale));
                _mm256_storeu_ps(sptr6 + j, _mm256_mul_ps(_sum6, _scale));
                _mm256_storeu_ps(sptr7 + j, _mm256_mul_ps(_sum7, _scale));
            }
        }
#endif // __AVX__
        {
            __m128 _scale = _mm_set1_ps(scale);
            for (; j + 3 < max_jj; j += 4)
            {
                const float* kptr = packed_key + n_start + j;

                __m128 _sum0 = _mm_setzero_ps();
                __m128 _sum1 = _mm_setzero_ps();
                __m128 _sum2 = _mm_setzero_ps();
                __m128 _sum3 = _mm_setzero_ps();
                __m128 _sum4 = _mm_setzero_ps();
                __m128 _sum5 = _mm_setzero_ps();
                __m128 _sum6 = _mm_setzero_ps();
                __m128 _sum7 = _mm_setzero_ps();

                for (int k = 0; k < embed_dim; k++)
                {
                    __m128 _k = _mm_loadu_ps(kptr);
                    _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(qptr0[k]), _k, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_mm_set1_ps(qptr1[k]), _k, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_mm_set1_ps(qptr2[k]), _k, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_mm_set1_ps(qptr3[k]), _k, _sum3);
                    _sum4 = _mm_comp_fmadd_ps(_mm_set1_ps(qptr4[k]), _k, _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_mm_set1_ps(qptr5[k]), _k, _sum5);
                    _sum6 = _mm_comp_fmadd_ps(_mm_set1_ps(qptr6[k]), _k, _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_mm_set1_ps(qptr7[k]), _k, _sum7);
                    kptr += dst_seqlen;
                }

                _mm_storeu_ps(sptr0 + j, _mm_mul_ps(_sum0, _scale));
                _mm_storeu_ps(sptr1 + j, _mm_mul_ps(_sum1, _scale));
                _mm_storeu_ps(sptr2 + j, _mm_mul_ps(_sum2, _scale));
                _mm_storeu_ps(sptr3 + j, _mm_mul_ps(_sum3, _scale));
                _mm_storeu_ps(sptr4 + j, _mm_mul_ps(_sum4, _scale));
                _mm_storeu_ps(sptr5 + j, _mm_mul_ps(_sum5, _scale));
                _mm_storeu_ps(sptr6 + j, _mm_mul_ps(_sum6, _scale));
                _mm_storeu_ps(sptr7 + j, _mm_mul_ps(_sum7, _scale));
            }
        }
#endif // __SSE2__
        for (; j < max_jj; j++)
        {
            const float* kptr = packed_key + n_start + j;

            float sum0 = 0.f;
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            float sum4 = 0.f;
            float sum5 = 0.f;
            float sum6 = 0.f;
            float sum7 = 0.f;

            for (int k = 0; k < embed_dim; k++)
            {
                const float v = *kptr;
                sum0 += qptr0[k] * v;
                sum1 += qptr1[k] * v;
                sum2 += qptr2[k] * v;
                sum3 += qptr3[k] * v;
                sum4 += qptr4[k] * v;
                sum5 += qptr5[k] * v;
                sum6 += qptr6[k] * v;
                sum7 += qptr7[k] * v;
                kptr += dst_seqlen;
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

        int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        {
            __m512 _scale = _mm512_set1_ps(scale);
            for (; j + 15 < max_jj; j += 16)
            {
                const float* kptr = packed_key + n_start + j;

                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();
                __m512 _sum2 = _mm512_setzero_ps();
                __m512 _sum3 = _mm512_setzero_ps();

                for (int k = 0; k < embed_dim; k++)
                {
                    __m512 _k = _mm512_loadu_ps(kptr);
                    _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(qptr0[k]), _k, _sum0);
                    _sum1 = _mm512_fmadd_ps(_mm512_set1_ps(qptr1[k]), _k, _sum1);
                    _sum2 = _mm512_fmadd_ps(_mm512_set1_ps(qptr2[k]), _k, _sum2);
                    _sum3 = _mm512_fmadd_ps(_mm512_set1_ps(qptr3[k]), _k, _sum3);
                    kptr += dst_seqlen;
                }

                _mm512_storeu_ps(sptr0 + j, _mm512_mul_ps(_sum0, _scale));
                _mm512_storeu_ps(sptr1 + j, _mm512_mul_ps(_sum1, _scale));
                _mm512_storeu_ps(sptr2 + j, _mm512_mul_ps(_sum2, _scale));
                _mm512_storeu_ps(sptr3 + j, _mm512_mul_ps(_sum3, _scale));
            }
        }
#endif // __AVX512F__
        {
            __m256 _scale = _mm256_set1_ps(scale);
            for (; j + 7 < max_jj; j += 8)
            {
                const float* kptr = packed_key + n_start + j;

                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();
                __m256 _sum2 = _mm256_setzero_ps();
                __m256 _sum3 = _mm256_setzero_ps();

                int k = 0;
                for (; k + 3 < embed_dim; k += 4)
                {
                    const float* kptr0 = kptr;
                    const float* kptr1 = kptr0 + dst_seqlen;
                    const float* kptr2 = kptr1 + dst_seqlen;
                    const float* kptr3 = kptr2 + dst_seqlen;

                    __m256 _k0 = _mm256_loadu_ps(kptr0);
                    __m256 _k1 = _mm256_loadu_ps(kptr1);
                    __m256 _k2 = _mm256_loadu_ps(kptr2);
                    __m256 _k3 = _mm256_loadu_ps(kptr3);

                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr0[k]), _k0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr1[k]), _k0, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr2[k]), _k0, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr3[k]), _k0, _sum3);

                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr0[k + 1]), _k1, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr1[k + 1]), _k1, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr2[k + 1]), _k1, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr3[k + 1]), _k1, _sum3);

                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr0[k + 2]), _k2, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr1[k + 2]), _k2, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr2[k + 2]), _k2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr3[k + 2]), _k2, _sum3);

                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr0[k + 3]), _k3, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr1[k + 3]), _k3, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr2[k + 3]), _k3, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr3[k + 3]), _k3, _sum3);

                    kptr = kptr3 + dst_seqlen;
                }
                for (; k < embed_dim; k++)
                {
                    __m256 _k = _mm256_loadu_ps(kptr);
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr0[k]), _k, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr1[k]), _k, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr2[k]), _k, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr3[k]), _k, _sum3);
                    kptr += dst_seqlen;
                }

                _mm256_storeu_ps(sptr0 + j, _mm256_mul_ps(_sum0, _scale));
                _mm256_storeu_ps(sptr1 + j, _mm256_mul_ps(_sum1, _scale));
                _mm256_storeu_ps(sptr2 + j, _mm256_mul_ps(_sum2, _scale));
                _mm256_storeu_ps(sptr3 + j, _mm256_mul_ps(_sum3, _scale));
            }
        }
#endif // __AVX__
        {
            __m128 _scale = _mm_set1_ps(scale);
            for (; j + 3 < max_jj; j += 4)
            {
                const float* kptr = packed_key + n_start + j;

                __m128 _sum0 = _mm_setzero_ps();
                __m128 _sum1 = _mm_setzero_ps();
                __m128 _sum2 = _mm_setzero_ps();
                __m128 _sum3 = _mm_setzero_ps();

                for (int k = 0; k < embed_dim; k++)
                {
                    __m128 _k = _mm_loadu_ps(kptr);
                    _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(qptr0[k]), _k, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_mm_set1_ps(qptr1[k]), _k, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_mm_set1_ps(qptr2[k]), _k, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_mm_set1_ps(qptr3[k]), _k, _sum3);
                    kptr += dst_seqlen;
                }

                _mm_storeu_ps(sptr0 + j, _mm_mul_ps(_sum0, _scale));
                _mm_storeu_ps(sptr1 + j, _mm_mul_ps(_sum1, _scale));
                _mm_storeu_ps(sptr2 + j, _mm_mul_ps(_sum2, _scale));
                _mm_storeu_ps(sptr3 + j, _mm_mul_ps(_sum3, _scale));
            }
        }
#endif // __SSE2__
        for (; j < max_jj; j++)
        {
            const float* kptr = packed_key + n_start + j;

            float sum0 = 0.f;
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;

            for (int k = 0; k < embed_dim; k++)
            {
                const float v = *kptr;
                sum0 += qptr0[k] * v;
                sum1 += qptr1[k] * v;
                sum2 += qptr2[k] * v;
                sum3 += qptr3[k] * v;
                kptr += dst_seqlen;
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

        int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        {
            __m512 _scale = _mm512_set1_ps(scale);
            for (; j + 15 < max_jj; j += 16)
            {
                const float* kptr = packed_key + n_start + j;

                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();

                for (int k = 0; k < embed_dim; k++)
                {
                    __m512 _k = _mm512_loadu_ps(kptr);
                    _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(qptr0[k]), _k, _sum0);
                    _sum1 = _mm512_fmadd_ps(_mm512_set1_ps(qptr1[k]), _k, _sum1);
                    kptr += dst_seqlen;
                }

                _mm512_storeu_ps(sptr0 + j, _mm512_mul_ps(_sum0, _scale));
                _mm512_storeu_ps(sptr1 + j, _mm512_mul_ps(_sum1, _scale));
            }
        }
#endif // __AVX512F__
        {
            __m256 _scale = _mm256_set1_ps(scale);
            for (; j + 7 < max_jj; j += 8)
            {
                const float* kptr = packed_key + n_start + j;

                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();

                for (int k = 0; k < embed_dim; k++)
                {
                    __m256 _k = _mm256_loadu_ps(kptr);
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr0[k]), _k, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr1[k]), _k, _sum1);
                    kptr += dst_seqlen;
                }

                _mm256_storeu_ps(sptr0 + j, _mm256_mul_ps(_sum0, _scale));
                _mm256_storeu_ps(sptr1 + j, _mm256_mul_ps(_sum1, _scale));
            }
        }
#endif // __AVX__
        {
            __m128 _scale = _mm_set1_ps(scale);
            for (; j + 3 < max_jj; j += 4)
            {
                const float* kptr = packed_key + n_start + j;

                __m128 _sum0 = _mm_setzero_ps();
                __m128 _sum1 = _mm_setzero_ps();

                for (int k = 0; k < embed_dim; k++)
                {
                    __m128 _k = _mm_loadu_ps(kptr);
                    _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(qptr0[k]), _k, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_mm_set1_ps(qptr1[k]), _k, _sum1);
                    kptr += dst_seqlen;
                }

                _mm_storeu_ps(sptr0 + j, _mm_mul_ps(_sum0, _scale));
                _mm_storeu_ps(sptr1 + j, _mm_mul_ps(_sum1, _scale));
            }
        }
#endif // __SSE2__
        for (; j < max_jj; j++)
        {
            const float* kptr = packed_key + n_start + j;

            float sum0 = 0.f;
            float sum1 = 0.f;

            for (int k = 0; k < embed_dim; k++)
            {
                const float v = *kptr;
                sum0 += qptr0[k] * v;
                sum1 += qptr1[k] * v;
                kptr += dst_seqlen;
            }

            sptr0[j] = sum0 * scale;
            sptr1[j] = sum1 * scale;
        }
    }

    for (; ii < max_ii; ii++)
    {
        const float* qptr0 = query + (i + ii) * embed_dim;
        float* sptr0 = score + ii * BLOCK_N;

        int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        {
            __m512 _scale = _mm512_set1_ps(scale);
            for (; j + 15 < max_jj; j += 16)
            {
                const float* kptr = packed_key + n_start + j;
                __m512 _sum0 = _mm512_setzero_ps();

                for (int k = 0; k < embed_dim; k++)
                {
                    _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(qptr0[k]), _mm512_loadu_ps(kptr), _sum0);
                    kptr += dst_seqlen;
                }

                _mm512_storeu_ps(sptr0 + j, _mm512_mul_ps(_sum0, _scale));
            }
        }
#endif // __AVX512F__
        {
            __m256 _scale = _mm256_set1_ps(scale);
            for (; j + 7 < max_jj; j += 8)
            {
                const float* kptr = packed_key + n_start + j;
                __m256 _sum0 = _mm256_setzero_ps();

                for (int k = 0; k < embed_dim; k++)
                {
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(qptr0[k]), _mm256_loadu_ps(kptr), _sum0);
                    kptr += dst_seqlen;
                }

                _mm256_storeu_ps(sptr0 + j, _mm256_mul_ps(_sum0, _scale));
            }
        }
#endif // __AVX__
        {
            __m128 _scale = _mm_set1_ps(scale);
            for (; j + 3 < max_jj; j += 4)
            {
                const float* kptr = packed_key + n_start + j;
                __m128 _sum0 = _mm_setzero_ps();

                for (int k = 0; k < embed_dim; k++)
                {
                    _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(qptr0[k]), _mm_loadu_ps(kptr), _sum0);
                    kptr += dst_seqlen;
                }

                _mm_storeu_ps(sptr0 + j, _mm_mul_ps(_sum0, _scale));
            }
        }
#endif // __SSE2__
        for (; j < max_jj; j++)
        {
            const float* kptr = packed_key + n_start + j;
            float sum0 = 0.f;

            for (int k = 0; k < embed_dim; k++)
            {
                sum0 += qptr0[k] * *kptr;
                kptr += dst_seqlen;
            }

            sptr0[j] = sum0 * scale;
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

static void sdpa_flash_attention_tile_packed_fp32(const float* query, const float* packed_key, const float* value, const float* mask, float* outptr, float* score, float* m_vec, float* l_vec, int i, int max_ii, int n_begin, int n_end, int nstep, int dst_seqlen, int embed_dim, int out_embed_dim, int mask_stride, float scale, bool k_end)
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

        sdpa_qk_tile_packed_fp32(query, packed_key, score, i, max_ii, n_start, max_jj, embed_dim, dst_seqlen, scale);

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

        if (n_start == n_begin)
            sdpa_pv_tile_fp32(outptr, value, score, max_ii, n_start, max_jj, out_embed_dim);
        else
            sdpa_pv_tile_accumulate_fp32(outptr, value, score, max_ii, n_start, max_jj, out_embed_dim);
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

static int sdpa_pack_key_transpose_fp32(const Mat& cur_key, Mat& packed_key, const Option& opt)
{
    const int embed_dim = cur_key.w;
    const int cur_seqlen = cur_key.h;
    const int num_group = cur_key.c;

    packed_key.create(cur_seqlen, embed_dim, num_group, 4u, opt.workspace_allocator);
    if (packed_key.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_group; q++)
    {
        const Mat cur_key_head = cur_key.channel(q);
        Mat packed_key_head = packed_key.channel(q);

        int k = 0;
#if __AVX512F__
        for (; k + 15 < embed_dim; k += 16)
        {
            float* outptr0 = packed_key_head.row(k);
            float* outptr1 = packed_key_head.row(k + 1);
            float* outptr2 = packed_key_head.row(k + 2);
            float* outptr3 = packed_key_head.row(k + 3);
            float* outptr4 = packed_key_head.row(k + 4);
            float* outptr5 = packed_key_head.row(k + 5);
            float* outptr6 = packed_key_head.row(k + 6);
            float* outptr7 = packed_key_head.row(k + 7);
            float* outptr8 = packed_key_head.row(k + 8);
            float* outptr9 = packed_key_head.row(k + 9);
            float* outptra = packed_key_head.row(k + 10);
            float* outptrb = packed_key_head.row(k + 11);
            float* outptrc = packed_key_head.row(k + 12);
            float* outptrd = packed_key_head.row(k + 13);
            float* outptre = packed_key_head.row(k + 14);
            float* outptrf = packed_key_head.row(k + 15);

            int i = 0;
            for (; i + 15 < cur_seqlen; i += 16)
            {
                const float* kptr0 = cur_key_head.row(i) + k;
                const float* kptr1 = cur_key_head.row(i + 1) + k;
                const float* kptr2 = cur_key_head.row(i + 2) + k;
                const float* kptr3 = cur_key_head.row(i + 3) + k;
                const float* kptr4 = cur_key_head.row(i + 4) + k;
                const float* kptr5 = cur_key_head.row(i + 5) + k;
                const float* kptr6 = cur_key_head.row(i + 6) + k;
                const float* kptr7 = cur_key_head.row(i + 7) + k;
                const float* kptr8 = cur_key_head.row(i + 8) + k;
                const float* kptr9 = cur_key_head.row(i + 9) + k;
                const float* kptra = cur_key_head.row(i + 10) + k;
                const float* kptrb = cur_key_head.row(i + 11) + k;
                const float* kptrc = cur_key_head.row(i + 12) + k;
                const float* kptrd = cur_key_head.row(i + 13) + k;
                const float* kptre = cur_key_head.row(i + 14) + k;
                const float* kptrf = cur_key_head.row(i + 15) + k;

                __m512 _r0 = _mm512_loadu_ps(kptr0);
                __m512 _r1 = _mm512_loadu_ps(kptr1);
                __m512 _r2 = _mm512_loadu_ps(kptr2);
                __m512 _r3 = _mm512_loadu_ps(kptr3);
                __m512 _r4 = _mm512_loadu_ps(kptr4);
                __m512 _r5 = _mm512_loadu_ps(kptr5);
                __m512 _r6 = _mm512_loadu_ps(kptr6);
                __m512 _r7 = _mm512_loadu_ps(kptr7);
                __m512 _r8 = _mm512_loadu_ps(kptr8);
                __m512 _r9 = _mm512_loadu_ps(kptr9);
                __m512 _ra = _mm512_loadu_ps(kptra);
                __m512 _rb = _mm512_loadu_ps(kptrb);
                __m512 _rc = _mm512_loadu_ps(kptrc);
                __m512 _rd = _mm512_loadu_ps(kptrd);
                __m512 _re = _mm512_loadu_ps(kptre);
                __m512 _rf = _mm512_loadu_ps(kptrf);

                transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);

                _mm512_storeu_ps(outptr0 + i, _r0);
                _mm512_storeu_ps(outptr1 + i, _r1);
                _mm512_storeu_ps(outptr2 + i, _r2);
                _mm512_storeu_ps(outptr3 + i, _r3);
                _mm512_storeu_ps(outptr4 + i, _r4);
                _mm512_storeu_ps(outptr5 + i, _r5);
                _mm512_storeu_ps(outptr6 + i, _r6);
                _mm512_storeu_ps(outptr7 + i, _r7);
                _mm512_storeu_ps(outptr8 + i, _r8);
                _mm512_storeu_ps(outptr9 + i, _r9);
                _mm512_storeu_ps(outptra + i, _ra);
                _mm512_storeu_ps(outptrb + i, _rb);
                _mm512_storeu_ps(outptrc + i, _rc);
                _mm512_storeu_ps(outptrd + i, _rd);
                _mm512_storeu_ps(outptre + i, _re);
                _mm512_storeu_ps(outptrf + i, _rf);
            }

            for (; i < cur_seqlen; i++)
            {
                const float* kptr = cur_key_head.row(i) + k;

                outptr0[i] = kptr[0];
                outptr1[i] = kptr[1];
                outptr2[i] = kptr[2];
                outptr3[i] = kptr[3];
                outptr4[i] = kptr[4];
                outptr5[i] = kptr[5];
                outptr6[i] = kptr[6];
                outptr7[i] = kptr[7];
                outptr8[i] = kptr[8];
                outptr9[i] = kptr[9];
                outptra[i] = kptr[10];
                outptrb[i] = kptr[11];
                outptrc[i] = kptr[12];
                outptrd[i] = kptr[13];
                outptre[i] = kptr[14];
                outptrf[i] = kptr[15];
            }
        }
#endif // __AVX512F__
#if __AVX__
        for (; k + 7 < embed_dim; k += 8)
        {
            float* outptr0 = packed_key_head.row(k);
            float* outptr1 = packed_key_head.row(k + 1);
            float* outptr2 = packed_key_head.row(k + 2);
            float* outptr3 = packed_key_head.row(k + 3);
            float* outptr4 = packed_key_head.row(k + 4);
            float* outptr5 = packed_key_head.row(k + 5);
            float* outptr6 = packed_key_head.row(k + 6);
            float* outptr7 = packed_key_head.row(k + 7);

            int i = 0;
            for (; i + 7 < cur_seqlen; i += 8)
            {
                const float* kptr0 = cur_key_head.row(i) + k;
                const float* kptr1 = cur_key_head.row(i + 1) + k;
                const float* kptr2 = cur_key_head.row(i + 2) + k;
                const float* kptr3 = cur_key_head.row(i + 3) + k;
                const float* kptr4 = cur_key_head.row(i + 4) + k;
                const float* kptr5 = cur_key_head.row(i + 5) + k;
                const float* kptr6 = cur_key_head.row(i + 6) + k;
                const float* kptr7 = cur_key_head.row(i + 7) + k;

                __m256 _r0 = _mm256_loadu_ps(kptr0);
                __m256 _r1 = _mm256_loadu_ps(kptr1);
                __m256 _r2 = _mm256_loadu_ps(kptr2);
                __m256 _r3 = _mm256_loadu_ps(kptr3);
                __m256 _r4 = _mm256_loadu_ps(kptr4);
                __m256 _r5 = _mm256_loadu_ps(kptr5);
                __m256 _r6 = _mm256_loadu_ps(kptr6);
                __m256 _r7 = _mm256_loadu_ps(kptr7);

                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                _mm256_storeu_ps(outptr0 + i, _r0);
                _mm256_storeu_ps(outptr1 + i, _r1);
                _mm256_storeu_ps(outptr2 + i, _r2);
                _mm256_storeu_ps(outptr3 + i, _r3);
                _mm256_storeu_ps(outptr4 + i, _r4);
                _mm256_storeu_ps(outptr5 + i, _r5);
                _mm256_storeu_ps(outptr6 + i, _r6);
                _mm256_storeu_ps(outptr7 + i, _r7);
            }

            for (; i < cur_seqlen; i++)
            {
                const float* kptr = cur_key_head.row(i) + k;

                outptr0[i] = kptr[0];
                outptr1[i] = kptr[1];
                outptr2[i] = kptr[2];
                outptr3[i] = kptr[3];
                outptr4[i] = kptr[4];
                outptr5[i] = kptr[5];
                outptr6[i] = kptr[6];
                outptr7[i] = kptr[7];
            }
        }
#endif // __AVX__
        for (; k < embed_dim; k++)
        {
            float* outptr = packed_key_head.row(k);

            for (int i = 0; i < cur_seqlen; i++)
            {
                const float* kptr = cur_key_head.row(i);
                outptr[i] = kptr[k];
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
    if (past_seqlen == 0 && src_seqlen >= 32 && embed_dim <= 128)
    {
        int ret = sdpa_pack_key_transpose_fp32(cur_key, packed_key, opt);
        if (ret != 0)
            return ret;
    }

    const int use_packed_key = !packed_key.empty();

    const int num_kv_chunks = num_tiles < opt.num_threads && num_kv_blocks >= 2 ? std::min(opt.num_threads, num_kv_blocks) : 1;

    if (num_kv_chunks == 1)
    {
        if (!use_packed_key)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int task_id = 0; task_id < num_tiles; task_id++)
            {
                const int q = task_id / num_i_tiles;
                const int i_tile = task_id % num_i_tiles;
                const int g = q / num_heads_per_group;
                const int i = i_tile * TILE_M;
                const int max_ii = i + TILE_M < src_seqlen ? TILE_M : src_seqlen - i;

                const Mat query_head = query.channel(q);
                const Mat past_key_head = past_seqlen > 0 ? past_key.channel(g) : Mat();
                const Mat cur_key_head = cur_key.channel(g);
                const Mat past_value_head = past_seqlen > 0 ? past_value.channel(g) : Mat();
                const Mat cur_value_head = cur_value.channel(g);

                Mat mask_head;
                if (attn_mask)
                {
                    mask_head = attn_mask_blob;
                    if (attn_mask_blob.dims == 3)
                        mask_head = attn_mask_blob.c > 1 ? attn_mask_blob.channel(q) : attn_mask_blob.channel(0);
                }

                const float* query_ptr = query_head;
                const float* past_key_ptr = past_seqlen > 0 ? (const float*)past_key_head : 0;
                const float* cur_key_ptr = cur_key_head;
                const float* past_value_ptr = past_seqlen > 0 ? (const float*)past_value_head : 0;
                const float* cur_value_ptr = cur_value_head;
                const float* mask_ptr = mask_head.empty() ? 0 : (const float*)mask_head;
                const int mask_stride = mask_head.empty() ? 0 : mask_head.w;

                Mat top_blob_head = top_blob.channel(q);
                float* score = score_workspace.row(get_omp_thread_num());
                float* outptr0 = top_blob_head.row(i);
                float m_vec[MAX_BLOCK_M];
                float l_vec[MAX_BLOCK_M];

                sdpa_flash_attention_tile_fp32(query_ptr, past_key_ptr, cur_key_ptr, past_value_ptr, cur_value_ptr, mask_ptr, outptr0, score, m_vec, l_vec, i, max_ii, 0, dst_seqlen, TILE_N, past_seqlen, embed_dim, out_embed_dim, mask_stride, _scale, true);
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int task_id = 0; task_id < num_tiles; task_id++)
            {
                const int q = task_id / num_i_tiles;
                const int i_tile = task_id % num_i_tiles;
                const int g = q / num_heads_per_group;
                const int i = i_tile * TILE_M;
                const int max_ii = i + TILE_M < src_seqlen ? TILE_M : src_seqlen - i;

                const Mat query_head = query.channel(q);
                const Mat packed_key_head = packed_key.channel(g);
                const Mat cur_value_head = cur_value.channel(g);

                Mat mask_head;
                if (attn_mask)
                {
                    mask_head = attn_mask_blob;
                    if (attn_mask_blob.dims == 3)
                        mask_head = attn_mask_blob.c > 1 ? attn_mask_blob.channel(q) : attn_mask_blob.channel(0);
                }

                const float* query_ptr = query_head;
                const float* packed_key_ptr = packed_key_head;
                const float* cur_value_ptr = cur_value_head;
                const float* mask_ptr = mask_head.empty() ? 0 : (const float*)mask_head;
                const int mask_stride = mask_head.empty() ? 0 : mask_head.w;

                Mat top_blob_head = top_blob.channel(q);
                float* score = score_workspace.row(get_omp_thread_num());
                float* outptr0 = top_blob_head.row(i);
                float m_vec[MAX_BLOCK_M];
                float l_vec[MAX_BLOCK_M];

                sdpa_flash_attention_tile_packed_fp32(query_ptr, packed_key_ptr, cur_value_ptr, mask_ptr, outptr0, score, m_vec, l_vec, i, max_ii, 0, dst_seqlen, TILE_N, dst_seqlen, embed_dim, out_embed_dim, mask_stride, _scale, true);
            }
        }
    }
    else
    {
        const int partial_stride = MAX_BLOCK_M * 2 + MAX_BLOCK_M * out_embed_dim;
        Mat partials(partial_stride, num_kv_chunks, num_tiles, 4u, opt.workspace_allocator);
        if (partials.empty())
            return -100;

        if (!use_packed_key)
        {
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
                const int n_begin = chunk_id * dst_seqlen / num_kv_chunks;
                const int n_end = (chunk_id + 1) * dst_seqlen / num_kv_chunks;

                const Mat query_head = query.channel(q);
                const Mat past_key_head = past_seqlen > 0 ? past_key.channel(g) : Mat();
                const Mat cur_key_head = cur_key.channel(g);
                const Mat past_value_head = past_seqlen > 0 ? past_value.channel(g) : Mat();
                const Mat cur_value_head = cur_value.channel(g);

                Mat mask_head;
                if (attn_mask)
                {
                    mask_head = attn_mask_blob;
                    if (attn_mask_blob.dims == 3)
                        mask_head = attn_mask_blob.c > 1 ? attn_mask_blob.channel(q) : attn_mask_blob.channel(0);
                }

                const float* query_ptr = query_head;
                const float* past_key_ptr = past_seqlen > 0 ? (const float*)past_key_head : 0;
                const float* cur_key_ptr = cur_key_head;
                const float* past_value_ptr = past_seqlen > 0 ? (const float*)past_value_head : 0;
                const float* cur_value_ptr = cur_value_head;
                const float* mask_ptr = mask_head.empty() ? 0 : (const float*)mask_head;
                const int mask_stride = mask_head.empty() ? 0 : mask_head.w;

                float* score = score_workspace.row(get_omp_thread_num());
                float* partial = partials.channel(tile_id).row(chunk_id);
                float* m_vec = partial;
                float* l_vec = partial + MAX_BLOCK_M;
                float* outptr0 = partial + MAX_BLOCK_M * 2;

                sdpa_flash_attention_tile_fp32(query_ptr, past_key_ptr, cur_key_ptr, past_value_ptr, cur_value_ptr, mask_ptr, outptr0, score, m_vec, l_vec, i, max_ii, n_begin, n_end, TILE_N, past_seqlen, embed_dim, out_embed_dim, mask_stride, _scale, false);
            }
        }
        else
        {
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
                const int n_begin = chunk_id * dst_seqlen / num_kv_chunks;
                const int n_end = (chunk_id + 1) * dst_seqlen / num_kv_chunks;

                const Mat query_head = query.channel(q);
                const Mat packed_key_head = packed_key.channel(g);
                const Mat cur_value_head = cur_value.channel(g);

                Mat mask_head;
                if (attn_mask)
                {
                    mask_head = attn_mask_blob;
                    if (attn_mask_blob.dims == 3)
                        mask_head = attn_mask_blob.c > 1 ? attn_mask_blob.channel(q) : attn_mask_blob.channel(0);
                }

                const float* query_ptr = query_head;
                const float* packed_key_ptr = packed_key_head;
                const float* cur_value_ptr = cur_value_head;
                const float* mask_ptr = mask_head.empty() ? 0 : (const float*)mask_head;
                const int mask_stride = mask_head.empty() ? 0 : mask_head.w;

                float* score = score_workspace.row(get_omp_thread_num());
                float* partial = partials.channel(tile_id).row(chunk_id);
                float* m_vec = partial;
                float* l_vec = partial + MAX_BLOCK_M;
                float* outptr0 = partial + MAX_BLOCK_M * 2;

                sdpa_flash_attention_tile_packed_fp32(query_ptr, packed_key_ptr, cur_value_ptr, mask_ptr, outptr0, score, m_vec, l_vec, i, max_ii, n_begin, n_end, TILE_N, dst_seqlen, embed_dim, out_embed_dim, mask_stride, _scale, false);
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
            sdpa_flash_attention_reduce_fp32(top_blob_head.row(i), partials.channel(tile_id), max_ii, out_embed_dim, num_kv_chunks);
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
