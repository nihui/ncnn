// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void deconvolution_pack4to16_avx512(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_packed, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    int outch = top_blob.c;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const float* bias_data_ptr = bias_data;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* outptr = top_blob.channel(p);

        const int maxk = kernel_w * kernel_h;

        // shadowed variable for less openmp task args
        const int w = bottom_blob.w;
        const int h = bottom_blob.h;
        const int channels = bottom_blob.c;
        const int outw = top_blob.w;
        const int outh = top_blob.h;

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                __m512 _sum = _mm512_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum = _mm512_loadu_ps(bias_data_ptr + p * 16);
                }

                const float* kptr = weight_data_packed.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);

                    for (int y = 0; y < kernel_h; y++)
                    {
                        int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                        if (sys < 0 || sys % stride_h != 0)
                            continue;

                        int sy = sys / stride_h;
                        if (sy >= h)
                            continue;

                        for (int x = 0; x < kernel_w; x++)
                        {
                            int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                            if (sxs < 0 || sxs % stride_w != 0)
                                continue;

                            int sx = sxs / stride_w;
                            if (sx >= w)
                                continue;

                            const float* sptr = m.row(sy) + sx * 4;

                            int k = (y * kernel_w + x) * 64;

                            __m512 _val0 = _mm512_set1_ps(sptr[0]);
                            __m512 _val1 = _mm512_set1_ps(sptr[1]);
                            __m512 _val2 = _mm512_set1_ps(sptr[2]);
                            __m512 _val3 = _mm512_set1_ps(sptr[3]);
                            __m512 _w0 = _mm512_load_ps(kptr + k);
                            __m512 _w1 = _mm512_load_ps(kptr + k + 16);
                            __m512 _w2 = _mm512_load_ps(kptr + k + 32);
                            __m512 _w3 = _mm512_load_ps(kptr + k + 48);
                            _sum = _mm512_fmadd_ps(_val0, _w0, _sum);
                            _sum = _mm512_fmadd_ps(_val1, _w1, _sum);
                            _sum = _mm512_fmadd_ps(_val2, _w2, _sum);
                            _sum = _mm512_fmadd_ps(_val3, _w3, _sum);
                        }
                    }

                    kptr += maxk * 64;
                }

                _sum = activation_avx512(_sum, activation_type, activation_params);

                _mm512_storeu_ps(outptr, _sum);
                outptr += 16;
            }
        }
    }
}
