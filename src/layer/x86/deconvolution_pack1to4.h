// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void deconvolution_pack1to4_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_packed, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
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
                __m128 _sum = _mm_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum = _mm_loadu_ps(bias_data_ptr + p * 4);
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

                        const float* sptr = m.row(sy);

                        for (int x = 0; x < kernel_w; x++)
                        {
                            int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                            if (sxs < 0 || sxs % stride_w != 0)
                                continue;

                            int sx = sxs / stride_w;
                            if (sx >= w)
                                continue;

                            float val = sptr[sx];

                            int k = y * kernel_w + x;

                            __m128 _val = _mm_set1_ps(val);
                            __m128 _w = _mm_load_ps(kptr + k * 4);
                            _sum = _mm_comp_fmadd_ps(_val, _w, _sum);
                        }
                    }

                    kptr += maxk * 4;
                }

                _sum = activation_sse(_sum, activation_type, activation_params);

                _mm_storeu_ps(outptr, _sum);
                outptr += 4;
            }
        }
    }
}
