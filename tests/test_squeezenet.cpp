// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "platform.h"
#include "net.h"
#include "testutil.h"

#include <stdio.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

static ncnn::Mat generate_ncnn_logo(int pixel_type_to, int w, int h)
{
    // clang-format off
    // *INDENT-OFF*
    static const unsigned char ncnn_logo_data[16][16] =
    {
        {245, 245,  33, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245,  33, 245, 245},
        {245,  33,  33,  33, 245, 245, 245, 245, 245, 245, 245, 245,  33,  33,  33, 245},
        {245,  33, 158, 158,  33, 245, 245, 245, 245, 245, 245,  33, 158, 158,  33, 245},
        { 33, 117, 158, 224, 158,  33, 245, 245, 245, 245,  33, 158, 224, 158, 117,  33},
        { 33, 117, 224, 224, 224,  66,  33,  33,  33,  33,  66, 224, 224, 224, 117,  33},
        { 33, 189, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 189,  33},
        { 33, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224,  33},
        { 33, 224, 224,  97,  97,  97,  97, 224, 224,  97,  97,  97,  97, 224, 224,  33},
        { 33, 224, 224,  97,  33,   0, 189, 224, 224,  97,   0,  33,  97, 224, 224,  33},
        { 33, 224, 224,  97,  33,   0, 189, 224, 224,  97,   0,  33,  97, 224, 224,  33},
        { 33, 224, 224,  97,  97,  97,  97, 224, 224,  97, 189, 189,  97, 224, 224,  33},
        { 33,  66,  66,  66, 224, 224, 224, 224, 224, 224, 224, 224,  66,  66,  66,  33},
        { 66, 158, 158,  66,  66, 224, 224, 224, 224, 224, 224,  66, 158, 158,  66,  66},
        { 66, 158, 158, 208,  66, 224, 224, 224, 224, 224, 224,  66, 158, 158, 208,  66},
        { 66, 224, 202, 158,  66, 224, 224, 224, 224, 224, 224,  66, 224, 202, 158,  66},
        { 66, 158, 224, 158,  66, 224, 224, 224, 224, 224, 224,  66, 158, 224, 158,  66}
    };
    // *INDENT-ON*
    // clang-format on

    const unsigned char* p_ncnn_logo_data = (const unsigned char*)ncnn_logo_data;
    ncnn::Mat logo = ncnn::Mat::from_pixels(p_ncnn_logo_data, ncnn::Mat::PIXEL_GRAY | (pixel_type_to << ncnn::Mat::PIXEL_CONVERT_SHIFT), 16, 16);

    ncnn::Mat m;
    ncnn::Option opt;
    opt.num_threads = 1;
    ncnn::resize_nearest(logo, m, w, h, opt);
    return m;
}

struct compare_score_index
{
    inline bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b)
    {
        return a.first > b.first;
    }
};

static int check_top2(const std::vector<float>& cls_scores, float epsilon = 0.001)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + 2, vec.end(), compare_score_index());

    int expect_indexes[2] = {532, 920};
    float expect_scores[2] = {0.189459f, 0.082801f};

    for (int i = 0; i < 2; i++)
    {
        int index = vec[i].second;
        float score = vec[i].first;

        if (index != expect_indexes[i])
        {
            fprintf(stderr, "top %d index not match  expect %d but got %d\n", i, expect_indexes[i], index);
            return -1;
        }

        if (!NearlyEqual(score, expect_scores[i], epsilon))
        {
            fprintf(stderr, "top %d score not match  expect %f but got %f\n", i, expect_scores[i], score);
            return -1;
        }
    }

    return 0;
}

static void fread_or_error(void* buffer, size_t size, size_t count, FILE* fp, const char* s)
{
    if (count != fread(buffer, size, count, fp))
    {
        fprintf(stderr, "Couldn't read from file: %s\n", s);
        fclose(fp);
        exit(EXIT_FAILURE);
    }
}

static std::string read_file_string(const char* filepath)
{
    FILE* fp = fopen(filepath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", filepath);
        return std::string();
    }

    fseek(fp, 0, SEEK_END);
    int len = ftell(fp);
    rewind(fp);

    std::string s;
    s.resize(len + 1); // +1 for '\0'

    fread_or_error((char*)s.c_str(), 1, len, fp, filepath);
    fclose(fp);

    s[len] = '\0';

    return s;
}

static ncnn::Mat read_file_content(const char* filepath)
{
    FILE* fp = fopen(filepath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", filepath);
        return ncnn::Mat();
    }

    fseek(fp, 0, SEEK_END);
    int len = ftell(fp);
    rewind(fp);

    ncnn::Mat m(len, (size_t)1u, 1);

    fread_or_error(m, 1, len, fp, filepath);
    fclose(fp);

    return m;
}

static int test_squeezenet(const ncnn::Option& opt, int load_model_type, float epsilon = 0.001)
{
    ncnn::Net squeezenet;

    squeezenet.opt = opt;

#ifdef __EMSCRIPTEN__
#define MODEL_DIR "/working"
#else
#define MODEL_DIR "../../examples"
#endif

    std::string param_str;
    ncnn::Mat param_data;
    ncnn::Mat model_data;
    if (load_model_type == 0)
    {
        // load from plain model file
        squeezenet.load_param(MODEL_DIR "/squeezenet_v1.1.param");

        // test random feature disabled bits
        {
            std::vector<ncnn::Layer*>& layers = squeezenet.mutable_layers();
            for (size_t i = 0; i < layers.size(); i++)
            {
                layers[i]->featmask = i * 11 % 128;
            }
        }

        squeezenet.load_model(MODEL_DIR "/squeezenet_v1.1.bin");
    }
    if (load_model_type == 1)
    {
        // load from plain model memory
        param_str = read_file_string(MODEL_DIR "/squeezenet_v1.1.param");
        model_data = read_file_content(MODEL_DIR "/squeezenet_v1.1.bin");
        squeezenet.load_param_mem((const char*)param_str.c_str());
        squeezenet.load_model((const unsigned char*)model_data);
    }
    if (load_model_type == 2)
    {
        // load from binary model file
        squeezenet.load_param_bin(MODEL_DIR "/squeezenet_v1.1.param.bin");
        squeezenet.load_model(MODEL_DIR "/squeezenet_v1.1.bin");
    }
    if (load_model_type == 3)
    {
        // load from binary model memory
        param_data = read_file_content(MODEL_DIR "/squeezenet_v1.1.param.bin");
        model_data = read_file_content(MODEL_DIR "/squeezenet_v1.1.bin");
        squeezenet.load_param((const unsigned char*)param_data);
        squeezenet.load_model((const unsigned char*)model_data);
    }

    ncnn::Mat in = generate_ncnn_logo(ncnn::Mat::PIXEL_BGR, 227, 227);

    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = squeezenet.create_extractor();

    ncnn::Mat out;
    if (load_model_type == 0 || load_model_type == 1)
    {
        ex.input("data", in);
        ex.extract("prob", out);
    }
    if (load_model_type == 2 || load_model_type == 3)
    {
        ex.input(0, in);
        ex.extract(82, out);
    }

    std::vector<float> cls_scores;
    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return check_top2(cls_scores, epsilon);
}

class MyConvolution : public ncnn::Layer
{
public:
    MyConvolution()
    {
        impl = ncnn::create_layer("Convolution");

        one_blob_only = impl->one_blob_only;
        support_inplace = impl->support_inplace;

        support_packing = impl->support_packing;
        support_vulkan = impl->support_vulkan;
        support_bf16_storage = impl->support_bf16_storage;
        support_fp16_storage = impl->support_fp16_storage;
        support_int8_storage = impl->support_int8_storage;
    }

    ~MyConvolution()
    {
        delete impl;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
#if NCNN_VULKAN
        impl->vkdev = vkdev;
#endif // NCNN_VULKAN

        return impl->load_param(pd);
    }

    virtual int load_model(const ncnn::ModelBin& mb)
    {
        return impl->load_model(mb);
    }

    virtual int create_pipeline(const ncnn::Option& opt)
    {
        int ret = impl->create_pipeline(opt);

        one_blob_only = impl->one_blob_only;
        support_inplace = impl->support_inplace;

        support_packing = impl->support_packing;
        support_vulkan = impl->support_vulkan;
        support_bf16_storage = impl->support_bf16_storage;
        support_fp16_storage = impl->support_fp16_storage;
        support_int8_storage = impl->support_int8_storage;

        return ret;
    }

    virtual int destroy_pipeline(const ncnn::Option& opt)
    {
        return impl->destroy_pipeline(opt);
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        return impl->forward(bottom_blob, top_blob, opt);
    }

#if NCNN_VULKAN
    virtual int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
    {
        return impl->upload_model(cmd, opt);
    }

    virtual int forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const
    {
        return impl->forward(bottom_blob, top_blob, cmd, opt);
    }
#endif // NCNN_VULKAN

private:
    ncnn::Layer* impl;
};

DEFINE_LAYER_CREATOR(MyConvolution)
DEFINE_LAYER_DESTROYER(MyConvolution)

static int test_squeezenet_overwrite_softmax(const ncnn::Option& opt, int load_model_type, float epsilon = 0.001)
{
    ncnn::Net squeezenet;

    squeezenet.opt = opt;

#ifdef __EMSCRIPTEN__
#define MODEL_DIR "/working"
#else
#define MODEL_DIR "../../examples"
#endif

    std::string param_str;
    ncnn::Mat param_data;
    ncnn::Mat model_data;
    if (load_model_type == 0)
    {
        // load from plain model file
        squeezenet.register_custom_layer("Convolution", MyConvolution_layer_creator, MyConvolution_layer_destroyer);
        squeezenet.load_param(MODEL_DIR "/squeezenet_v1.1.param");

        // test random feature disabled bits
        {
            std::vector<ncnn::Layer*>& layers = squeezenet.mutable_layers();
            for (size_t i = 0; i < layers.size(); i++)
            {
                layers[i]->featmask = i * 11 % 128;
            }
        }

        squeezenet.load_model(MODEL_DIR "/squeezenet_v1.1.bin");
    }
    if (load_model_type == 1)
    {
        // load from plain model memory
        squeezenet.register_custom_layer("Convolution", MyConvolution_layer_creator, MyConvolution_layer_destroyer);
        param_str = read_file_string(MODEL_DIR "/squeezenet_v1.1.param");
        model_data = read_file_content(MODEL_DIR "/squeezenet_v1.1.bin");
        squeezenet.load_param_mem((const char*)param_str.c_str());
        squeezenet.load_model((const unsigned char*)model_data);
    }
    if (load_model_type == 2)
    {
        // load from binary model file
        squeezenet.register_custom_layer(ncnn::layer_to_index("Convolution"), MyConvolution_layer_creator, MyConvolution_layer_destroyer);
        squeezenet.load_param_bin(MODEL_DIR "/squeezenet_v1.1.param.bin");
        squeezenet.load_model(MODEL_DIR "/squeezenet_v1.1.bin");
    }
    if (load_model_type == 3)
    {
        // load from binary model memory
        squeezenet.register_custom_layer(ncnn::layer_to_index("Convolution"), MyConvolution_layer_creator, MyConvolution_layer_destroyer);
        param_data = read_file_content(MODEL_DIR "/squeezenet_v1.1.param.bin");
        model_data = read_file_content(MODEL_DIR "/squeezenet_v1.1.bin");
        squeezenet.load_param((const unsigned char*)param_data);
        squeezenet.load_model((const unsigned char*)model_data);
    }

    ncnn::Mat in = generate_ncnn_logo(ncnn::Mat::PIXEL_BGR, 227, 227);

    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = squeezenet.create_extractor();

    ncnn::Mat out;
    if (load_model_type == 0 || load_model_type == 1)
    {
        ex.input("data", in);
        ex.extract("prob", out);
    }
    if (load_model_type == 2 || load_model_type == 3)
    {
        ex.input(0, in);
        ex.extract(82, out);
    }

    std::vector<float> cls_scores;
    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return check_top2(cls_scores, epsilon);
}

int main()
{
    SRAND(7767517);

#ifdef __EMSCRIPTEN__
    EM_ASM(
        FS.mkdir('/working');
        FS.mount(NODEFS, {root: '../../examples'}, '/working'););
#endif // __EMSCRIPTEN__

    ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    ncnn::PoolAllocator g_workspace_pool_allocator;

    ncnn::Option opts[4];

    opts[0].use_packing_layout = false;
    opts[0].use_fp16_packed = false;
    opts[0].use_fp16_storage = false;
    opts[0].use_fp16_arithmetic = false;
    opts[0].use_shader_pack8 = false;

    opts[1].use_packing_layout = true;
    opts[1].use_fp16_packed = true;
    opts[1].use_fp16_storage = false;
    opts[1].use_fp16_arithmetic = false;
    opts[1].use_shader_pack8 = true;

    opts[2].use_packing_layout = true;
    opts[2].use_fp16_packed = true;
    opts[2].use_fp16_storage = true;
    opts[2].use_fp16_arithmetic = false;
    opts[2].use_bf16_storage = false; // FIXME enable me
    opts[2].use_shader_pack8 = true;
    opts[2].blob_allocator = &g_blob_pool_allocator;
    opts[2].workspace_allocator = &g_workspace_pool_allocator;

    opts[3].use_packing_layout = true;
    opts[3].use_fp16_packed = true;
    opts[3].use_fp16_storage = true;
    opts[3].use_fp16_arithmetic = false; // FIXME enable me
    opts[3].use_bf16_storage = false;
    opts[3].use_shader_pack8 = true;
    opts[3].blob_allocator = &g_blob_pool_allocator;
    opts[3].workspace_allocator = &g_workspace_pool_allocator;

    int load_model_types[4] = {0, 1, 2, 3};

    for (int i = 0; i < 4; i++)
    {
        opts[i].num_threads = 1;
    }

    for (int i = 0; i < 4; i++)
    {
        const ncnn::Option& opt = opts[i];

        float epsilon;
        if (opt.use_bf16_storage || opt.use_fp16_packed || opt.use_fp16_storage)
        {
            epsilon = 0.1;
        }
        else
        {
            epsilon = 0.01;
        }

        int ret;

        ncnn::Option opt_cpu = opt;
        opt_cpu.use_vulkan_compute = false;
        ret = test_squeezenet(opt_cpu, load_model_types[i], epsilon);
        if (ret != 0)
        {
            fprintf(stderr, "test_squeezenet cpu failed use_packing_layout=%d use_fp16_packed=%d use_fp16_storage=%d use_shader_pack8=%d use_bf16_storage=%d\n", opt.use_packing_layout, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_shader_pack8, opt.use_bf16_storage);
            return ret;
        }

#if NCNN_VULKAN
        ncnn::Option opt_gpu = opt;
        opt_gpu.use_vulkan_compute = true;
        ret = test_squeezenet(opt_gpu, load_model_types[i], epsilon);
        if (ret != 0)
        {
            fprintf(stderr, "test_squeezenet gpu failed use_packing_layout=%d use_fp16_packed=%d use_fp16_storage=%d use_shader_pack8=%d use_bf16_storage=%d\n", opt.use_packing_layout, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_shader_pack8, opt.use_bf16_storage);
            return ret;
        }
#endif // NCNN_VULKAN

        ret = test_squeezenet_overwrite_softmax(opt_cpu, load_model_types[i], epsilon);
        if (ret != 0)
        {
            fprintf(stderr, "test_squeezenet_overwrite_softmax cpu failed use_packing_layout=%d use_fp16_packed=%d use_fp16_storage=%d use_shader_pack8=%d use_bf16_storage=%d\n", opt.use_packing_layout, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_shader_pack8, opt.use_bf16_storage);
            return ret;
        }

#if NCNN_VULKAN
        ret = test_squeezenet_overwrite_softmax(opt_gpu, load_model_types[i], epsilon);
        if (ret != 0)
        {
            fprintf(stderr, "test_squeezenet_overwrite_softmax gpu failed use_packing_layout=%d use_fp16_packed=%d use_fp16_storage=%d use_shader_pack8=%d use_bf16_storage=%d\n", opt.use_packing_layout, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_shader_pack8, opt.use_bf16_storage);
            return ret;
        }
#endif // NCNN_VULKAN
    }

    return 0;
}
