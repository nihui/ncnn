// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef LAYER_RESHAPE_H
#define LAYER_RESHAPE_H

#include "layer.h"

namespace ncnn {

class Reshape : public Layer
{
public:
    Reshape();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    int eval_shape_expr(const std::vector<Mat>& bottom_blobs, int& outw, int& outh, int& outd, int& outc) const;

public:
    // reshape flag
    // 0 = copy from bottom
    // -1 = remaining
    // -233 = drop this dim (default)
    int w;
    int h;
    int d;
    int c;

    int ndim;

    // see docs/developer-guide/expression.md
    std::string shape_expr;
};

} // namespace ncnn

#endif // LAYER_RESHAPE_H
