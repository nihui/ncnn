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

#include "pass_level2.h"

namespace pnnx {

class F_tanhshrink : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 8 value=1
aten::tanh              op_1        1 1 input 7
aten::sub               op_2        3 1 input 7 8 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.tanhshrink";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_tanhshrink, 100)

class F_tanhshrink_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
aten::tanh              op_0        1 1 input 7
aten::sub               op_1        2 1 input 7 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.tanhshrink";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_tanhshrink_onnx, 100)

} // namespace pnnx
