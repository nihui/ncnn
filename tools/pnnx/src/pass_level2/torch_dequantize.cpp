// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_dequantize : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
aten::dequantize        op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.dequantize";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_dequantize, 40)

} // namespace pnnx
