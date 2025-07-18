// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_hardswish : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::hardswish         op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish, 100)

class F_hardswish_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
11 10
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 392 value=3
prim::Constant          op_1        0 1 393 value=1
aten::add               op_2        3 1 input 392 393 a
prim::Constant          op_3        0 1 394 value=0
prim::Constant          op_4        0 1 395 value=6
F.hardtanh              op_5        3 1 a 394 395 b
aten::mul               op_6        2 1 input b c
prim::Constant          op_7        0 1 391 value=6
aten::div               op_8        2 1 c 391 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_1, 101)

class F_hardswish_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
F.hardsigmoid           op_0        1 1 input a
aten::mul               op_1        2 1 input a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_2, 101)

class F_hardswish_2_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
F.hardsigmoid           op_0        1 1 input a
aten::mul               op_1        2 1 a input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_2_1, 101)

class F_hardswish_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
11 10
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 12 value=3
prim::Constant          op_1        0 1 13 value=1
aten::add               op_2        3 1 input 12 13 a
prim::Constant          op_3        0 1 17 value=0
prim::Constant          op_4        0 1 18 value=6
F.hardtanh              op_5        3 1 a 17 18 b
aten::mul               op_6        2 1 input b c
prim::Constant          op_7        0 1 22 value=6
aten::div               op_8        2 1 c 22 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_3, 101)

class F_hardswish_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
11 10
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 25 value=3
prim::Constant          op_1        0 1 47 value=1
aten::add               op_2        3 1 input 25 47 a
prim::Constant          op_3        0 1 48 value=0
prim::Constant          op_4        0 1 49 value=6
F.hardtanh              op_5        3 1 a 48 49 b
prim::Constant          op_6        0 1 50 value=6
aten::div               op_7        2 1 b 50 c
aten::mul               op_8        2 1 c input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_4, 101)

class F_hardswish_4_1 : public F_hardswish_4
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
10 9
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 25 value=3
aten::add               op_2        2 1 input 25 a
prim::Constant          op_3        0 1 48 value=0
prim::Constant          op_4        0 1 49 value=6
torch.clamp             op_5        3 1 a 48 49 b
prim::Constant          op_6        0 1 50 value=6
aten::div               op_7        2 1 b 50 c
aten::mul               op_8        2 1 c input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_4_1, 101)

class F_hardswish_5 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 25 value=3
prim::Constant          op_1        0 1 48 value=1
aten::add               op_2        3 1 input 25 48 a
F.relu6                 op_3        1 1 a b
prim::Constant          op_4        0 1 49 value=6
aten::div               op_5        2 1 b 49 c
aten::mul               op_6        2 1 c input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_5, 101)

class F_hardswish_6 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 25 value=3
prim::Constant          op_1        0 1 48 value=1
aten::add               op_2        3 1 input 25 48 a
F.relu6                 op_3        1 1 a b
aten::mul               op_4        2 1 input b c
prim::Constant          op_5        0 1 49 value=6
aten::div               op_6        2 1 c 49 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_6, 101)

class F_hardswish_7 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
10 9
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 25 value=3
aten::add               op_1        2 1 input 25 a
prim::Constant          op_2        0 1 min value=0
prim::Constant          op_3        0 1 max value=6
torch.clamp             op_4        3 1 a min max b
aten::mul               op_5        2 1 input b c
prim::Constant          op_6        0 1 49 value=6
aten::div               op_7        2 1 c 49 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_7, 101)

class F_hardswish_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
HardSwish               op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_onnx, 100)

static bool NearlyEqual(float a, float b, float epsilon)
{
    if (a == b)
        return true;

    float diff = (float)fabs(a - b);
    if (diff <= epsilon)
        return true;

    // relative error
    return diff < epsilon * std::max(fabs(a), fabs(b));
}

class F_hardswish_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
HardSigmoid             op_0        1 1 input h alpha=%alpha
aten::mul               op_1        2 1 input h out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        float alpha = captured_params.at("alpha").f;
        return NearlyEqual(alpha, 1.f / 6, 0.001);
    }

    void write(Operator* /*op*/, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_onnx_1, 101)

class F_hardswish_onnx_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 20 value=3
aten::add               op_1        2 1 input 20 8
torch.clamp             op_2        1 1 8 9 max=6 min=0
prim::Constant          op_3        0 1 23 value=6
aten::div               op_4        2 1 9 23 10
aten::mul               op_5        2 1 input 10 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_onnx_2, 101)

class F_hardswish_onnx_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 v3 value=3
aten::add               op_1        2 1 input v3 a
torch.clamp             op_2        1 1 a b max=6 min=0
aten::mul               op_3        2 1 input b c
prim::Constant          op_4        0 1 v6 value=6
aten::div               op_5        2 1 c v6 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_onnx_3, 101)

} // namespace pnnx
