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

#include "fuse_module_pass.h"

namespace pnnx {

class Linear : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.linear.Linear";
    }

    const char* type_str() const
    {
        return "nn.Linear";
    }

    void write(Operator* op, const TorchGraphProxy& /*graph*/, const TorchModuleProxy& mod) const
    {
        // const TorchNodeProxy* addmm = graph.find_node_by_kind("aten::addmm");

        const TorchTensorProxy& weight = mod.attr("weight");

        op->params["in_features"] = weight.size(1);
        op->params["out_features"] = weight.size(0);
        op->params["bias"] = mod.hasattr("bias");

        op->attrs["weight"] = weight;
        if (mod.hasattr("bias"))
        {
            op->attrs["bias"] = mod.attr("bias");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Linear)

} // namespace pnnx
