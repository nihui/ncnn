// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <stdio.h>

#include <string>
#include <vector>

#include "ir.h"
#include "pass_ncnn.h"
#include "save_ncnn.h"

static std::string get_basename(const std::string& path)
{
    return path.substr(0, path.find_last_of('.'));
}

static void show_usage()
{
    fprintf(stderr, "Usage: pnnx2ncnn [pnnx.param] [pnnx.bin] [ncnn.param] [ncnn.bin] [ncnn.py]\n");
    fprintf(stderr, "Sample usage: pnnx2ncnn mobilenet_v2.pnnx.param mobilenet_v2.pnnx.bin\n");
    fprintf(stderr, "              pnnx2ncnn mobilenet_v2.pnnx.param mobilenet_v2.pnnx.bin mobilenet_v2.ncnn.param mobilenet_v2.ncnn.bin mobilenet_v2_ncnn.py\n");
}

int main(int argc, char** argv)
{
    if (argc < 3 || argc > 6)
    {
        show_usage();
        return -1;
    }

    std::string pnnxparampath = std::string(argv[1]);
    std::string pnnxbinpath = std::string(argv[2]);

    std::string ptbase = get_basename(pnnxparampath);

    if (ptbase.size() > 5 && ptbase.substr(ptbase.size() - 5) == ".pnnx")
    {
        ptbase = ptbase.substr(0, ptbase.size() - 5);
    }

    std::string ncnnparampath = ptbase + ".ncnn.param";
    std::string ncnnbinpath = ptbase + ".ncnn.bin";
    std::string ncnnpypath = ptbase + "_ncnn.py";

    pnnx::Graph pnnx_graph;

    pnnx_graph.load(pnnxparampath, pnnxbinpath);

    pnnx::pass_ncnn(pnnx_graph);

    pnnx::save_ncnn(pnnx_graph, ncnnparampath, ncnnbinpath, ncnnpypath);

    return 0;
}