# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.avg_pool1d(x, kernel_size=3)
        x = F.avg_pool1d(x, kernel_size=4, stride=2, padding=2)
        x = F.avg_pool1d(x, kernel_size=3, stride=1, padding=(0), ceil_mode=False, count_include_pad=True)
        x = F.avg_pool1d(x, kernel_size=5, stride=2, padding=(2), ceil_mode=True, count_include_pad=False)
        x = F.avg_pool1d(x, kernel_size=3, stride=2, padding=1, ceil_mode=False, count_include_pad=True)
        x = F.avg_pool1d(x, kernel_size=2, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        x = F.avg_pool1d(x, kernel_size=4, stride=1, padding=2, ceil_mode=False, count_include_pad=False)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 128)

    a = net(x)

    # export onnx
    torch.onnx.export(net, (x,), "test_F_avg_pool1d.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_F_avg_pool1d.onnx inputshape=[1,12,128]")

    # pnnx inference
    import test_F_avg_pool1d_pnnx
    b = test_F_avg_pool1d_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
