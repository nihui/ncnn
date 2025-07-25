# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.w2 = nn.Parameter(torch.rand(6, 12, 4, 4, 4))
        self.b2 = nn.Parameter(torch.rand(12))
        self.w3 = nn.Parameter(torch.rand(12, 2, 3, 3, 3))

    def forward(self, x, w0, w1, b1, y):
        x = F.conv_transpose3d(x, w0, None, stride=(2,2,2), padding=(1,0,1), output_padding=(1,1,0))
        x = F.conv_transpose3d(x, w1, b1, stride=(1,1,2), padding=(2,2,1), dilation=(2,2,1), groups=2)

        y = F.conv_transpose3d(y, self.w2, self.b2, stride=(2,2,2), padding=(1,0,1), output_padding=(1,1,0))
        y = F.conv_transpose3d(y, self.w3, None, stride=(1,1,2), padding=(2,2,1), dilation=(2,2,1), groups=3)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 10, 12, 14)
    w0 = torch.rand(12, 16, 3, 2, 3)
    w1 = torch.rand(16, 8, 5, 4, 5)
    b1 = torch.rand(16)
    y = torch.rand(1, 6, 4, 5, 6)

    a0, a1 = net(x, w0, w1, b1, y)

    # export onnx
    torch.onnx.export(net, (x, w0, w1, b1, y), "test_F_conv_transpose3d.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_F_conv_transpose3d.onnx inputshape=[1,12,10,12,14],[12,16,3,2,3],[16,8,5,4,5],[16],[1,6,4,5,6]")

    # pnnx inference
    import test_F_conv_transpose3d_pnnx
    b0, b1 = test_F_conv_transpose3d_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
