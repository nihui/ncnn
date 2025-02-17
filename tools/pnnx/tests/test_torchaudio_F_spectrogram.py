# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x0 = torchaudio.functional.spectrogram(x, n_fft=64, window=torch.hann_window(44), win_length=44, hop_length=16, pad=0, center=True, normalized='window', power=1)
        if version.parse(torchaudio.__version__) < version.parse('0.11.0'):
            # return_complex=False with power=None, skip it
            x1 = torchaudio.functional.spectrogram(x, n_fft=128, window=torch.hann_window(128), win_length=128, hop_length=3, pad=0, center=False, onesided=True, normalized=False, power=1)
        else:
            x1 = torchaudio.functional.spectrogram(x, n_fft=128, window=torch.hann_window(128), win_length=128, hop_length=3, pad=0, center=False, onesided=True, normalized=False, power=None)
        x2 = torchaudio.functional.spectrogram(x, n_fft=512, window=torch.hamming_window(256), win_length=256, hop_length=128, pad=0, center=True, pad_mode='constant', onesided=True, normalized='frame_length', power=2)
        x3 = torchaudio.functional.spectrogram(x, n_fft=512, window=torch.hamming_window(512), win_length=512, hop_length=128, pad=32, center=True, onesided=False, normalized=False, power=2)
        y0 = torchaudio.functional.spectrogram(y, n_fft=64, window=torch.hann_window(44), win_length=44, hop_length=16, pad=0, center=True, normalized='window', power=1)
        if version.parse(torchaudio.__version__) < version.parse('0.11.0'):
            # return_complex=False with power=None, skip it
            y1 = torchaudio.functional.spectrogram(y, n_fft=128, window=torch.hann_window(128), win_length=128, hop_length=3, pad=0, center=False, onesided=True, normalized=False, power=1)
        else:
            y1 = torchaudio.functional.spectrogram(y, n_fft=128, window=torch.hann_window(128), win_length=128, hop_length=3, pad=0, center=False, onesided=True, normalized=False, power=None)
        y2 = torchaudio.functional.spectrogram(y, n_fft=512, window=torch.hamming_window(256), win_length=256, hop_length=128, pad=0, center=True, pad_mode='constant', onesided=True, normalized='frame_length', power=2)
        y3 = torchaudio.functional.spectrogram(y, n_fft=512, window=torch.hamming_window(512), win_length=512, hop_length=128, pad=32, center=True, onesided=False, normalized=False, power=2)
        return x0, x1, x2, x3, y0, y1, y2, y3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 2560)
    y = torch.rand(1000)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_torchaudio_F_spectrogram.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torchaudio_F_spectrogram.pt inputshape=[3,2560],[1000]")

    # pnnx inference
    import test_torchaudio_F_spectrogram_pnnx
    b = test_torchaudio_F_spectrogram_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
