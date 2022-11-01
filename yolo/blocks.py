import torch
import torch.nn as nn

class Convblock(nn.Module):
    def __init__(self, input_c, output_c ,kernel=3, stride=1,padding=1, bn=True) -> None:
        super().__init__()

        tmp_seq = [
            nn.Conv2d(input_c, output_c, kernel,stride, padding),
            nn.LeakyReLU()
        ]
        if bn:
            tmp_seq.append(
                nn.BatchNorm2d(output_c)
            )

        self.conv = nn.Sequential(*tmp_seq)
        del tmp_seq
    
    def forward(self, x):
        return self.conv(x)


class YoloBlock(nn.Module):
    def __init__(self, in_channel=3,out_channels=[],kernels=[], strides=[],bn=True, max_pool=None) -> None:
        super().__init__()

        layers = []
        for out, kernel, stride in zip(out_channels, kernels, strides):
            layers.append(
                Convblock(in_channel,out,kernel,stride,bn=bn)
            )
            in_channel = out
        if max_pool:
            layers.append(
                nn.MaxPool2d(max_pool)
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)