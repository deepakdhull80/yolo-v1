import torch
import torch.nn as nn
from yolo.blocks import YoloBlock

class BaseYolo(nn.Module):
    def __init__(self, input_c, blocks):
        """
        ## Yolo base block:
        input_c: (int)
        blocks: (List[dict]), each dict have config for a block (channels, kernels, strided, bn, max_pool)
        """
        super().__init__()
        layers = []
        for block in blocks:
            layers.append(
                YoloBlock(
                    in_channel=input_c,
                    out_channels=block['channels'],
                    kernels=block['kernels'],
                    strides=block['strides'],
                    bn=block['bn'],
                    max_pool=block['max_pool']
                )
            )

            input_c = block['channels'][-1]
        # self.layers = layers
        self.base = nn.Sequential(*layers)


    def forward(self, x):
        return self.base(x)