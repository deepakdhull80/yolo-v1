import torch
import torch.nn as nn

from yolo.base_block import BaseYolo

class ImageClassifier(nn.Module):
    def __init__(self, input_channel, blocks, bottleneck_features, n_class) -> None:
        super().__init__()

        self.base = BaseYolo(input_channel, blocks)
        self.base_output_size = blocks[-1]['channels'][-1]

        self.fc = nn.Sequential(
            nn.Linear(self.base_output_size, bottleneck_features),
            nn.Dropout(),
            nn.Linear(bottleneck_features, n_class)
        )

    def forward(self, x):
        b = x.shape[0]
        x = self.base(x)
        x = x.view(b,self.base_output_size, -1)
        x = torch.mean(x,axis=-1)

        return self.fc(x)

class Yolo(nn.Module):
    """
    ## yOLOv1 model 
    https://arxiv.org/pdf/1506.02640.pdf

    input_channel: (int)
    blocks: ()
    bottleneck_features: ()
    s: (int), default [7], patches
    b: (int), default [2], this is for bounding box
    n_class: (int) default [10]
    """
    def __init__(self, input_channel, blocks, bottleneck_features, s=7, b=2, n_class=10) -> None:
        super().__init__()
        self.s=s
        self.b=b
        self.n_class=n_class
        self.base = BaseYolo(input_channel, blocks)
        self.base_output_size = blocks[-1]['channels'][-1] * 14 * 14

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.base_output_size, bottleneck_features),
            # nn.Tanh(),
            nn.Dropout(),
            nn.Linear(bottleneck_features, self.s*self.s*(self.b * 5 + self.n_class)),
            nn.Sigmoid()
        )

    def forward(self, x):
        b = x.shape[0]
        x = self.base(x)
        x = self.fc(x)
        return x.view(b, self.s, self.s, self.b * 5 + self.n_class) # [16, 7, 7, 20]
