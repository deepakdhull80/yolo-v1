from turtle import forward
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


class BaseYolo(nn.Module):
    def __init__(self, input_c, blocks):
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

class ImageClassifier(nn.Module):
    def __init__(self, input_channel, blocks, bottleneck_features, n_classes) -> None:
        super().__init__()

        self.base = BaseYolo(input_channel, blocks)
        self.base_output_size = blocks[-1]['channels'][-1]

        self.fc = nn.Sequential(
            nn.Linear(self.base_output_size, bottleneck_features),
            nn.Dropout(),
            nn.Linear(bottleneck_features, n_classes),
            nn.Softmax(dim=1)
        )



    def forward(self, x):
        b = x.shape[0]
        x = self.base(x)
        x = x.view(b,self.base_output_size, -1)
        x = torch.mean(x,axis=-1)

        return self.fc(x)
        

        return x

def test_model():
    blocks = [
        {
            'channels':[64],
            'kernels':[7],
            'strides':[2],
            'bn':True,
            'max_pool':2
        },
        {
            'channels':[192],
            'kernels':[3],
            'strides':[1],
            'bn':True,
            'max_pool':2
        },
        {
            'channels':[128,256,256,512],
            'kernels':[1,3,1,3],
            'strides':[1,1,1,1],
            'bn':True,
            'max_pool':2
        },
        {
            'channels':[256,512,256,512,256,512,256,512,512,1024],
            'kernels':[1,3,1,3,1,3,1,3,1,3],
            'strides':[1,1,1,1,1,1,1,1,1,1],
            'bn':True,
            'max_pool':2
        },
        {
            'channels':[512,1024,512,1024,1024,1024],
            'kernels':[1,3,1,3,3,3],
            'strides':[1,1,1,1,1,2],
            'bn':True,
            'max_pool':False
        },
        {
            'channels':[1024,1024],
            'kernels':[3,3],
            'strides':[1,1],
            'bn':True,
            'max_pool':False
        }
    ]

    model = ImageClassifier(3, blocks, 4096, 1024)
    x = model(x)
    print(x[0,:5])
    print(x.shape)

if __name__ == '__main__':
    test_model()
