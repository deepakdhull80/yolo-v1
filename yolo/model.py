import torch
import torch.nn as nn

from .base_block import BaseYolo

class ImageClassifier(nn.Module):
    def __init__(self, input_channel, blocks, bottleneck_features, n_class) -> None:
        super().__init__()

        self.base = BaseYolo(input_channel, blocks)
        self.base_output_size = blocks[-1]['channels'][-1]

        self.fc = nn.Sequential(
            nn.Linear(self.base_output_size, bottleneck_features),
            nn.Dropout(),
            nn.Linear(bottleneck_features, n_class),
            nn.Softmax(dim=1)
        )



    def forward(self, x):
        b = x.shape[0]
        x = self.base(x)
        x = x.view(b,self.base_output_size, -1)
        x = torch.mean(x,axis=-1)

        return self.fc(x)

class Yolo(nn.Module):
    def __init__(self, input_channel, blocks, bottleneck_features, s=7, b=2, n_class=10) -> None:
        super().__init__()
        self.s=s
        self.b=b
        self.n_class=n_class
        self.base = BaseYolo(input_channel, blocks)
        self.base_output_size = blocks[-1]['channels'][-1] * 12 * 12

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.base_output_size, bottleneck_features),
            nn.Dropout(),
            nn.Linear(bottleneck_features, self.s*self.s*(self.b * 5 + self.n_class)),
            nn.Sigmoid()
        )



    def forward(self, x):
        x = self.base(x)
        return self.fc(x)


def print_test(t):
    print(f"***** TESTING BLOCK {t} ******")

def test_classifier_model():
    print_test("START")
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
    x = torch.randn(1,3,448,448)
    x = model(x)
    print(x.shape)
    print_test("END")

def test_yolo_model():
    print_test("START")
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

    model = Yolo(3, blocks, 4096)
    x = torch.randn(1,3,448,448)
    x = model(x)
    print(x.shape)
    print_test("END")


if __name__ == '__main__':
    # test_classifier_model()
    test_yolo_model()
