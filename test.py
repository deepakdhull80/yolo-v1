import random
import torch
from yolo.data import ImageDataset
from yolo.model import ImageClassifier, Yolo
import config
from tqdm import tqdm

from torch.utils.data import DataLoader, default_collate


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

    model = ImageClassifier(3, config.blocks, 4096, 1024)
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
            'bn':False,
            'max_pool':2
        },
        {
            'channels':[192],
            'kernels':[3],
            'strides':[1],
            'bn':False,
            'max_pool':2
        },
        {
            'channels':[128,256,256,512],
            'kernels':[1,3,1,3],
            'strides':[1,1,1,1],
            'bn':False,
            'max_pool':2
        },
        {
            'channels':[256,512,256,512,256,512,256,512,512,1024],
            'kernels':[1,3,1,3,1,3,1,3,1,3],
            'strides':[1,1,1,1,1,1,1,1,1,1],
            'bn':False,
            'max_pool':2
        },
        {
            'channels':[512,1024,512,1024,1024,1024],
            'kernels':[1,3,1,3,3,3],
            'strides':[1,1,1,1,1,2],
            'bn':False,
            'max_pool':False
        },
        {
            'channels':[1024,1024],
            'kernels':[3,3],
            'strides':[1,1],
            'bn':False,
            'max_pool':False
        }
    ]

    blocks = config.blocks
    model = Yolo(
            config.input_channel, 
            blocks, 
            config.bottle_neck_feature_size,
            s=config.yolo_patches,
            b=config.yolo_bounding_box,
            n_class=config.n_classes
        )
    x = torch.randn(config.batch_size,config.input_channel,config.image_size,config.image_size)
    x = model(x)
    print(x.shape)
    print_test("END")

def test_image_dataset():
    print_test("Start")
    val = ImageDataset(
        data_path="data",
        image_path="data/val2017",
        fold='train'
    )

    def collate_fn(batch):
        tbatch = []
        for i in batch:
            if i[0] != None:
                tbatch.append(i)
        
        return default_collate(tbatch)

    dl = DataLoader(val, batch_size=10, collate_fn=lambda x: collate_fn(x), shuffle=True)
    
    for row in dl:
        i = row
        print(i[0].shape, i[1])
    print_test("end")

if __name__ == '__main__':
    # test_classifier_model()
    # test_yolo_model()
    test_image_dataset()
