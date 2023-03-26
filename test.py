import random
import torch
from yolo.data import ImageDataset, YoloDataset
from yolo.model import ImageClassifier, Yolo
from yolo.loss import YoloLoss
import config
from tqdm import tqdm

from torch.utils.data import DataLoader, default_collate


def print_test(t):
    print(f"***** TESTING BLOCK {t} ******")

def test_classifier_model():
    print_test("START")
    
    model = ImageClassifier(3, config.blocks, 4096, 5)
    x = torch.randn(1,3,448,448)
    print(model)
    x = model(x)
    print(x.shape)
    print_test("END")

def test_yolo_model():
    print_test("START")

    blocks = config.blocks
    print(f'bottle_neck_feature_size={config.bottle_neck_feature_size}\ns={config.yolo_patches}\nb={config.yolo_bounding_box}\nn_class={config.n_classes}')
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

def test_yolo_dataset():
    print_test("Start")
    val = YoloDataset(
        data_path="data",
        image_path="data/val2017",
        fold='val',
        image_size=config.image_size,
        s=config.yolo_patches,
        b=config.yolo_bounding_box
    )

    def collate_fn(batch):
        tbatch = []
        for i in batch:
            if i[0] != None:
                tbatch.append(i)
        
        return default_collate(tbatch)

    dl = DataLoader(val, batch_size=10, collate_fn=lambda x: collate_fn(x), shuffle=False)
    
    for row in dl:
        i = row
        print(i[0].shape, i[1].shape)
    print_test("end")

# @torch.no_grad()
def test_yolo_loss():
    loss = YoloLoss(
        s=config.yolo_patches,
        b=config.yolo_bounding_box,
        n_class=config.n_classes,
        lambda_coord=config.lambda_coord,
        lambda_noobj=config.lambda_noobj
    )

    val = YoloDataset(
        data_path="data",
        image_path="data/val2017",
        fold='train',
        image_size=config.image_size,
        s=config.yolo_patches,
        b=config.yolo_bounding_box
    )

    def collate_fn(batch):
        tbatch = []
        for i in batch:
            if i[0] != None:
                tbatch.append(i)
        
        return default_collate(tbatch)

    dl = DataLoader(val, batch_size=10, collate_fn=lambda x: collate_fn(x), shuffle=False)
    
    blocks = config.blocks
    print(f'bottle_neck_feature_size={config.bottle_neck_feature_size}\ns={config.yolo_patches}\nb={config.yolo_bounding_box}\nn_class={config.n_classes}')
    
    model = Yolo(
            config.input_channel, 
            blocks, 
            config.bottle_neck_feature_size,
            s=config.yolo_patches,
            b=config.yolo_bounding_box,
            n_class=config.n_classes
        )
    model = model.eval()

    for row in dl:
        image, target = row
        o = model(image)
        l = loss(target, o)
        print(l)





if __name__ == '__main__':
    # test_classifier_model()
    # test_yolo_model()
    # test_image_dataset()
    # test_yolo_dataset()
    test_yolo_loss()