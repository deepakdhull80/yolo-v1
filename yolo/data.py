import os
import json
from typing import *

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomVerticalFlip, RandomHorizontalFlip
from sklearn.utils.class_weight import compute_class_weight

class ImageCrop:
    def __init__(self, transform=None) -> None:
        self.transform = transform

    def __call__(self, image: Image, box: Union[List, np.ndarray]):
        """
        params:
            image: (PIL.Image)
            box: (Union[List, np.ndarray]), box should have shape of 4, [x,y,w,h]
        """
        c = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
        image = image.crop(c)
        return self.transform(image) if self.transform is not None else image


class ImageDataset(Dataset):
    def __init__(self, data_path, image_path, fold="train", image_size=448, augment=False):
        super().__init__()
        self.data_path = data_path
        self.image_path = image_path
        self.image_size = image_size
        self.fold = fold

        self.df = pd.read_parquet(f"{data_path}/{fold}.parquet")
        transform = Compose([
            Resize((image_size, image_size)),
            ToTensor()
        ])
        if augment:
            self.augment = Compose(
                [
                RandomHorizontalFlip(0.3),
                RandomVerticalFlip(0.2)
                ]
            )

        self.transform = ImageCrop(transform=transform)

        self.category = json.load(open(f'{self.data_path}/sampled_categories.json','r'))
        self.val2indx = {
            v:i for i,v in enumerate(self.category.keys())
        }

        self.class_weights = self.get_class_weights()
    
    def get_class_weights(self):
        y = self.df['category_id'].map(lambda x: self.val2indx[str(x)])
        class_weights = compute_class_weight(
                        class_weight='balanced',
                        classes=np.unique(y.to_numpy()), 
                        y = y.to_numpy()
                    )
        return torch.tensor(class_weights, dtype=torch.float)

    def __getitem__(self, index):
        
        row = self.df.loc[index]
        image = Image.open(f"{self.image_path}/{row['file_name']}")
        if len(image.getbands())!=3:
            return None, None
        image = self.transform(image,row['bbox'])
        if hasattr(self, "augment"):
            image = self.augment(image)
        return image, self.val2indx[str(row['category_id'])]
        
    def __len__(self):
        return self.df.shape[0]


class YoloDataset(Dataset):
    def __init__(self, data_path, image_path, fold="train", image_size=448, s=7, b=2):
        super().__init__()
        self.data_path = data_path
        self.image_path = image_path
        self.image_size = image_size
        self.fold = fold
        self.s = s
        self.b = b

        self.df = pd.read_parquet(f"{data_path}/{fold}.parquet")
        
        self.transform = Compose([
            Resize((image_size, image_size)),
            ToTensor()
        ])

        self.category = json.load(open(f'{self.data_path}/sampled_categories.json','r'))
        self.val2indx = {
            v:i for i,v in enumerate(self.category.keys())
        }
        self.n_class = len(list(self.val2indx.keys()))
        self.image_ids = list(set(self.df['image_id'].to_list()))

    def __getitem__(self, index):
        """
        Args:
            index (_type_): _description_
        
        returns:
            image, target
            -------------------------------------------------
            image -> (3,image_size, image_size)
            target -> (s,s,b*5+classes), s is patch, b no of object can be present in single 
                                        patch, 5 -> (is_present, x, y, w, h), classes is no 
                                        of classes
        """
        image_id = self.image_ids[index]

        annotate = self.df[self.df['image_id'] == image_id].reset_index(drop=True)
        file_name = annotate['file_name'].iloc[0]
        image = Image.open(f"{self.image_path}/{file_name}")
        if len(image.getbands())!=3:
            return None, None
        h,w = image.height, image.width

        image = self.transform(image)
        target = torch.zeros(
            self.s, self.s, self.b*5 + self.n_class
        )
        for _, row in annotate.iterrows():
            if str(row['category_id']) not in self.category:
                continue
            bbox = self.scale_box(
                row['bbox'], w, h
            )
            sx = int(bbox[0] * self.s) 
            sy = int(bbox[1] * self.s)
            v = torch.zeros(self.b*5 + self.n_class)
            class_idx = self.val2indx[str(row['category_id'])]
            v[self.b*5 + class_idx] = 1
            for i in range(self.b):
                v[i * 5] = 1
                v[i*5 + 1 : i*5 + 5] = torch.tensor(bbox)
            target[sx,sy,:] = v.clone()
        return image, target

    def scale_box(self, box, original_width, original_height):
        x,y,wi,hi = box
        x /= original_width
        wi /= original_width
        y /= original_height
        hi /= original_height
        return [x,y,wi,hi]

    def __len__(self):
        return len(self.image_ids)


def collate_fn(batch):
    tbatch = []
    for i in batch:
        if i[0] != None:
            tbatch.append(i)
    return default_collate(tbatch)

def get_data_loader(data_path, image_path ,yolo_train=False, image_size=448, batch_size=8, n_worker=1, pin_memory=False, **kwargs):
    ### it should return train and val set, train and val split is done randomly with evenly distributed data
    """# DataLoader

    Args:
        data_path (_type_): _description_
        image_path (_type_): _description_
        yolo_train (bool, optional): _description_. Defaults to False.
        image_size (int, optional): _description_. Defaults to 448.
        batch_size (int, optional): _description_. Defaults to 8.
        n_worker (int, optional): _description_. Defaults to 1.
        pin_memory (bool, optional): _description_. Defaults to False.
        s (int): Yolo number of patches, required when yolo training. Defaults to 7
        b (int): Yolo bounding box, required when yolo training. Defaults to 1

    Returns:
        tuple: (DataLoader,DataLoader,Tensor), train_dataloader, val_dataloader, class_weights
    """
    train_ds, val_ds = None, None
    if yolo_train:
        train_ds, val_ds = YoloDataset(
            data_path="data",
            image_path="data/val2017",
            fold='train',
            image_size=image_size,
            s=kwargs.get("yolo_patches",7),
            b=kwargs.get("yolo_bounding_box",1)
        ), YoloDataset(
            data_path="data",
            image_path="data/val2017",
            fold='val',
            image_size=image_size,
            s=kwargs.get("yolo_patches",7),
            b=kwargs.get("yolo_bounding_box",1)
        )
    else:
        train_ds, val_ds = ImageDataset(
            data_path=data_path,
            image_path=image_path,
            fold='train',image_size=image_size
        ), ImageDataset(
            data_path=data_path,
            image_path=image_path,
            fold='val',image_size=image_size
        )
    assert train_ds != None or val_ds != None, "dataloader strategy failed for yolo_train={yolo_train}"
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_worker, collate_fn=lambda x:collate_fn(x), pin_memory=pin_memory), \
            DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=n_worker, collate_fn=lambda x:collate_fn(x), pin_memory=pin_memory), \
            train_ds.class_weights if not yolo_train else None