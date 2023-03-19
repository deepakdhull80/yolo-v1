import os
import json
from typing import *

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, default_collate
import torchvision
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
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
    def __init__(self, data_path, image_path, fold="train", image_size=448):
        super().__init__()
        self.data_path = data_path
        self.image_path = image_path
        self.image_size = image_size
        self.fold = fold

        self.df = pd.read_parquet(f"{data_path}/{fold}.parquet")
        transform = Compose([
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize(0.5, 0.5) #(0.1307,), (0.3081,) try this one also
        ])

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
        image = self.transform(image,row['bbox'])
        if image.shape[0]!=3:
            return None, None
        return image, self.val2indx[str(row['category_id'])]
        
    def __len__(self):
        return self.df.shape[0]


class YoloDataset(Dataset):
    def __init__(self, data_path, image_path, fold="train", t2i=None, image_size=448, s=7, b=2, n_class=15):
        super().__init__()
        self.data_path = data_path
        self.image_path = image_path
        self.image_size = image_size
        self.s = s
        self.b = b
        self.n_class = n_class

        self.image_df = pd.read_json(os.path.join(data_path,f"{fold}_image_data.json"))
        self.annotation_df = pd.read_json(os.path.join(data_path,"final_data.json"))

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size))
        ])

        self.target2indx = t2i

    def __getitem__(self, index):
        data = self.image_df.loc[index].to_dict()
        
        image = torchvision.io.read_image(f"{self.image_path}/{data['file_name']}")
        
        annotation = self.annotation_df[self.annotation_df['image_id'] == data['id']]
        target = torch.zeros((self.s,self.s,self.b *5 + self.n_class))

        for _, row in annotation.iterrows():
            bbox = row['bbox']
            image, ij, t = self.decode(image, bbox)
            for i in range(self.b):
                target[ij[0], ij[1], i:(i+1)*5] = t
            
            target[ij[0], ij[1], self.target2indx[row['supercategory']]] = 1

        return image, target

    def decode(self, image, bbox):
        _, h, w = image.shape
        scale_x, scale_y, scale_w, scale_h = bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h
        center_xy = torch.Tensor([scale_x + scale_w/2, scale_y + scale_h/2])
        ij = torch.ceil(center_xy * self.s) - 1
        ij = ij.int()
        image = self.transform(image)
        image = image/ 255.0
        return image, ij, [center_xy[0], center_xy[1], scale_w, scale_h, 1]

    def __len__(self):
        return self.image_df.shape[0]


def collate_fn(batch):
    tbatch = []
    for i in batch:
        if i[0] != None:
            tbatch.append(i)
    return default_collate(tbatch)

def get_data_loader(data_path, image_path ,yolo_train=False, image_size=448, batch_size=8, n_worker=1):
    ### it should return train and val set, train and val split is done randomly with evenly distributed data
    """
    
    return TrainDataLoader, ValDataLoader, class_weights #ImageClassification task.
    """
    if yolo_train:
        raise NotImplementedError()
    
    train_ds, val_ds = ImageDataset(
        data_path=data_path,
        image_path=image_path,
        fold='train',image_size=image_size
    ), ImageDataset(
        data_path=data_path,
        image_path=image_path,
        fold='val',image_size=image_size
    )

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True,num_workers=n_worker, collate_fn=lambda x:collate_fn(x), pin_memory=False), \
        DataLoader(val_ds, batch_size=batch_size, shuffle=True,num_workers=n_worker, collate_fn=lambda x:collate_fn(x), pin_memory=False), train_ds.class_weights