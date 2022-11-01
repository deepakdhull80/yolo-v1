import pandas as pd
import numpy as np
import sys
import os
import json

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision

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


class ImageDataset(Dataset):
    def __init__(self, data_path, image_path, t2i=None, fold="train", image_size=448, n_class=15):
        super().__init__()
        self.data_path = data_path
        self.image_path = image_path
        self.image_size = image_size
        self.n_class = n_class
        self.fold = fold
        self.image_df = pd.read_json(os.path.join(data_path,f"{fold}_image_data.json"))
        self.annotation_df = pd.read_json(os.path.join(data_path,"final_data.json"))

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size))
        ])

        self.target2indx = t2i
        if not self.target2indx:
            self.target2indx = self.get_tar2indx()
        

    def get_tar2indx(self):
        return {name:idx for idx,name in enumerate(self.annotation_df['supercategory'])}


    def __getitem__(self, index):
        data = self.image_df.loc[index].to_dict()
        
        image = torchvision.io.read_image(f"{self.image_path}/{data['file_name']}")
        
        annotation = self.annotation_df[self.annotation_df['image_id'] == data['id']]
        t = []
        areas = []

        for _, row in annotation.iterrows():
            bbox = row['bbox']
            _image = self.decode(image, bbox)
            area = row['area']
            _class = row['supercategory']
            t.append([_image, _class])
            areas.append(area)
        
        idx = torch.argmax(areas)
        image, target = t[idx]
        del t, area
        target = self.target2indx(target)
        return image, target

    def decode(self, image, bbox):
        _, h, w = image.shape
        image = image[:,bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]]
        image = self.transform(image)
        image = image/ 255.0
        return image

    def __len__(self):
        return self.image_df.shape[0]