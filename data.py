import pandas as pd
import numpy as np
import sys
import os
import json

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision

class YoloDataset(Dataset):
    def __init__(self, data_path, image_path, image_size=448, s=7, b=2, n_class=15):
        super().__init__()
        self.data_path = data_path
        self.image_path = image_path
        self.image_size = image_size
        self.s = s
        self.b = b
        self.n_class = n_class

        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.image_df = data['images']
        self.annotation_df = data['annotations']
        self.cat_df = data['categories']

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size))
        ])

        self.target2indx = {} # TODO [write target2index dict]

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
            
            target[ij[0], ij[1], self.target2indx[row['name']]] = 1

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
    def __init__(self) -> None:
        super().__init__()
    
    def __getitem__(self, index):
        return
    
    def __len__(self):
        returngit a