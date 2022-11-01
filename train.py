import pandas as pd

import torch
import torchvision

from yolo.model import ImageClassifier, Yolo
from yolo.data import ImageDataset, YoloDataset
import config

# model
model = Yolo(config.input_channel, config.blocks, config.bottle_neck_feature_size, n_class=config.n_classes) if config.yolo_training_enable else \
    ImageClassifier(config.input_channel, config.blocks, config.bottle_neck_feature_size, n_class=config.n_classes)
print(model)

# Data

anno_df = pd.read_json(f"{config.data_base_path}/final_data.json")
t2i = {
    name:idx for idx, name in enumerate(anno_df['supercategory'])
}
i2t = {
    idx:name for idx, name in enumerate(anno_df['supercategory'])
}

train_ds = YoloDataset(
    data_path = config.data_base_path,
    image_path = config.image_base_path,
    t2i = t2i,
    fold = "train",
    image_size = config.image_size,
    n_class = config.n_classes

) if config.yolo_training_enable else \
    ImageDataset(
        data_path = config.data_base_path,
        image_path = config.image_base_path,
        t2i = t2i,
        fold = "train",
        image_size = config.image_size,
        n_class = config.n_classes
    )

val_ds = YoloDataset(
    data_path = config.data_base_path,
    image_path = config.image_base_path,
    t2i = t2i,
    fold = "val",
    image_size = config.image_size,
    n_class = config.n_classes

) if config.yolo_training_enable else \
    ImageDataset(
        data_path = config.data_base_path,
        image_path = config.image_base_path,
        t2i = t2i,
        fold = "val",
        image_size = config.image_size,
        n_class = config.n_classes
    )

train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size = config.batch_size,
    shuffle=True
)

val_dl = torch.utils.data.DataLoader(
    val_ds, batch_size = config.batch_size,
)

