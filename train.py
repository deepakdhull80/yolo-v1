import pandas as pd
from tqdm import tqdm
import json

import torch
import torchvision

from yolo.model import ImageClassifier, Yolo
from yolo.data import ImageDataset, YoloDataset
import config
from yolo.loss import YoloLoss

# model
model = Yolo(config.input_channel, config.blocks, config.bottle_neck_feature_size, n_class=config.n_classes) if config.yolo_training_enable else \
    ImageClassifier(config.input_channel, config.blocks, config.bottle_neck_feature_size, n_class=config.n_classes)
print(model)

# Data

anno_df = pd.read_json(f"{config.data_base_path}/final_data.json")
t2i = {
    name:idx for idx, name in enumerate(anno_df['supercategory'].unique())
}
i2t = {
    idx:name for idx, name in enumerate(anno_df['supercategory'].unique())
}
with open("data/i2t.json", "w") as f:
    json.dump(i2t, f)

print(t2i)
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

def collate_fn(data):
    len_batch = len(data) # original data length
    data = list(filter (lambda x:(x[0] is not None)&(x[0][0]!=3), data)) # filter out all the Nones
    # if len_batch > len(batch): # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
    #     diff = len_batch - len(batch)
    #     for i in range(diff):
    #         batch = batch + batch[:diff]
    return torch.utils.data.dataloader.default_collate(data)

train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size = config.batch_size,
    shuffle=True
)

val_dl = torch.utils.data.DataLoader(
    val_ds, batch_size = config.batch_size,
)

loss_fn = YoloLoss(config.s,config.b,config.n_class,config.lambda_coord, config.noobj) if config.yolo_training_enable else \
    torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
val_loss = 1e5

for epoch in range(config.epochs):
    ## train part
    model = model.train()
    print(f"EPOCH {epoch+1}")
    iter = tqdm(train_dl, total=len(train_dl))
    total_train_loss = 0
    for idx, batch in enumerate(iter):
        optimizer.zero_grad()

        images, targets = batch[0], batch[1]
        y_h = model(images)
        train_loss = loss_fn(y_h, targets)

        train_loss.backward()
        optimizer.step()
        _loss = train_loss.detach().item()
        total_train_loss+=_loss
        iter.set_description(f"loss: {_loss:.2f} total_loss: {total_train_loss/(idx+1):.2f}")

    ## val part
    model = model.eval()
    iter = tqdm(val_dl, total=len(val_dl))
    total_val_loss = 0
    print(f"EPOCH {epoch+1} -validation step")
    for idx, batch in enumerate(iter):
        
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            y_h = model(images)
        
        val_loss = loss_fn(y_h, targets)
        _loss = val_loss.item()
        total_val_loss+=_loss
        iter.set_description(f"loss: {_loss:.2f} total_loss: {total_val_loss/(idx+1):.2f}")
    ## model save