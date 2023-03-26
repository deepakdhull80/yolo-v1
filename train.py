import os
import logging

from tqdm import tqdm
import torch
import torchmetrics

from yolo.model import ImageClassifier, Yolo
from yolo.data import get_data_loader
from yolo.loss import YoloLoss
import config

### LOGGER
# Create a custom logger
logger = logging.getLogger(__name__)
# Create handlers
f_handler = logging.FileHandler(f"{config.chkpt_dir}/train.logs", mode='w')
c_handler = logging.StreamHandler()

f_handler.setLevel(logging.INFO)
c_handler.setLevel(logging.INFO)
# Create formatters and add it to handlers
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)

c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)

# Add handlers to the logger
logger.addHandler(f_handler)
logger.addHandler(c_handler)

# model
device = torch.device(config.device)

model = Yolo(config.input_channel, config.blocks, config.bottle_neck_feature_size, n_class=config.n_classes) if config.yolo_training_enable else \
    ImageClassifier(config.input_channel, config.blocks, config.bottle_neck_feature_size, n_class=config.n_classes)


# load weights
if not os.path.exists(config.chkpt_dir):
    os.makedirs(config.chkpt_dir)

if os.path.exists(config.classifier_model_save_path):
    state_dict = torch.load(config.classifier_model_save_path)
    r = model.load_state_dict(state_dict)
    # print(f"model weights loaded: {config.classifier_model_save_path}, status: {r}")
    logger.info(f"model weights loaded: {config.classifier_model_save_path}, status: {r}")

if config.yolo_training_enable and os.path.exists(f"{config.chkpt_dir}/classifier.pt"):
    cp_state_dict = torch.load(f"{config.chkpt_dir}/classifier.pt",map_location=device)
    state_dict = model.state_dict()
    for k in state_dict:
        if k in cp_state_dict:
            state_dict[k] = cp_state_dict[k]
    
    f = model.load_state_dict(state_dict)
    print("yolo base-model load status",f)

model = model.to(device)

# Data
train_dl, val_dl, c_weigh = get_data_loader(
    data_path=config.data_base_path,
    image_path=config.image_base_path,
    yolo_train=config.yolo_training_enable,
    image_size=config.image_size,
    batch_size=config.batch_size,
    n_worker=config.n_worker,
    s=config.yolo_patches,
    b=config.yolo_bounding_box
)

loss_fn = YoloLoss(config.s, config.b, config.n_class, config.lambda_coord, config.noobj) if config.yolo_training_enable else \
    torch.nn.CrossEntropyLoss(weight=c_weigh.to(device),reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
val_loss = 1e5
train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=c_weigh.shape[0]).to(device)
val_metric = torchmetrics.Accuracy(task="multiclass", num_classes=c_weigh.shape[0]).to(device)

for epoch in range(config.epochs):
    ## train part
    model = model.train()
    # print(f"EPOCH {epoch+1}")
    logging.info(f"EPOCH {epoch+1}")
    iter = tqdm(train_dl, total=len(train_dl))
    total_train_loss = 0
    for idx, batch in enumerate(iter):
        optimizer.zero_grad()

        images, targets = batch[0].to(device), batch[1].to(device)
        y_h = model(images)
        train_loss = loss_fn(y_h, targets)

        train_loss.backward()
        optimizer.step()
        _loss = train_loss.detach().item()
        total_train_loss+=_loss
        with torch.no_grad():
            auc = train_metric(y_h.softmax(dim=-1), targets)
            total_auc = train_metric.compute()

        iter.set_description(f"loss: {_loss:.2f} total_loss: {total_train_loss/(idx+1):.2f}, auc:{auc:.2f} ,total_auc:{total_auc:.2f}")
    
    ## val part
    model = model.eval()
    iter = tqdm(val_dl, total=len(val_dl))
    total_val_loss = 0
    # print(f"EPOCH {epoch+1} -validation step")
    logging.info(f"EPOCH {epoch+1} -validation step")
    for idx, batch in enumerate(iter):
        
        images, targets = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            y_h = model(images)
        
        _val_loss = loss_fn(y_h, targets)
        _loss = _val_loss.item()
        total_val_loss+=_loss
        with torch.no_grad():
            auc = val_metric(y_h.softmax(dim=-1), targets)
            total_auc = val_metric.compute()

        iter.set_description(f"loss: {_loss:.2f} total_loss: {total_val_loss/(idx+1):.2f}, auc:{auc:.2f} ,total_auc:{total_auc:.2f}")
    ## model save
    if val_loss > total_val_loss:
        val_loss = total_val_loss
        torch.save(model.state_dict(),config.classifier_model_save_path)
        # print(f"Model saved, {config.classifier_model_save_path}")
        logging.info(f"Model saved, {config.classifier_model_save_path}")
    
    train_metric.reset()
    val_metric.reset()