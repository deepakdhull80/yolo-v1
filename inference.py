import torch
from yolo.model import ImageClassifier
from yolo.data import ImageDataset
import config
import os
import sys
import random
from torchvision import transforms

if __name__ == '__main__':
    model = ImageClassifier(config.input_channel, config.blocks, config.bottle_neck_feature_size, n_class=config.n_classes)
    config.classifier_model_save_path = "/content/drive/MyDrive/models/yolov1/classifiert1.pt"
    if os.path.exists(config.classifier_model_save_path):
        state_dict = torch.load(config.classifier_model_save_path, map_location='cpu')
        r = model.load_state_dict(state_dict)
        print(f"model weights loaded: {config.classifier_model_save_path}, status: {r}")

    model = model.eval()
    device = torch.device(sys.argv[1])
    train_ds = ImageDataset(
        data_path="./data",
        image_path="./data/val2017/",
        fold='train',image_size=448
    )

    image, label = train_ds.__getitem__(random.randint(0, 500))
    with torch.no_grad():
        pred = model(image.unsqueeze(0))
    print('prediction: ',pred)
    print(f"predicted label: {torch.argmax(pred, dim=1)}, actual: {label}")
    image = transforms.ToPILImage()(image[0].cpu())
    image.save("predict.png")