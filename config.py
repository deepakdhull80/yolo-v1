import os
import json
# data
image_size = 448
data_base_path = "./data"
dataset_name = "val2017"
image_base_path = f"{data_base_path}/{dataset_name}"

if os.path.exists(f"{data_base_path}/sampled_categories.json"):
    d = json.load(open(f"{data_base_path}/sampled_categories.json",'r'))
    n_classes = len(list(d.keys()))

#model
device='cuda'
yolo_training_enable = False

input_channel = 3
bottle_neck_feature_size = 4096
epochs = 30
chkpt_dir = "/content/drive/MyDrive/models/yolov1"
classifier_model_save_path = f"{chkpt_dir}/yolo.pt" \
                            if yolo_training_enable \
                                else f"{chkpt_dir}/classifier.pt"

yolo_bounding_box = 2
yolo_patches = 7
n_worker = 1
lr=1e-3
batch_size = 16

### yolo model architecture
bn_enable = False
blocks = [
        {
            'channels':[16],
            'kernels':[3],
            'strides':[1],
            'bn':bn_enable,
            'max_pool':2
        },
        {
            'channels':[32],
            'kernels':[3],
            'strides':[1],
            'bn':bn_enable,
            'max_pool':2
        },
        {
            'channels':[64],
            'kernels':[3],
            'strides':[1],
            'bn':bn_enable,
            'max_pool':2
        },
        {
            'channels':[128],
            'kernels':[3],
            'strides':[1],
            'bn':bn_enable,
            'max_pool':2
        },
        {
            'channels':[256],
            'kernels':[3],
            'strides':[1],
            'bn':bn_enable,
            'max_pool':2
        }
    ]