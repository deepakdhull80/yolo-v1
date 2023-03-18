import json
import os

# data
image_size = 448
data_base_path = "./data"
dataset_name = "val2017"
image_base_path = f"{data_base_path}/{dataset_name}"


#model
device='cuda'
yolo_training_enable = False

assert os.path.exists(image_base_path), "data not found, run prepare_data.py first."
assert os.path.exists(f'{data_base_path}/sampled_categories.json'), "prepare_data.py not run properly, make sure it run without any error or warning."

category_id = json.load(open(f'{data_base_path}/sampled_categories.json'))
n_classes = len(category_id)

input_channel = 3
bottle_neck_feature_size = 4096
epochs = 30
classifier_model_save_path = "./checkpoints/yolo.pt" \
                            if yolo_training_enable \
                                else "./checkpoints/classifier.pt"

yolo_bounding_box = 2
yolo_patches = 7
n_worker = 1
lr=1e-3
batch_size = 16

### yolo model architecture
bn_enable = False
blocks = [
        {
            'channels':[64],
            'kernels':[7],
            'strides':[2],
            'bn':bn_enable,
            'max_pool':2
        },
        {
            'channels':[192],
            'kernels':[3],
            'strides':[1],
            'bn':bn_enable,
            'max_pool':2
        },
        {
            'channels':[128,256,256,512],
            'kernels':[1,3,1,3],
            'strides':[1,1,1,1],
            'bn':bn_enable,
            'max_pool':2
        },
        {
            'channels':[256,512,256,512,256,512,256,512,512,1024],
            'kernels':[1,3,1,3,1,3,1,3,1,3],
            'strides':[1,1,1,1,1,1,1,1,1,1],
            'bn':bn_enable,
            'max_pool':2
        },
        {
            'channels':[512,1024,512,1024,1024,1024],
            'kernels':[1,3,1,3,3,3],
            'strides':[1,1,1,1,1,2],
            'bn':bn_enable,
            'max_pool':False
        },
        {
            'channels':[1024,1024],
            'kernels':[3,3],
            'strides':[1,1],
            'bn':bn_enable,
            'max_pool':False
        }
    ]