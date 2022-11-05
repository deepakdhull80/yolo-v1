
yolo_training_enable = False
#model
n_classes = 12
input_channel = 3
bottle_neck_feature_size = 4096
epochs = 150
classifier_model_save_path = "/home/deepakdhull/practice/computer_vision/object-detection/yolo-v1/checkpoints/classifier.pt"

# data
image_size = 224
data_base_path = "/home/deepakdhull/practice/computer_vision/object-detection/data"
image_base_path = "/home/deepakdhull/practice/computer_vision/object-detection/data/train2017"
batch_size = 14

blocks = [
        {
            'channels':[64],
            'kernels':[7],
            'strides':[2],
            'bn':True,
            'max_pool':2
        },
        {
            'channels':[192],
            'kernels':[3],
            'strides':[1],
            'bn':True,
            'max_pool':2
        },
        {
            'channels':[128,256,256,512],
            'kernels':[1,3,1,3],
            'strides':[1,1,1,1],
            'bn':True,
            'max_pool':2
        },
        {
            'channels':[256,512,256,512,256,512,256,512,512,1024],
            'kernels':[1,3,1,3,1,3,1,3,1,3],
            'strides':[1,1,1,1,1,1,1,1,1,1],
            'bn':True,
            'max_pool':2
        },
        {
            'channels':[512,1024,512,1024,1024,1024],
            'kernels':[1,3,1,3,3,3],
            'strides':[1,1,1,1,1,2],
            'bn':True,
            'max_pool':False
        },
        {
            'channels':[1024,1024],
            'kernels':[3,3],
            'strides':[1,1],
            'bn':True,
            'max_pool':False
        }
    ]