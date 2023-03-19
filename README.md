# yolo-v1
paper https://arxiv.org/pdf/1506.02640.pdf

![](https://miro.medium.com/v2/resize:fit:735/1*9nikM2b0u-m67SJpQXftKA.png)

## Model Architecture
![model](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-24_at_12.22.30_PM.png)

## dataset
  coco dataset : https://cocodataset.org/

## Todo
- [x] complete dataloader.
- [ ] add LR schedular.
- [x] train classifier
- [ ] train yolov1
- [ ] Add metrics (mAP, mAR).


#### Learning
- With crossentropyloss, never use softmax on logits, nllloss(neg log likelihood loss) need log_softmax on logits.
- For image normalization, always use standard mean and std RGB values mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225).