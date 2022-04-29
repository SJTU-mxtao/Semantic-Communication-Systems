The semantic extraction part of the proposed method in "Deep Learning-Enabled Semantic Communication Systems with Task-Unaware Transmitter and Dynamic Data"

## Notes
The folder is for image classification task with the MNIST and CIFAR10 datasets. The image segmentation task with the PASCAL-VOC dataset is in the sub-folder [VOC](./VOC).

## Download Pretrained Models
All pretrained models: [Dropbox](https://www.dropbox.com/sh/w3z9z8lqpi8b2w7/AAB0vkl4F5vy6HdIhmRCTKHSa?dl=0), [Tencent Weiyun](https://share.weiyun.com/qqx78Pv5)


## Quick Start
### Train the Semantic Extraction Part
```bash
$ python main_semantic_encoding_diff_compre.py --model deeplabv3plus_mobilenet --gpu_id 0 --year 2012_aug --crop_val --lr 0.005 --crop_size 513 --batch_size 8 --output_stride 16
```

### Save the Visualized Results
```bash
$ python main_semantic_encoding_save_figure.py --model deeplabv3plus_mobilenet --gpu_id 0 --year 2012_aug --crop_val --lr 0.005 --crop_size 513 --batch_size 8 --output_stride 16
```

## Acknowledgement
This part is based on [DeepLabv3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)

