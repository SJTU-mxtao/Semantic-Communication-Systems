The semantic extraction part of the proposed method in "Deep Learning-Enabled Semantic Communication Systems with Task-Unaware Transmitter and Dynamic Data".

## Important Update
MLP_MNIST_model.py, MNIST.py, googlenet_train.py, and CIFAR.py have been updated on 2024.11.26.


## Other Instructions
This is an example of semantic communication using a small-sized dataset based on MLP and CNN.
**If you require a more advanced neural network framework or a system with better performance, we recommend using our [another code repository based on Swin Transformer](https://github.com/SJTU-mxtao/semantic-communication-w-codebook)**

## Notes
The folder is for image classification task with the MNIST and CIFAR10 datasets. The image segmentation task with the PASCAL-VOC dataset is in the sub-folder [VOC](./VOC).

## Quick Start
### Train the Classifier (Pragmatic Function)

#### 1) For the MNIST dataset
```bash
$ python MLP_MNIST_model.py 
```

#### 2) For the CIFAR10 dataset
```bash
$ python googlenet_train.py 
```

### Train the Semantic Extraction Part
#### 1) For the MNIST dataset
```bash
$ python MNIST.py --alpha xx --pretrain_epoch xx --random_seed xx
```

#### 2) For the CIFAR10 dataset
```bash
$ python CIFAR.py --alpha xx --pretrain_epoch xx --random_seed xx
```


## Citation

Please use the following BibTeX citation if you use this repository in your work:

```
@ARTICLE{9953099,
  author={Zhang, Hongwei and Shao, Shuo and Tao, Meixia and Bi, Xiaoyan and Letaief, Khaled B.},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={Deep Learning-Enabled Semantic Communication Systems With Task-Unaware Transmitter and Dynamic Data}, 
  year={2023},
  volume={41},
  number={1},
  pages={170-185},
  doi={10.1109/JSAC.2022.3221991}}
```

