The semantic extraction part of the proposed method in "Deep Learning-Enabled Semantic Communication Systems with Task-Unaware Transmitter and Dynamic Data".

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

## Some Results

![1.](http://latex.codecogs.com/svg.latex?\\lambda=0.1): ![image](./image_recover_combing/mnist_train_15_0.600000_lambda_1.000000.jpg)  

![2.](http://latex.codecogs.com/svg.latex?\\lambda=0.2): ![image](./image_recover_combing/mnist_train_15_0.600000_lambda_2.000000.jpg)  

![3.](http://latex.codecogs.com/svg.latex?\\lambda=0.3): ![image](./image_recover_combing/mnist_train_15_0.600000_lambda_3.000000.jpg)  

![4.](http://latex.codecogs.com/svg.latex?\\lambda=0.4): ![image](./image_recover_combing/mnist_train_15_0.600000_lambda_4.000000.jpg)  

![5.](http://latex.codecogs.com/svg.latex?\\lambda=0.5): ![image](./image_recover_combing/mnist_train_15_0.600000_lambda_5.000000.jpg)  

![6.](http://latex.codecogs.com/svg.latex?\\lambda=0.6): ![image](./image_recover_combing/mnist_train_15_0.600000_lambda_6.000000.jpg)  

![7.](http://latex.codecogs.com/svg.latex?\\lambda=0.7): ![image](./image_recover_combing/mnist_train_15_0.600000_lambda_7.000000.jpg)  

![8.](http://latex.codecogs.com/svg.latex?\\lambda=0.8): ![image](./image_recover_combing/mnist_train_15_0.600000_lambda_8.000000.jpg)  

![9.](http://latex.codecogs.com/svg.latex?\\lambda=0.9): ![image](./image_recover_combing/mnist_train_15_0.600000_lambda_9.000000.jpg)  

![10.](http://latex.codecogs.com/svg.latex?\\lambda=1.0): ![image](./image_recover_combing/mnist_train_15_0.600000_lambda_10.000000.jpg)  


## Citation

Please use the following BibTeX citation if you use this repository in your work:

```
@article{Deep_semantic_comm_2022,
  title={Deep Learning-Enabled Semantic Communication Systems with Task-Unaware Transmitter and Dynamic Data},
  author={Zhang, Hongwei and Shao, Shuo and Tao, Meixia and Bi, Xiaoyan and Letaief, Khaled B},
  journal={arXiv preprint arXiv:2205.00271},
  year={2022}
}
```

