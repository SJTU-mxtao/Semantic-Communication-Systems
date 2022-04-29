The semantic extraction part of the proposed method in "Deep Learning-Enabled Semantic Communication Systems with Task-Unaware Transmitter and Dynamic Data"

## Notes
The folder is for image classification task with the MNIST and CIFAR10 datasets. The image segmentation task with the PASCAL-VOC dataset is in the sub-folder [VOC](./VOC)

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
$ python MNIST.py
```

#### 2) For the CIFAR10 dataset
```bash
$ python CIFAR.py
```

## Some Results
With $\lambda = 0.1$: ![alt text](image_recover_combing/mnist_train_15_0.600000_lambda_1.000000.jpg)

