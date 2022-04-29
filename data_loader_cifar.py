#!/usr/bin/env python
# encoding: utf-8
"""
@author: hongwei zhang
@contact: zhanghwei@sjtu.edu.cn
@file: data_loader_cifar.py
@time: 2022/3/8 15:56
"""
import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np


def data_tf(x):
    x = x.resize((96, 96), 2)  # shape of x: (96, 96, 3)
    x = np.array(x, dtype='float32') / 255
    # print('shape of x:', np.shape(x))
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x


def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""

    transform = transforms.Compose([
        transforms.Scale(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # cifar = datasets.CIFAR10(root=config.cifar_path, download=True, transform=transform)
    # stl = datasets.STL10(root=config.stl_path, download=True, transform=transform)

    cifar = datasets.CIFAR10(root=config.cifar_path, download=True, transform=data_tf)
    stl = datasets.STL10(root=config.stl_path, download=True, transform=data_tf)

    cifar_loader = torch.utils.data.DataLoader(dataset=cifar,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    stl_loader = torch.utils.data.DataLoader(dataset=stl,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)

    return stl_loader, cifar_loader

