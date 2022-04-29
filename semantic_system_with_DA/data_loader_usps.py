#!/usr/bin/env python
# encoding: utf-8
"""
@author: hongwei zhang
@contact: zhanghwei@sjtu.edu.cn
@file: data_loader_usps.py
@time: 2022/3/6 19:16
"""
import torch
from torchvision import datasets
from torchvision import transforms


def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""

    transform = transforms.Compose([
        transforms.Scale(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Scale(config.image_size),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    mnist = datasets.MNIST(root=config.mnist_path, download=True, transform=transform_mnist)
    usps = datasets.USPS(root=config.mnist_path, download=True, transform=transform_mnist)

    usps_loader = torch.utils.data.DataLoader(dataset=usps,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)
    return usps_loader, mnist_loader
