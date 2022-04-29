#!/usr/bin/env python
# encoding: utf-8
"""
@author: hongwei zhang
@contact: zhanghwei@sjtu.edu.cn
@file: model_stl_cifar.py
@time: 2022/3/8 16:03
"""
import torch.nn as nn
import torch.nn.functional as F


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = [nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False)]
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = [nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)]
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class G12(nn.Module):
    """Generator for transfering from mnist to svhn"""

    def __init__(self, conv_dim=64):
        super(G12, self).__init__()
        # encoding blocks
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv2_2 = conv(conv_dim * 2, conv_dim * 4, 4)

        # residual blocks
        self.conv3 = conv(conv_dim * 4, conv_dim * 4, 3, 1, 1)
        self.conv4 = conv(conv_dim * 4, conv_dim * 4, 3, 1, 1)

        # decoding blocks
        self.deconv1_1 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv2_2(out), 0.05)

        out = F.leaky_relu(self.conv3(out), 0.05)  # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)  # ( " )

        out = F.leaky_relu(self.deconv1_1(out), 0.05)
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = F.tanh(self.deconv2(out))  # (?, 3, 32, 32)
        # print('shape G12 (svhn):', out.size())
        return out


class G21(nn.Module):
    """Generator for transfering from svhn to mnist"""

    def __init__(self, conv_dim=64):
        super(G21, self).__init__()
        # encoding blocks
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv2_2 = conv(conv_dim * 2, conv_dim * 4, 4)

        # residual blocks
        self.conv3 = conv(conv_dim * 4, conv_dim * 4, 3, 1, 1)
        self.conv4 = conv(conv_dim * 4, conv_dim * 4, 3, 1, 1)

        # decoding blocks
        self.deconv1_1 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv2_2(out), 0.05)

        out = F.leaky_relu(self.conv3(out), 0.05)  # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)  # ( " )

        out = F.leaky_relu(self.deconv1_1(out), 0.05)
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = F.tanh(self.deconv2(out))  # (?, 1, 32, 32)
        # print('shape G21 (mnist):', out.size())
        return out


class D1(nn.Module):
    """Discriminator for mnist."""

    def __init__(self, conv_dim=64, use_labels=False):
        super(D1, self).__init__()
        # self.conv1 = conv(3, conv_dim, 4, bn=False)
        # self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        # self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        # n_out = 11 if use_labels else 1
        # # self.fc = conv(conv_dim*4, n_out, 4, 1, 0, False)
        # self.fc = conv(conv_dim * 4, n_out, 4, 1, 0, False)

        self.conv1 = conv(3, conv_dim, 6, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 6)
        self.conv2_2 = conv(conv_dim * 2, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 6)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim * 4, n_out, 4, 1, 0, False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv2_2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)

        out = self.fc(out).squeeze()
        # out = self.fc(out)
        return out


class D2(nn.Module):
    """Discriminator for svhn."""

    def __init__(self, conv_dim=64, use_labels=False):
        super(D2, self).__init__()
        # self.conv1 = conv(3, conv_dim, 4, bn=False)
        # self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        # self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        # n_out = 11 if use_labels else 1
        # self.fc = conv(conv_dim*4, n_out, 4, 1, 0, False)

        self.conv1 = conv(3, conv_dim, 6, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 6)
        self.conv2_2 = conv(conv_dim * 2, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 6)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim * 4, n_out, 4, 1, 0, False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv2_2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        # out = self.fc(out)
        return out
