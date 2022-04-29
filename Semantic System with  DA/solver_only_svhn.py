#!/usr/bin/env python
# encoding: utf-8
"""
@author: hongwei zhang
@contact: zhanghwei@sjtu.edu.cn
@file: solver_only_svhn.py
@time: 2022/3/7 13:57
"""
import torch
import torch.nn as nn
import torchvision
import os
import pickle
import scipy.io
import numpy as np

from torch.autograd import Variable
from torch import optim
import scipy.misc
import imageio
import torch.nn.functional as F
import pandas as pd


def rgb2gray(rgb):
    rgb_np = rgb.detach().cpu().numpy()
    r, g, b = rgb_np[:, 0], rgb_np[:, 1], rgb_np[:, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray_tensor = torch.from_numpy(gray)
    # gray_tensor = gray_tensor.to(device)
    return gray_tensor


class MLP_MNIST(nn.Module):
    def __init__(self):
        super(MLP_MNIST, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 125)
        # self.fc4 = nn.Linear(125, 10)
        self.fc4 = nn.Linear(125, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Solver(object):
    def __init__(self, config, svhn_loader, mnist_loader):
        self.svhn_loader = svhn_loader
        self.mnist_loader = mnist_loader
        self.g12 = None
        self.g21 = None
        self.d1 = None
        self.d2 = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.use_reconst_loss = config.use_reconst_loss
        self.use_labels = config.use_labels
        self.num_classes = config.num_classes
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.train_iters = config.train_iters
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.build_model()
        self.train_acc = []
        self.train_acc_local = []

    def build_model(self):
        """Builds a generator and a discriminator."""

    def merge_images(self, sources, targets, k=10):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([3, row * h, row * w * 2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
            merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
        return merged.transpose(1, 2, 0)

    def merge_images_encoder(self, sources, targets, decoder, k=10):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([3, row * h, row * w * 3])
        for idx, (s, t, de) in enumerate(zip(sources, targets, decoder)):
            i = idx // row
            j = idx % row
            merged[:, i * h:(i + 1) * h, (j * 3) * h:(j * 3 + 1) * h] = s
            merged[:, i * h:(i + 1) * h, (j * 3 + 1) * h:(j * 3 + 2) * h] = t
            merged[:, i * h:(i + 1) * h, (j * 3 + 2) * h:(j * 3 + 3) * h] = de
        return merged.transpose(1, 2, 0)

    def to_var(self, x):
        """Converts numpy to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def to_data(self, x):
        """Converts variable to numpy."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()

    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def train(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: %s" % device)
        flag = 0

        acc_all_np = []
        acc_all_all = []
        for iii in range(9):
            compression_rate = (iii + 1) * 0.1
            print('compression rate:', compression_rate)

            class MLP(nn.Module):
                def __init__(self):
                    compre_rate = compression_rate
                    super(MLP, self).__init__()
                    self.fc1 = nn.Linear(28 * 28, int(28 * 28 * compre_rate))
                    self.fc2 = nn.Linear(int(28 * 28 * compre_rate), 28 * 28)

                def forward(self, x):
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    # x = F.relu(self.fc1(x))
                    x = self.fc1(x)

                    # out_np = x.detach().cpu().numpy()
                    # print(out_np)
                    #
                    # out_pd = np.array(out_np)

                    # scale and quantize
                    x = x.detach().cpu()
                    x_max = torch.max(x)
                    x_tmp = copy.deepcopy(torch.div(x, x_max))

                    # quantize
                    x_tmp = copy.deepcopy(torch.mul(x_tmp, 256))
                    x_tmp = copy.deepcopy(x_tmp.clone().type(torch.int))
                    x_tmp = copy.deepcopy(x_tmp.clone().type(torch.float32))
                    x_tmp = copy.deepcopy(torch.div(x_tmp, 256))

                    x = copy.deepcopy(torch.mul(x_tmp, x_max))

                    aver_tmp = torch.mean(x, dim=0, out=None)
                    aver = torch.mean(aver_tmp, dim=0, out=None)
                    aver = abs(aver.item())

                    snr = 3  # dB
                    aver_noise = aver / 10 ** (snr / 10)
                    noise = torch.randn(size=x.shape) * aver_noise

                    x = x + noise

                    # forward
                    x_np = x.detach().cpu().numpy()
                    out_square = np.square(x_np)
                    aver = np.sum(out_square) / np.size(out_square)

                    snr = 3  # dB
                    aver_noise = aver / 10 ** (snr / 10)
                    noise = np.random.random(size=x_np.shape) * np.sqrt(aver_noise)
                    # noise = noise.to(device)

                    x_np = x_np + noise
                    x = torch.from_numpy(x_np)
                    x = x.to(torch.float32)
                    x = x.to(device)

                    x = self.fc2(x)
                    return x

            mlp_encoder = MLP()  # coder
            mlp_mnist = MLP_MNIST()  # classifier
            mlp_encoder = mlp_encoder.to(device)
            mlp_mnist = mlp_mnist.to(device)

            mlp_encoder.load_state_dict(torch.load('MLP_MNIST'
                                                   '_encoder_combining_%.6f.pkl' % compression_rate))
            mlp_mnist.load_state_dict(torch.load('MLP_MNIST.pkl'))

            svhn_iter = iter(self.svhn_loader)
            mnist_iter = iter(self.mnist_loader)
            iter_per_epoch = min(len(svhn_iter), len(mnist_iter))

            fixed_svhn = self.to_var(svhn_iter.next()[0])
            fixed_mnist = self.to_var(mnist_iter.next()[0])

            # loss if use_labels = True
            criterion = nn.CrossEntropyLoss()
            # criterion = nn.BCELoss()

            for step in range(self.train_iters + 1):
                if flag == 1:
                    flag = 0
                    break
                # reset data_iter for each epoch
                if (step + 1) % iter_per_epoch == 0:
                    mnist_iter = iter(self.mnist_loader)
                    svhn_iter = iter(self.svhn_loader)

                # load svhn and mnist dataset
                svhn, s_labels = svhn_iter.next()
                svhn, s_labels = self.to_var(svhn), self.to_var(s_labels).long().squeeze()
                # mnist, m_labels = mnist_iter.next()
                # mnist, m_labels = self.to_var(mnist), self.to_var(m_labels)
                svhn = rgb2gray(svhn)
                svhn = svhn.to(device)
                # print(svhn.size())

                if self.use_labels:
                    svhn_fake_labels = self.to_var(
                        torch.Tensor([self.num_classes] * svhn.size(0)).long())

                xx = svhn.detach().cpu().numpy()
                xxx = int(np.size(xx))
                if xxx != 50176:
                    # print('There is an error in shape of data, the true batch size:', xxx)
                    continue
                svhn_reshape = svhn.view(64, -1)
                svhn_reshape = svhn_reshape.to(device)
                out_encoder = mlp_encoder(svhn_reshape)
                out_encoder_reshape = torch.reshape(out_encoder, (64, 1, 28, 28))
                # print('max of out:', torch.max(out_encoder_reshape))
                out_mnist = mlp_mnist(out_encoder)

                _, pred = out_mnist.max(1)
                num_correct = (pred == s_labels).sum().item()
                acc = num_correct / svhn.shape[0]
                self.train_acc_local.append(acc)
                # if counter % 10 == 0:
                if step % 100 == 0:
                    # counter = 0
                    self.train_acc.append(np.max(self.train_acc_local))
                    self.train_acc_local = []
                # counter += 1

                # print the log info
                if (step + 1) % self.log_step == 0:
                    print('Step [%d/%d], acc: %.6f'
                          % (step + 1, self.train_iters, acc))

                if (step + 1) % (self.log_step * 10) == 0:
                    # acc_all_np.append(np.array(self.train_acc))
                    # acc_all_np = np.array(acc_all_np)
                    acc_all_all.append(self.train_acc)
                    if compression_rate > 0.8:
                        acc_all_all = np.array(acc_all_all)
                        file = './results/acc.csv'
                        data = pd.DataFrame(acc_all_all)
                        data.to_csv(file, index=False)
                    break
