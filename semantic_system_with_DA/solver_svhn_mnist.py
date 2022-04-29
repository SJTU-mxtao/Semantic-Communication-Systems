#!/usr/bin/env python
# encoding: utf-8
"""
@author: hongwei zhang
@contact: zhanghwei@sjtu.edu.cn
@file: solver_svhn_mnist.py
@time: 2022/3/6 14:31
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
from model_svhn_mnist import G12, G21
from model_svhn_mnist import D1, D2
import scipy.misc
import imageio
import torch.nn.functional as F
import pandas as pd


class MLP_MNIST(nn.Module):
    # classifier
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
        self.g12 = G12(conv_dim=self.g_conv_dim)
        self.g21 = G21(conv_dim=self.g_conv_dim)
        self.d1 = D1(conv_dim=self.d_conv_dim, use_labels=self.use_labels)
        self.d2 = D2(conv_dim=self.d_conv_dim, use_labels=self.use_labels)

        g_params = list(self.g12.parameters()) + list(self.g21.parameters())
        d_params = list(self.d1.parameters()) + list(self.d2.parameters())

        self.g_optimizer = optim.Adam(g_params, self.lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(d_params, self.lr, [self.beta1, self.beta2])

        if torch.cuda.is_available():
            self.g12.cuda()
            self.g21.cuda()
            self.d1.cuda()
            self.d2.cuda()

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

        acc_all_all = []
        for iii in range(9):
            acc_all_np = []
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

                    x = x.detach().cpu()
                    x_max = torch.max(x)
                    x_tmp = copy.deepcopy(torch.mul(x, x_max))
                    x_tmp = copy.deepcopy(x_tmp.clone().type(torch.int))
                    x_tmp = copy.deepcopy(x_tmp.clone().type(torch.float32))
                    x = copy.deepcopy(torch.div(x_tmp, x_max))

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

            # fixed mnist and svhn for sampling
            # print('svhn_iter.next()[0]', svhn_iter.next()[0].size())
            # print('mnist_iter.next()[0]', mnist_iter.next()[0].size())
            fixed_svhn = self.to_var(svhn_iter.next()[0])  # iterator(可迭代对象的下一个值)
            fixed_mnist = self.to_var(mnist_iter.next()[0])

            # loss if use_labels = True
            criterion = nn.CrossEntropyLoss()
            # criterion = nn.BCELoss()
            counter = 0

            for step in range(self.train_iters + 1):
                # reset data_iter for each epoch
                if (step + 1) % iter_per_epoch == 0:
                    mnist_iter = iter(self.mnist_loader)
                    svhn_iter = iter(self.svhn_loader)

                # load svhn and mnist dataset
                svhn, s_labels = svhn_iter.next()
                svhn, s_labels = self.to_var(svhn), self.to_var(s_labels).long().squeeze()
                mnist, m_labels = mnist_iter.next()
                mnist, m_labels = self.to_var(mnist), self.to_var(m_labels)

                if self.use_labels:
                    mnist_fake_labels = self.to_var(
                        torch.Tensor([self.num_classes] * svhn.size(0)).long())
                    svhn_fake_labels = self.to_var(
                        torch.Tensor([self.num_classes] * mnist.size(0)).long())

                # ============ train D ============#

                # train with real images
                self.reset_grad()
                out = self.d1(mnist)
                if self.use_labels:
                    d1_loss = criterion(out, m_labels)
                else:
                    d1_loss = torch.mean((out - 1) ** 2)

                out = self.d2(svhn)
                if self.use_labels:
                    d2_loss = criterion(out, s_labels)
                else:
                    d2_loss = torch.mean((out - 1) ** 2)

                d_mnist_loss = d1_loss
                d_svhn_loss = d2_loss
                d_real_loss = d1_loss + d2_loss
                d_real_loss.backward()
                self.d_optimizer.step()

                # train with fake images
                self.reset_grad()
                fake_svhn = self.g12(mnist)
                out = self.d2(fake_svhn)
                if self.use_labels:
                    d2_loss = criterion(out, svhn_fake_labels)
                else:
                    d2_loss = torch.mean(out ** 2)

                fake_mnist = self.g21(svhn)
                out = self.d1(fake_mnist)
                if self.use_labels:
                    d1_loss = criterion(out, mnist_fake_labels)
                else:
                    d1_loss = torch.mean(out ** 2)

                d_fake_loss = d1_loss + d2_loss
                d_fake_loss.backward()
                self.d_optimizer.step()

                # ============ train G ============#

                # train mnist-svhn-mnist cycle
                self.reset_grad()
                fake_svhn = self.g12(mnist)
                out = self.d2(fake_svhn)
                reconst_mnist = self.g21(fake_svhn)
                if self.use_labels:
                    g_loss = criterion(out, m_labels)
                else:
                    g_loss = torch.mean((out - 1) ** 2)

                if self.use_reconst_loss:
                    g_loss += torch.mean((mnist - reconst_mnist) ** 2)

                g_loss.backward()
                self.g_optimizer.step()

                # train svhn-mnist-svhn cycle
                self.reset_grad()
                fake_mnist = self.g21(svhn)
                out = self.d1(fake_mnist)
                reconst_svhn = self.g12(fake_mnist)

                fake_mnist_reshape = fake_mnist.view(64, -1)
                fake_mnist_reshape = fake_mnist_reshape.to(device)
                out_encoder = mlp_encoder(fake_mnist_reshape)
                out_encoder_reshape = torch.reshape(out_encoder, (64, 1, 28, 28))
                out_mnist = mlp_mnist(out_encoder)

                _, pred = out_mnist.max(1)
                num_correct = (pred == s_labels).sum().item()
                acc = num_correct / fake_mnist.shape[0]
                self.train_acc_local.append(acc)
                # if counter % 10 == 0:
                if step % 100 == 0:
                    # counter = 0
                    self.train_acc.append(np.max(self.train_acc_local))
                    self.train_acc_local = []
                # counter += 1

                if self.use_labels:
                    g_loss = criterion(out, s_labels)
                else:
                    g_loss = torch.mean((out - 1) ** 2)

                if self.use_reconst_loss:
                    g_loss += torch.mean((svhn - reconst_svhn) ** 2)

                g_loss.backward()
                self.g_optimizer.step()

                # print the log info
                if (step + 1) % self.log_step == 0:
                    # print('Step [%d/%d], d_real_loss: %.4f, d_mnist_loss: %.4f, d_svhn_loss: %.4f, '
                    #       'd_fake_loss: %.4f, g_loss: %.4f'
                    #       %(step+1, self.train_iters, d_real_loss.data[0], d_mnist_loss.data[0],
                    #         d_svhn_loss.data[0], d_fake_loss.data[0], g_loss.data[0]))
                    # print(d_real_loss.size())
                    print('Step [%d/%d], acc: %.4f, d_real_loss: %.6f, d_mnist_loss: %.6f, d_svhn_loss: %.6f, '
                          'd_fake_loss: %.6f, g_loss: %.4f'
                          % (step + 1, self.train_iters, acc, d_real_loss.item(), d_mnist_loss.item(),
                             d_svhn_loss.item(), d_fake_loss.item(), g_loss.item()))

                # save the sampled images
                if (step + 1) % self.sample_step == 0:
                    # save images

                    # fake_svhn = self.g12(fixed_mnist)
                    # fake_mnist = self.g21(fixed_svhn)
                    #
                    # fake_mnist_reshape = fake_mnist.view(64, -1)
                    # fake_mnist_reshape = fake_mnist_reshape.to(device)
                    # out_encoder = mlp_encoder(fake_mnist_reshape)
                    # out_encoder_reshape = torch.reshape(out_encoder, (64, 1, 28, 28))
                    #
                    # mnist, fake_mnist = self.to_data(fixed_mnist), self.to_data(fake_mnist)
                    # svhn, fake_svhn = self.to_data(fixed_svhn), self.to_data(fake_svhn)
                    #
                    # out_encoder_reshape_data = self.to_data(out_encoder_reshape)
                    #
                    # # mnist to svhn
                    # merged = self.merge_images(mnist, fake_svhn)
                    # path = os.path.join(self.sample_path, 'sample-%d-m-s.png' % (step + 1))
                    # # scipy.misc.imsave(path, merged)
                    # imageio.imwrite(path, merged)
                    # print('saved %s' % path)
                    #
                    # # svhn to mnist
                    # merged = self.merge_images(svhn, fake_mnist)
                    # path = os.path.join(self.sample_path, 'sample-%d-s-m.png' % (step + 1))
                    # # scipy.misc.imsave(path, merged)
                    # imageio.imwrite(path, merged)
                    # print('saved %s' % path)
                    #
                    # merged = self.merge_images_encoder(svhn, fake_mnist, out_encoder_reshape_data)
                    # path = os.path.join(self.sample_path, 'sample-%d-s-m-encoder.png' % (step + 1))
                    # # scipy.misc.imsave(path, merged)
                    # imageio.imwrite(path, merged)
                    # print('saved %s' % path)

                    acc_all_np = np.array(self.train_acc)
                    file = ('./results/acc_svhn_mnist_%.2f.csv' % compression_rate)
                    data = pd.DataFrame(acc_all_np)
                    data.to_csv(file, index=False)

                if (step + 1) % 15000 == 0:
                    # save models
                    # save the model parameters for each epoch
                    # g12_path = os.path.join(self.model_path, 'g12-%d.pkl' % (step + 1))
                    # g21_path = os.path.join(self.model_path, 'g21-%d.pkl' % (step + 1))
                    # d1_path = os.path.join(self.model_path, 'd1-%d.pkl' % (step + 1))
                    # d2_path = os.path.join(self.model_path, 'd2-%d.pkl' % (step + 1))
                    # torch.save(self.g12.state_dict(), g12_path)
                    # torch.save(self.g21.state_dict(), g21_path)
                    # torch.save(self.d1.state_dict(), d1_path)
                    # torch.save(self.d2.state_dict(), d2_path)
                    break
