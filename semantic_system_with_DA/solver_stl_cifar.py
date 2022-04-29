#!/usr/bin/env python
# encoding: utf-8
"""
@author: hongwei zhang
@contact: zhanghwei@sjtu.edu.cn
@file: solver_stl_cifar.py
@time: 2022/3/8 16:01
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
from model_stl_cifar import G12, G21
from model_stl_cifar import D1, D2
import scipy.misc
import imageio
import pandas as pd


def data_tf(x):
    x = x.resize((96, 96), 2)
    x = np.array(x, dtype='float32') / 255
    # print('shape of x:', np.shape(x))
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x


def conv_relu(in_channels, out_channels, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.BatchNorm2d(out_channels, eps=1e-3),
        nn.ReLU(True)
    )
    return layer


class inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(inception, self).__init__()
        # the first line
        self.branch1x1 = conv_relu(in_channel, out1_1, 1)

        # the second line
        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)
        )

        # the thrid line
        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)
        )

        # the fourth line
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1)
        )

    def forward(self, x):
        # forward
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output


class googlenet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(googlenet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Sequential(
            conv_relu(in_channel, out_channels=64, kernel=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2)
        )
        self.block2 = nn.Sequential(
            conv_relu(64, 64, kernel=1),
            conv_relu(64, 192, kernel=3, padding=1),
            nn.MaxPool2d(3, 2)
        )
        self.block3 = nn.Sequential(
            inception(192, 64, 96, 128, 16, 32, 32),
            inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2)
        )
        self.block4 = nn.Sequential(
            inception(480, 192, 96, 208, 16, 48, 64),
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2)
        )
        self.block5 = nn.Sequential(
            inception(832, 256, 160, 320, 32, 128, 128),
            inception(832, 384, 182, 384, 48, 128, 128),
            nn.AvgPool2d(2)
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))

        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


class RED_CNN(nn.Module):
    def __init__(self, out_ch=16):
        super(RED_CNN, self).__init__()
        channel = 2
        self.conv1 = nn.Conv2d(3, out_ch, kernel_size=channel, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 3, kernel_size=channel, stride=1, padding=0)

        # self.relu = nn.ReLU()

    def forward(self, x):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # encoder
        out = self.conv1(x)
        out = self.conv2(out)
        # out = self.conv3(out)

        # scale and quantize
        out = out.detach().cpu()
        out_max = torch.max(out)
        out_tmp = copy.deepcopy(torch.div(out, out_max))

        # quantize
        out_tmp = copy.deepcopy(torch.mul(out_tmp, 256))
        out_tmp = copy.deepcopy(out_tmp.clone().type(torch.int))
        out_tmp = copy.deepcopy(out_tmp.clone().type(torch.float32))
        out_tmp = copy.deepcopy(torch.div(out_tmp, 256))

        out = copy.deepcopy(torch.mul(out_tmp, out_max))

        out_tmp = out.detach().cpu().numpy()
        out_square = np.square(out_tmp)
        aver = np.sum(out_square) / np.size(out_square)

        snr = 3  # dB
        aver_noise = aver / 10 ** (snr / 10)
        noise = torch.randn(size=out.shape) * np.sqrt(aver_noise)
        noise = noise.to(device)

        out = out + noise
        # out = torch.from_numpy(out)
        # out = out.to(torch.float32)
        # out = out.to(device)

        # print('out_after:', out.shape)

        # decoder
        # print('out_4:', out.shape)
        # out = self.tconv3(out)
        out = self.tconv4(out)
        out = self.tconv5(out)
        # print('out_5:', out.shape)
        # out += residual_1
        # out = self.relu(out)
        # print('shape of out:', out.size())
        return out


class Solver(object):
    def __init__(self, config, stl_loader, cifar_loader):
        self.stl_loader = stl_loader
        self.cifar_loader = cifar_loader
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

        for iii in range(2):
            acc_all_np = []
            compression_rate = (iii + 8) * 0.1
            channel = max(np.sqrt(32 * (1 - compression_rate) / 2), 1)
            channel = int(channel)
            print('compression rate:', compression_rate)
            flag = 0

            class RED_CNN(nn.Module):
                def __init__(self, out_ch=16):
                    super(RED_CNN, self).__init__()
                    # channel = 2
                    self.conv1 = nn.Conv2d(3, out_ch, kernel_size=channel, stride=1, padding=0)
                    self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

                    self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
                    self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
                    self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
                    self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)

                    self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
                    self.unpool = nn.MaxUnpool2d(2, stride=2)
                    self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
                    self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
                    self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
                    self.tconv5 = nn.ConvTranspose2d(out_ch, 3, kernel_size=channel, stride=1, padding=0)

                    # self.relu = nn.ReLU()

                def forward(self, x):
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    # encoder
                    out = self.conv1(x)
                    out = self.conv2(out)
                    # out = self.conv3(out)

                    out = out.detach().cpu()
                    out_max = torch.max(out)
                    out_tmp = copy.deepcopy(torch.mul(out, out_max))
                    out_tmp = copy.deepcopy(out_tmp.clone().type(torch.int))
                    out_tmp = copy.deepcopy(out_tmp.clone().type(torch.float32))
                    out = copy.deepcopy(torch.div(out_tmp, out_max))

                    out_tmp = out.detach().cpu().numpy()
                    out_square = np.square(out_tmp)
                    aver = np.sum(out_square) / np.size(out_square)

                    snr = 3  # dB
                    aver_noise = aver / 10 ** (snr / 10)
                    noise = torch.randn(size=out.shape) * np.sqrt(aver_noise)
                    noise = noise.to(device)

                    out = out + noise
                    # out = torch.from_numpy(out)
                    # out = out.to(torch.float32)
                    # out = out.to(device)

                    # print('out_after:', out.shape)

                    # decoder
                    # print('out_4:', out.shape)
                    # out = self.tconv3(out)
                    out = self.tconv4(out)
                    out = self.tconv5(out)
                    # print('out_5:', out.shape)
                    # out += residual_1
                    # out = self.relu(out)
                    # print('shape of out:', out.size())
                    return out

            mlp_encoder = RED_CNN()
            mlp_encoder.load_state_dict(torch.load('mlp_encoder_cifar-lambda-0.80-compre-%.2f.pkl' % compression_rate))
            mlp_encoder = mlp_encoder.to(device)
            classifier = googlenet(3, 10)
            classifier.load_state_dict(torch.load('google_net.pkl'))
            classifier.to(device)

            # print('type:', type(self.cifar_loader))
            stl_iter = iter(self.stl_loader)
            cifar_iter = iter(self.cifar_loader)
            iter_per_epoch = min(len(stl_iter), len(cifar_iter))

            # fixed cifar and stl for sampling
            # print('stl_iter.next()[0]', stl_iter.next()[0].size())
            # print('cifar_iter.next()[0]', cifar_iter.next()[0].size())
            fixed_stl = self.to_var(stl_iter.next()[0])  # iterator(可迭代对象的下一个值)
            fixed_cifar = self.to_var(cifar_iter.next()[0])

            # loss if use_labels = True
            criterion = nn.CrossEntropyLoss()
            # criterion = nn.BCELoss()

            for step in range(self.train_iters + 1):
                # reset data_iter for each epoch
                if (step + 1) % iter_per_epoch == 0:
                    cifar_iter = iter(self.cifar_loader)
                    stl_iter = iter(self.stl_loader)

                # load stl and cifar dataset
                stl, s_labels = stl_iter.next()
                stl, s_labels = self.to_var(stl), self.to_var(s_labels).long().squeeze()
                cifar, m_labels = cifar_iter.next()
                cifar, m_labels = self.to_var(cifar), self.to_var(m_labels)

                if self.use_labels:
                    cifar_fake_labels = self.to_var(
                        torch.Tensor([self.num_classes] * stl.size(0)).long())
                    stl_fake_labels = self.to_var(
                        torch.Tensor([self.num_classes] * cifar.size(0)).long())

                # ============ train D ============#

                # train with real images
                self.reset_grad()
                out = self.d1(cifar)
                if self.use_labels:
                    d1_loss = criterion(out, m_labels)
                else:
                    d1_loss = torch.mean((out - 1) ** 2)

                out = self.d2(stl)
                if self.use_labels:
                    d2_loss = criterion(out, s_labels)
                else:
                    d2_loss = torch.mean((out - 1) ** 2)

                d_cifar_loss = d1_loss
                d_stl_loss = d2_loss
                d_real_loss = d1_loss + d2_loss
                d_real_loss.backward()
                self.d_optimizer.step()

                # train with fake images
                self.reset_grad()
                fake_stl = self.g12(cifar)
                out = self.d2(fake_stl)
                if self.use_labels:
                    d2_loss = criterion(out, stl_fake_labels)
                else:
                    d2_loss = torch.mean(out ** 2)

                fake_cifar = self.g21(stl)
                out = self.d1(fake_cifar)
                if self.use_labels:
                    d1_loss = criterion(out, cifar_fake_labels)
                else:
                    d1_loss = torch.mean(out ** 2)

                d_fake_loss = d1_loss + d2_loss
                d_fake_loss.backward()
                self.d_optimizer.step()

                # ============ train G ============#

                # train cifar-stl-cifar cycle
                self.reset_grad()
                fake_stl = self.g12(cifar)
                out = self.d2(fake_stl)
                reconst_cifar = self.g21(fake_stl)
                if self.use_labels:
                    g_loss = criterion(out, m_labels)
                else:
                    g_loss = torch.mean((out - 1) ** 2)

                if self.use_reconst_loss:
                    g_loss += torch.mean((cifar - reconst_cifar) ** 2)

                g_loss.backward()
                self.g_optimizer.step()

                # train stl-cifar-stl cycle
                self.reset_grad()
                fake_cifar = self.g21(stl)

                out_encoder = mlp_encoder(fake_cifar)
                out_cifar = classifier(out_encoder)

                _, pred = out_cifar.max(1)
                num_correct = (pred == s_labels).sum().item()
                acc = num_correct / fake_cifar.shape[0]
                self.train_acc_local.append(acc)

                # if counter % 10 == 0:
                if step % 100 == 0:
                    # counter = 0
                    self.train_acc.append(np.max(self.train_acc_local))
                    self.train_acc_local = []
                # counter += 1

                out = self.d1(fake_cifar)
                reconst_stl = self.g12(fake_cifar)
                if self.use_labels:
                    g_loss = criterion(out, s_labels)
                else:
                    g_loss = torch.mean((out - 1) ** 2)

                if self.use_reconst_loss:
                    g_loss += torch.mean((stl - reconst_stl) ** 2)

                g_loss.backward()
                self.g_optimizer.step()

                # print the log info
                if (step + 1) % self.log_step == 0:
                    # print('Step [%d/%d], d_real_loss: %.4f, d_cifar_loss: %.4f, d_stl_loss: %.4f, '
                    #       'd_fake_loss: %.4f, g_loss: %.4f'
                    #       %(step+1, self.train_iters, d_real_loss.data[0], d_cifar_loss.data[0],
                    #         d_stl_loss.data[0], d_fake_loss.data[0], g_loss.data[0]))
                    # print(d_real_loss.size())
                    print('Step [%d/%d], acc: %.4f, d_real_loss: %.6f, d_mnist_loss: %.6f, d_svhn_loss: %.6f, '
                          'd_fake_loss: %.6f, g_loss: %.4f'
                          % (step + 1, self.train_iters, acc, d_real_loss.item(), d_cifar_loss.item(),
                             d_stl_loss.item(), d_fake_loss.item(), g_loss.item()))

                # save the sampled images
                if (step + 1) % self.sample_step == 0:
                    # 保存图，需要的时候再打开

                    # fake_stl = self.g12(fixed_cifar)
                    # fake_cifar = self.g21(fixed_stl)
                    #
                    # out_encoder = mlp_encoder(fake_cifar)
                    #
                    # cifar, fake_cifar = self.to_data(fixed_cifar), self.to_data(fake_cifar)
                    # stl, fake_stl = self.to_data(fixed_stl), self.to_data(fake_stl)
                    #
                    # out_encoder = self.to_data(out_encoder)
                    #
                    # merged = self.merge_images(cifar, fake_stl)
                    # path = os.path.join(self.sample_path, 'sample-%d-m-s.png' % (step + 1))
                    # # scipy.misc.imsave(path, merged)
                    # imageio.imwrite(path, merged)
                    # print('saved %s' % path)
                    #
                    # merged = self.merge_images(stl, fake_cifar)
                    # path = os.path.join(self.sample_path, 'sample-%d-s-m.png' % (step + 1))
                    # # scipy.misc.imsave(path, merged)
                    # imageio.imwrite(path, merged)
                    # print('saved %s' % path)
                    #
                    # merged = self.merge_images_encoder(stl, fake_cifar, out_encoder)
                    # path = os.path.join(self.sample_path, 'sample-%d-s-m-encoder.png' % (step + 1))
                    # # scipy.misc.imsave(path, merged)
                    # imageio.imwrite(path, merged)
                    # print('saved %s' % path)

                    acc_all_np = np.array(self.train_acc)
                    file = ('./results/acc_stl_cifar_%.2f.csv' % compression_rate)
                    data = pd.DataFrame(acc_all_np)
                    data.to_csv(file, index=False)

                if (step + 1) % 15000 == 0:
                    # save the model parameters for each epoch
                    # g12_path = os.path.join(self.model_path, 'g12-%d.pkl' % (step + 1))
                    # g21_path = os.path.join(self.model_path, 'g21-%d.pkl' % (step + 1))
                    # d1_path = os.path.join(self.model_path, 'd1-%d.pkl' % (step + 1))
                    # d2_path = os.path.join(self.model_path, 'd2-%d.pkl' % (step + 1))
                    # torch.save(self.g12.state_dict(), g12_path)
                    # torch.save(self.g21.state_dict(), g21_path)
                    # torch.save(self.d1.state_dict(), d1_path)
                    # torch.save(self.d2.state_dict(), d2_path)
                    break  # 跳出这次循环，开始下一个压缩率下的处理

