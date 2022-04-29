#!/usr/bin/env python
# encoding: utf-8
"""
@author: hongwei zhang
@contact: zhanghwei@sjtu.edu.cn
@file: MNIST.py
@time: 2021/12/29 21:36
"""
import torch
from torchvision.datasets import mnist
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import copy
import torch
from torch import nn
import scipy
from torch.autograd import Variable
from PIL import Image

raw_dim = 28 * 28  # shape of the raw image

# for rate in range(50):
for rate in range(1):
    # compression_rate = (rate + 1) * 0.02
    compression_rate = min((rate + 10) * 0.1, 1)
    channel = int(compression_rate * raw_dim)

    lambda1 = 1 - compression_rate
    lambda2 = compression_rate

    class MLP(nn.Module):
        # coders
        def __init__(self):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(28 * 28, channel)
            self.fc2 = nn.Linear(channel, 28 * 28)

        def forward(self, x):
            # x = F.relu(self.fc1(x))
            # encoder
            x = self.fc1(x)

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

            # add noise
            x_np = x.detach().numpy()
            # x_np = x.numpy()
            out_square = np.square(x_np)
            aver = np.sum(out_square) / np.size(out_square)

            # snr = 3  # dB
            snr = 10  # dB
            aver_noise = aver / 10 ** (snr / 10)
            noise = np.random.random(size=x_np.shape) * np.sqrt(aver_noise)
            # noise = noise.to(device)

            x_np = x_np + noise
            x = torch.from_numpy(x_np)
            x = x.to(torch.float32)

            # decoder
            x = self.fc2(x)
            return x


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


    mlp_encoder = MLP()
    mlp_mnist = MLP_MNIST()

    # load the MNIST classifier
    mlp_mnist.load_state_dict(torch.load('MLP_MNIST.pkl'))

    # mlp_mnist_ini = copy.deepcopy(mlp_mnist)

    def data_transform(x):
        x = np.array(x, dtype='float32') / 255
        x = (x - 0.5) / 0.5
        # x = x / 0.5
        x = x.reshape((-1,))
        x = torch.from_numpy(x)
        return x


    def data_inv_transform(x):
        """
        :param x:
        :return:
        """
        recover_data = x * 0.5 + 0.5
        # recover_data = x * 0.5
        recover_data = recover_data * 255
        recover_data = recover_data.reshape((28, 28))
        recover_data = recover_data.detach().numpy()
        return recover_data


    # load data
    trainset = mnist.MNIST('./dataset/mnist', train=True, transform=data_transform, download=True)
    testset = mnist.MNIST('./dataset/mnist', train=False, transform=data_transform, download=True)
    train_data = DataLoader(trainset, batch_size=64, shuffle=True)
    test_data = DataLoader(testset, batch_size=128, shuffle=False)

    # loss function
    def criterion(x_in, y_in, raw_in):
        out_tmp1 = nn.CrossEntropyLoss()
        out_tmp2 = nn.MSELoss()
        z_in = mlp_mnist(x_in)
        mse_in = lambda2 * out_tmp2(x_in, raw_in)
        loss_channel = lambda1 * out_tmp1(z_in, y_in) + lambda2 * mse_in
        return loss_channel

    # SGD or Adam
    optimizer = torch.optim.SGD(mlp_encoder.parameters(), 1e-3)

    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    psnr_all = []
    psnr = None

    model_dict = mlp_mnist.state_dict()

    print('Training Start')
    print('Compression Rate:', compression_rate)
    epoch_len = 500
    out = None
    for e in range(epoch_len):
        train_loss = 0
        train_acc = 0
        psnr_aver = 0
        mlp_encoder.train()
        for im, label in train_data:
            im = Variable(im)
            label = Variable(label)

            # forward
            out = mlp_encoder(im)
            out_mnist = mlp_mnist(out)

            loss = criterion(out, label, im)
            cr1 = nn.MSELoss()
            mse = cr1(out, im)
            out_np = out.detach().numpy()
            psnr = 10 * np.log10(np.max(out_np) ** 2 / mse.detach().numpy() ** 2)
            psnr_aver += psnr

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # calculate accuracy
            _, pred = out_mnist.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / im.shape[0]
            train_acc += acc

        # mlp_mnist_diff = copy.deepcopy(mlp_mnist.state_dict())
        # counter = 0
        # print('counter', counter)
        # for kk in model_dict.keys():
        #     mlp_mnist_diff[kk] = torch.sub(mlp_mnist.state_dict()[kk], mlp_mnist_ini.state_dict()[kk])
        #     if counter == 0:
        #         print('diff of mlp_mnist', mlp_mnist_diff[kk])
        #     counter += 1

        losses.append(train_loss / len(train_data))
        acces.append(train_acc / len(train_data))
        psnr_all.append(psnr_aver/ len(train_data))

        eval_loss = 0
        eval_acc = 0
        mlp_encoder.eval()
        for im, label in test_data:
            # image_recover = data_inv_transform(im[0])
            # pil_img = Image.fromarray(np.uint8(image_recover))
            # pil_img.show()

            im = Variable(im)
            label = Variable(label)

            out = mlp_encoder(im)

            out_mnist = mlp_mnist(out)

            # if e % 4 == 0:
            #     print('decoder input:', im)
            #     print('decoder output:', out)

            loss = criterion(out, label, im)
            eval_loss += loss.item()

            _, pred = out_mnist.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / im.shape[0]
            eval_acc += acc

        eval_losses.append(eval_loss / len(test_data))
        eval_acces.append(eval_acc / len(test_data))
        print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}, PSNR: {:.6f}'
              .format(e, train_loss / len(train_data), train_acc / len(train_data),
                      eval_loss / len(test_data), eval_acc / len(test_data), psnr))

    # save model and results
    torch.save(mlp_encoder.state_dict(), ('MLP_MNIST_encoder_combining_%f.pkl' % compression_rate))

    file = ('./results/MLP_sem_MNIST/loss_semantic_combining_%f.csv' % compression_rate)
    data = pd.DataFrame(eval_losses)
    data.to_csv(file, index=False)

    file = ('./results/MLP_sem_MNIST/acc_semantic_combining_%f.csv' % compression_rate)
    data = pd.DataFrame(eval_acces)
    data.to_csv(file, index=False)

    eval_psnr = np.array(psnr_all)
    file = ('./results/MLP_sem_MNIST/psnr_semantic_combining_%f.csv' % compression_rate)
    data = pd.DataFrame(eval_psnr)
    data.to_csv(file, index=False)

    # save the recovered images
    for ii in range(len(out)):
        image_recover = data_inv_transform(out[ii])
        pil_img = Image.fromarray(np.uint8(image_recover))
        pil_img.save("image_recover_combing/mnist_train_%d_%f.jpg" % (ii, compression_rate))



