#!/usr/bin/env python
# encoding: utf-8
"""
@author: hongwei zhang
@contact: zhanghwei@sjtu.edu.cn
@file: MNIST.py
@time: 2021/12/29 21:36
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
import pdb

raw_dim = 28 * 28  # shape of the raw image

def data_transform(x):
    x = np.array(x, dtype='float32') / 255
    # x = (x - 0.5) / 0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x


def data_inv_transform(x):
    """
    :param x:
    :return:
    """
    # recover_data = x * 0.5 + 0.5
    recover_data = recover_data * 255
    recover_data = recover_data.reshape((28, 28))
    recover_data = recover_data.detach().numpy()
    return recover_data


# load data
trainset = mnist.MNIST('./dataset/mnist', train=True, transform=data_transform, download=True)
testset = mnist.MNIST('./dataset/mnist', train=False, transform=data_transform, download=True)
train_data = DataLoader(trainset, batch_size=64, shuffle=True)
test_data = DataLoader(testset, batch_size=128, shuffle=False)


for rate in range(3):
    compression_rate = min((rate + 1) * 0.1, 1)
    channel = int(compression_rate * raw_dim)

    lambda1 = 0.4 - compression_rate * 0.2
    lambda2 = 0.6 + compression_rate * 0.2

    class MLP(nn.Module):
        # coders
        def __init__(self):
            super(MLP, self).__init__()
            self.fc1_1 = nn.Linear(28 * 28, 1024)
            self.fc1_2 = nn.Linear(1024, channel)
            self.fc2_1 = nn.Linear(channel, 1024)
            self.fc2_2 = nn.Linear(1024, 28 * 28)

        def forward(self, x):
            # x = F.relu(self.fc1(x))
            # encoder
            x = x.view(-1, 28 * 28)
            x = F.relu(self.fc1_1(x))
            x = F.relu(self.fc1_2(x))

            # # scale and quantize (only test)
            # x = x.detach().cpu()
            # x_max = torch.max(x)
            # x_tmp = copy.deepcopy(torch.div(x, x_max))
            # # quantize 
            # x_tmp = copy.deepcopy(torch.mul(x_tmp, 256))
            # x_tmp = copy.deepcopy(x_tmp.clone().type(torch.int))
            # x_tmp = copy.deepcopy(x_tmp.clone().type(torch.float32))
            # x_tmp = copy.deepcopy(torch.div(x_tmp, 256))
            # x = copy.deepcopy(torch.mul(x_tmp, x_max))

            # add noise
            x_np = x.detach().cpu().numpy()
            # x_np = x.numpy()
            out_square = np.square(x_np)
            aver = np.sum(out_square) / np.size(out_square)
            # snr = 3  # dB
            snr = 10  # dB
            aver_noise = aver / 10 ** (snr / 10)
            noise = np.random.random(size=x_np.shape) * np.sqrt(aver_noise)
            noise = torch.from_numpy(noise).cuda().to(torch.float32)

            # x_np = x_np + noise
            # x = torch.from_numpy(x_np)
            # x = x.to(torch.float32)

            x = torch.add(x, noise)

            # decoder
            x = F.relu(self.fc2_1(x))
            x = F.relu(self.fc2_2(x))
            return x


    class MLP_MNIST(nn.Module):
        # classifier
        def __init__(self):
            super(MLP_MNIST, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 500)
            self.fc2 = nn.Linear(500, 250)
            self.fc3 = nn.Linear(250, 125)
            self.fc4 = nn.Linear(125, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x


    mlp_encoder = MLP().cuda()
    mlp_mnist = MLP_MNIST().cuda()

    # loss function
    def criterion(x_in, y_in, raw_in):
        out_tmp1 = nn.CrossEntropyLoss()
        out_tmp2 = nn.MSELoss()
        z_in = mlp_mnist(x_in)
        mse_in = out_tmp2(x_in, raw_in)

        # print('loss1:', out_tmp1(z_in, y_in))
        # print('loss2:', mse_in)
        # pdb.set_trace()

        # loss_channel = lambda1 * out_tmp1(z_in, y_in) / 20 + lambda2 * mse_in
        loss_channel = lambda2 * mse_in
        return loss_channel

    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    psnr_all = []
    psnr = None

    # load the MNIST classifier
    mlp_mnist.load_state_dict(torch.load('saved_model/MLP_MNIST.pkl'))

    # load the MNIST coder
    model_path = "./saved_model/MLP_MNIST_coder_" + str(compression_rate) + ".pkl"
    pre_model_exist = os.path.isfile(model_path)  # if the pre-trained model exists

    if pre_model_exist:
        print('load model parameters ...')
        mlp_encoder.load_state_dict(torch.load(model_path))
        
    else:
        print('No Well-Trained Model!')

    model_dict = mlp_mnist.state_dict()

    print('Training Start')
    print('Compression Rate:', compression_rate)
    epoch_len = 150
    out = None
    for e in range(epoch_len):
        torch.cuda.empty_cache()

        # SGD or Adam
        if epoch_len < 80:
            optimizer = torch.optim.Adam(mlp_encoder.parameters(), 1e-3)
        elif epoch_len < 120:
            optimizer = torch.optim.Adam(mlp_encoder.parameters(), 1e-4)
        else:
            optimizer = torch.optim.Adam(mlp_encoder.parameters(), 1e-5)

        train_loss = 0
        train_acc = 0
        psnr_aver = 0
        counter = 0
        mlp_encoder.train()
        for im, label in train_data:
            counter += 1
            # im = Variable(im).cuda()
            # label = Variable(label).cuda()
            im = im.cuda()
            label = label.cuda()

            # forward
            out = mlp_encoder(im)
            out_mnist = mlp_mnist(out)

            loss = criterion(out, label, im)
            cr1 = nn.MSELoss()
            mse = cr1(out, im)
            out_np = out.detach().cpu().numpy()
            psnr = 10 * (np.log(1. / mse.item()) / np.log(10))
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

        losses.append(train_loss / counter)
        acces.append(train_acc / counter)
        psnr_all.append(psnr_aver/ counter)

        eval_loss = 0
        eval_acc = 0
        counter = 0
        mlp_encoder.eval()
        for im, label in test_data:
            # image_recover = data_inv_transform(im[0])
            # pil_img = Image.fromarray(np.uint8(image_recover))
            # pil_img.show()

            counter += 1
            # im = Variable(im).cuda()
            # label = Variable(label).cuda()
            im = im.cuda()
            label = label.cuda()

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

        eval_losses.append(eval_loss / counter)
        eval_acces.append(eval_acc / counter)
        print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}, PSNR: {:.6f}'
              .format(e, train_loss / len(train_data), train_acc / len(train_data),
                      eval_loss / len(test_data), eval_acc / len(test_data), psnr))

        # save model and results
        torch.save(mlp_encoder.state_dict(), ('saved_model/MLP_MNIST_coder_%f.pkl' % compression_rate))

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

        # # save the recovered images
        # for ii in range(len(out)):
        #     image_recover = data_inv_transform(out[ii])
        #     pil_img = Image.fromarray(np.uint8(image_recover))
        #     pil_img.save("image_recover_combing/mnist/mnist_train_%d_%f.jpg" % (ii, compression_rate))



