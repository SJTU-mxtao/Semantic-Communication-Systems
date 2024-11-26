#!/usr/bin/env python
# encoding: utf-8
"""
@author: hongwei zhang
@contact: zhanghwei@sjtu.edu.cn
@file: MLP_MNIST_model.py
@time: 2021/12/15 20:38
"""
# train and save the MNIST classifier
import torch
import numpy as np
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pdb


def data_transform(x):
    x = np.array(x, dtype='float32') / 255
    # x = (x - 0.5) / 0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x


# load data
trainset = mnist.MNIST('./dataset/mnist', train=True, transform=data_transform, download=True)
testset = mnist.MNIST('./dataset/mnist', train=False, transform=data_transform, download=True)
train_data = DataLoader(trainset, batch_size=64, shuffle=True)
test_data = DataLoader(testset, batch_size=128, shuffle=False)


class MLP(nn.Module):
    # classifier
    def __init__(self):
        super(MLP, self).__init__()
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


mlp = MLP()

criterion = nn.CrossEntropyLoss()

losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(10):
    # SGD or Adam
    if e < 7:
        optimizer = torch.optim.Adam(mlp.parameters(), 1e-3)
    else:
        optimizer = torch.optim.Adam(mlp.parameters(), 1e-4)

    train_loss = 0
    train_acc = 0
    mlp.train()
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        # forward
        out = mlp(im)

        loss = criterion(out, label)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))

    eval_loss = 0
    eval_acc = 0
    mlp.eval()
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = mlp(im)
        loss = criterion(out, label)

        eval_loss += loss.item()

        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_data), train_acc / len(train_data),
                  eval_loss / len(test_data), eval_acc / len(test_data)))

# save the model
torch.save(mlp.state_dict(), 'saved_model/MLP_MNIST.pkl')

# file = './results/MLP_MNIST_model/acc.csv'
# data = pd.DataFrame(eval_acces)
# data.to_csv(file, index=False)
#
# file = './results/MLP_MNIST_model/loss.csv'
# data = pd.DataFrame(eval_losses)
# data.to_csv(file, index=False)






