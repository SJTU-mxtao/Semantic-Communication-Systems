#!/usr/bin/env python
# encoding: utf-8
"""
@author: hongwei zhang
@contact: zhanghwei@sjtu.edu.cn
@file: main_semantic_encoding_diff_compre.py
@time: 2021/12/31 22:23
"""
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from torch import cuda

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn.metrics as skm
import copy

warnings.filterwarnings("ignore")


def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
        network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=800,
                        help="epoch number (default: 0.8k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=None,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    parser.add_argument("--alpha", type=float, default=None,
                        help="parameter in loss function")
    parser.add_argument("--pretrain_epoch", type=int, default=None,
                        help='epochs of the pretraining stage')

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # # Visdom options
    # parser.add_argument("--enable_vis", action='store_true', default=False,
    #                     help="use visdom for visualization")
    # parser.add_argument("--vis_port", type=str, default='13570',
    #                     help='port for visdom')
    # parser.add_argument("--vis_env", type=str, default='main',
    #                     help='env for visdom')
    # parser.add_argument("--vis_num_samples", type=int, default=8,
    #                     help='number of samples for visualization (default: 8)')
    return parser


# def PSNR(x_in, y_in):
#     x_in_np = x_in.numpy()
#     y_in_np = y_in.numpy()
#     diff_xy = x_in_np - y_in_np


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])

        # load data
        # train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
        #                             image_set='train', download=opts.download, transform=train_transform)
        #
        # val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
        #                           image_set='val', download=False, transform=val_transform)

        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=True, transform=train_transform)

        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=True, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst


def main(ee):
    def criterion_encoder(x_in, y_in, z_in, label_in, lambda0):
        out_tmp = nn.MSELoss()
        out_tmp2 = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        MSE_tmp = out_tmp(x_in, y_in)
        psnr_tmp = 10 * np.log10((torch.max(y_in)).detach().cpu().numpy() ** 2 / MSE_tmp.detach().cpu().numpy() ** 2)
        loss_channel = opts.alpha * lambda0 * out_tmp2(z_in, label_in) + (1 - lambda0) * MSE_tmp
        # print('MSE:', out_tmp2(z_in, label_in))
        return loss_channel, psnr_tmp

    def criterion_encoder_pretrain(x_in, y_in, z_in, label_in, lambda0):
        out_tmp = nn.MSELoss()
        out_tmp2 = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        MSE_tmp = out_tmp(x_in, y_in)
        psnr_tmp = 10 * np.log10((torch.max(y_in)).detach().cpu().numpy() ** 2 / MSE_tmp.detach().cpu().numpy() ** 2)
        loss_channel = (1 - lambda0) * MSE_tmp
        # print('MSE:', out_tmp2(z_in, label_in))
        return loss_channel, psnr_tmp

    class RED_CNN(nn.Module):
        def __init__(self, out_ch=96):
            super(RED_CNN, self).__init__()
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
            # # encoder
            # residual_1 = x
            # out = self.relu(self.conv1(x))
            # out = self.relu(self.conv2(out))
            # residual_2 = out
            # out = self.relu(self.conv3(out))
            # out = self.relu(self.conv4(out))
            # residual_3 = out
            # out = self.relu(self.conv5(out))
            # # decoder
            # out = self.tconv1(out)
            # out += residual_3
            # out = self.tconv2(self.relu(out))
            # out = self.tconv3(self.relu(out))
            # out += residual_2
            # out = self.tconv4(self.relu(out))
            # out = self.tconv5(self.relu(out))
            # out += residual_1
            # out = self.relu(out)

            # encoder
            residual_1 = x
            out = self.conv1(x)
            # out, index = self.pool(out)
            out = self.conv2(out)
            # residual_2 = out
            out = self.conv3(out)
            out = self.conv4(out)
            # residual_3 = out
            out = self.conv5(out)

            out = out.detach().cpu()
            out_max = torch.max(out)
            out_tmp = copy.deepcopy(torch.mul(out, out_max))
            out_tmp = copy.deepcopy(out_tmp.clone().type(torch.int))
            out_tmp = copy.deepcopy(out_tmp.clone().type(torch.float32))
            out = copy.deepcopy(torch.div(out_tmp, out_max))

            out = out.detach().cpu()
            out_max = torch.max(out)
            out_tmp = copy.deepcopy(torch.mul(out, out_max))
            out_tmp = copy.deepcopy(out_tmp.clone().type(torch.int))
            out_tmp = copy.deepcopy(out_tmp.clone().type(torch.float32))
            out = copy.deepcopy(torch.div(out_tmp, out_max))

            # print('out_ini:', out.shape)

            out_tmp = out.detach().cpu().numpy()
            out_square = np.square(out_tmp)
            aver = np.sum(out_square) / np.size(out_square)

            # aver_tmp = torch.mean(out, dim=0, out=None)
            # aver = torch.mean(aver_tmp, dim=0, out=None)
            # aver = abs(aver.item())

            # snr = 3  # dB
            snr = 10  # dB
            aver_noise = aver / 10 ** (snr / 10)
            noise = torch.randn(size=out.shape) * np.sqrt(aver_noise)
            noise = noise.to(device)

            out = out + noise

            # print('out_after:', out.shape)

            # decoder
            out = self.tconv1(out)
            # print('out_1:', out.shape)
            # out = self.unpool(out, index)
            # out += residual_3
            out = self.tconv2(out)
            # print('out_2:', out.shape)
            out = self.tconv3(out)
            # print('out_3:', out.shape)
            # out += residual_2
            out = self.tconv4(out)
            # print('out_4:', out.shape)
            out = self.tconv5(out)
            # print('out_5:', out.shape)
            # out += residual_1
            # out = self.relu(out)
            return out

    def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
        """Do validation and return specified samples"""
        metrics.reset()
        ret_samples = []
        if opts.save_val_results:
            if not os.path.exists('results'):
                os.mkdir('results')
            denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
            img_id = 0

        with torch.no_grad():
            counter = 0
            for i, (images, labels) in tqdm(enumerate(loader)):

                if counter < 16:
                    images = images.to(device, dtype=torch.float32)
                    labels = labels.to(device, dtype=torch.long)

                    # forward
                    mlp_encoder = RED_CNN().to(device)
                    # mlp_encoder.load_state_dict(torch.load('MLP_MNIST_encoder_combining.pkl'))
                    out = mlp_encoder(images)
                    out = out.to(device)

                    outputs = model(out)
                    # outputs.to(device)
                    preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                    targets = labels.cpu().numpy()

                    metrics.update(targets, preds)
                    if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                        ret_samples.append(
                            (images[0].detach().cpu().numpy(), targets[0], preds[0]))

                    if opts.save_val_results:
                        for i in range(len(images)):
                            image = images[i].detach().cpu().numpy()
                            recover = out[i].detach().cpu().numpy()
                            target = targets[i]
                            pred = preds[i]

                            image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                            recover = (denorm(recover) * 255).transpose(1, 2, 0).astype(np.uint8)
                            target = loader.dataset.decode_target(target).astype(np.uint8)
                            pred = loader.dataset.decode_target(pred).astype(np.uint8)

                            # save the recovered image
                            Image.fromarray(image).save(
                                'results/Combining_Encoder_Recover/%d_%f_image.png' % (img_id, compre_rate))
                            Image.fromarray(recover).save(
                                'results/Combining_Encoder_Recover/%d_%f_recover.png' % (img_id, compre_rate))
                            Image.fromarray(target).save(
                                'results/Combining_Encoder_Recover/%d_%f_target.png' % (img_id, compre_rate))
                            Image.fromarray(pred).save(
                                'results/Combining_Encoder_Recover/%d_%f_pred.png' % (img_id, compre_rate))
                            #
                            # fig = plt.figure()
                            # plt.imshow(image)
                            # plt.axis('off')
                            # plt.imshow(pred, alpha=0.7)
                            # ax = plt.gca()
                            # ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                            # ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                            # plt.savefig('results/Combining_Encoder_Recover/%d_overlay.png' % img_id, bbox_inches='tight',
                            #             pad_inches=0)
                            # plt.close()
                            img_id += 1
                else:
                    score = metrics.get_results()
                    return score
                counter += 1

            score = metrics.get_results()
        return score


    if ee != 100:
        compre_rate = 0.1 * (ee + 1) + 0.1

        channel = max(np.sqrt(513 * (1 - compre_rate) / 5), 1)
        channel = int(channel)

        print('compression rate:', compre_rate)
        print('Size of kernel:', channel)

        loss_add = 0
        val_score_all = []
        overall_acc_all = []
        mean_acc_all = []
        freq_acc_all = []
        class_iou_all = []
        mean_iou_all = []

        psnr_all = []
        loss_all = []
        torch.cuda.empty_cache()

        opts = get_argparser().parse_args()
        if opts.dataset.lower() == 'voc':
            opts.num_classes = 21
        elif opts.dataset.lower() == 'cityscapes':
            opts.num_classes = 19

        os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: %s" % device)

        mlp_encoder = RED_CNN().to(device)
        optimizer_encoder = torch.optim.SGD(mlp_encoder.parameters(), 1e-3)
        scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=opts.step_size, gamma=0.1)

        # Setup random seed
        torch.manual_seed(opts.random_seed)
        np.random.seed(opts.random_seed)
        random.seed(opts.random_seed)

        # Setup dataloader
        if opts.dataset == 'voc' and not opts.crop_val:
            opts.val_batch_size = 1

        train_dst, val_dst = get_dataset(opts)
        print('type of vla_dst', type(val_dst))
        train_loader = data.DataLoader(
            train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
            drop_last=True)  # drop_last=True to ignore single-image batches.
        val_loader = data.DataLoader(
            val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
        print("Dataset: %s, Train set: %d, Val set: %d" %
              (opts.dataset, len(train_dst), len(val_dst)))

        # opts = get_argparser().parse_args()
        # Set up model (all models are 'constructed at network.modeling)
        model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('Segmentation_Network.pkl'))
        model = model.to(device)

        # Set up metrics
        metrics = StreamSegMetrics(opts.num_classes)

        # Set up criterion
        # criterion = utils.get_loss(opts.loss_type)
        # if opts.loss_type == 'focal_loss':
        #     criterion = utils.FocalLoss(ignore_index=255, size_average=True)
        # elif opts.loss_type == 'cross_entropy':
        #     criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

        # Restore
        cur_itrs = 0
        cur_epochs = 0

        # utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        interval_loss = 0
        torch.cuda.empty_cache()

        while True:  # cur_itrs < opts.total_itrs:
            # =====  Train  =====
            cur_epochs += 1
            for (images, labels) in train_loader:
                cur_itrs += 1

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                # forward
                optimizer_encoder.zero_grad()

                out = mlp_encoder(images)

                # out_np = out.detach().cpu().numpy()
                # print(np.shape(out_np))

                out_model = model(out)
                loss_encoder, psnr_local = criterion_encoder_pretrain(out, images, out_model, labels,
                                                             lambda0=1 - compre_rate)  # 尽可能无损
                psnr_all.append(psnr_local)
                # backward
                loss_encoder.backward()
                optimizer_encoder.step()
                # train_loss += loss.item()
                np_loss = loss_encoder.detach().cpu().numpy()

                # torch.save(mlp_encoder.state_dict(), ('MLP_MNIST_encoder_semantic_%f.pkl' % compre_rate))

                scheduler_encoder.step()

                if cur_itrs >= opts.pretrain_epoch:
                    # torch.save(mlp_encoder.state_dict(), ('MLP_MNIST_encoder_semantic_%f.pkl' % compre_rate))
                    return

    if ee != 100:
        compre_rate = 0.1 * (ee + 1) + 0.1

        channel = max(np.sqrt(513 * (1 - compre_rate) / 5), 1)
        channel = int(channel)

        print('compression rate:', compre_rate)
        print('Size of kernel:', channel)

        loss_add = 0
        val_score_all = []
        overall_acc_all = []
        mean_acc_all = []
        freq_acc_all = []
        class_iou_all = []
        mean_iou_all = []

        psnr_all = []
        loss_all = []
        torch.cuda.empty_cache()

        opts = get_argparser().parse_args()
        if opts.dataset.lower() == 'voc':
            opts.num_classes = 21
        elif opts.dataset.lower() == 'cityscapes':
            opts.num_classes = 19

        os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: %s" % device)

        mlp_encoder = RED_CNN().to(device)
        optimizer_encoder = torch.optim.SGD(mlp_encoder.parameters(), 1e-3)
        scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=opts.step_size, gamma=0.1)

        # Setup random seed
        torch.manual_seed(opts.random_seed)
        np.random.seed(opts.random_seed)
        random.seed(opts.random_seed)

        # Setup dataloader
        if opts.dataset == 'voc' and not opts.crop_val:
            opts.val_batch_size = 1

        train_dst, val_dst = get_dataset(opts)
        print('type of vla_dst', type(val_dst))
        train_loader = data.DataLoader(
            train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
            drop_last=True)  # drop_last=True to ignore single-image batches.
        val_loader = data.DataLoader(
            val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
        print("Dataset: %s, Train set: %d, Val set: %d" %
              (opts.dataset, len(train_dst), len(val_dst)))

        # opts = get_argparser().parse_args()
        # Set up model (all models are 'constructed at network.modeling)
        model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('Segmentation_Network.pkl'))
        model = model.to(device)

        # Set up metrics
        metrics = StreamSegMetrics(opts.num_classes)

        # Set up criterion
        # criterion = utils.get_loss(opts.loss_type)
        # if opts.loss_type == 'focal_loss':
        #     criterion = utils.FocalLoss(ignore_index=255, size_average=True)
        # elif opts.loss_type == 'cross_entropy':
        #     criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

        # Restore
        cur_itrs = 0
        cur_epochs = 0

        # utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        interval_loss = 0
        torch.cuda.empty_cache()

        while True:  # cur_itrs < opts.total_itrs:
            # =====  Train  =====
            cur_epochs += 1
            for (images, labels) in train_loader:
                cur_itrs += 1

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                # forward
                optimizer_encoder.zero_grad()

                out = mlp_encoder(images)

                # out_np = out.detach().cpu().numpy()
                # print(np.shape(out_np))

                out_model = model(out)
                loss_encoder, psnr_local = criterion_encoder(out, images, out_model, labels,
                                                             lambda0=1 - compre_rate)  # 尽可能无损
                psnr_all.append(psnr_local)
                # backward
                loss_encoder.backward()
                optimizer_encoder.step()
                # train_loss += loss.item()
                np_loss = loss_encoder.detach().cpu().numpy()

                loss_add += np_loss

                if cur_itrs % 4 == 0:
                    loss_add = loss_add / 4
                    loss_all.append(loss_add)
                    print("Epoch %d, Iteration %d, Loss_combining=%f, PSNR=%f" % (
                        cur_epochs, cur_itrs, loss_add, psnr_local))

                    loss_add = 0

                if cur_itrs % 20 == 0:

                    model.eval()
                    val_score = validate(
                        opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
                    print(metrics.to_str(val_score))
                    # print(val_score.keys())
                    # print(val_score.values())
                    overall_acc_all.append(val_score['Overall Acc'])
                    mean_acc_all.append(val_score['Mean Acc'])
                    freq_acc_all.append(val_score['FreqW Acc'])
                    mean_iou_all.append(val_score['Mean IoU'])
                    class_iou_all.append(val_score['Class IoU'])

                if cur_itrs % int(10 * 4) == 0:
                    loss_all_np = np.array(loss_all)
                    val_score_all_np = np.array(val_score_all)
                    psnr_all_np = np.array(psnr_all)
                    overall_acc_all_np = np.array(overall_acc_all)
                    mean_acc_all_np = np.array(mean_acc_all)
                    freq_acc_all_np = np.array(freq_acc_all)
                    mean_iou_all_np = np.array(mean_iou_all)
                    class_iou_all_np = np.array(class_iou_all)

                    # save results
                    file = (
                                './results_data/Semantic_Encoding/loss_semantic_%f.csv' % compre_rate)
                    data_save = pd.DataFrame(loss_all_np)
                    data_save.to_csv(file, index=False)

                    file = (
                                './results_data/Semantic_Encoding/acc_semantic_%f.csv' % compre_rate)
                    data_save = pd.DataFrame(val_score_all_np)
                    data_save.to_csv(file, index=False)

                    file = (
                                './results_data/Semantic_Encoding/psnr_semantic_%f.csv' % compre_rate)
                    data_save = pd.DataFrame(psnr_all_np)
                    data_save.to_csv(file, index=False)

                    file = (
                                './results_data/Semantic_Encoding/overall_acc_semantic_%f.csv' % compre_rate)
                    data_save = pd.DataFrame(overall_acc_all_np)
                    data_save.to_csv(file, index=False)

                    file = (
                                './results_data/Semantic_Encoding/mean_acc_semantic_%f.csv' % compre_rate)
                    data_save = pd.DataFrame(mean_acc_all_np)
                    data_save.to_csv(file, index=False)

                    file = (
                                './results_data/Semantic_Encoding/freq_acc_semantic_%f.csv' % compre_rate)
                    data_save = pd.DataFrame(freq_acc_all_np)
                    data_save.to_csv(file, index=False)

                    file = (
                                './results_data/Semantic_Encoding/mean_iou_semantic_%f.csv' % compre_rate)
                    data_save = pd.DataFrame(mean_iou_all_np)
                    data_save.to_csv(file, index=False)

                    file = (
                                './results_data/Semantic_Encoding/class_iou_semantic_%f.csv' % compre_rate)
                    data_save = pd.DataFrame(class_iou_all_np)
                    data_save.to_csv(file, index=False)

                    torch.save(mlp_encoder.state_dict(), ('MLP_MNIST_encoder_semantic_%f.pkl' % compre_rate))

                scheduler_encoder.step()

                if cur_itrs >= 800:
                    torch.save(mlp_encoder.state_dict(), ('MLP_MNIST_encoder_semantic_%f.pkl' % compre_rate))
                    return


if __name__ == '__main__':
    for ee in range(10):
        main(ee)
