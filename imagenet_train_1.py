#!/usr/bin/python3.6

import os, os.path as osp
import datetime, logging, pprint, random, time
from typing import *

import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from easydict import EasyDict as edict

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

from imagenet_datagen import Datagen
from imagenet_logger import create_logger, AverageMeter, rmse

import torchvision.models as models
import pretrainedmodels


SEED = 1
KFOLDS = 10


timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
opt = edict()

opt.MODEL = edict()
opt.MODEL.ARCH = 'resnet50'
# opt.MODEL.ARCH = 'se_resnext50_32x4d'
opt.MODEL.PRETRAINED = True
opt.MODEL.INPUT_SIZE = 224 # crop size

opt.EXPERIMENT = edict()
opt.EXPERIMENT.CODENAME = '2B'
opt.EXPERIMENT.TASK = 'finetune'
opt.EXPERIMENT.DIR = osp.join("../images", opt.EXPERIMENT.CODENAME)

opt.LOG = edict()
opt.LOG.LOG_FILE = osp.join(opt.EXPERIMENT.DIR, 'log_{}.txt'.format(opt.EXPERIMENT.TASK))

opt.TRAIN = edict()
opt.TRAIN.BATCH_SIZE = 32
opt.TRAIN.SHUFFLE = True
opt.TRAIN.WORKERS = 12
opt.TRAIN.PRINT_FREQ = 20
opt.TRAIN.SEED = None
opt.TRAIN.LEARNING_RATE = 1e-4
opt.TRAIN.LR_GAMMA = 0.5
opt.TRAIN.EPOCHS = 3
opt.TRAIN.SAVE_FREQ = 1
opt.TRAIN.RESUME = None
# opt.TRAIN.IMAGE_DIR = "../input/train_jpg"
# opt.TRAIN.CSV = "../input/train.csv"

opt.NUM_FEATURES = 8
opt.VALIDATION_SIZE = 0.1

if opt.TRAIN.SEED is None:
    opt.TRAIN.SEED = int(time.time())


def save_checkpoint(state: Any, filename: str = 'checkpoint.pk') -> None:
    torch.save(state, osp.join(opt.EXPERIMENT.DIR, filename))
    logger.info('A snapshot was saved to {}.'.format(filename))

def train(train_loader: Any, model: Any, criterion: Any, optimizer: Any,
          epoch: int, relu: Any, last_fc: Any, train_losses: List[float],
          train_top1s: List[float]) -> None:
    logger.info('Epoch {}'.format(epoch))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)

        # compute output
        output = model(input)

        output = relu(output)
        output = last_fc(output)
        output = relu(output)

        loss = criterion(output, target)

        # measure RMSE and record loss
        err = rmse(output.data, target)
        losses.update(loss.data.item(), input.size(0))
        top1.update(err, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.TRAIN.PRINT_FREQ == 0:
            logger.info('[{1}/{2}]\t'
                        'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'RMSE {top1.val:.4f} ({top1.avg:.4f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1))

    train_losses.append(losses.avg)
    train_top1s.append(top1.avg)

def validate(val_loader: Datagen, model: Any, criterion: Any,
             relu: Any, last_fc: Any, test_losses: List[float],
             test_top1s: List[float]) -> float:
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)

        with torch.no_grad():
            # compute output
            output = model(input)

            output = relu(output)
            output = last_fc(output)
            output = relu(output)

            loss = criterion(output, target)

            # measure RMSE and record loss
            err = rmse(output.data, target)
            losses.update(loss.data.item(), input.size(0))
            top1.update(err, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % opt.TRAIN.PRINT_FREQ == 0:
            logger.info('test: [{0}/{1}]\t'
                        'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'RMSE {top1.val:.4f} ({top1.avg:.4f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

    logger.info(' * RMSE {top1.avg:.4f}'.format(top1=top1))

    test_losses.append(losses.avg)
    test_top1s.append(top1.avg)

    return top1.avg


def train_one_fold(train_images: Any, train_scores: Any, val_images: Any,
                   val_scores: Any, fold: int) -> None:
    logger.info('searching images')

    logger.info('loading train dataset')
    train_dataset = Datagen(opt.TRAIN.IMAGE_DIR, train_images, train_scores,
                                transform=transform_train)

    logger.info('loading test dataset')
    val_dataset = Datagen(opt.TRAIN.IMAGE_DIR, val_images, val_scores, transform=transform_val)

    logger.info('{} images are used to train'.format(len(train_dataset)))
    logger.info('{} images are used to val'.format(len(val_dataset)))


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.TRAIN.BATCH_SIZE, shuffle=opt.TRAIN.SHUFFLE, num_workers=opt.TRAIN.WORKERS)

    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.TRAIN.BATCH_SIZE, shuffle=False, num_workers=opt.TRAIN.WORKERS)


    # create model
    logger.info("=> using pre-trained model '{}'".format(opt.MODEL.ARCH ))
    if opt.MODEL.ARCH.startswith('resnet'):
        model = models.__dict__[opt.MODEL.ARCH](pretrained=True)
    else:
        model = pretrainedmodels.__dict__[opt.MODEL.ARCH](pretrained='imagenet')


    if opt.MODEL.ARCH.startswith('resnet'):
        assert(opt.MODEL.INPUT_SIZE % 32 == 0)
        model.avgpool = nn.AvgPool2d(opt.MODEL.INPUT_SIZE // 32, stride=1)
        model.fc = nn.Linear(model.fc.in_features, opt.NUM_FEATURES)
        model = torch.nn.DataParallel(model).cuda()
    elif opt.MODEL.ARCH.startswith('se'):
        assert(opt.MODEL.INPUT_SIZE % 32 == 0)
        model.avgpool = nn.AvgPool2d(opt.MODEL.INPUT_SIZE // 32, stride=1)
        model.last_linear = nn.Linear(model.last_linear.in_features, opt.NUM_FEATURES)
        model = torch.nn.DataParallel(model).cuda()
    elif opt.MODEL.ARCH.startswith('se'):
        assert(opt.MODEL.INPUT_SIZE % 32 == 0)
        model.avgpool = nn.AvgPool2d(opt.MODEL.INPUT_SIZE // 32, stride=1)
        model.last_linear = nn.Linear(model.last_linear.in_features, opt.NUM_FEATURES)
        model = torch.nn.DataParallel(model).cuda()
    else:
        raise NotImplementedError


    optimizer = optim.Adam(model.module.parameters(), opt.TRAIN.LEARNING_RATE)
    lr_scheduler = ExponentialLR(optimizer, gamma=opt.TRAIN.LR_GAMMA)

    if opt.TRAIN.RESUME is None:
        last_epoch = 0
        logger.info("Training will start from Epoch {}".format(last_epoch+1))
    else:
        last_checkpoint = torch.load(opt.TRAIN.RESUME)
        assert(last_checkpoint['arch']==opt.MODEL.ARCH)
        model.module.load_state_dict(last_checkpoint['state_dict'])
        optimizer.load_state_dict(last_checkpoint['optimizer'])
        logger.info("Checkpoint '{}' was loaded.".format(opt.TRAIN.RESUME))

        last_epoch = last_checkpoint['epoch']
        logger.info("Training will be resumed from Epoch {}".format(last_checkpoint['epoch']))


    criterion = nn.MSELoss()
    last_fc = nn.Linear(in_features=opt.NUM_FEATURES, out_features=1).cuda()
    relu = nn.ReLU(inplace=True).cuda()

    best_err = 1.0
    best_epoch = 0

    train_losses: List[float] = []
    train_top1s: List[float] = []
    test_losses: List[float] = []
    test_top1s: List[float] = []


    for epoch in range(last_epoch+1, opt.TRAIN.EPOCHS+1):
        logger.info('-'*50)
        lr_scheduler.step(epoch)
        logger.info('lr: {}'.format(lr_scheduler.get_lr()))
        train(train_loader, model, criterion, optimizer, epoch,
              relu, last_fc, train_losses, train_top1s)
        err = validate(test_loader, model, criterion,
                       relu, last_fc, test_losses, test_top1s)
        is_best = err < best_err
        if is_best:
            best_epoch = epoch
            best_err = err

        if epoch % opt.TRAIN.SAVE_FREQ == 0:
            save_checkpoint({
                'epoch': epoch,
                'arch': opt.MODEL.ARCH,
                'state_dict': model.module.state_dict(),
                'best_err': best_err,
                'err': err,
                'optimizer' : optimizer.state_dict(),
            }, '{}_[{}]_{:.04f}_fold{}.pk'.format(opt.MODEL.ARCH, epoch, err, fold))

        if is_best:
            save_checkpoint({
                'epoch': epoch,
                'arch': opt.MODEL.ARCH,
                'state_dict': model.module.state_dict(),
                'best_err': best_err,
                'err': err,
                'optimizer' : optimizer.state_dict(),
            }, 'best_model_fold{}.pk'.format(fold))

    logger.info('best RMSE: {:.04f}'.format(best_err))
    #best_checkpoint_path = osp.join(opt.EXPERIMENT.DIR, 'best_model.pk')
    #logger.info("Loading parameters from the best checkpoint '{}',".format(best_checkpoint_path))
    #checkpoint = torch.load(best_checkpoint_path)
    #logger.info("which has a single crop err {:.02f}%.".format(checkpoint['err']))
    #model.load_state_dict(checkpoint['state_dict'])

    best_epoch = np.argmin(test_losses)
    best_loss = test_losses[best_epoch]
    plt.figure(0)
    x = np.arange(last_epoch+1, opt.TRAIN.EPOCHS+1)
    plt.plot(x, train_losses, '-+')
    plt.plot(x, test_losses, '-+')
    plt.scatter(best_epoch+1, best_loss, c='C1', marker='^', s=80)
    # plt.ylim(ymin=0, ymax=5)
    plt.grid(linestyle=':')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.title('Loss over epoch')
    plt.savefig(osp.join(opt.EXPERIMENT.DIR, 'loss_curves.png'))

    best_epoch = np.argmin(test_top1s)
    best_top1 = test_top1s[best_epoch]
    plt.figure(1)
    plt.plot(x, train_top1s, '-+')
    plt.plot(x, test_top1s, '-+')
    plt.scatter(best_epoch+1, best_top1, c='C1', marker='^', s=80)
    # plt.ylim(ymin=0, ymax=100)
    plt.grid(linestyle=':')
    plt.xlabel('epoch')
    plt.ylabel('err')
    plt.title('err over epoch')
    plt.savefig(osp.join(opt.EXPERIMENT.DIR, 'accuracy_curves.png'))

if __name__ == "__main__":
    cudnn.benchmark = True

    random.seed(opt.TRAIN.SEED)
    torch.manual_seed(opt.TRAIN.SEED)
    torch.cuda.manual_seed(opt.TRAIN.SEED)

    if not osp.exists(opt.EXPERIMENT.DIR):
        os.makedirs(opt.EXPERIMENT.DIR)

    logger = create_logger(opt.LOG.LOG_FILE)
    logger.info('\n\nOptions:')
    logger.info(pprint.pformat(opt))

    msg = 'Use time as random seed: {}'.format(opt.TRAIN.SEED)
    logger.info(msg)

    # data-loader of training set
    transform_train = transforms.Compose([
        transforms.Resize((opt.MODEL.INPUT_SIZE)), #Smaller edge
        transforms.RandomCrop(opt.MODEL.INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                              std = [ 0.229, 0.224, 0.225 ]),
    ])

    # data-loader of testing set
    transform_val = transforms.Compose([
        transforms.Resize((opt.MODEL.INPUT_SIZE)),
        transforms.CenterCrop(opt.MODEL.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                              std = [ 0.229, 0.224, 0.225 ]),
    ])

    logger.info('reading CSV')
    train_dataset = pd.read_csv(opt.TRAIN.CSV, index_col = "item_id")
    logger.info('shape of dataset: {}'.format(train_dataset.shape))

    x, y = train_dataset["image"].values, train_dataset["deal_probability"].values
    is_missing = pd.isnull(x)
    x = x[is_missing == False]
    y = y[is_missing == False]

    print("dataset after filtering of NA:")
    print(x)
    print(y)

    ntrain, ntest = x.shape[0], y.shape[0]
    kf = KFold(n_splits=KFOLDS, shuffle=False, random_state=SEED)
    test_pred = np.empty((KFOLDS, ntest))
    rmse_history = np.zeros(KFOLDS)

    for k, (train_idx, val_idx) in enumerate(kf.split(x)):
        logger.info("fold {}".format(k))
        logger.info("train_idx {}".format(train_idx))
        logger.info("val_idx {}".format(val_idx))

        train_images = x[train_idx]
        train_scores = y[train_idx]
        val_images = x[val_idx]
        val_scores = y[val_idx]

        train_one_fold(train_images, train_scores, val_images, val_scores, k)