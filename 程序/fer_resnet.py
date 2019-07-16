'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from fer import FER2013
from torch.autograd import Variable
from models import resnet_prune_1

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def updateBN():
    #print('!!!prune!!!\r\n')
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(opt.s*torch.sign(m.weight.data))

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='resnet18', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--bs', default=512, type=int, help='batch_size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true', help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.00001, help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH', help='refine from prune model')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', type=int, default=550, metavar='N',help='number of epochs to train (default: 250)')
parser.add_argument('--depth', default=164, type=int, help='depth of the neural network')
opt = parser.parse_args()
print('==> Options:',opt)

use_cuda = torch.cuda.is_available()
best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 80  # 150,80,50,30
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

cut_size = 44

path = os.path.join(opt.dataset + '_' + opt.model)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

trainset = FER2013(split = 'Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=1)
PublicTestset = FER2013(split = 'PublicTest', transform=transform_test)
#PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=20, shuffle=False, num_workers=1)
PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
#PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=20, shuffle=False, num_workers=1)

if opt.refine:
    print('==> Refine model ...')
    checkpoint = torch.load(opt.refine)
    #net = vgg_prune.VGG(cfg=checkpoint['cfg'])
    net = resnet_prune_1.resnet(depth=opt.depth, cfg=checkpoint['cfg'])
    net.load_state_dict(checkpoint['state_dict'])
elif opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint1 = torch.load('prune_models/fer_Pri_resnet164_prune_1.pth')
    net = resnet_prune_1.resnet(cfg=checkpoint1['cfg'])
    checkpoint2 = torch.load('prune_models/fer_Pri_resnet164_refine_1.pth')
    #net.load_state_dict(checkpoint2['state_dict'])
    #net = resnet_prune_1.resnet(depth=opt.depth)
    #assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    #checkpoint = torch.load('ori_models/fer_Pri_resnet164_0.pth')
    #checkpoint = torch.load('prune_models/fer_Pri_resnet164_preprune_0.pth')
    
    net.load_state_dict(checkpoint2['state_dict'])
    best_PublicTest_acc = checkpoint2['best_PublicTest_acc']
    best_PrivateTest_acc = checkpoint2['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = checkpoint2['best_PublicTest_acc_epoch']
    best_PrivateTest_acc_epoch = checkpoint2['best_PrivateTest_acc_epoch']
    start_epoch = checkpoint2['best_PrivateTest_acc_epoch'] + 1
else:
    # Model
    print('==> Building model...')
    #if opt.model == 'resnet18':
    #    net = resnet_prune.resnet18()
    #elif opt.model == 'resnet34':
    #    net = resnet_prune.resnet34()
    #elif opt.model == 'resnet50':
    #    net = resnet_prune.resnet50()
    #elif opt.model == 'resnet101':
    #    net = resnet_prune.resnet101()
    #elif opt.model == 'resnet152':
    #    net = resnet_prune.resnet152()
    net = resnet_prune_1.resnet(depth=opt.depth)

print(net)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay= 5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)

        if opt.sr:
            updateBN()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100. * float(correct) / total, correct, total))

    Train_acc = 100. * float(correct) / total

def PublicTest(epoch):
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    net.eval()
    PublicTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PublicTest_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PublicTest_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    # Save checkpoint.
    PublicTest_acc = 100. * float(correct) /total
    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f%%" % PublicTest_acc)
        state = {
            'state_dict': net.state_dict(),
            'best_PublicTest_acc': PublicTest_acc
        }
        if not os.path.isdir(path):
            os.mkdir(path)

        state_copy = state['state_dict'].copy()

        for key in state_copy.keys():
            if 'module' in key:
                state['state_dict'][key.replace('module.', '')] = state['state_dict'].pop(key)

        #torch.save(state, 'ori_models/fer_Pub_resnet18_0.pth')
        #torch.save(state, 'ori_models/fer_Pub_resnet101_0.pth')
        torch.save(state, 'ori_models/fer_Pub_resnet164_0.pth')

        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch

def PrivateTest(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))
    
    # Save checkpoint.
    PrivateTest_acc = 100. * float(correct) /total
    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f%%" % PrivateTest_acc)
        state = {
            'state_dict': net.state_dict(),
            'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': PrivateTest_acc,
            'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        
        state_copy = state['state_dict'].copy()

        for key in state_copy.keys():
            if 'module' in key:
                state['state_dict'][key.replace('module.', '')] = state['state_dict'].pop(key)

        #torch.save(state,'ori_models/fer_Pri_resnet18_0.pth')
        #torch.save(state,'ori_models/fer_Pri_resnet101_0.pth')
        #torch.save(state,'ori_models/fer_Pri_resnet164_0.pth')

        #torch.save(state,'prune_models/fer_Pri_resnet164_preprune_0.pth')

        torch.save(state, 'prune_models/fer_Pri_resnet164_refine_1.pth')

        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch

for epoch in range(start_epoch, opt.epochs):
    train(epoch)
    #PublicTest(epoch)
    PrivateTest(epoch)

print("best_PublicTest_acc: %0.3f%%" % best_PublicTest_acc)
print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
print("best_PrivateTest_acc: %0.3f%%" % best_PrivateTest_acc)
print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)
