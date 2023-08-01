from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import scipy.io
import os
from sklearn import manifold, datasets
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
from loss import OriTripletLoss
from random_erasing import RandomErasing
from inner_id import IDA_classifier
from proxy_mse import proxyMSE
import hdf5storage

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='sysu_baselinest.t', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=10, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='base', type=str, metavar='m', help='method type: base or agw')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--margin_cc', default=0.1, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=10, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=10, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--dist-type', default='l2', type=str, help='type of distance')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--part_num', default=12, type=int, help='number of attention map')
parser.add_argument('--w_sas', default=0.5, type=float, help='weight of Cross-Center Loss')
parser.add_argument('--w_hc', default=2.0, type=float, help='weight of Cross-Center Loss')
parser.add_argument('--train_mode', default='AST', type=str, help='weight of Cross-Center Loss')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/home/tan/data/sysu/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = '/home/tan/data/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset + '_{}_p{}_n{}_lr_{}_weight_{}_{}'.format(args.train_mode, args.num_pos, args.batch_size, args.lr,
                                                        args.w_sas, args.w_hc, args.part_num)

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)

print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_map = 0  # best test map
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train_list = [
    transforms.ToPILImage(),
    transforms.ToTensor(),
    normalize,
]
transform_train = transforms.Compose(transform_train_list)
print(transform_train_list)

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal=test_mode[0])
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal=test_mode[1])

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = 395#len(np.unique(trainset.train_color_label))
nquery = len(trainset.train_color_label)
ngall = len(trainset.train_thermal_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')

net = embed_net(n_class, args.part_num, arch=args.arch)
net = torch.nn.DataParallel(net).to(device)
add_net = IDA_classifier(args.part_num, n_class * 2)
add_net = torch.nn.DataParallel(add_net).to(device)
cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss()
loader_batch = args.batch_size * args.num_pos
criterion_tri = OriTripletLoss(batch_size=loader_batch, margin=args.margin)
criterion_cc = proxyMSE()

criterion_id.to(device)
criterion_tri.to(device)
criterion_cc.to(device)

if args.optim == 'sgd':
    ignored_params = list(map(id, net.module.classifier.parameters())) \

base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

optimizer = optim.SGD([
    {'params': base_params, 'lr': 0.1 * args.lr},
    {'params': net.module.classifier.parameters(), 'lr': args.lr}],
    weight_decay=5e-4, momentum=0.868, nesterov=True)

optimizer_add = optim.SGD(add_net.parameters(), lr=args.lr ,weight_decay=5e-4, momentum=0.9, nesterov=True)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 40:
        lr = args.lr
    elif epoch >= 40 and epoch < 80:
        lr = args.lr * 0.1
    elif epoch >= 80:
        lr = args.lr * 0.01
    
    optimizer_add.param_groups[0]['lr'] = lr
    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def plot_embedding(X, y, z, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8), dpi=100)
    cx, cy = [], []

    r = []
    for i in range(X.shape[0]):
        cx.append(X[i, 0])
        cy.append(X[i, 1])
        if z[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], s=160, color=color[y[i]], marker='^', linewidths=0, edgecolors='#000000' ,alpha=0.8)
        else:
            plt.scatter(X[i, 0], X[i, 1], s=200, color=color[y[i]], marker='.', linewidths=0, edgecolors='#000000' ,alpha=0.8)

color = ['deepskyblue', 'plum', 'tomato', 'steelblue', 'gold', 'silver', 'sienna', 'lightpink', 'seagreen', 'cornflowerblue']

def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    cc_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0
    query_feat = np.zeros((3000, args.part_num * 2048)) 
    gall_feat = np.zeros((3000, args.part_num * 2048))
    query_label = np.zeros((3000)) 
    gall_label = np.zeros((3000)) 
    ptr = 0 
    # switch to train mode
    net.eval()
    end = time.time()
    with torch.no_grad():
      for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):
          labels = torch.cat((label1, label2), 0)
          input1 = Variable(input1.cuda())
          input2 = Variable(input2.cuda())
          batch_num = input1.size(0)
          data_time.update(time.time() - end)
          feat = net(input1, input2)
          feat1, feat2 = torch.chunk(feat, 2, )
          query_feat[ptr:ptr + batch_num, :] = feat2.detach().cpu().numpy()
          query_label[ptr:ptr + batch_num] = label2.detach().cpu().numpy()
          gall_feat[ptr:ptr + batch_num, :] = feat1.detach().cpu().numpy()
          gall_label[ptr:ptr + batch_num] = label1.detach().cpu().numpy()
          ptr = ptr + batch_num
          if ptr==6000:
              result = {'gallery_f':gall_feat,'gallery_label':gall_label, 'query_f':query_feat,'query_label':query_label}
              break
      query_feature = torch.FloatTensor(result['query_f'])
      query_label = result['query_label']
      gallery_feature = torch.FloatTensor(result['gallery_f'])
      gallery_label = result['gallery_label']
      
      query_feature = query_feature.cuda()
      gallery_feature = gallery_feature.cuda()
      print(len(query_label), len(gallery_label))
      intra = []
      inter = []
      mscore = torch.matmul(query_feature, gallery_feature.t()).cpu().numpy()
      
      for i in range(len(query_label)):
          for j in range(len(gallery_label)):
              score = mscore[i,j]
              if query_label[i] == gallery_label[j]:
                  intra.append(score)
              else:
                  inter.append(score)
        
      print(len(intra), len(inter), len(intra) + len(inter))            
      
      plt.rc('font',family='Times New Roman') 
      
      fig, ax = plt.subplots()
      b = np.linspace(0.3, 1.35, num=400)
      
      ax.hist(intra, b, histtype="stepfilled", alpha=0.5, color = 'deepskyblue', density=True, label='Intra-class')
      ax.hist(inter, b, histtype="stepfilled", alpha=0.5, color = 'lightpink', density=True, label='Inter-class')
      
      ax.set_title('Scores by group and gender')
      ax.set_xlabel('Feature Distance')
      ax.set_ylabel('Frequency')
      ax.legend()
      
      
      fig.savefig('scatter.svg',dpi=600,format='svg')
      plt.show()
              

    
        
# training
print('==> Start Training...')
print('==> Preparing Data Loader...')
# identity sampler
epoch=0
sampler = IdentitySampler(trainset.train_color_label, \
                          trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                          epoch)

trainset.cIndex = sampler.index1  # color index
trainset.tIndex = sampler.index2  # thermal index
print(epoch)
print(trainset.cIndex)
print(trainset.tIndex)

loader_batch = args.batch_size * args.num_pos

trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                              sampler=sampler, num_workers=args.workers, drop_last=True)
# training
train(epoch)