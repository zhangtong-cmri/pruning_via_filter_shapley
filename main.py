import argparse
import os
import sys
sys.path.append('/root/codes/flops-counter.pytorch-master')
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import math
import resnet
import data

from datetime import datetime
from torchsummary import summary

from tools import *
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--is_train', action='store_true',
                    help='If training')
parser.add_argument('--test_code', action='store_true',
                    help='Test code')
parser.add_argument('--no_flip', action='store_true',
                    help='No flip')
parser.add_argument('--dir_data', default='./Dataset',
                    help='dataset directory')
parser.add_argument('--n_threads', type=int, default=10,
                    help='number of threads for data loading')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--model', metavar='MODEL', default='resnet',
                    help='model name')
parser.add_argument('--depth', default=20, type=int, metavar='N',
                    help='model depth')
parser.add_argument('--pruned_model_dir', default='./',
                    help='pruned model directory')

parser.add_argument('--origin', action='store_true',
                    help='If origin')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--gammas',default=[0.2,0.2,0.2],type=float,metavar='N',
                    help="change lr")
parser.add_argument('--schedule',default=[60,120,160],type=int,metavar='N',
                    help="the epoch of change lr")




def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch <40):
            lr = 0.01
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
#     print("lr:",lr)
    return lr



def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    
    for i, (inputs, target) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda(async=True)
        input_var = Variable(inputs.type(args.type), volatile=not training)
        target_var = Variable(target)

        # Compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        if type(output) is list:
            output = output[0]

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))
        
    return losses.avg, top1.avg, top5.avg, inputs.type(args.type)


def train(data_loader, model, criterion, epoch, optimizer):
    # Switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    # Switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)

if __name__ == '__main__':
    global args, best_prec1
    best_prec1 = 0.
    args = parser.parse_args()
    
    save_fold_name = [
        args.model, str(args.depth), args.dataset, 'BS%d'%args.batch_size
    ]
    if args.origin:
        save_fold_name.insert(0, 'Origin')
        
    if args.model == 'resnet':
        if args.depth == 20:
            network = resnet.resnet20()
        if args.depth == 32:
            network = resnet.resnet32()
        if args.depth == 44:
            network = resnet.resnet44()
        if args.depth == 56:
            network = resnet.resnet56()
        if args.depth == 110:
            network = resnet.resnet110()
 
    if not args.origin:
        print('Pruning the model in %s' %args.pruned_model_dir)
        check_point = torch.load(args.pruned_model_dir + "model_best.pth.tar")
        network.load_state_dict(check_point['state_dict'])
        codebook_index_list = np.load(args.pruned_model_dir + "codebook.npy", allow_pickle=True).tolist()
        m_l = []
        b_l = []
        
        for i in network.modules():
            if isinstance(i, nn.Conv2d):
                m_l.append(i)
            if isinstance(i, nn.BatchNorm2d):
                b_l.append(i)
                
        for i in range(len(codebook_index_list)):
            cn_layer_out= m_l[2*i+1]
            bn_layer=b_l[2*i+1]
            cn_layer_in=m_l[2*i+2]
            get_new_conv_out(cn_layer_out,0, codebook_index_list[i])
            get_new_norm(bn_layer, codebook_index_list[i])
            get_new_conv_in(cn_layer_in, codebook_index_list[i])
           
    if args.test_code:
        if args.dataset == 'cifar10':
            summary(network.cuda(),(3,32,32))
        elif args.dataset == 'imagenet':
            summary(network.cuda(),(3,224,224))
        raise Exception(1)
        
    if args.dataset == 'cifar10':
        loader_train, loader_test = data.get_loader_cifar10(args)
        
    save_fold_name.append(datetime.now().strftime('%Y-%m-%d_%H-%M'))
    save_fold_name = '_'.join(save_fold_name)    
    save_path = os.path.join(args.results_dir, save_fold_name)
    os.makedirs(save_path)
    
    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')
    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)
    
    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None
    
    logging.info("creating model %s", args.model)
    
    criterion = nn.CrossEntropyLoss()
    criterion.type(args.type)
    network.type(args.type)
    
    optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=True)
    
    
    if args.origin:
        regime = {
                0: {'optimizer': args.optimizer, 'lr': 1e-2,
                    'weight_decay': 5e-4, 'momentum': 0.9},
                40: {'lr': 1e-1},
                60: {'lr': 2e-2},
                120: {'lr': 4e-3},
                160: {'lr': 8e-4},
         } 
    else:
        regime = {
                0: {'optimizer': args.optimizer, 'lr': 1e-2,
                    'weight_decay': 5e-4, 'momentum': 0.9},
                40: {'lr': 5e-3},
                60: {'lr': 25e-4},
                120: {'lr': 625e-5}
         }     
    
    for epoch in range(args.start_epoch, args.epochs):
        optimizer = adjust_optimizer(optimizer, epoch, regime)
#         gammas = args.gammas
#         schedule = args.schedule
#         adjust_learning_rate(optimizer,epoch,gammas,schedule)     
        train_loss, train_prec1, train_prec5, tmp_x = train(
            loader_train, network, criterion, epoch, optimizer)
        
        val_loss, val_prec1, val_prec5, _ = validate(
            loader_test, network, criterion, epoch)
    
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        
        if args.origin & is_best:
            conv_list = []
            bn_list = []
            res = []
            hook_list = []
            codebook_index_list ={}
            
            for i in network.modules():
                if isinstance(i, nn.Conv2d):
                    conv_list.append(i)
                if isinstance(i, nn.BatchNorm2d):
                    bn_list.append(i)
            
            def get_features_hook(self, input, output):
                res.append(output)    
            
            for i,j in enumerate(conv_list):
                if i % 2 == 1:
                    m = conv_list[i]
                    hook_list.append(m.register_forward_hook(get_features_hook))
            
            _ = network(tmp_x)
            
            for i in hook_list:
                i.remove()
            
            for i in range(len(res)):
                feature1 = feature_extract(res[i])
                feature2 = filter_shapely(feature1)
                feature3 = get_small_value_filter_shapely(feature2,0.6)
                codebook_index_list[i]  = feature3
                
            np.save(save_path + '/codebook.npy', codebook_index_list)
        print("lr:",regime[0]['lr'])
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'state_dict': network.state_dict(),
            'best_prec1': best_prec1,
            'regime': regime
        }, is_best, path=save_path)
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))
        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                    train_error5=100 - train_prec5, val_error5=100 - val_prec5)
        results.save()
        
        
        
#     print(save_fold_name)
    
