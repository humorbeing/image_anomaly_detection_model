import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F


from dare_models import model_list

from loss import OptiHardTripletLoss
from dataset_loader import get_train_loader
def main(model):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = dare_R().to(device)
    model = model.to(device)
    criterion = OptiHardTripletLoss(mean_loss=False, margin=2.0, eps=1e-08).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    
    train_loader = get_train_loader()

    iter_count = 1
    while iter_count < 5:
        # train for one epoch
        loss_train, iter_count = train(train_loader, model, criterion, optimizer, iter_count)
        print(loss_train)
        


def train(train_loader, model, criterion, optimizer, iter_count):
    device = next(model.parameters()).device
    batch_size = 2
    num_sample_persons = 2
    num_sample_imgs = 3
    """Train Function"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, input_var in enumerate(train_loader):
        # adjust_lr_adam(optimizer, iter_count)
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = input_var.to(device)
        # compute output
        outs = model(input_var)
        if type(outs) is list or type(outs) is tuple:
            loss_list = []
            for out in outs:
                loss_list.append(
                    criterion(out, num_sample_persons=num_sample_persons, num_sample_imgs=num_sample_imgs)
                )
            loss = sum(loss_list)
            losses.update(loss.item(), input_var.size(0))
        else:
            loss = criterion(outs, num_sample_persons=num_sample_persons, num_sample_imgs=num_sample_imgs)
            losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        
        iter_count += 1

    return losses.avg, iter_count


def save_checkpoint(state, is_best, folder, filename='checkpoint.pth.tar'):
    filename = os.path.join(folder, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(folder, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





if __name__ == '__main__':
    for model_ in model_list:
        main(model_())
