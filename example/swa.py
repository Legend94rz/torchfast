import torch
import torchvision
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torchfast import T, F, nn, Learner, StochasticWeightAveraging, ModelCheckpoint, BaseCallback
from pathlib import Path
from functools import partial
from torch.optim import lr_scheduler

import numpy as np
import os


def get_loaders():
    cache_dir = str(Path.home())
    trainset = torchvision.datasets.MNIST(root=cache_dir, download=True, train=True, transform=torchvision.transforms.ToTensor())
    testset = torchvision.datasets.MNIST(root=cache_dir, download=True, train=False, transform=torchvision.transforms.ToTensor())
    
    train_sampler = DistributedSampler(trainset, shuffle=True)
    test_sampler = DistributedSampler(testset)
    #train_sampler = test_sampler = None
    
    trainloader = DataLoader(trainset, batch_size=256, shuffle=False, sampler=train_sampler, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, sampler=test_sampler, num_workers=4, pin_memory=True)
    
    return trainloader, testloader


# Reference:
# https://github.com/LTS4/hold-me-tight/blob/main/model_classes/mnist/lenet.py
class LeNet(nn.Module):
    def __init__(self, ch=1, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(ch, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class CosineAnnealingLR(BaseCallback):
    def __init__(self, T_max, max_epoch=10, eta_min=0, last_epoch=- 1, verbose=False):
        super().__init__()
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.max_epoch = max_epoch
        self.sch = None

    def on_fit_begin(self, **kwargs):
        super().on_fit_begin(**kwargs)
        self.sch = lr_scheduler.CosineAnnealingLR(kwargs['optimizer'], self.T_max, self.eta_min, self.last_epoch, self.verbose)

    def on_epoch_end(self, training_log, validation_log):
        if len(training_log) > self.max_epoch:
            return
        self.sch.step()


# How to use swa & ddp & amp
#  https://gist.github.com/sayakpaul/97e20e0a18a03f8c960b57a59188bd8b
#  https://github.com/hellojialee/Improved-Body-Parts/blob/master/train_distributed_SWA.py
#  https://github.com/pytorch/pytorch/issues/59363
if __name__ == "__main__":
    # bash: torchrun --nproc_per_node=5 swa.py
    local_rank = Learner.init_distributed_training(seed=0, dummy=False)
    m = Learner(LeNet(), partial(T.optim.Adam, lr=5*1e-3), nn.CrossEntropyLoss(), amp=True)
    train_dl, _ = get_loaders()
    m.fit(train_dl, 20, None, callbacks=[StochasticWeightAveraging(0.05, 'swa.pt', swa_start=10, verbose=True), CosineAnnealingLR(20)], device='cuda', verbose=local_rank==0)
    dist.barrier()
    m.load('swa.pt')
