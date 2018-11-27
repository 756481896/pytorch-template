import torch
import pandas as pd
import numpy as np
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torch.autograd as autograd
from torch.autograd import Variable
from random import random, randint
from time import sleep
from tqdm import tqdm_notebook as tqdm
torch.set_default_tensor_type('torch.DoubleTensor')
print('hello world')


class Generator(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )

    def forward(self, input):
        return self.main(input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

ngpu =0
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
netG = Generator(ngpu).to(device)
netD = Discriminator(ngpu).to(device)
netG.apply(weights_init)
netD.apply(weights_init)
print(netD)
print(netG)

#加载一维CD方程的部分点的准确解[x,t,phi]
data = np.load('./data/CDR_1dim_100times.npy')
index = np.random.choice(len(data),size = len(data),replace=False)
data = data[index]
X = data[:,0]
T = data[:,1]
Phi = data[:,2]
#检查数据是否正确
assert np.linalg.norm(100*np.exp(-1*T)*X**2-Phi)<0.001

def pair_iter(data,batch_size = 64):
    batches = (len(data) + batch_size - 1)//batch_size
    X = data[:,0]
    T = data[:,1]
    Phi = data[:,2]
    for i in range(batches-1):
        x_batch = X[i*batch_size:(i+1)*batch_size]
        x_batch = torch.from_numpy(x_batch)
        x_batch = x_batch.to(device)
        t_batch = T[i*batch_size:(i+1)*batch_size]
        t_batch = torch.from_numpy(t_batch)
        t_batch = t_batch.to(device)
        phi_batch = Phi[i*batch_size:(i+1)*batch_size]
        phi_batch = torch.from_numpy(phi_batch)
        phi_batch = phi_batch.to(device)
        yield (x_batch,t_batch,phi_batch)

