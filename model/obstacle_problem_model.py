import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import pdb
import torch

class NormalResNetModel(BaseModel):
    def __init__(self, hidden_size,input_size = 2):
        super().__init__()
        h = hidden_size
        self.linear1 = nn.Linear(input_size,h)
        self.linear2 = nn.Linear(h, h)
        self.linear3 = nn.Linear(h, h)
        self.linear4 = nn.Linear(h, h)
        self.linear5 = nn.Linear(h,1)
    def forward(self, x):
        #只有一个残差块
        tmp = x
        tmp2 = F.relu(self.linear1(tmp))**3
        tmp = F.relu(self.linear2(tmp2))**3
        tmp2 = F.relu(self.linear3(tmp)) ** 3
        tmp = tmp + F.relu(self.linear4(tmp2)) ** 3
        output = self.linear5(tmp)
        return output