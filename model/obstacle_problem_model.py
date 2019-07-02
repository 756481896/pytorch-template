import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import pdb
import torch

# class ResBlock(nn.Module):
#     def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
#         super(UNetDown, self).__init__()
#         layers = [nn.Linear(in_size,out_size)]
#         if normalize:
#             layers.append(nn.InstanceNorm2d(out_size))
#         layers.append(nn.LeakyReLU(0.2))
#         if dropout:
#             layers.append(nn.Dropout(dropout))
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.model(x)

# class NormalResNetModel(BaseModel):
#     def __init__(self, hidden_size,input_size = 2):
#         super().__init__()
#         h = hidden_size
#         self.h = h
#         self.linear1 = nn.Linear(input_size,h)
#         self.linear2 = nn.Linear(h, h)
#         self.linear3 = nn.Linear(h, h)
#         self.linear4 = nn.Linear(h, h)
#         self.linear5 = nn.Linear(h,1)
#
#     def linear_layer(self):
#         return nn.Linear(self.h,self.h)
#
#     def res_block(self,tmp):
#         tmp2 = F.relu(self.linear_layer()(tmp)) ** 3
#         return tmp + F.relu(self.linear_layer()(tmp2)) ** 3
#
#     def forward(self, x):
#         #只有一个残差块
#         tmp = x
#         tmp2 = F.relu(self.linear1(tmp))**3
#         tmp = F.relu(self.linear2(tmp2))**3
#         tmp = self.res_block(tmp)
#         tmp = self.res_block(tmp)
#         tmp = self.res_block(tmp)
#         output = self.linear5(tmp)
#         return output

class NormalMultiLayersModel(BaseModel):
    def __init__(self, hidden_size,input_size = 2):
        super().__init__()
        h = hidden_size
        self.h = h
        self.linear1 = nn.Linear(input_size,h)
        self.linear2 = nn.Linear(h, h)
        self.linear3 = nn.Linear(h, h)
        self.linear4 = nn.Linear(h, h)
        # self.linear5 = nn.Linear(h, h)
        # self.linear6 = nn.Linear(h, h)
        # self.linear7 = nn.Linear(h, h)
        # self.linear8 = nn.Linear(h, h)
        self.linear9 = nn.Linear(h,1)

    def forward(self, x):
        tmp = x
        tmp = F.relu(self.linear1(tmp)) ** 3
        tmp2 = F.relu(self.linear2(tmp)) ** 3
        tmp = F.relu(self.linear3(tmp2)) ** 3
        tmp = tmp2 +F.relu(self.linear4(tmp)) ** 3
        # tmp = F.relu(self.linear5(tmp)) ** 3
        # tmp = F.relu(self.linear6(tmp)) ** 3
        # tmp = F.relu(self.linear7(tmp)) ** 3
        # tmp = F.relu(self.linear8(tmp)) ** 3
        output = self.linear9(tmp)
        return output