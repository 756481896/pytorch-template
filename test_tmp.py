import torch
import torch.nn as nn
from torch.optim import Adam

class NN_Network(nn.Module):
    def __init__(self,in_dim,hid,out_dim):
        super(NN_Network, self).__init__()
        self.linear1 = nn.Linear(in_dim,hid)
        # self.linear2 = nn.Linear(hid,out_dim)
        self.W_x = torch.nn.Parameter(torch.randn(3, 3))
        # self.weight = torch.nn.Parameter(torch.zeros(5,5))
        # self.linear1.bias = torch.nn.Parameter(torch.ones(hid))
        # self.linear2.weight = torch.nn.Parameter(torch.zeros(in_dim,hid))
        # self.linear2.bias = torch.nn.Parameter(torch.ones(hid))

    def forward(self, input_array):
        h = self.linear1(input_array)
        # y_pred = self.linear2(h)
        return h

in_d = 5
hidn = 2
out_d = 3
net = NN_Network(in_d, hidn, out_d)
print(list(net.parameters()))