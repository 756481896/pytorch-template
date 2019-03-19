import torch
import pdb
from torch import nn

def g_metric(output, target):
    """
    :param output: G的输出 [batch_size,1,64,64]
    :param target: G的正确值 [batch_size,1,64,64]
    :return: L2(output-target)
    """
    with torch.no_grad():
        assert output.shape == target.shape
        g_metric = torch.norm(output - target)
    return correct / len(target)

def d_metric(output, target):
    """
    判断D成功分辨出多少图的真假
    :param output: 是真的概率值,[batch_size,1,1,1]
    :param target: 全部1 或者全部0，[batch_size,1,1,1]
    :return:acc
    """
    with torch.no_grad():
        pred = output>0.5
        pred = pred.type_as(target)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()

    return correct / len(target)

def L2_metric(output, target):
    with torch.no_grad():
        loss = nn.MSELoss()
        l = loss(output, target)
    return l

def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
