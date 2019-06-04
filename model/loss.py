import torch.nn.functional as F
from torch import nn
import numpy as np
import torch
def BCE_loss(predict,target):
    loss = nn.BCELoss()
    return loss(predict,target)

def MSE_loss(predict,target):
    loss = nn.MSELoss()
    return loss(predict,target)
# def nll_loss(output, target):
#     return F.nll_loss(output, target)

def F_loss(u_t,u_x, u_y, u_xx, u_yy):
    F = u_t - u_x - u_y - u_xx - u_yy
    loss = nn.MSELoss()
    return loss(F, torch.zeros_like(F))

def L1_loss(predict,target):
    loss = nn.L1Loss()
    return loss(predict,target)
def equation_loss(InputG,predictG):
    U = InputG[:,0]

    #calculate u_xx
    dist_x = 1/128
    U_xx = (U[:,:,2:]+U[:,:,:-2]-2*U[:,:,1:-1])/dist_x**2
    u_xx0 = U_xx[:,:,0].view(-1,128,1)
    u_xxN = U_xx[:,:,-1].view(-1,128,1)
    U_xx = torch.cat((u_xx0,U_xx,u_xxN),2)

    dist_y = dist_x
    U_yy = (U[:, 2:, :] + U[:, :-2, :] - 2 * U[:, 1:-1, :]) / dist_y ** 2
    #暂时不对边界进行额外处理
    u_yy0 = U_xx[:,0,:].view(-1,1,128)
    u_yyN = U_xx[:,-1,:].view(-1,1,128)
    U_yy = torch.cat((u_yy0, U_yy, u_yyN), 1)
    F = (U_xx + U_yy)*(-1) - predictG
    loss = nn.MSELoss()
    return loss(F,torch.zeros_like(F))

def diff_loss(Input_D):
    X = np.linspace(-np.pi, np.pi, 130)
    dist_x = (X[1] - X[0]) * 8
    T = np.linspace(0, 1, 512)
    dist_t = (T[1] - T[0]) * 2

    U = Input_D[:,0]
    alpha = Input_D[:,1]
    #用差分拟合导数u_t,倒数第二层用中心差分，最后一层用向前差分，其他用向后差分
    u_t1 = (U[:,1:-1,:] - U[:,:-2,:])/dist_t
    #0到n-2的u_t，dist_t是每个点之间的实际距离
    u_t2 = (U[:,-1,:] - U[:,-3,:])/(2*dist_t)
    #倒数第二层
    u_t2 = u_t2.view(-1,1,64)
    u_t3 = (U[:,-1,:] - U[:,-2,:])/dist_t
    #最后一层
    u_t3 = u_t3.view(-1, 1, 64)
    U_t = torch.cat((u_t1,u_t2,u_t3),1)
    #shape: [batch_size,64,64]

    #用差分拟合导数u_xx，使用中心差分，第一层和最后一层暂时用第二层和倒数第二层表示
    U_xx = (U[:,:,2:]+U[:,:,:-2]-2*U[:,:,1:-1])/dist_x**2
    u_xx0 = U_xx[:,:,0].view(-1,64,1)
    u_xxN = U_xx[:,:,-1].view(-1,64,1)
    U_xx = torch.cat((u_xx0,U_xx,u_xxN),2)
    #shape: [batch_size,64,64]
    F = U_t-alpha*U_xx
    #这里用l1范数作为损失，可以考虑用其他的
    loss = nn.L1Loss()
    return loss(F,torch.zeros_like(F))


def diff_loss(Input_D):
    X = np.linspace(-np.pi, np.pi, 130)
    dist_x = (X[1] - X[0]) * 8
    T = np.linspace(0, 1, 512)
    dist_t = (T[1] - T[0]) * 2

    U = Input_D[:,0]
    alpha = Input_D[:,1]
    #用差分拟合导数u_t,倒数第二层用中心差分，最后一层用向前差分，其他用向后差分
    u_t1 = (U[:,1:-1,:] - U[:,:-2,:])/dist_t
    #0到n-2的u_t，dist_t是每个点之间的实际距离
    u_t2 = (U[:,-1,:] - U[:,-3,:])/(2*dist_t)
    #倒数第二层
    u_t2 = u_t2.view(-1,1,64)
    u_t3 = (U[:,-1,:] - U[:,-2,:])/dist_t
    #最后一层
    u_t3 = u_t3.view(-1, 1, 64)
    U_t = torch.cat((u_t1,u_t2,u_t3),1)
    #shape: [batch_size,64,64]

    #用差分拟合导数u_xx，使用中心差分，第一层和最后一层暂时用第二层和倒数第二层表示
    U_xx = (U[:,:,2:]+U[:,:,:-2]-2*U[:,:,1:-1])/dist_x**2
    u_xx0 = U_xx[:,:,0].view(-1,64,1)
    u_xxN = U_xx[:,:,-1].view(-1,64,1)
    U_xx = torch.cat((u_xx0,U_xx,u_xxN),2)
    #shape: [batch_size,64,64]
    F = U_t-alpha*U_xx
    #这里用l1范数作为损失，可以考虑用其他的
    loss = nn.L1Loss()
    return loss(F,torch.zeros_like(F))
