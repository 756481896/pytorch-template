import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import pdb
import torch
import math

class AttentionPDEModel(nn.Module):
    def __init__(self,ndim,T,device_num = 0):
        super(AttentionPDEModel, self).__init__()
        # W_x = torch.randn(ndim, ndim, requires_grad=True)
        # torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        #默认全部都使用cuda
        self.device_num = device_num
        self.ndim = ndim
        self.ndim = ndim
        self.T = T
        self.dt = 0.015
        self.dx = 0.1257
        self.dy = 0.1257
        self.build()
        # xy_path = "/home1/shenxing/Attention_PDE/data/xy.pt"
        # self.xy = torch.load(xy_path)
        # xy = xy.double()


        # Input = xy[:, :, :, 0]
        # Input = Input.repeat(20,1,1,1)
        # Input = Input.permute(1,2,3,0)
        # u_t = self.get_u_t(Input)

        #
        # Input = xy[:, :, :, 0]
        # Input = Input.repeat(20,1,1,1)
        # Input = Input.permute(1,2,3,0)
        #
        # u_x = self.get_u_x(Input)
        # u_xx = self.get_u_xx(Input)
        #
        # Input = xy[:, :, :, 1]
        # Input = Input.repeat(20, 1, 1, 1)
        # Input = Input.permute(1, 2, 3, 0)
        #
        # u_y = self.get_u_y(Input)
        # u_yy = self.get_u_yy(Input)

    def build(self):
        """变量转到gpu device之后运行，构建W"""
        self.build_W_t()
        self.build_W_x()
        self.build_W_y()
        self.build_W_xx()
        self.build_W_yy()

    def build_W_t(self):
        #计算u_t这里没有可训练参数
        dt = 0.015
        #暂时用固定值，之后添加其他
        W_t_diag1 = torch.ones(self.T-1)*(1 / 2)
        W_t_diag1[0] = 1
        #上对角
        W_t_diag2 = torch.ones(self.T-1) * (- 1 / 2)
        W_t_diag2[-1] = -1
        # 下对角

        W_t = torch.diag(W_t_diag1,1) + torch.diag(W_t_diag2,-1)
        W_t[0,0] = -1
        W_t[-1,-1] = 1
        self.W_t = W_t*(1/dt)
        # self.W_t = torch.nn.Parameter(W_t, requires_grad=False)
        #shape [20,20]

    def build_W_x(self):
        # self.W_x_2D = torch.nn.Parameter(torch.DoubleTensor(self.ndim, self.ndim))
        # torch.nn.init.xavier_normal(self.W_x_2D)

        #用准确解来验证模型是否正确
        K = torch.ones(49,49)
        K = K*(2*self.dx)
        K[:,0] = K[:,0] *(1/2)
        K[:, -1] = K[:, -1] * (1 / 2)
        r_ux = self.get_real_coefficient()[0]
        r_W_x_2d = r_ux / K
        # self.W_x_2D = torch.nn.init.constant(self.W_x_2D,r_W_x_2d)
        self.W_x_2D = r_W_x_2d
        self.W_x_2D.requires_grad = True

        # self.W_x_2D = torch.arange(1, 49 * 49+1).view(49, 49)
        #shape 49*49
        W_x_tmp = self.W_x_2D[:,:-1]
        #0 to n-1
        W_x_tmp2 = self.W_x_2D[:,1:]
        #1 to end
        #shape 49*48
        W_x_diag1 = torch.diag_embed(W_x_tmp, offset = 1,dim1=0, dim2=1)
        W_x_diag2 = -1 *torch.diag_embed(W_x_tmp2, offset = -1,dim1=0, dim2=1)
        self.W_x_3D = W_x_diag1 + W_x_diag2
        self.W_x_3D[0,0,:] = - self.W_x_2D[:,0]
        self.W_x_3D[-1,-1,:] = self.W_x_2D[:,-1]
        # self.W_x_3D = torch.nn.Parameter(self.W_x_3D,requires_grad=True)
        #变为parameter后这里就不能往后求导了
        self.W_x_3D = self.W_x_3D.double()
        #shape [49,49,49]


    def build_W_xx(self):
        """
        因为方程u_xx,u_yy前都是固定的参数，所以只有一个需要更新的参数
        :return:
        """
        # self.W_xx_k = torch.nn.Parameter(torch.randn(1))
        c = 0.2/0.1257
        self.W_xx_k = torch.tensor([c],requires_grad = False)
        # self.W_xx_k = torch.ones(1)
        W_xx_diag1 = torch.ones(self.ndim)*(-2)
        W_xx_diag1[0] = W_xx_diag1[-1] = 1
        #中心对角
        W_xx_diag2 = torch.ones(self.ndim-1)
        W_xx_diag2[0] = -2
        #上1对角
        W_xx_diag3 = torch.ones(self.ndim-1)
        W_xx_diag3[-1] = -2
        #下1对角
        W_xx = torch.diag(W_xx_diag1) + torch.diag(W_xx_diag2,1) + torch.diag(W_xx_diag3,-1)
        W_xx[0,2] = 1
        W_xx[-1,-3] = 1
        # W_xx = torch.nn.Parameter(W_xx, requires_grad=True)
        self.W_xx = (self.W_xx_k * W_xx).double()
        # self.W_xx = torch.nn.Parameter(self.W_xx)

    def build_W_y(self):
        # self.W_y_2D = torch.nn.Parameter(torch.DoubleTensor(self.ndim, self.ndim))
        # torch.nn.init.xavier_normal(self.W_y_2D)

        #用准确解来验证模型是否正确
        K = torch.ones(49,49)
        K = K * (2 * self.dy)
        K[:, 0] = K[:, 0] * (1 / 2)
        K[:, -1] = K[:, -1] * (1 / 2)
        r_uy = self.get_real_coefficient()[1]
        r_W_y_2d = r_uy / K
        self.W_y_2D = r_W_y_2d
        self.W_y_2D.requires_grad = True
        #shape 49*49
        W_y_tmp = self.W_y_2D[:,:-1]
        #0 to n-1
        W_y_tmp2 = self.W_y_2D[:,1:]
        #1 to end
        #shape 49*48
        W_y_diag1 = torch.diag_embed(W_y_tmp, offset = 1,dim1=0, dim2=1)
        W_y_diag2 = -1 *torch.diag_embed(W_y_tmp2, offset = -1,dim1=0, dim2=1)
        self.W_y_3D = W_y_diag1 + W_y_diag2
        self.W_y_3D[0,0,:] = - self.W_y_2D[:,0]
        self.W_y_3D[-1,-1,:] = self.W_y_2D[:,-1]
        # self.W_y_3D = torch.nn.Parameter(self.W_y_3D,requires_grad=True)
        self.W_y_3D = self.W_y_3D.double()

    def build_W_yy(self):
        # self.W_yy_k = torch.nn.Parameter(torch.randn(1))
        d = 0.3 / 0.1257
        self.W_yy_k = torch.tensor([d], requires_grad=False)
        W_yy_diag1 = torch.ones(self.ndim)*(-2)
        W_yy_diag1[0] = W_yy_diag1[-1] = 1
        #中心对角
        W_yy_diag2 = torch.ones(self.ndim-1)
        W_yy_diag2[0] = -2
        #上1对角
        W_yy_diag3 = torch.ones(self.ndim-1)
        W_yy_diag3[-1] = -2
        #下1对角
        W_yy = torch.diag(W_yy_diag1) + torch.diag(W_yy_diag2,1) + torch.diag(W_yy_diag3,-1)
        W_yy[0,2] = 1
        W_yy[-1,-3] = 1
        # W_yy = torch.nn.Parameter(W_yy,requires_grad=True)
        self.W_yy = (self.W_yy_k * W_yy).double()
        # self.W_yy = torch.nn.Parameter(self.W_yy)

    def get_u_t(self,Input):
        """
        计算偏导u_t，在所有空间上
        :param Input: shape[batch_size,49,49,20]
        :return: u_t: shape [batch_size,49,49,20]
        """
        batch_size = Input.shape[0]
        T = Input.shape[-1]
        size =  Input.shape[1]
        Input = Input.view(batch_size,size*size,T)
        #shape [28,2401,20]
        W_t = torch.transpose(self.W_t,0,1)
        #shape [20,20]
        u_t = torch.matmul(Input,W_t)
        #shape [28,2401,20]
        u_t = u_t.view(batch_size,size,size,T)
        return u_t

    def get_u_x(self,Input):
        """
        计算偏导u_x，在所有时间空间
        :param Input: shape[batch_size,49,49,20]
        :return: u_x: shape [batch_size,49,49,20]
        """
        batch_size = Input.shape[0]
        T = Input.shape[-1]
        size =  Input.shape[1]
        Input = Input.permute(0,3,1,2)
        #shape[batch_size,20, 49, 49]
        Input = Input.contiguous().view(-1,self.ndim,self.ndim)
        # shape[batch_size*20, 49, 49]
        Input = Input.permute(1,0,2)
        # shape[49,batch_size*20, 49]
        W_x_tmp = self.W_x_3D.permute(2, 0, 1)
        # shape[49, 49, 49]
        W_x_tmp = torch.transpose(W_x_tmp, 1, 2)
        u_x = torch.bmm(Input, W_x_tmp)
        #batch matmul 每个49*49的网格点都计算一次u_x,因为alpha(x,y)与t无关，所以用同一个W_X_3D
        #shape [49,batch_size*20,49]
        u_x = u_x.permute(1,0,2)
        u_x = u_x.view(batch_size,T,size,size)
        u_x = u_x.permute(0,2,3,1)
        return u_x

    def get_u_xx(self,Input):
        """
        计算偏导u_xx，在所有时间空间
        :param Input: shape[batch_size,49,49,20]
        :return: u_xx: shape [batch_size,49,49,20]
        """
        batch_size = Input.shape[0]
        T = Input.shape[-1]
        size =  Input.shape[1]
        Input = Input.permute(0,3,1,2)
        #shape[batch_size,20, 49, 49]
        Input = Input.contiguous().view(-1,self.ndim,self.ndim)
        # shape[batch_size*20, 49, 49]
        W_xx_tmp = torch.transpose(self.W_xx, 0, 1)
        # shape [49,49]
        u_xx = torch.matmul(Input, W_xx_tmp)
        #shape [batch_size*20,49,49]
        u_xx = u_xx.view(batch_size,T,size,size)
        u_xx = u_xx.permute(0,2,3,1)
        return u_xx


    def get_u_y(self, Input):
        """
        计算偏导u_y，在所有时间空间
        :param Input: shape[batch_size,49,49,20]
        :return: u_y: shape [batch_size,49,49,20]
        """
        batch_size = Input.shape[0]
        T = Input.shape[-1]
        size = Input.shape[1]
        Input = Input.permute(0, 3, 1, 2)
        # shape[batch_size,20, 49, 49]
        Input = Input.contiguous().view(-1, self.ndim, self.ndim)
        # shape[batch_size*20, 49, 49]
        Input = Input.permute(2, 1, 0)
        # shape[49, 49, batch_size*20]
        W_y_tmp = self.W_y_3D.permute(2, 0, 1)
        # shape[49, 49, 49]
        u_y = torch.bmm(W_y_tmp,Input)
        # batch matmul 每个49*49的网格点都计算一次u_x,因为alpha(x,y)与t无关，所以用同一个W_X_3D
        # shape [49,49,batch_size*20]
        u_y = u_y.permute(2, 1, 0)
        u_y = u_y.view(batch_size, T, size, size)
        u_y = u_y.permute(0, 2, 3, 1)
        return u_y

    def get_u_yy(self,Input):
        """
        计算偏导u_yy，在所有时间空间
        :param Input: shape[batch_size,49,49,20]
        :return: u_yy: shape [batch_size,49,49,20]
        """
        batch_size = Input.shape[0]
        T = Input.shape[-1]
        size =  Input.shape[1]
        Input = Input.permute(0,3,1,2)
        #shape[batch_size,20, 49, 49]
        Input = Input.contiguous().view(-1,self.ndim,self.ndim)
        Input = Input.permute(1,2,0)
        # shape[49, 49,batch_size*20]
        u_yy = torch.matmul(self.W_yy,Input)
        #shape [49,49,batch_size*20]
        u_yy = u_yy.permute(2,0,1)
        u_yy = u_yy.view(batch_size,T,size,size)
        u_yy = u_yy.permute(0,2,3,1)
        return u_yy

    def cal_from_u0(self,Input):
        """
        从u0计算u1,u2,...uT
        :param Input: u0 shape [batch_size,size,size,1]
        :return: u_all shape [batch_size,size,size,21]
        """
        batch_size = Input.shape[0]
        T = Input.shape[-1]
        size =  Input.shape[1]
        u0 = Input[:,:,:,0]
        u0 = u0.view(batch_size,size,size,1)
        u_tmp = u0
        u_all = u_tmp
        for i in range(20):
            u_xi, u_yi, u_xxi, u_yyi = self.get_u_x(u_tmp), self.get_u_y(u_tmp), self.get_u_xx(u_tmp), self.get_u_yy(u_tmp)
            G = u_xi + u_yi + u_xxi + u_yyi
            u_tmp = self.dt * G + u0
            u_all = torch.cat((u_all,u_tmp),3)
        return u_all

    def coefficient_ux(self):
        """
        从self.W_x_2D算出方程u_x对应的参数
        :return:
        """
        K = torch.ones_like(self.W_x_2D)
        K = K*(2*self.dx)
        K[:,0] = K[:,0] *(1/2)
        self.p_ux = K*self.W_x_2D
        return self.p_ux

    def coefficient_uy(self):
        """
        从self.W_y_2D算出方程u_y对应的参数
        :return:
        """
        K = torch.ones_like(self.W_y_2D)
        K = K*(2*self.dy)
        K[:,0] = K[:,0] *(1/2)
        self.p_uy = K*self.W_y_2D
        return self.p_uy

    def coefficient_uxx(self):
        """
        从self.W_xx_k算出方程u_xx对应的参数
        :return:
        """
        self.p_uxx = (self.dx)*self.W_xx_k
        return self.p_uxx

    def coefficient_uyy(self):
        """
        从self.W_yy_k算出方程u_yy对应的参数
        :return:
        """
        self.p_uyy = (self.dy)*self.W_yy_k
        return self.p_uyy


    def get_coefficient(self):
        """
        :return:算出的方程参数
        """
        return [self.coefficient_ux(),self.coefficient_uy(),self.coefficient_uxx(),self.coefficient_uyy()]

    def get_real_coefficient(self,xy_batch = None):
        """
        :return:方程参数的真实解
        """
        if xy_batch == None:
            xy_path = "/home1/shenxing/Attention_PDE/data/xy.pt"
            xy_batch = torch.load(xy_path)
        x = xy_batch[0,:,:,0]
        y = xy_batch[0,:,:,1]
        r_ux = 0.5 * torch.cos(y) + 0.5 * x * (2 * math.pi - x) * torch.sin(x) + 0.6
        r_uy = 2 * (torch.cos(y) + torch.sin(x)) +0.8
        r_uxx = torch.tensor([0.2])
        r_uyy = torch.tensor([0.3])
        return [r_ux,r_uy,r_uxx,r_uyy]


    def forward(self, Input):
        """
        输入:当前u0生成u在t在0.015到0.3的所有的u(x,y)
        input:uT_batch shape:[batch_size,49,49,20]"""
        return self.get_u_t(Input),self.get_u_x(Input), self.get_u_y(Input), self.get_u_xx(Input), self.get_u_yy(Input)

    # def build_W_x(self,device = None):
    #     # self.W_x_2D = torch.arange(1, 49 * 49+1).view(49, 49)
    #     self.W_x_2D = torch.nn.Parameter(torch.randn(self.ndim, self.ndim))
    #     if device != None:
    #         self.W_x_2D = self.W_x_2D.to(device)
    #     #shape 49*49
    #     W_x_tmp = self.W_x_2D[:,:-1]
    #     #0 to n-1
    #     W_x_tmp2 = self.W_x_2D[:,1:]
    #     #1 to end
    #     #shape 49*48
    #     W_x_diag1 = torch.diag_embed(W_x_tmp, offset = 1,dim1=0, dim2=1)
    #     W_x_diag2 = -1 *torch.diag_embed(W_x_tmp2, offset = -1,dim1=0, dim2=1)
    #     self.W_x_3D_1 = W_x_diag1 + W_x_diag2
    #     self.W_x_3D_2 = self.W_x_3D_1
    #     self.W_x_3D_2[0,0,:] = - self.W_x_2D[:,0]
    #     self.W_x_3D_2[-1,-1,:] = self.W_x_2D[:,-1]
    #     # self.W_x_3D_3 = torch.nn.Parameter(self.W_x_3D_2)
    #     #?测试这里能否顺利求导
    #     self.W_x_3D = self.W_x_3D_2.double()
    # shape [49,49,49]



        # # W_y = torch.randn(ndim, ndim, requires_grad=True)
        # self.W_y = torch.nn.Parameter(torch.randn(ndim, ndim))
        # ones_tmp = torch.ones(ndim)
        # mask = torch.diag(ones_tmp, 0) + torch.diag(ones_tmp[1:], 1) + torch.diag(ones_tmp[1:], -1)
        # mask = torch.nn.Parameter(mask,requires_grad=False)
        # W_x_mask = torch.mul(mask, self.W_x)
        # self.W_x_mask = torch.nn.Parameter(W_x_mask)
        # #shape 49,49
        # # 三对角阵.点乘
        # W_y_mask = torch.mul(mask,self.W_y)
        # self.W_y_mask = torch.nn.Parameter(W_y_mask)

    # def get_u_x(self,Input):
    #     """
    #     计算偏导u_x，在所有时间空间
    #     :param Input: shape[batch_size,49,49,20]
    #     :return: u_x: shape [batch_size,49,49,20]
    #     """
    #     Input = Input.permute(0,3,1,2)
    #     # shape [batch_size,20,49,49]
    #     u_x = torch.matmul(Input,self.W_x_mask)
    #     # u_x: shape[batch_size, 20, 49, 49]
    #     u_x = u_x.permute(0,2,3,1)
    #     #matmul可以broadcast
    #     return u_x
    #
    # def get_u_y(self,Input):
    #     """
    #     计算偏导u_y，在所有时间空间
    #     :param Input: shape[batch_size,49,49,20]
    #     :return: u_y: shape [batch_size,49,49,20]
    #     """
    #     Input = Input.permute(1, 2, 3, 0)
    #     # shape [49,49,20,batch_size]
    #     Input = Input.contiguous().view(self.ndim,self.ndim,-1)
    #     # shape [49,49,20*batch_size]
    #     u_y = torch.matmul(self.W_y_mask,Input)
    #     # shape [49,49,20*batch_size]
    #     u_y = u_y.view(self.ndim,self.ndim,20,-1)
    #     # shape [49,49,20,batch_size]
    #     u_y = u_y.permute(3,0,1,2)
    #     return u_y
    #     Input_x = Input.view(49, 1, 49)
    #
    #     W_x_tmp = self.W_x_3D.permute(2,0,1)
    #     W_x_tmp = torch.transpose(W_x_tmp,1,2)
    #     u_x = torch.bmm(Input_x,W_x_tmp)
    #     u_x = torch.squeeze(u_x)
    #
    #     self.W_y_2D = torch.nn.Parameter(torch.randn(ndim, ndim))
    #     # shape 49*49
    #     W_y_tmp = self.W_y_2D[:, :-1]
    #     # 0 to n-1
    #     W_y_tmp2 = self.W_y_2D[:, 1:]
    #     # 1 to end
    #     # shape 49*48
    #     W_x_diag1 = torch.diag_embed(W_y_tmp, offset=1)
    #     W_y_diag2 = -1 * torch.diag_embed(W_y_tmp2, offset=-1)
    #     self.W_y_3D = W_x_diag1 + W_x_diag2
    #     self.W_y_3D[0, 0, :] = self.W_y_2D[0, 0]
    #     self.W_y_3D[-1, -1, :] = self.W_y_2D[-1, -1]
    #     self.W_y_3D = torch.nn.Parameter(self.W_y_3D)
    #     print(self.W_y_3D)
