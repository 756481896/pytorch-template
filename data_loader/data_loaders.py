from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import torch
import torch.utils.data as utils
import pdb
import matplotlib.tri as tri
import matplotlib.pyplot as plt
class PoissonMembranceDataLoader(BaseDataLoader):
    """
    MyDataLoader to get data
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        self.data_dir = data_dir
        w_path = os.path.join(data_dir, 'w_data.npy')
        p_path = os.path.join(data_dir, 'p_data.npy')
        mesh_path = os.path.join(data_dir,'mesh_coordinates.npy')
        triangles_path = os.path.join(data_dir,'triangles.npy')

        w_data = np.load(w_path)
        w_data = torch.from_numpy(w_data)
        w_data = w_data.view(-1,1,128,128)
        p_data = np.load(p_path)
        p_data = torch.from_numpy(p_data)
        p_data = p_data.view(-1, 1, 128, 128)
        InputD = torch.cat((w_data,p_data),dim=1)
        TargetD = torch.ones((InputD.shape[0], 1, 1, 1))
        InputG = w_data
        TargetG = p_data
        # torch.transpose(p_data, 0, 1)
        self.dataset = utils.TensorDataset(InputD, TargetD, InputG, TargetG)

        #mesh 的numpy的值，用来后面的画图部分
        mesh_coordinates = np.load(mesh_path)
        mesh_coordinates = np.reshape(mesh_coordinates,(-1,2))
        #shape [128*128,2]  (x,y)的坐标
        triangles = np.load(triangles_path)
        self.triangulation = tri.Triangulation(mesh_coordinates[:, 0],
                                          mesh_coordinates[:, 1],
                                          triangles)
        super(PoissonMembranceDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class VariantCoeLinearDataLoader(BaseDataLoader):
    """
    使用pdenet文章中的代码生成的数据，
    xy: [28,49,49,2]
    uT: [28,49,49,20]
    u0: [28,49,49]
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        self.data_dir = data_dir
        xy_path = os.path.join(data_dir, 'xy.pt')
        u0_path = os.path.join(data_dir, 'u0.pt')
        uT_path = os.path.join(data_dir, 'uT.pt')
        xy = torch.load(xy_path)
        u0 = torch.load(u0_path)
        u0 = u0.view(-1,49,49,1)
        uT = torch.load(uT_path)
        uT = torch.cat((u0,uT),3)

        self.dataset = utils.TensorDataset(xy,u0,uT)
        super(VariantCoeLinearDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

# class MyCustomDataset(Dataset):
#     def __init__(self, data_dir='/home1/shenxing/GAN_PDE'):
#         # 自己定义的dataset函数，用于读取生成好的热方程的64*64的数据
#         U_path = os.path.join(data_dir,'U_data.npy')
#         alpha_path = os.path.join(data_dir,'alpha_data.npy')
#         self.U_data = np.load(U_path)
#         self.U_data = torch.from_numpy(self.U_data)
#         self.Alpha_data = np.load(alpha_path)
#         self.Alpha_data = torch.from_numpy(self.Alpha_data)
#
#     def __getitem__(self, index):
#         """
#         原始大小是512*128，转换为64*64
#         :param index:
#         :return: D,G的输入输出
#         """
#         U_sample = self.U_data[index][::8,::2]
#         alpha_sample = self.Alpha_data[index][::8,::2]
#         InputD = torch.cat((U_sample,alpha_sample),dim=1)
#         OutputD = torch.ones_like(InputD)
#         InputG = U_sample
#         OutputG = alpha_sample
#
#         #每个64*64的网格点之间的距离
#         X = np.linspace(-np.pi, np.pi, 130)
#         self.dist_x = (X[1] - X[0]) * 8
#         T = np.linspace(0, 1, 512)
#         self.dist_t = (T[1] - T[0]) * 2
#
#         return (InputD,OutputD,InputG,OutputG)
#
#     def __len__(self):
#         return len(self.U_data)

class MyDataLoader(BaseDataLoader):
    """
    MyDataLoader to get data
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        self.data_dir = data_dir
        U_path = os.path.join(data_dir, 'U_data.npy')
        alpha_path = os.path.join(data_dir, 'alpha_data.npy')


        U_data = np.load(U_path)
        U_data = torch.from_numpy(U_data)
        Alpha_data = np.load(alpha_path)
        Alpha_data = torch.from_numpy(Alpha_data)
        U_sample = U_data[:, ::8, ::2].view(-1, 1, 64, 64)
        alpha_sample = Alpha_data[:, ::8, ::2].view(-1, 1, 64, 64)
        InputD = torch.cat((U_sample, alpha_sample), dim=1)
        TargetD = torch.ones((InputD.shape[0],1,1,1))
        InputG = U_sample
        TargetG = alpha_sample
        self.dataset = utils.TensorDataset(InputD, TargetD, InputG, TargetG)

        super(MyDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class MyDataLoaderFixU0(BaseDataLoader):
    """
    MyDataLoader to get data
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        self.data_dir = data_dir
        U_path = os.path.join(data_dir, 'U_data_fix_u0.npy')
        alpha_path = os.path.join(data_dir, 'alpha_data_fix_u0.npy')
        U_data = np.load(U_path)
        U_data = torch.from_numpy(U_data)
        Alpha_data = np.load(alpha_path)
        Alpha_data = torch.from_numpy(Alpha_data)
        U_sample = U_data[:, ::8, 34:].view(-1, 1, 64, 64)
        alpha_sample = Alpha_data[:, ::8, 34:].view(-1, 1, 64, 64)
        InputD = torch.cat((U_sample, alpha_sample), dim=1)
        TargetD = torch.ones((InputD.shape[0],1,1,1))
        InputG = U_sample
        TargetG = alpha_sample
        self.dataset = utils.TensorDataset(InputD, TargetD, InputG, TargetG)
        super(MyDataLoaderFixU0, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

