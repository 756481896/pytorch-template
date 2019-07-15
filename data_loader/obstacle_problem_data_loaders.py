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
import math

class ObstacleProblemDataLoader(BaseDataLoader):
    """
    obstacle problem data loader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, num_internal_points,num_boundary_points,training=True):
        """
        $- \Delta u - f \geq 0$
        $ u \geq \psi$
        $(- \Delta u - f) ( u - \psi) = 0$
        $u = 0, x,y \in \partial \Omega$
        """
        # considering random points datasets,now is all linspace
        x_internal = np.random.random([num_internal_points,1])
        y_internal = np.random.random([num_internal_points,1])
        xy_internal = np.concatenate((x_internal,y_internal),axis=1)

        # boundary [0,random],[1,random],[random,0],[random,1]
        x_boundary = np.concatenate((np.zeros(num_boundary_points//4),
                                     np.ones(num_boundary_points//4),
                                     np.random.random(num_boundary_points//4),
                                     np.random.random(num_boundary_points//4)))
        x_boundary = x_boundary.reshape([-1,1])
        y_boundary = np.concatenate((np.random.random(num_boundary_points//4),
                                     np.random.random(num_boundary_points//4),
                                     np.ones(num_boundary_points//4),
                                     np.zeros(num_boundary_points//4)))
        y_boundary = y_boundary.reshape([-1, 1])
        xy_boundary = np.concatenate((x_boundary,y_boundary),axis=1)

        x_all = np.concatenate((x_internal,x_boundary),axis=0)
        y_all = np.concatenate((y_internal,y_boundary),axis=0)
        self.xy_all = np.concatenate((xy_internal,xy_boundary),axis=0)
        self.xy_all = torch.Tensor(self.xy_all)

        self.boundary_data = torch.Tensor(xy_boundary)
        self.dataset = utils.TensorDataset(self.xy_all)

        #可以尝试每个batch的数据都是随机新生成的，每个epoch的数据都不一样.
        super(ObstacleProblemDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split,
                                                        num_workers)


class ObstacleProblemValidDataLoader(BaseDataLoader):
    """
    creat valid data of mesh grid in [0,1]*[0,1]
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers,num_internal_points,num_boundary_points,training=True):
        """
        $- \Delta u - f \geq 0$
        $ u \geq \psi$
        $(- \Delta u - f) ( u - \psi) = 0$
        $u = 0, x,y \in \partial \Omega$
        """
        # considering random points datasets,now is all linspace
        # x_internal = np.random.random([num_internal_points,1])
        x_internal = np.linspace(0, 1, int(math.sqrt(num_internal_points)))
        y_internal = np.linspace(0, 1, int(math.sqrt(num_internal_points)))
        xs, ys = np.meshgrid(x_internal, y_internal)
        x_internal = np.reshape(xs, [-1,1])
        y_internal = np.reshape(ys, [-1,1])
        xy_internal = np.concatenate((x_internal,y_internal),axis=1)
        # boundary [0,random],[1,random],[random,0],[random,1]
        x_boundary = np.concatenate((np.zeros(num_boundary_points // 4),
                                     np.ones(num_boundary_points // 4),
                                     np.linspace(0, 1, num_boundary_points // 4),
                                     np.linspace(0, 1, num_boundary_points // 4)))
        x_boundary = x_boundary.reshape([-1,1])
        y_boundary = np.concatenate((np.linspace(0, 1, num_boundary_points // 4),
                                     np.linspace(0, 1, num_boundary_points // 4),
                                     np.ones(num_boundary_points//4),
                                     np.zeros(num_boundary_points//4)))
        y_boundary = y_boundary.reshape([-1, 1])
        xy_boundary = np.concatenate((x_boundary,y_boundary),axis=1)

        x_all = np.concatenate((x_internal,x_boundary),axis=0)
        y_all = np.concatenate((y_internal,y_boundary),axis=0)
        self.xy_all = np.concatenate((xy_internal,xy_boundary),axis=0)
        self.xy_all = torch.Tensor(self.xy_all)

        self.boundary_data = torch.Tensor(xy_boundary)
        self.dataset = utils.TensorDataset(self.xy_all)

        #可以尝试每个batch的数据都是随机新生成的，每个epoch的数据都不一样.
        super(ObstacleProblemValidDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split,
                                                        num_workers)
