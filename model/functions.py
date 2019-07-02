import torch
import numpy as np
#
# def f(xy_data):
#



# def phi(xy_data):
#     """
#     -distant (xy_data)
#     xy_data : [x,y] shape [m,2]
#     return : [m,]
#     """
#     b = torch.zeros_like(xy_data)
#     b[:,0] = torch.min(1-xy_data,dim=1)[0]
#     b[:,1] = torch.min(xy_data,dim=1)[0]
#     return -torch.min(b,dim=1)[0]

def phi(xy_data):
    x = xy_data[:,0]
    y = xy_data[:,1]
    return 10*x*(1-x)*y*(1-y)-1/2

# def f(xy_data):
#     return -20

# def f(xy_data):
#     return -8

def f(xy_data):
    return 11*(xy_data[:,0]+xy_data[:,1]-1)

def creat_random_train_data(batch_size):
    """creat one batch random train data"""
    x_internal = np.random.random([batch_size, 1])
    y_internal = np.random.random([batch_size, 1])
    xy_internal = np.concatenate((x_internal, y_internal), axis=1)
    
    # boundary [0,random],[1,random],[random,0],[random,1]
    # x_boundary = np.concatenate((np.zeros(batch_size),
    #                              np.ones(batch_size),
    #                              np.random.random(batch_size),
    #                              np.random.random(batch_size)))
    # x_boundary = x_boundary.reshape([-1, 1])
    # y_boundary = np.concatenate((np.random.random(batch_size),
    #                              np.random.random(batch_size),
    #                              np.ones(batch_size),
    #                              np.zeros(batch_size)))

    # x_boundary = np.concatenate((np.zeros(batch_size // 4),
    #                              np.ones(batch_size // 4),
    #                              np.random.random(batch_size // 4),
    #                              np.random.random(batch_size // 4)))
    # x_boundary = x_boundary.reshape([-1, 1])
    # y_boundary = np.concatenate((np.random.random(batch_size // 4),
    #                              np.random.random(batch_size // 4),
    #                              np.ones(batch_size // 4),
    #                              np.zeros(batch_size // 4)))
    # y_boundary = y_boundary.reshape([-1, 1])
    # xy_boundary = np.concatenate((x_boundary, y_boundary), axis=1)

    x_boundary = np.concatenate((np.zeros(batch_size // 4),
                                 np.ones(batch_size // 4),
                                 np.linspace(0, 1, batch_size // 4),
                                 np.linspace(0, 1, batch_size // 4)))
    x_boundary = x_boundary.reshape([-1, 1])
    y_boundary = np.concatenate((np.linspace(0, 1, batch_size // 4),
                                 np.linspace(0, 1, batch_size // 4),
                                 np.ones(batch_size // 4),
                                 np.zeros(batch_size // 4)))
    y_boundary = y_boundary.reshape([-1, 1])
    xy_boundary = np.concatenate((x_boundary, y_boundary), axis=1)
    xy_all = np.concatenate((xy_internal, xy_boundary), axis=0)
    xy_all = torch.Tensor(xy_all)
    
    boundary_data = torch.Tensor(xy_boundary)
    return xy_all, boundary_data