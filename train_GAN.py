import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.poisson_membrance_model as module_arch
from trainer.pix2pixtrainer import TrainerGAN

from utils import Logger
torch.set_default_tensor_type('torch.DoubleTensor')
def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    train_logger = Logger()

    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader', config)
    valid_data_loader = data_loader.split_validation()

    # build model architecture
    modelD = get_instance(module_arch, 'archD', config)
    modelG = get_instance(module_arch, 'archG', config)
    print(modelD)
    print(modelG)

    # get function handles of loss and metrics
    loss = [getattr(module_loss, l) for l in config['loss']]
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_paramsD = filter(lambda p: p.requires_grad, modelD.parameters())
    trainable_paramsG = filter(lambda p: p.requires_grad, modelG.parameters())
    optimizerD = get_instance(torch.optim, 'optimizerD', config, trainable_paramsD)
    optimizerG = get_instance(torch.optim, 'optimizerD', config, trainable_paramsG)
    Lambda = [l for l in config['Lambda']]

    lr_schedulerD = get_instance(torch.optim.lr_scheduler, 'lr_schedulerD', config, optimizerD)
    lr_schedulerG = get_instance(torch.optim.lr_scheduler, 'lr_schedulerG', config, optimizerG)

    trainer = TrainerGAN(modelD, modelG, loss[0], loss[1], Lambda, metrics, optimizerD, optimizerG,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_schedulerD=lr_schedulerD,
                      lr_schedulerG=lr_schedulerG,
                      train_logger=train_logger)

    trainer.train()

if __name__ == '__main__':
    config = json.load(open("./poisson_membrance_config.json"))
    path = os.path.join(config['trainer']['save_dir'], config['name'])
    device = "1"
    os.environ["CUDA_VISIBLE_DEVICES"]=device

    main(config,resume=False)
