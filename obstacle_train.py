import os
import json
import argparse
import torch
import data_loader.obstacle_problem_data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.obstacle_problem_model as module_arch
# from trainer.pix2pixtrainer import TrainerGAN
from trainer.obstacle_trainer import Trainer
from utils import Logger

torch.set_default_tensor_type('torch.DoubleTensor')
def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    train_logger = Logger()

    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader', config)
    # valid_data_loader = data_loader.split_validation()
    # valid_data_loader = get_instance(module_data, 'valid_data_loader', config)
    valid_data_loader = None
    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    print(model)

    # get function handles of loss and metrics
    loss = [getattr(module_loss, l) for l in config['loss']]
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    Lambda = [l for l in config['Lambda']]

    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    trainer = Trainer(model, loss, Lambda, metrics, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)

    trainer.train()

if __name__ == '__main__':
    config = json.load(open("./obstacle_problem_config.json"))
    path = os.path.join(config['trainer']['save_dir'], config['name'])
    device = "1"
    os.environ["CUDA_VISIBLE_DEVICES"]=device

    main(config,resume=False)
