import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.attentionPDE_model as module_arch
from trainer.attentionPDE_trainer import AttentionPDETrainer
import datetime
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
    model = get_instance(module_arch, 'arch', config)
    print(model)

    # get function handles of loss and metrics
    loss = [getattr(module_loss, l) for l in config['loss']]
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam([model.W_x_2D,model.W_y_2D,model.W_xx_k,model.W_yy_k], lr=0.01)
    # optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    Lambda = [l for l in config['Lambda']]

    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    trainer = AttentionPDETrainer(model, loss, metrics, optimizer, resume, config, data_loader, train_logger = train_logger, Lambda=Lambda)
    # trainer.lbfgs_train()
    trainer.train()

if __name__ == '__main__':
    config = json.load(open("./attentionPDEconfig.json"))
    path = os.path.join(config['trainer']['save_dir'], config['name'])
    device = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]=device
    start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
    try:
        os.renames(path,path+start_time)
    except:
       print('no rename')
    main(config,resume=False)
