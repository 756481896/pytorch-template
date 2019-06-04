import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from model.functions import *
from torch import nn

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, Lambda, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.Lambda = Lambda

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        # Lambda[0] is epsilon in F(u)
        # Lambda[1] is superparameter of loss function
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx in range(1000):
            xy_data, boundary_data = creat_random_train_data(self.data_loader.batch_size)
            xy_data, boundary_data = xy_data.to(self.device), boundary_data.to(self.device)
            xy_data.requires_grad = True
            self.optimizer.zero_grad()
            u_hat = self.model(xy_data)
            nabla_u = torch.autograd.grad(u_hat, xy_data, grad_outputs=torch.ones(u_hat.size()).to(self.device), create_graph=True)[0]
            #u关于x,y的一阶导
            Delta_u = torch.autograd.grad(nabla_u, xy_data, grad_outputs=torch.ones(nabla_u.size()).to(self.device), create_graph=True)[0]
            #u关于x,y的二阶导
            F_u = -torch.sum(Delta_u, 1) - f(xy_data) - (phi(xy_data) - u_hat.squeeze() + self.Lambda[0] * (-torch.sum(Delta_u, 1) - f(xy_data)) / self.Lambda[0])

            #considering batch boundary data
            u_boundary = self.model(boundary_data)

            loss_l2 = nn.MSELoss()(F_u, torch.zeros_like(F_u))
            loss_l1 = nn.L1Loss()(torch.abs(F_u),torch.zeros_like(F_u))
            loss_boundary = nn.MSELoss()(u_boundary,torch.zeros_like(u_boundary))
            loss = loss_l2 + self.Lambda[1] * loss_l1 + self.Lambda[2] * loss_boundary
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            self.writer.add_scalar('loss_l1', loss_l1.item())
            self.writer.add_scalar('loss_l2', loss_l2.item())
            self.writer.add_scalar('loss_boundary', loss_boundary.item())
            total_loss += loss.item()
            # total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.3f} Loss_l2: {:.3f} Loss_l1: {:.3f} Loss_boudary: {:.3f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item(),
                    loss_l2.item(),
                    loss_l1.item(),
                    loss_boundary.item())
                )
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader)
        }
        # 'metrics': (total_metrics / len(self.data_loader)).tolist()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        # with torch.no_grad():
        boundary_data = self.data_loader.boundary_data.to(self.device)
        xy_data = self.data_loader.xy_all.to(self.device)
        xy_data.requires_grad = True
        self.optimizer.zero_grad()
        u_hat = self.model(xy_data)
        nabla_u = torch.autograd.grad(u_hat, xy_data, grad_outputs=torch.ones(u_hat.size()).to(self.device), create_graph=True)[0]
        # u关于x,y的一阶导
        Delta_u = torch.autograd.grad(nabla_u, xy_data, grad_outputs=torch.ones(nabla_u.size()).to(self.device), create_graph=True)[0]
        # u关于x,y的二阶导
        F_u = -torch.sum(Delta_u, 1) - f(xy_data) - (
                    phi(xy_data) - u_hat.squeeze() + self.Lambda[0] * (-torch.sum(Delta_u, 1) - f(xy_data)) /
                    self.Lambda[0])
        u_boundary = self.model(boundary_data)

        loss_l2 = nn.MSELoss()(F_u, torch.zeros_like(F_u))
        loss_l1 = nn.L1Loss()(torch.abs(F_u),torch.zeros_like(F_u))
        loss_boundary = nn.MSELoss()(u_boundary,torch.zeros_like(u_boundary))
        loss = loss_l2 + self.Lambda[1] * loss_l1 + self.Lambda[2] * loss_boundary

        self.writer.set_step(epoch - 1, 'valid')
        self.writer.add_scalar('valid_loss', loss.item())
        self.writer.add_scalar('valid_loss_l1', loss_l1.item())
        self.writer.add_scalar('loss_l2', loss_l2.item())
        self.writer.add_scalar('loss_boundary', loss_boundary.item())
        total_val_loss += loss.item()
        # total_val_metrics += self._eval_metrics(output, target)
        # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        self.optimizer.zero_grad()
        self.logger.info(
            'Valid Epoch: {}  Loss: {:.3f} Loss_l2: {:.3f} Loss_l1: {:.3f} Loss_boudary: {:.3f}'.format(
                epoch,
                loss.item(),
                loss_l2.item(),
                loss_l1.item(),
                loss_boundary.item())
        )
        return {
            'val_loss': total_val_loss / len(self.valid_data_loader)
        }
        # 'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()



    # def _train_epoch(self, epoch):
    #     """
    #     Training logic for an epoch
    #
    #     :param epoch: Current training epoch.
    #     :return: A log that contains all information you want to save.
    #
    #     Note:
    #         If you have additional information to record, for example:
    #             > additional_log = {"x": x, "y": y}
    #         merge it with log before return. i.e.
    #             > log = {**log, **additional_log}
    #             > return log
    #
    #         The metrics in log must have the key 'metrics'.
    #     """
    #     self.model.train()
    #     # Lambda[0] is epsilon in F(u)
    #     # Lambda[1] is superparameter of loss function
    #     total_loss = 0
    #     total_metrics = np.zeros(len(self.metrics))
    #     boundary_data = self.data_loader.boundary_data.to(self.device)
    #     for batch_idx, (xy_data) in enumerate(self.data_loader):
    #         xy_data = xy_data[0].to(self.device)
    #         xy_data.requires_grad = True
    #         self.optimizer.zero_grad()
    #         u_hat = self.model(xy_data)
    #         nabla_u = torch.autograd.grad(u_hat, xy_data, grad_outputs=torch.ones(u_hat.size()).to(self.device), create_graph=True)[0]
    #         #u关于x,y的一阶导
    #         Delta_u = torch.autograd.grad(nabla_u, xy_data, grad_outputs=torch.ones(nabla_u.size()).to(self.device), create_graph=True)[0]
    #         #u关于x,y的二阶导
    #         F_u = -torch.sum(Delta_u, 1) - f(xy_data) - (phi(xy_data) - u_hat.squeeze() + self.Lambda[0] * (-torch.sum(Delta_u, 1) - f(xy_data)) / self.Lambda[0])
    #
    #         #considering batch boundary data
    #         u_boundary = self.model(boundary_data)
    #
    #         loss_l2 = nn.MSELoss()(F_u, torch.zeros_like(F_u))
    #         loss_l1 = nn.L1Loss()(torch.abs(F_u),torch.zeros_like(F_u))
    #         loss_boundary = nn.MSELoss()(u_boundary,torch.zeros_like(u_boundary))
    #         loss = loss_l2 + self.Lambda[1] * loss_l1 + self.Lambda[2] * loss_boundary
    #         loss.backward()
    #         self.optimizer.step()
    #
    #         self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
    #         self.writer.add_scalar('loss', loss.item())
    #         self.writer.add_scalar('loss_l1', loss_l1.item())
    #         self.writer.add_scalar('loss_l2', loss_l2.item())
    #         self.writer.add_scalar('loss_boundary', loss_boundary.item())
    #         total_loss += loss.item()
    #         # total_metrics += self._eval_metrics(output, target)
    #
    #         if self.verbosity >= 2 and batch_idx % self.log_step == 0:
    #             self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.3f} Loss_l2: {:.3f} Loss_l1: {:.3f} Loss_boudary: {:.3f}'.format(
    #                 epoch,
    #                 batch_idx * self.data_loader.batch_size,
    #                 self.data_loader.n_samples,
    #                 100.0 * batch_idx / len(self.data_loader),
    #                 loss.item(),
    #                 loss_l2.item(),
    #                 loss_l1.item(),
    #                 loss_boundary.item())
    #             )
    #             # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
    #
    #     log = {
    #         'loss': total_loss / len(self.data_loader)
    #     }
    #     # 'metrics': (total_metrics / len(self.data_loader)).tolist()
    #
    #     if self.do_validation:
    #         val_log = self._valid_epoch(epoch)
    #         log = {**log, **val_log}
    #
    #     if self.lr_scheduler is not None:
    #         self.lr_scheduler.step()
    #
    #     return log