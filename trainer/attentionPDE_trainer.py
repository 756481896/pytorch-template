import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
import os

class AttentionPDETrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None, Lambda = None):
        super(AttentionPDETrainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        # self.model.build()
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.loss0 = self.loss[0]
        self.loss1 = self.loss[1]
        self.Lambda0 = Lambda[0]
        # self.optimizer = torch.optim.LBFGS([self.model.W_x_2D,self.model.W_y_2D,self.model.W_xx_k,self.model.W_yy_k], lr=0.8)

    def _eval_metrics(self, output, target):
        """
        :param output:[p_ux,p_uy,p_uxx,p_uyy]
        :param target: u_x等前面的参数的真实值。
        :return:
        """
        names = ["p_ux","p_uy","p_uxx","p_uyy"]
        metric = self.metrics[0]
        acc_metrics = np.zeros(len(output))
        for i in range(len(output)):
            acc_metrics[i] += metric(output[i],target[i])
            self.writer.add_scalar(names[i], acc_metrics[i])
        return acc_metrics

    def lbfgs_train(self):
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
        # self.model.train()
        data_dir = "/home1/shenxing/Attention_PDE/data/"
        total_loss = 0
        total_metrics = np.zeros(4)
        xy_path = os.path.join(data_dir, 'xy.pt')
        u0_path = os.path.join(data_dir, 'u0.pt')
        uT_path = os.path.join(data_dir, 'uT.pt')
        xy = torch.load(xy_path)
        u0 = torch.load(u0_path)
        u0 = u0.view(28,49,49,1)
        uT = torch.load(uT_path)
        uT = torch.cat((u0,uT),3)
        xy_batch, u0_batch, uT_batch = xy.to(self.device), u0.to(self.device), uT.to(self.device)
            # torch.autograd.Variable()
            # optimizer = torch.optim.Adam([self.model.W_xx_k], lr=0.0001)
            # optimizer.zero_grad()
        batch_idx = 1
        for i in range(100):
            print("STEP: ",i)
            def closure():
                self.optimizer.zero_grad()

                u_forward = self.model.cal_from_u0(uT_batch)
                u_t,u_x, u_y, u_xx, u_yy = self.model(uT_batch)

                loss0 = self.loss0(u_forward, uT_batch)
                loss1 = self.loss1(u_t,u_x, u_y, u_xx, u_yy)
                # + self.Lambda0 * loss1
                loss_sum = loss0 + self.Lambda0 * loss1
                loss_sum.backward(retain_graph=True)
                self.writer.set_step((i - 1) * len(self.data_loader) + batch_idx)
                self.writer.add_scalar('loss0', loss0.item())
                self.writer.add_scalar('loss1', loss1.item())
                self.writer.add_scalar('loss_sum', loss_sum.item())

                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] '
                             'Loss0: {:.5f} Loss1: {:.5f} Loss_sum: {:.3f} '
                             .format(
                i,
                batch_idx * self.data_loader.batch_size,
                self.data_loader.n_samples,
                100.0 * batch_idx / len(self.data_loader),
                loss0.item(),loss1.item(),loss_sum.item()
                ))
                return loss_sum
            # retain_graph = True
            self.optimizer.step(closure)
            print(self.model.W_xx_k)
            with torch.no_grad():
                predict_coefficient = self.model.get_coefficient()
                real_coefficient = self.model.get_real_coefficient(xy_batch)
                real_coefficient = [i.to(self.device) for i in real_coefficient]

                # total_loss += loss_sum.item()
                metrics = self._eval_metrics(predict_coefficient, real_coefficient)
                # total_metrics += metrics

                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] '
                                 'p_ux: {:.3f} p_uy: {:.3f} p_uxx: {:.3f} p_uyy: {:.3f}'.format(
                    i,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    metrics[0].item(),metrics[1].item(),metrics[2].item(),metrics[3].item()
                ))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # log = {
        #     'loss': total_loss / len(self.data_loader),
        #     'metrics': (total_metrics / len(self.data_loader)).tolist()
        # }
        #
        # if self.do_validation:
        #     val_log = self._valid_epoch(epoch)
        #     log = {**log, **val_log}
        #
        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()
        #
        # return log

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

        total_loss = 0
        total_metrics = np.zeros(4)
        for batch_idx, (xy_batch,u0_batch,uT_batch) in enumerate(self.data_loader):
            xy_batch, u0_batch, uT_batch = xy_batch.to(self.device), u0_batch.to(self.device), uT_batch.to(self.device)

            torch.autograd.Variable()
            # optimizer = torch.optim.Adam([self.model.W_xx_k], lr=0.0001)
            # optimizer.zero_grad()
            self.optimizer.zero_grad()

            real_coefficient = self.model.get_real_coefficient()
            alpha = real_coefficient[0]
            U_x = []
            U_y = []
            for i in range(49):
                alpha_i = alpha[i,:]
                Diag_m  = torch.ones(48) / (self.model.dx * 2)
                Diag_m[0] = Diag_m[0]*2
                D1 = alpha_i[0:-1] *Diag_m
                Diag_m = (-1) * torch.ones(48) / (self.model.dx * 2)
                Diag_m[-1] = Diag_m[-1] * 2
                D2 = alpha_i[1:] *Diag_m
                W = torch.diag(D1,1) + torch.diag(D2,-1)
                W[0,0] = (-1) * alpha_i[0] / self.model.dx
                W[-1,-1] = alpha_i[-1] / self.model.dx
                W = torch.transpose(W,0,1)
                u_sample = uT_batch[0, i, :, 0]
                u_sample_x = torch.matmul(u_sample,W)
                U_x.append(u_sample_x)
            # u_x = torch.cat(u_x,)
            U_x = torch.stack(U_x)

            alpha = real_coefficient[1]
            for i in range(49):
                alpha_i = alpha[:,i]
                Diag_m  = torch.ones(48) / (self.model.dx * 2)
                Diag_m[0] = Diag_m[0]*2
                D1 = alpha_i[0:-1] *Diag_m
                Diag_m = (-1) * torch.ones(48) / (self.model.dx * 2)
                Diag_m[-1] = Diag_m[-1] * 2
                D2 = alpha_i[1:] *Diag_m
                W = torch.diag(D1,1) + torch.diag(D2,-1)
                W[0,0] = (-1) * alpha_i[0] / self.model.dx
                W[-1,-1] = alpha_i[-1] / self.model.dx
                u_sample = uT_batch[0, :, i, 0]
                u_sample_y = torch.matmul(W,u_sample)
                U_y.append(u_sample_y)
            # u_x = torch.cat(u_x,)
            U_y = torch.stack(U_y,1)

            U_xx = []
            for i in range(49):
                W = self.model.W_xx
                W = torch.transpose(W,0,1)
                u_sample = uT_batch[0, i, :, 0]
                u_sample_xx = torch.matmul(u_sample,W)
                U_xx.append(u_sample_xx)
            U_xx = torch.stack(U_xx)

            U_yy = []
            for i in range(49):
                W = self.model.W_yy
                u_sample = uT_batch[0, :, i, 0]
                u_sample_yy = torch.matmul(W,u_sample)
                U_yy.append(u_sample_yy)
            U_yy = torch.stack(U_yy,1)

            u_t0 = uT_batch[0,:,:,0]
            u_t1 = uT_batch[0,:,:,1]
            U_t = (u_t1 - u_t0) /self.model.dt

            G = U_x + U_y + U_xx + U_yy
            err = torch.sum(torch.abs(U_t - G))
            print(err)

            #计算正确的alpha_ux
            # ???????????
            one = torch.ones(48)
            W = torch.diag(one,1) - torch.diag(one,-1)
            W[0,0] = -1
            W[-1,-1] = 1
            W = torch.transpose(W,0,1)
            u_sample = uT_batch[0,:,:,0]
            real_ux = torch.matmul(u_sample,W)
            # real_ux = real_ux *         \

            u_forward = self.model.cal_from_u0(uT_batch)
            u_t,u_x, u_y, u_xx, u_yy = self.model(uT_batch)

            loss0 = self.loss0(u_forward, uT_batch)
            loss1 = self.loss1(u_t,u_x, u_y, u_xx, u_yy)
            loss_sum = loss1
            loss_sum.backward(retain_graph=True)
            # retain_graph = True
            # a5 = self.model.W_x_2D.grad
            # a6 = self.model.W_xx_k.grad
            # a7 = self.model.W_yy_k.grad
            # a8 = self.model.W_y_2D.grad
            # check grad
            # optimizer.step()
            self.optimizer.step()

            # print(self.model.W_x_2D.grad[0])
            # print(self.model.W_y_2D.grad[0])
            # print(self.model.W_xx_k)
            # print(self.model.W_yy_k)
            # print(self.model.W_xx_k.grad)
            # print(self.model.W_yy_k.grad)
            with torch.no_grad():
                predict_coefficient = self.model.get_coefficient()
                real_coefficient = self.model.get_real_coefficient(xy_batch)
                real_coefficient = [i.to(self.device) for i in real_coefficient]

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss0', loss0.item())
            self.writer.add_scalar('loss1', loss1.item())
            self.writer.add_scalar('loss_sum', loss_sum.item())
            total_loss += loss_sum.item()
            metrics = self._eval_metrics(predict_coefficient, real_coefficient)
            total_metrics += metrics

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] '
                                 'Loss0: {:.3f} Loss1: {:.3f} Loss_sum: {:.3f} '
                                 'p_ux: {:.3f} p_uy: {:.3f} p_uxx: {:.3f} p_uyy: {:.3f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss0.item(),loss1.item(),loss_sum.item(),
                    metrics[0].item(),metrics[1].item(),metrics[2].item(),metrics[3].item()
                ))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

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
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
