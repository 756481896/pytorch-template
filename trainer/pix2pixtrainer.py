import numpy as np
import torch
from torchvision.utils import make_grid
from base.BaseTrainerGAN import BaseTrainerGAN
from model.loss import equation_loss
import pdb
import os
import matplotlib.tri as tri
import matplotlib.pyplot as plt
class TrainerGAN(BaseTrainerGAN):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, modelD, modelG, loss1, loss2, Lambda, metrics, optimizerD, optimizerG, resume, config,
                 data_loader, valid_data_loader=None, lr_schedulerD=None, lr_schedulerG=None, train_logger=None):
        super(TrainerGAN, self).__init__(modelD, modelG, loss1, loss2, metrics, optimizerD, optimizerG, resume, config,train_logger)
        #loss1 BCEloss,loss2 MSEloss
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_schedulerD = lr_schedulerD
        self.lr_schedulerG = lr_schedulerG
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.Lambda = Lambda
    def _eval_metrics(self, output, target,metrics):
        acc_metrics = np.zeros(len(metrics))
        for i, metric in enumerate(metrics):
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
        self.modelD.train()
        self.modelG.train()

        total_lossD = 0
        total_lossG = 0
        total_metricsD = np.zeros(len(self.metrics))
        # total_metrics_G = np.zeros(len(self.metricsG))

        for batch_idx, (InputD,TargetD,InputG,TargetG) in enumerate(self.data_loader):
            #InputD:[u,alpha] shape:[batch_size,2,64,64],TargetD:概率值 shape:[batch_size,1,1,1]
            #InputG:u shape:[batch_size,1,64,64],TargetG:alpha shape:[batch_size,1,64,64]
            InputD, TargetD, InputG, TargetG = \
                InputD.to(self.device), TargetD.to(self.device), InputG.to(self.device), TargetG.to(self.device)
            # print(InputD.shape,TargetD.shape,InputG.shape,TargetG.shape)
            #G生成结果
            Fake_alpha = self.modelG(InputG)
            Fake_InputD = torch.cat((InputG,Fake_alpha),dim=1)
            Fake_TargetD = torch.zeros_like(TargetD)
            #D把G生成的结果全为判断为0

            #检查模型D的架构与输出
            assert Fake_InputD.shape == InputD.shape

            #更新D
            self.optimizerD.zero_grad()
            predictD = self.modelD(InputD)
            #[64,1,1,1]
            Fake_predictD = self.modelD(Fake_InputD)
            lossDtrue = self.loss1(predictD, TargetD)
            lossDfake = self.loss1(Fake_predictD, Fake_TargetD)
            lossD = 0.5*(lossDtrue + lossDfake)
            lossD.backward()
            self.optimizerD.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('lossDtrue', lossDtrue.item())
            self.writer.add_scalar('lossDfake', lossDfake.item())
            self.writer.add_scalar('lossD', lossD.item())

            cmap = plt.cm.rainbow
            fig = plt.figure()
            w_vertex = InputG[0,0,:,:].view(-1)
            plt.tripcolor(self.data_loader.triangulation, w_vertex, cmap=cmap)
            self.writer.add_figure(fig)
            total_lossD += lossD.item()
            #暂时只记录真实样本的metrics
            total_metricsD += self._eval_metrics(predictD, TargetD,self.metrics)


            #更新G
            # train_rate = 3
            # for i in range(train_rate):
            self.optimizerG.zero_grad()
            predictG = self.modelG(InputG)
            Fake_InputD = torch.cat((InputG, predictG), dim = 1)
            Fake_predictD = self.modelD(Fake_InputD)
            Fake_TargetD = torch.ones_like(TargetD)
            lossG1 = self.loss1(Fake_predictD,Fake_TargetD)
            lossG2 = self.loss1(predictG,TargetG)
            lossG3 = equation_loss(InputG,predictG)
            # 这里参数要调
            lossG = self.Lambda[0] * lossG1 + self.Lambda[1] * lossG2 + self.Lambda[2] * lossG3

            lossG.backward()
            self.optimizerG.step()

            self.writer.add_scalar('lossG1', lossG1.item())
            self.writer.add_scalar('lossG2', lossG2.item())
            self.writer.add_scalar('lossG3', lossG3.item())
            self.writer.add_scalar('lossG', lossG.item())
            total_lossG += lossG.item()
            #暂时只记录真实样本的metrics

            if self.verbosity >= 2 and batch_idx*2 % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] LossD: {:.6f} LossG: {:.6f} LossG1: {:.6f} LossG2: {:.6f} LossG3: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    lossD.item(),
                    lossG.item(),
                    lossG1.item(),
                    lossG2.item(),
                    lossG3.item()
                ))


        log = {
            'lossD': total_lossD / len(self.data_loader),
            'lossG': total_lossG / len(self.data_loader),
            'metrics': (total_metricsD / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_schedulerD is not None:
            self.lr_schedulerD.step()
        if self.lr_schedulerG is not None:
            self.lr_schedulerG.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.modelD.eval()
        self.modelG.eval()
        total_val_lossD = 0
        total_val_lossG = 0
        with torch.no_grad():
            for batch_idx, (InputD, TargetD, InputG, TargetG) in enumerate(self.valid_data_loader):
                # InputD:[u,alpha],TargetD:概率值,InputG:u,TargetG:alpha
                InputD, TargetD, InputG, TargetG = \
                    InputD.to(self.device), TargetD.to(self.device), InputG.to(self.device), TargetG.to(self.device)

                # G生成结果
                Fake_alpha = self.modelG(InputG)
                Fake_InputD = torch.cat((InputG, Fake_alpha), dim = 1)
                Fake_TargetD = torch.zeros_like(TargetD)
                # D把G生成的结果全为判断为0
                assert Fake_InputD.shape == InputD.shape


                predictD = self.modelD(InputD)
                Fake_predictD = self.modelD(Fake_InputD)
                lossDtrue = self.loss1(predictD, TargetD)
                lossDfake = self.loss1(Fake_predictD, Fake_TargetD)

                # lossDtrue = -torch.mean(predictD)
                # lossDfake = torch.mean(Fake_predictD)

                lossD = lossDtrue + lossDfake

                predictG = self.modelG(InputG)
                lossG1 = self.loss1(Fake_predictD, Fake_TargetD)
                lossG2 = self.loss2(predictG, TargetG)
                lossG3 = equation_loss(InputG,predictG)
                # 这里参数要调
                lossG = self.Lambda[0] * lossG1 + self.Lambda[1] * lossG2 + self.Lambda[2] * lossG3
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('lossDtrue', lossDtrue.item())
                self.writer.add_scalar('lossDfake', lossDfake.item())
                self.writer.add_scalar('lossD', lossD.item())
                self.writer.add_scalar('lossG1', lossG1.item())
                self.writer.add_scalar('lossG2', lossG2.item())
                self.writer.add_scalar('lossG3', lossG3.item())
                self.writer.add_scalar('lossG', lossG.item())

                total_val_lossD += lossD.item()
                total_val_lossG += lossG.item()


        return {
            'val_lossD': total_val_lossD / len(self.valid_data_loader),
            'val_lossG': total_val_lossG / len(self.valid_data_loader)
        }

