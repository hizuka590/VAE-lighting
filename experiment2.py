import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from helper_misc_tensor import check_mkdir, cal_precision_recall_mae, AvgMeter, cal_fmeasure, cal_Jaccard, cal_BER

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        # print('======')
        # decoded, inputted, mued, log_vared = results[0],results[1],results[2],results[3]
        # print(decoded.shape,torch.unique(decoded),torch.max(decoded,torch.min(decoded)))
        # print('mu  log :', mued,log_vared)
        train_loss = self.model.loss_function(*results,labels=labels,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        precision_record, recall_record, mae_record, Jaccard_record, BER_record, shadow_BER_record, non_shadow_BER_record = eval_metrics_init()
        real_img, labels = batch
        # print(real_img.shape,labels.shape)
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        shadow = results[4]
        precision, recall, mae = cal_precision_recall_mae(shadow, labels)
        Jaccard = cal_Jaccard(shadow, labels)

        Jaccard_record.update(Jaccard)
        BER, shadow_BER, non_shadow_BER = cal_BER(shadow, labels)
        BER_record.update(BER)
        shadow_BER_record.update(shadow_BER)
        non_shadow_BER_record.update(non_shadow_BER)

        for pidx, pdata in enumerate(zip(precision, recall)):
            p, r = pdata
            precision_record[pidx].update(p)
            recall_record[pidx].update(r)
        mae_record.update(mae)
        # break

        fmeasure = cal_fmeasure([precord.avg for precord in precision_record],[rrecord.avg for rrecord in recall_record])
        # print('=====validation====')
        # decoded, inputted, mued, log_vared = results[0],results[1],results[2],results[3]
        # print(decoded.shape)
        # print('mu  log :', mued.shape,log_vared.shape)
        # print(len(results),print(results[4].shape,results[5].shape))
        val_loss = self.model.loss_function(*results,labels=labels,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
        log = 'MAE:{}, F-beta:{}, Jaccard:{}, BER:{}, SBER:{}, non-SBER:{}'.format(mae_record.avg, fmeasure,
                                                                                       Jaccard_record.avg,
                                                                                       BER_record.avg,
                                                                                       shadow_BER_record.avg,
                                                                                       non_shadow_BER_record.avg)
        critia_dict = {
            'MAE':mae_record.avg,
            'F-measure':fmeasure,
            'Jaccard':Jaccard_record.avg,
            'BER':BER_record.avg,
            's_BER':shadow_BER_record.avg,
            'ns_BER':non_shadow_BER_record.avg
        }

        val_loss = dict(list(val_loss.items()) + list(critia_dict.items()))
        # print(val_loss)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)


        
    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons, shadow = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}_r.png"),
                          normalize=True,
                          nrow=12)
        vutils.save_image(test_input.data,
                          os.path.join(self.logger.log_dir ,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}_o.png"),
                          normalize=True,
                          nrow=12)

        # try:
        #     samples_recons,samples_shadow = self.model.sample(144,
        #                                 self.curr_device,
        #                                 labels = test_label)
        #     vutils.save_image(samples_recons.cpu().data,
        #                       os.path.join(self.logger.log_dir ,
        #                                    "Samples",
        #                                    f"{self.logger.name}_Epoch_{self.current_epoch}_r.png"),
        #                       normalize=True,
        #                       nrow=12)
        #     vutils.save_image(samples_shadow.cpu().data,
        #                       os.path.join(self.logger.log_dir ,
        #                                    "Samples",
        #                                    f"{self.logger.name}_Epoch_{self.current_epoch}_s.png"),
        #                       normalize=True,
        #                       nrow=12)
        # except Warning:
        #     print('saving failed')

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims


def eval_metrics_init():
    precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
    mae_record = AvgMeter()
    Jaccard_record = AvgMeter()
    BER_record = AvgMeter()
    shadow_BER_record = AvgMeter()
    non_shadow_BER_record = AvgMeter()
    return precision_record, recall_record, mae_record,Jaccard_record, BER_record, shadow_BER_record,non_shadow_BER_record