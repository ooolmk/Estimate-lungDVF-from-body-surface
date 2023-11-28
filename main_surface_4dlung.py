import numpy as np

import model
import dataset_surface_4dlung as dataset
import train_surface_4dlung as train
import os
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from test_surface_4dlung import test_model


def TRE_one(gt, pre, mask, part=False):
    dim = gt.shape
    pre = F.interpolate(pre.unsqueeze(dim=0), size=[dim[-3], dim[-2], dim[-1]], mode='trilinear').squeeze(dim=0)
    if part:
        L1 = ((gt - pre).pow(2).sum(0).pow(0.5) * mask)
        return L1

    else:
        loss = ((gt - pre).pow(2).sum(0).pow(0.5) * mask).sum() / mask.sum()
        return loss


class TRE(nn.Module):
    def __init__(self):
        super(TRE, self).__init__()

    def forward(self, gt, pre, mask, part=False):
        loss = TRE_one(gt[0, :, :, :, :], pre[0, :, :, :, :], mask[0, 0, :, :, :], part=part)
        return loss


cuda = 6
for dataset_r in [1]:
    for fold in [0, 1, 2, 3, 4, 'all']:  # 0,1,2,3,4
        for part in ['all']:
            sur = 'x'
            epoch = int(60 / dataset_r)
            resolution = 24  # 48-24-12-6
            multi_phase = True
            other_loss = False
            path = '/opt/data/private/lmk/4dct_4dlung_full_supervise/ft'
            head = f'{part}_{dataset_r}_sur_{sur}_multiphase{multi_phase}_other_loss{other_loss}_{epoch}'  # learnable_pos_emb_
            print(head)
            filename = f'{head}_fold_{fold}.txt'
            if fold == 'all':
                train_data = dataset.ct_data(dir='/opt/data/private/lmk/4D-Lung-mat', multi_phase=multi_phase,
                                             test=False,
                                             d=(0, 0), sur=sur, dataset_r=dataset_r, part=part)
                test_data = dataset.ct_data(dir='/opt/data/private/lmk/4D-Lung-mat', multi_phase=multi_phase, test=True,
                                            d=(0, 0), sur=sur, part=part)
            else:
                train_data = dataset.ct_data(dir='/opt/data/private/lmk/4D-Lung-mat', multi_phase=multi_phase,
                                             test=False,
                                             d=(fold * 16, (fold + 1) * 16), sur=sur, dataset_r=dataset_r, part=part)
                test_data = dataset.ct_data(dir='/opt/data/private/lmk/4D-Lung-mat', multi_phase=multi_phase, test=True,
                                            d=(fold * 16, (fold + 1) * 16), sur=sur, part=part)
            # 使用dataset0时学习率*10，epoch数翻倍
            dataloader = {'train': DataLoader(train_data, batch_size=1, shuffle=True),
                          'test': DataLoader(test_data, batch_size=1, shuffle=False)}

            regnet = model.regnet_2dsur_FCN(resolution=resolution, multi_phase=multi_phase, sur=sur)

            criterion = TRE()
            optimizer = optim.Adam(regnet.parameters(),
                                   lr=2 * 1e-3 if other_loss else 1 * 1e-3)  # norm=False时学习率减少10**3
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(10 / dataset_r), gamma=0.5)

            model_ft = train.train_model(regnet, dataloader, criterion, optimizer, scheduler, num_epochs=epoch,
                                         outlog=os.path.join(path, filename), device_ids=[cuda], other_loss=other_loss)
            torch.save(model_ft, os.path.join(path, f'{head}_fold_{fold}.pth'))

            test_model(model_ft, dataloader['test'], criterion, os.path.join(path, f'test_fold{fold}_{head}.txt'),
                       device_ids=[cuda])
