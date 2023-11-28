import numpy as np
import model
import os
import dataset_surface_dir as dataset
import train_surface_dir as train
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F


def writelog(txt, filepath, w=True):
    print(txt)
    if w:
        with open(filepath, "a") as f:
            f.write(str(txt) + '\n')


def TRE_one(pos_list, flow_pre, img_dim, img_res):
    flow_pre = F.interpolate(flow_pre.unsqueeze(dim=0), size=[img_dim[-3], img_dim[-2], img_dim[-1]],
                             mode='trilinear').squeeze(dim=0)
    flow_pre = flow_pre.permute(1, 2, 3, 0)
    flow_gt = (pos_list[1, :, :] - pos_list[0, :, :]) * img_res.unsqueeze(dim=0)
    flow_sample = flow_pre[pos_list[0, :, 0].long(), pos_list[0, :, 1].long(), pos_list[0, :, 2].long(), :]
    # 考虑dir_lab中对于像素点的表示是从0开始数还是从1开始数
    loss = (flow_gt - flow_sample).pow(2).sum(1).pow(0.5).mean()
    return loss


class TRE(nn.Module):
    def __init__(self):
        super(TRE, self).__init__()

    def forward(self, pos_dict, flow_pre, img_dim, img_res):
        s = flow_pre.shape
        b = s[0]
        loss = 0
        for i in range(b):
            loss += TRE_one(pos_dict[i, :, :, :], flow_pre[i, :, :, :, :], img_dim[i, :], img_res[i, :])
        return loss / b


for sur in ['x']:
    for leave in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        rr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        del rr[leave]
        epoch = 60
        resolution = 24  # 48-24-12-6
        cuda = 7
        multiphase = True
        other_loss = False
        # model_name = f'lung4d_sur_{sur}_multiphaseTrue_other_lossFalse_conv_pos_emb_seed2023_sigmoid1.8_add_60.pth'
        # model_path = f'/opt/data/private/lmk/4dct_4dlung_full_supervise/out/{model_name}'

        # 迁移学习
        # model_name = f'all_1_sur_x_multiphaseTrue_other_lossFalse_60_fold_0.pth'
        # model_path = f'/opt/data/private/lmk/4dct_4dlung_full_supervise/ft/{model_name}'
        # regnet = torch.load(model_path, map_location=f'cuda:{cuda}')

        # 从头训练
        regnet = model.regnet_2dsur_FCN(resolution=resolution, multi_phase=multiphase, sur=sur)

        path = '/opt/data/private/lmk/4dct_dir_2dsur_finetune/trans'
        head = f'1128_revise_congtou{epoch}'
        filename = f'{head}.txt'
        writelog('=' * 50, os.path.join(path, filename))
        writelog('\n', os.path.join(path, filename))
        writelog(f'leave : {leave}', os.path.join(path, filename))

        train_data = dataset.ct_data(dir='/opt/data/private/lmk/4dct/dir_lab', test=False, r=rr, multiphase=multiphase,
                                     sur=sur)
        test_data = dataset.ct_data(dir='/opt/data/private/lmk/4dct/dir_lab', test=True, r=[leave],
                                    multiphase=multiphase,
                                    sur=sur)
        # 使用dataset0时学习率*10，epoch数翻倍
        dataloader = {'train': DataLoader(train_data, batch_size=1, shuffle=True),
                      'test': DataLoader(test_data, batch_size=1, shuffle=True)}

        criterion = [TRE(), nn.L1Loss()]
        optimizer = optim.Adam(regnet.parameters(), lr=1 * 1e-3)  # norm=False时学习率减少10**3
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        model_ft = train.train_model(regnet, dataloader, criterion, optimizer, scheduler,
                                     num_epochs=epoch, outlog=os.path.join(path, filename), device_ids=[cuda])
        # torch.save(model_ft, os.path.join(path, f'{head}.pth'))
