from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image
import torch
from torch import nn
import random
import torch.optim as optim
import time
from torch.nn import functional as F


def writelog(txt, filepath, w=True):
    print(txt)
    if w:
        with open(filepath, "a") as f:
            f.write(str(txt) + '\n')


def train_model(regnet,
                dataloader,
                criterion,
                optimizer,
                scheduler,
                num_epochs,
                outlog,
                device_ids=[0, 1],
                ):
    if len(device_ids) > 1:
        USE_MULTI_GPU = True
        # 检测机器是否有多张显卡
        if USE_MULTI_GPU and torch.cuda.device_count() > 1:
            MULTI_GPU = True
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        else:
            MULTI_GPU = False

        if MULTI_GPU:
            optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
            optimizer = optimizer.module
            scheduler = nn.DataParallel(scheduler, device_ids=device_ids)
            scheduler = scheduler.module
            regnet = nn.DataParallel(regnet, device_ids=device_ids)

    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    regnet.to(device)

    print(f'train on {device}')
    for epoch in range(num_epochs):
        writelog('Epoch {}/{}'.format(epoch, num_epochs - 1), outlog)
        writelog('-' * 10, outlog)
        # 训练和验证

        for phase in ['test', 'train', 'test'] if epoch == 0 | (((epoch + 1) % (num_epochs // 3)) == 0) else ['train']:

            if phase == 'train':
                regnet.train()  # 训练
            else:
                regnet.eval()

            running_loss0 = 0.0
            running_loss1 = 0.0
            running_iniloss0 = 0.0
            running_iniloss1 = 0.0

            # running_hiloss0 = 0.0

            num = 0
            for img_dims, pos_dict, res, flow, ct_EOE, ct_EOI, ct_r, sub_2dsur, flow_r in dataloader[phase]:

                img_dims = img_dims.to(device)
                pos_dict = pos_dict.to(device)
                res = res.to(device)
                flow = flow.to(device)
                ct_EOE = ct_EOE.to(device)
                ct_EOI = ct_EOI.to(device)
                ct_r = ct_r.to(device)
                sub_2dsur = sub_2dsur.to(device)
                flow_r = flow_r.to(device)

                # 清零

                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    flowx = regnet(flow, ct_EOE, sub_2dsur)
                    pre_flow = flow * flowx
                    # pre_flow = flowx

                    iloss0 = criterion[0](pos_dict, torch.zeros_like(pre_flow), img_dims, res)
                    # iloss0 = criterion[0](pos_dict, 0.5 * flow, img_dims, res)

                    iloss1 = criterion[1](flow_r, 0.5 * flow)

                    loss0 = criterion[0](pos_dict, pre_flow, img_dims, res)
                    loss1 = criterion[1](flow_r, pre_flow)

                    if phase == 'train':
                        loss_s = loss0 + loss1
                        # loss_s = loss0 * 2
                        # loss_s = loss1 * 4
                        loss_s.backward()
                        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                        optimizer.step()
                # 计算损失
                running_loss0 += loss0.detach() * ct_EOE.shape[0]
                running_loss1 += loss1.detach() * ct_EOE.shape[0]
                running_iniloss0 += iloss0.detach() * ct_EOE.shape[0]
                running_iniloss1 += iloss1.detach() * ct_EOE.shape[0]
                # running_hiloss0 += hiloss0.detach() * ct_EOE.shape[0]
                num += ct_EOE.shape[0]
            epoch_loss0 = running_loss0 / num
            epoch_loss1 = running_loss1 / num
            epoch_iniloss0 = running_iniloss0 / num
            epoch_iniloss1 = running_iniloss1 / num
            # epoch_hiloss0 = running_hiloss0 / num
            writelog(
                '{} TRE_Loss:{:.4f} TRE_ini_Loss:{:.4f}  flow_L1_Loss:{:.4f} flow_L1_ini_Loss:{:.4f}'.format(
                    phase,
                    epoch_loss0,
                    epoch_iniloss0,
                    # epoch_hiloss0,
                    epoch_loss1,
                    epoch_iniloss1), outlog)

        scheduler.step()
        writelog('', outlog)
    return regnet
