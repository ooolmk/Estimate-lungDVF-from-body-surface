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
                other_loss=True
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
        for phase in ['train', 'test'] if (epoch + 1) % (num_epochs//3) == 0 else ['train']:
        # for phase in ['train']:
            if phase == 'train':
                regnet.train()  # 训练

            else:
                regnet.eval()

            running_loss = 0.0
            running_iniloss = 0.0

            num = 0
            for img_dims, res, flow, ct_EOE, sub_2dsur, flow_r, mask in dataloader[phase]:
                # print(flow_r.shape)
                # img_dims = img_dims.to(device)
                # res = res.to(device)
                flow = flow.to(device)
                ct_EOE = ct_EOE.to(device)
                # ct_EOI = ct_EOI.to(device)
                # ct_r = ct_r.to(device)
                sub_2dsur = sub_2dsur.to(device)
                flow_r = flow_r.to(device)
                mask = mask.to(device)
                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    flowx = regnet(flow, ct_EOE, sub_2dsur)
                    pre_flow = flow * flowx

                    if (phase == 'train') & other_loss:
                        loss = criterion(flow_r, pre_flow, torch.ones_like(mask))
                        # iloss = criterion(flow_r, torch.zeros_like(pre_flow), torch.ones_like(mask))
                    else:
                        # iloss = criterion(flow_r, torch.zeros_like(pre_flow), mask)
                        loss = criterion(flow_r, pre_flow, mask)

                    if phase == 'train':
                        loss_s = loss
                        loss_s.backward()
                        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                        optimizer.step()
                # 计算损失
                running_loss += loss.detach() * ct_EOE.shape[0]
                # running_iniloss += iloss.detach() * ct_EOE.shape[0]

                num += ct_EOE.shape[0]
            epoch_loss = running_loss / num
            # epoch_iniloss = running_iniloss / num
            # epoch_hiloss0 = running_hiloss0 / num
            writelog(
                '{} flow_L1_Loss:{:.4f}'.format(
                    phase,
                    epoch_loss), outlog)

        scheduler.step()
        writelog('', outlog)
    return regnet
