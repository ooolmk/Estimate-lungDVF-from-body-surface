from torch.nn import functional as F
from torch.utils.data import Dataset
import os
import torch
import random
import numpy as np
import skimage
import nibabel as nib
import scipy.io as scio


def load_img(folder, filename):
    path = f"{folder}{filename}"
    with open(folder + "info.txt") as infofile:
        img_dims = np.array(infofile.readline().rstrip().split(": ")[1].split(" x "), dtype=int)
    dataobj = np.memmap(path, dtype=np.int16, mode='r', shape=tuple(img_dims), order='F')
    img = nib.AnalyzeImage(dataobj, affine=None).get_fdata()
    img = img.transpose((1, 0, 2))
    return img


def load_res(folder):
    with open(folder + "info.txt") as infofile:
        infofile.readline()
        img_res = np.array(infofile.readline().rstrip().split(": ")[1].split(" x "), dtype=float)
        img_res = torch.tensor(list(map(float, img_res)))
    return img_res


def load_pos(folder, idx, r, crop):
    path = folder
    phase_list = ['T00', f'T{r}0']
    pos = []
    for phase in phase_list:
        with open(path + f"case{idx}_4D-75_{phase}.txt") as posfile:
            posfile = posfile.read().split('\n')[0:-1]
            pos_list = []
            for i in posfile:
                pos_list.append(list(map(float, i.split('\t')[0:3])))
            pos.append(torch.tensor(pos_list).unsqueeze(dim=0))
    pos = torch.cat([pos[0], pos[1]], dim=0)

    pos[:, :, 0] = pos[:, :, 1] - crop[0, 0]
    pos[:, :, 1] = pos[:, :, 0] - crop[1, 0]
    pos[:, :, 2] = pos[:, :, 2] - crop[2, 0]
    return pos


def get_surface(ct, closing=True):
    mask = (ct > 600 / (900 - 80)) * 1
    label_map = skimage.measure.label(mask, 1)
    img = (label_map == 1) * 1

    if closing:
        kernel = skimage.morphology.ball(4)
        img = skimage.morphology.closing(img, kernel)
    kernel = skimage.morphology.ball(1)
    img_dialtion = skimage.morphology.dilation(img, kernel)
    edge = img_dialtion - img

    return edge


def get_2dsurface(edge):
    edge = torch.tensor(edge)
    xd, yd, zd = edge.shape
    sur2d = torch.zeros_like(edge[:, 0, :])

    for i in range(xd):
        for j in range(zd):
            x = edge[i, :, j]
            xx = (x > 0).nonzero()
            if len(xx) == 0:
                x0 = xd - 1
            else:
                x0 = xx[0]
            sur2d[i, j] = torch.tensor(x0)
    return sur2d.unsqueeze(dim=0)


def get_mat(path):
    mat = scio.loadmat(path)
    flow = mat['Tptv']
    crop = mat['crop_v'] - 1

    flow = flow.transpose((1, 0, 2, 3))
    a = crop[0, :] + 0
    crop[0, :] = crop[1, :]
    crop[1, :] = a

    return flow, crop


# F:\4dct\manifest-ObLxS9Wd1073675925233948759\4D-Lung
class ct_data(Dataset):
    def __init__(self, dir='/opt/data/private/lmk/4dct/dir_lab', test=False, r=(0, 0.7), multiphase=False, sur='x'):
        self.r = r
        self.dir = dir
        self.test = test
        self.multiphase = multiphase
        self.sur = sur

        person_file = []
        for idx in range(1, 11):
            folder = f"{dir}/Case" + str(idx) + "Pack/Images/"
            filelist = os.listdir(folder)
            phase_file = []
            for n in range(10):
                phase_file.append([name for name in filelist if f"T{n}0" in name][0])
            person_file.append(phase_file)

        person_file = np.array(person_file)  # 10,10 第一维为10个被试，第二位为每次4dct包含10帧率
        self.person_file = person_file[r]
        self.l = len(self.person_file)
        print(len(self.person_file))

    def __len__(self):
        if self.test:
            return 4
        else:
            return self.l

    def __getitem__(self, index):
        # print(index)
        ini = 1
        phase_list = self.person_file[0] if self.test else self.person_file[index]
        # print(self.time_list[index])

        if self.test:
            r = index + 1

        else:
            # ls = [0, 1, 2, 3, 4, 5]
            # r = random.randint(1, 5)
            # r = random.randint(1, 4)
            r = random.randint(1, 4)
            # r = ls[r]

        folder = f"{self.dir}/Case{self.r[0] + 1}Pack/Images/" if self.test \
            else f"{self.dir}/Case{self.r[index] + ini}Pack/Images/"
        ct_EOE = load_img(folder, phase_list[0])
        ct_EOI = load_img(folder, phase_list[5])
        ct_r = load_img(folder, phase_list[r])
        res = load_res(folder)

        # path = f"{self.dir}/ft/DIR_0to5_idx{self.r[0] + 1}use_refinement0_resize0_fast_lcc1.mat" if self.test \
        #     else f"{self.dir}/ft/DIR_0to5_idx{self.r[index] + ini}use_refinement0_resize0_fast_lcc1.mat"
        path = f"{self.dir}/DIR_0to5_idx{self.r[0] + 1}use_refinement0_resize0_fast_lcc1.mat" if self.test \
            else f"{self.dir}/DIR_0to5_idx{self.r[index] + ini}use_refinement0_resize0_fast_lcc1.mat"
        flow, crop = get_mat(path)
        # path_r = f"{self.dir}/ft/DIR_0to{r}_idx{self.r[0] + 1}use_refinement0_resize0_fast_lcc1.mat" if self.test \
        #     else f"{self.dir}/ft/DIR_0to{r}_idx{self.r[index] + ini}use_refinement0_resize0_fast_lcc1.mat"
        path_r = f"{self.dir}/DIR_0to{r}_idx{self.r[0] + 1}use_refinement0_resize0_fast_lcc1.mat" if self.test \
            else f"{self.dir}/DIR_0to{r}_idx{self.r[index] + ini}use_refinement0_resize0_fast_lcc1.mat"
        if r != 0:
            flow_r, _ = get_mat(path_r)
        else:
            flow_r = np.zeros_like(flow)

        folder_pos = f"{self.dir}/Case{self.r[0] + 1}Pack/Sampled4D/" if self.test \
            else f"{self.dir}/Case{self.r[index] + ini}Pack/Sampled4D/"
        pos_list = load_pos(folder_pos, (self.r[0] + 1) if self.test else (self.r[index] + 1), r, crop)

        max = 900
        min = 80
        ct_EOE[ct_EOE > max] = max
        ct_EOE[ct_EOE < min] = min
        ct_EOE = (ct_EOE - min) / (max - min)

        ct_EOI[ct_EOI > max] = max
        ct_EOI[ct_EOI < min] = min
        ct_EOI = (ct_EOI - min) / (max - min)

        ct_r[ct_r > max] = max
        ct_r[ct_r < min] = min
        ct_r = (ct_r - min) / (max - min)

        # edge_EOE = get_surface(ct_EOE)
        # sur2d_EOE = get_2dsurface(edge_EOE[crop[0, 0]:crop[0, 1] + 1, :, crop[2, 0]:crop[2, 1] + 1])
        #
        # edge_EOI = get_surface(ct_EOI)
        # sur2d_EOI = get_2dsurface(edge_EOI[crop[0, 0]:crop[0, 1] + 1, :, crop[2, 0]:crop[2, 1] + 1])
        #
        # edge_r = get_surface(ct_r)
        # sur2d_r = get_2dsurface(edge_r[crop[0, 0]:crop[0, 1] + 1, :, crop[2, 0]:crop[2, 1] + 1])
        v = '' if self.sur == 'x' else '_y'
        sur2d_EOE = torch.tensor(np.load(
            f'{self.dir}/sur2d/case{self.r[0] + ini}_phase{0}{v}.npy' if self.test else f'{self.dir}/sur2d/case{self.r[index] + ini}_phase{0}{v}.npy'))
        sur2d_EOI = torch.tensor(np.load(
            f'{self.dir}/sur2d/case{self.r[0] + ini}_phase{5}{v}.npy' if self.test else f'{self.dir}/sur2d/case{self.r[index] + ini}_phase{5}{v}.npy'))
        sur2d_r = torch.tensor(np.load(
            f'{self.dir}/sur2d/case{self.r[0] + ini}_phase{r}{v}.npy' if self.test else f'{self.dir}/sur2d/case{self.r[index] + ini}_phase{r}{v}.npy'))
        sur2d_r0 = torch.tensor(np.load(
            f'{self.dir}/sur2d/case{self.r[0] + ini}_phase{r - 1}{v}.npy' if self.test else f'{self.dir}/sur2d/case{self.r[index] + ini}_phase{r - 1}{v}.npy'))

        if self.sur == 'x':
            sur2d_EOE = sur2d_EOE[:, crop[1, 0]:crop[1, 1] + 1, crop[2, 0]:crop[2, 1] + 1]
            sur2d_EOI = sur2d_EOI[:, crop[1, 0]:crop[1, 1] + 1, crop[2, 0]:crop[2, 1] + 1]
            sur2d_r = sur2d_r[:, crop[1, 0]:crop[1, 1] + 1, crop[2, 0]:crop[2, 1] + 1]
            sur2d_r0 = sur2d_r0[:, crop[1, 0]:crop[1, 1] + 1, crop[2, 0]:crop[2, 1] + 1]
        else:
            sur2d_EOE = sur2d_EOE[:, crop[0, 0]:crop[0, 1] + 1, crop[2, 0]:crop[2, 1] + 1]
            sur2d_EOI = sur2d_EOI[:, crop[0, 0]:crop[0, 1] + 1, crop[2, 0]:crop[2, 1] + 1]
            sur2d_r = sur2d_r[:, crop[0, 0]:crop[0, 1] + 1, crop[2, 0]:crop[2, 1] + 1]
            sur2d_r0 = sur2d_r0[:, crop[0, 0]:crop[0, 1] + 1, crop[2, 0]:crop[2, 1] + 1]

        # a = sur2d_EOE.shape
        # l1 = a[1]
        # s1 = l1 / 3
        # min1 = int(l1 / 2 - s1)
        # max1 = int(l1 / 2 + s1)
        # l2 = a[2]
        # s2 = l2 / 3
        # min2 = int(l2 / 2 - s2)
        # max2 = int(l2 / 2 + s2)
        # sur2d_EOE = sur2d_EOE[:, min1:max1, min2:max2]
        # sur2d_EOI = sur2d_EOI[:, min1:max1, min2:max2]
        # sur2d_r = sur2d_r[:, min1:max1, min2:max2]
        # sur2d_r0 = sur2d_r0[:, min1:max1, min2:max2]

        if self.multiphase:
            sub_2dsur = torch.cat([sur2d_EOE - sur2d_EOI, sur2d_EOE - sur2d_r, sur2d_EOE - sur2d_r0], dim=0)
        else:
            sub_2dsur = torch.cat([sur2d_EOE - sur2d_EOI, sur2d_EOE - sur2d_r], dim=0)
        sub_2dsur = F.interpolate(sub_2dsur.unsqueeze(dim=0).float(), size=[192, 192], mode='bilinear').squeeze(dim=0)

        ct_EOE = ct_EOE[crop[0, 0]:crop[0, 1] + 1, crop[1, 0]:crop[1, 1] + 1, crop[2, 0]:crop[2, 1] + 1]
        ct_EOI = ct_EOI[crop[0, 0]:crop[0, 1] + 1, crop[1, 0]:crop[1, 1] + 1, crop[2, 0]:crop[2, 1] + 1]
        ct_r = ct_r[crop[0, 0]:crop[0, 1] + 1, crop[1, 0]:crop[1, 1] + 1, crop[2, 0]:crop[2, 1] + 1]
        img_dims = torch.tensor(ct_r.shape)

        flow = torch.tensor(flow.transpose((3, 0, 1, 2))).float()
        flow_r = torch.tensor(flow_r.transpose((3, 0, 1, 2))).float()
        ct_EOE = torch.tensor(ct_EOE).unsqueeze(dim=0).float()
        ct_EOI = torch.tensor(ct_EOI).unsqueeze(dim=0).float()
        ct_r = torch.tensor(ct_r).unsqueeze(dim=0).float()

        flow = F.interpolate(flow.unsqueeze(dim=0), size=[192, 192, 192], mode='trilinear').squeeze(dim=0)
        flow_r = F.interpolate(flow_r.unsqueeze(dim=0), size=[192, 192, 192], mode='trilinear').squeeze(dim=0)
        ct_EOE = F.interpolate(ct_EOE.unsqueeze(dim=0), size=[192, 192, 192], mode='trilinear').squeeze(dim=0)
        ct_EOI = F.interpolate(ct_EOI.unsqueeze(dim=0), size=[192, 192, 192], mode='trilinear').squeeze(dim=0)
        ct_r = F.interpolate(ct_r.unsqueeze(dim=0), size=[192, 192, 192], mode='trilinear').squeeze(dim=0)

        return img_dims, pos_list, res, flow, ct_EOE, ct_EOI, ct_r, sub_2dsur, flow_r

# test = ct_data()
# img_dims, pos_list, res, flow_EOI, sub_2dsur = test.__getitem__(0)
# print(img_dims, pos_dict, res)
