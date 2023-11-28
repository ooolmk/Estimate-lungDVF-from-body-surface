from torch.nn import functional as F
from torch.utils.data import Dataset
import os
import torch
import random
import numpy as np
import skimage
import nibabel as nib
import scipy.io as scio


def lung_crop(crop, mask, flow, flow_r):
    axis = []
    axis.append(mask[0].sum(1).sum(1))
    axis.append(mask[0].sum(0).sum(1))
    axis.append(mask[0].sum(0).sum(0))
    crop2 = np.zeros_like(crop)
    for i in [0, 1, 2]:
        st = 0
        ed = 0
        a = True
        b = True
        for j in range(len(axis[i])):
            if axis[i][j] > 0:
                a = False
            else:
                st = j
            if axis[i][-j] > 0:
                b = False
            else:
                ed = j
            if not a | b:
                crop2[i, 0] = st
                crop2[i, 1] = len(axis[i]) - 1 - ed

                crop[i, 0] = crop[i, 0] + st
                crop[i, 1] = crop[i, 1] - ed
                break
    mask = mask[:, crop2[0, 0]:crop2[0, 1], crop2[1, 0]:crop2[1, 1], crop2[2, 0]:crop2[2, 1]]
    flow = flow[:, crop2[0, 0]:crop2[0, 1], crop2[1, 0]:crop2[1, 1], crop2[2, 0]:crop2[2, 1]]
    flow_r = flow_r[:, crop2[0, 0]:crop2[0, 1], crop2[1, 0]:crop2[1, 1], crop2[2, 0]:crop2[2, 1]]
    return crop, mask, flow, flow_r


def load_img(folder, filename):
    path = f"{folder}{filename}"
    with open(folder + "info.txt") as infofile:
        img_dims = np.array(infofile.readline().rstrip().split(": ")[1].split(" x "), dtype=int)
    dataobj = np.memmap(path, dtype=np.int16, mode='r', shape=tuple(img_dims), order='F')
    img = nib.AnalyzeImage(dataobj, affine=None).get_fdata()
    # img = img.transpose((1, 0, 2))
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
    pos[:, :, 0] = pos[:, :, 0] - crop[0, 0]
    pos[:, :, 1] = pos[:, :, 1] - crop[1, 0]
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
    crop = mat['crop'] - 1
    return flow, crop


# F:\4dct\manifest-ObLxS9Wd1073675925233948759\4D-Lung
class ct_data(Dataset):
    def __init__(self, dir='/opt/data/private/lmk/4D-Lung-mat', multi_phase=False, test=False, d=(0, 60), sur='x',
                 dataset_r=1, part='all'):
        # 0 60 / 60 80
        self.sur = sur
        self.d = d
        self.dir = dir
        self.test = test
        self.multi_phase = multi_phase

        time_list = []
        for time in os.listdir(dir):
            folder = f"{dir}/{time}"
            time_list.append(folder)
        l = 80  # 共80组4dct
        random.seed(2023)
        random.shuffle(time_list)
        if self.test:
            self.time_list = time_list[d[0]:d[1]]
        else:
            self.time_list = [x for x in time_list if x not in time_list[d[0]:d[1]]]
            if dataset_r != 1:
                self.time_list = time_list[0:int(len(self.time_list) * dataset_r)]
        print(len(self.time_list))
        self.l = len(self.time_list)
        self.part = part
        self.t = 8 if part == 'all' else 4

    def __len__(self):
        if self.test:
            return self.l * self.t
        else:
            return self.l

    def __getitem__(self, index):
        ls = [1, 2, 3, 4, 6, 7, 8, 9]
        if self.test:
            index = index // self.t
            r = index % self.t + (4 if self.part == '6-9' else 0)
            r = ls[r]

        else:
            if self.part == '1-4':
                r = random.randint(0, 3)
            elif self.part == '6-9':
                r = random.randint(4, 7)
            else:
                r = random.randint(0, 7)
            r = ls[r]

        ct_EOE = scio.loadmat(f"{self.time_list[index]}/phase_0.mat")['ct']

        res = torch.tensor([1, 1, 3])

        flow, crop = get_mat(f"{self.time_list[index]}/dvf_0to{5}.mat")
        flow_r, _ = get_mat(f"{self.time_list[index]}/dvf_0to{r}.mat")

        flow = torch.tensor(flow.transpose((3, 0, 1, 2))).float()
        flow_r = torch.tensor(flow_r.transpose((3, 0, 1, 2))).float()

        max = 900
        min = 80
        ct_EOE[ct_EOE > max] = max
        ct_EOE[ct_EOE < min] = min
        ct_EOE = (ct_EOE - min) / (max - min)

        # mask = torch.tensor(np.load(f"{self.time_list[index]}/lung_mask0.npy")).unsqueeze(dim=0)
        # mask = mask[:, crop[0, 0]:crop[0, 1] + 1, crop[1, 0]:crop[1, 1] + 1, crop[2, 0]:crop[2, 1] + 1]

        # mask = torch.tensor(np.load(f"{self.time_list[index]}/crop_lung_mask0.npy")).unsqueeze(dim=0)

        dir = '/opt/data/private/lmk/4D-Lung-label'
        mask_path = f"{dir}/{self.time_list[index].split('/')[-1]}/crop_lung_mask0.nii.gz"
        mask_obj = nib.load(mask_path)
        mask = torch.tensor(mask_obj.get_fdata()).unsqueeze(dim=0)
        crop, mask, flow, flow_r = lung_crop(crop, mask, flow, flow_r)
        # print(mask.shape, mask.max(), mask.min())

        v = '' if self.sur == 'x' else '_y'

        sur2d_EOE = torch.tensor(np.load(f"{self.time_list[index]}/surface_phase0{v}.npy"))
        sur2d_EOI = torch.tensor(np.load(f"{self.time_list[index]}/surface_phase5{v}.npy"))
        sur2d_r = torch.tensor(np.load(f"{self.time_list[index]}/surface_phase{r}{v}.npy"))
        sur2d_r0 = torch.tensor(np.load(f"{self.time_list[index]}/surface_phase{r - 1}{v}.npy"))

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

        # full_sur
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

        if self.multi_phase:
            sub_2dsur = torch.cat([sur2d_EOE - sur2d_EOI, sur2d_EOE - sur2d_r, sur2d_EOE - sur2d_r0], dim=0)
        else:
            sub_2dsur = torch.cat([sur2d_EOE - sur2d_EOI, sur2d_EOE - sur2d_r], dim=0)
        sub_2dsur = F.interpolate(sub_2dsur.unsqueeze(dim=0).float(), size=[192, 192], mode='bilinear').squeeze(dim=0)

        ct_EOE = ct_EOE[crop[0, 0]:crop[0, 1] + 1, crop[1, 0]:crop[1, 1] + 1, crop[2, 0]:crop[2, 1] + 1]

        img_dims = torch.tensor(ct_EOE.shape)



        flow = F.interpolate(flow.unsqueeze(dim=0), size=[192, 192, 192], mode='trilinear').squeeze(dim=0)
        # flow_r = F.interpolate(flow_r.unsqueeze(dim=0), size=[192, 192, 192], mode='trilinear').squeeze(dim=0)
        # print(f"{self.time_list[index]} \t {torch.tensor(ct_EOE).shape}")
        ct_EOE = F.interpolate(torch.tensor(ct_EOE).unsqueeze(dim=0).unsqueeze(dim=0), size=[192, 192, 192],
                               mode='trilinear').squeeze(dim=0)

        # return img_dims, res, flow, ct_EOE, ct_EOI, ct_r, sub_2dsur, flow_r
        return img_dims, res, flow.float(), ct_EOE.float(), sub_2dsur.float(), flow_r.float(), mask.float()

# test = ct_data()
# img_dims, pos_list, res, flow_EOI, sub_2dsur = test.__getitem__(0)
# print(img_dims, pos_dict, res)
