import torch

from Unet import *


class regnet_2dsur_FCN(nn.Module):
    def __init__(self, resolution=24, multi_phase=False, sur='x'):
        super(regnet_2dsur_FCN, self).__init__()
        self.resolution = resolution
        c = 32
        d = 192 // resolution
        st = ['start']
        for i in range(5):
            st.append('down' if (d // 2) > 0 else 'end')
            d = d // 2

        self.Generate = FCN([4, c, 2 * c, 4 * c, 8 * c, 8 * c],
                            [c, 2 * c, 4 * c, 8 * c, 8 * c, 8 * c],
                            st,
                            # ['start', 'down', 'down', 'down', 'end', 'end'],
                            #  192      96      48       24     12     6
                            depth=6)
        i = 3 if multi_phase else 2
        self.VGG = VGG([i, 1 * c, 2 * c],
                       [c, 2 * c, 4 * c], sur=sur)

    def forward(self, flow, ct_EOE, sur2d):
        sur_feature = self.VGG(sur2d, self.resolution)
        ct_in = torch.cat([flow, ct_EOE], dim=1)
        # ct_in = flow
        flowx = self.Generate(ct_in, sur_feature)
        # print(flowx.shape)
        flowx = F.interpolate(flowx, size=[192, 192, 192], mode='trilinear')
        # warper3d = Warper3d(ct_EOE.shape[-3::])
        # pre_ct_r = warper3d(ct_EOE, flowx * flow)
        # return flowx, pre_ct_r, phase
        return flowx


class regnet_2dsur_xy_FCN(nn.Module):
    def __init__(self, resolution=24, multi_phase=False):
        super(regnet_2dsur_xy_FCN, self).__init__()
        self.resolution = resolution
        c = 32
        d = 192 // resolution
        st = ['start']
        for i in range(5):
            st.append('down' if (d // 2) > 0 else 'end')
            d = d // 2

        self.Generate = FCN([4, c, 2 * c, 4 * c, 8 * c, 8 * c],
                            [c, 2 * c, 4 * c, 8 * c, 8 * c, 8 * c],
                            st,
                            # ['start', 'down', 'down', 'down', 'end', 'end'],
                            #  192      96      48       24     12     6
                            depth=6)
        i = 3 if multi_phase else 2
        self.VGG_xy = VGG_xy([i, 1 * c, 2 * c],
                             [c, 2 * c, 4 * c])

    def forward(self, flow, ct_EOE, sur2d_x, sur2d_y):
        sur_feature = self.VGG_xy(sur2d_x, sur2d_y, self.resolution)
        ct_in = torch.cat([flow, ct_EOE], dim=1)
        # ct_in = flow
        flowx = self.Generate(ct_in, sur_feature)
        # print(flowx.shape)
        flowx = F.interpolate(flowx, size=[192, 192, 192], mode='trilinear')
        # warper3d = Warper3d(ct_EOE.shape[-3::])
        # pre_ct_r = warper3d(ct_EOE, flowx * flow)
        # return flowx, pre_ct_r, phase
        return flowx


class regnet_2dsur_FCN_no3d(nn.Module):
    def __init__(self, cuda, resolution=24):
        super(regnet_2dsur_FCN_no3d, self).__init__()
        self.resolution = resolution
        c = 32
        self.out = nn.Sequential(
            nn.Conv3d(256, 1, 1, bias=True),
            nn.Sigmoid()
        )

        self.VGG = VGG([2, 1 * c, 2 * c],
                       [c, 2 * c, 4 * c], cuda=cuda)

    def forward(self, flow, ct_EOE, sur2d):
        sur_feature = self.VGG(sur2d, self.resolution)

        flowx = self.out(sur_feature) * 1.8
        # print(flowx.shape)
        flowx = F.interpolate(flowx, size=[192, 192, 192], mode='trilinear')
        # warper3d = Warper3d(ct_EOE.shape[-3::])
        # pre_ct_r = warper3d(ct_EOE, flowx * flow)
        # return flowx, pre_ct_r, phase
        return flowx
