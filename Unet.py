import torch
import torch.nn as nn
import torch.nn.functional as F


class layer_norm(nn.Module):
    def __init__(self):
        super(layer_norm, self).__init__()

    def forward(self, x):
        x = F.layer_norm(x, x.shape[-3::])
        return x


class conv33(nn.Module):
    def __init__(self, idim, odim):
        super(conv33, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(idim, odim, 3, 1, 1, bias=True),
            layer_norm(),
            nn.LeakyReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(odim, odim, 3, 1, 1, bias=True),
            layer_norm(),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Upsample_conv(nn.Module):
    # replace nn.ConvTranspose3d(odim, odim // 2, 2, 2, 0)
    # Upsample(odim)
    def __init__(self, odim):
        super(Upsample_conv, self).__init__()
        self.conv = nn.Conv3d(odim, odim // 2, 3, 1, 1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=[2, 2, 2], mode='trilinear')
        return x


class Ublock(nn.Module):
    def __init__(self, idim, odim, st):
        super(Ublock, self).__init__()
        if st == 'start':
            self.conv = nn.Sequential(
                conv33(idim, odim)
            )
        elif st == 'down':
            self.conv = nn.Sequential(
                nn.MaxPool3d(2),
                conv33(idim, odim)
            )
        elif st == 'mid':
            self.conv = nn.Sequential(
                nn.MaxPool3d(2),
                conv33(idim, odim),
                # nn.ConvTranspose3d(odim, odim // 2, 2, 2, 0)
                Upsample_conv(odim)
            )
        elif st == 'up':
            self.conv = nn.Sequential(
                conv33(idim, odim),
                # nn.ConvTranspose3d(odim, odim // 2, 2, 2, 0)
                Upsample_conv(odim)
            )
        elif st == 'end':
            self.conv = nn.Sequential(
                conv33(idim, odim),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, idim, odim, st, depth=7):
        super(UNet, self).__init__()
        self.depth = depth
        self.idim = idim
        self.odim = odim
        self.blocks = nn.ModuleList([
            Ublock(idim[i], odim[i], st[i])
            for i in range(depth)])
        self.out = nn.Sequential(
            nn.Conv3d(odim[-1], 3, 1, bias=True),
        )

    def forward(self, x):
        y = []
        for i in range(self.depth):
            if i > self.depth // 2:
                x = self.blocks[i](torch.cat((x, y[self.depth - i - 1]), dim=1))

            else:
                x = self.blocks[i](x)
            y.append(x)
        return self.out(x)


class FCN(nn.Module):
    def __init__(self, idim, odim, st, depth=6):
        super(FCN, self).__init__()
        self.depth = depth
        self.idim = idim
        self.odim = odim
        self.blocks = nn.ModuleList([
            Ublock(idim[i], odim[i], st[i])
            for i in range(depth)])
        self.out = nn.Sequential(
            nn.Conv3d(odim[-1], 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, sur_f):
        for i in range(self.depth):
            x = self.blocks[i](x)
        # sur_f = sur_f + 1
        # print(x.shape, sur_f.shape)
        return self.out(x + sur_f) * 1.8


class Warper3d(nn.Module):
    def __init__(self, img_size):
        super(Warper3d, self).__init__()
        """
        warp an image, according to the optical flow
        image: [B, 1, D, H, W] image for sampling
        flow: [B, 3, D, H, W] flow predicted from source image pair
        """
        self.img_size = img_size
        D, H, W = img_size
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, -1).repeat(D, H, 1).view(1, D, H, W)
        yy = torch.arange(0, H).view(1, -1, 1).repeat(D, 1, W).view(1, D, H, W)
        zz = torch.arange(0, D).view(-1, 1, 1).repeat(1, H, W).view(1, D, H, W)
        self.grid = torch.cat((xx, yy, zz), 0).float()  # [3, D, H, W]

    def forward(self, img, flow):
        grid = self.grid.repeat(flow.shape[0], 1, 1, 1, 1)  # [bs, 3, D, H, W]
        #        mask = torch.ones(img.size())
        if img.is_cuda:
            grid = grid.cuda()
        #            mask = mask.cuda()
        vgrid = grid + flow

        # scale grid to [-1,1]
        D, H, W = self.img_size
        vgrid[:, 0] = 2.0 * vgrid[:, 0] / (W - 1) - 1.0  # max(W-1,1)
        vgrid[:, 1] = 2.0 * vgrid[:, 1] / (H - 1) - 1.0  # max(H-1,1)
        vgrid[:, 2] = 2.0 * vgrid[:, 2] / (D - 1) - 1.0  # max(H-1,1)

        vgrid = vgrid.permute(0, 2, 3, 4, 1)  # [bs, D, H, W, 3]
        output = F.grid_sample(img, vgrid, mode='bilinear', align_corners=True)  # , mode='nearest'
        #        mask = F.grid_sample(mask, vgrid)#, mode='nearest'
        #        mask[mask<0.9999] = 0
        #        mask[mask>0] = 1
        return output  # *mask


class conv_down(nn.Module):
    def __init__(self, idim, odim, k=(3, 3)):
        super(conv_down, self).__init__()
        p = [int((i - 1) / 2) for i in k]
        stride = 2 if k[0] == 3 else 1
        self.conv = nn.Sequential(
            nn.Conv3d(idim, odim, (k[0], k[0], k[0]), padding=p[0], stride=(stride, stride, stride)),
            nn.BatchNorm3d(odim),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(odim, odim, (k[1], k[1], k[1]), padding=p[1], stride=(1, 1, 1)),
            nn.BatchNorm3d(odim),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class layer_norm2d(nn.Module):
    def __init__(self):
        super(layer_norm2d, self).__init__()

    def forward(self, x):
        x = F.layer_norm(x, x.shape[-2::])
        return x


class conv33_2d(nn.Module):
    def __init__(self, idim, odim, down=True):
        super(conv33_2d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(idim, odim, 3, 1, 1, bias=True),
            layer_norm2d(),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(odim, odim, 3, 1, 1, bias=True),
            layer_norm2d(),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2) if down else nn.Identity()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class VGG(nn.Module):
    def __init__(self, idim, odim, sur='x'):
        super(VGG, self).__init__()
        self.depth = len(idim)
        self.blocks = nn.ModuleList([
            conv33_2d(idim[i], odim[i]) for i in range(self.depth)
        ])
        self.out = nn.Sequential(conv33_2d(odim[-1], 2 * odim[-1], down=False))
        self.pos_emb = True
        self.conv3d = nn.Conv3d(256 + 3, 256, 1, 1, 0, bias=True)
        self.sur = sur

    def forward(self, x, r):
        x = F.interpolate(x, size=[8 * r, 8 * r], mode='bilinear')
        for i in range(self.depth):
            x = self.blocks[i](x)
        x = self.out(x)
        x = x.unsqueeze(dim=2).repeat(1, 1, r, 1, 1) if self.sur == 'x' else x.unsqueeze(dim=3).repeat(1, 1, 1, r, 1)
        if self.pos_emb:
            xx, yy, zz = torch.meshgrid(torch.arange(0, r) / r - 0.5, torch.arange(0, r) / r - 0.5,
                                        torch.arange(0, r) / r - 0.5)
            xx = torch.nn.Parameter(xx, requires_grad=False).unsqueeze(dim=0).unsqueeze(dim=0)
            yy = torch.nn.Parameter(yy, requires_grad=False).unsqueeze(dim=0).unsqueeze(dim=0)
            zz = torch.nn.Parameter(zz, requires_grad=False).unsqueeze(dim=0).unsqueeze(dim=0)

            if x.is_cuda:
                # print(x.device)
                device = x.device
                xx = xx.to(device)
                yy = yy.to(device)
                zz = zz.to(device)
            x = torch.cat([x, xx, yy, zz], dim=1)

            # pos_emb = torch.nn.Parameter(torch.zeros_like(x), requires_grad=True)
            # x = x + pos_emb
            x = self.conv3d(x)
        return x


class VGG_xy(nn.Module):
    def __init__(self, idim, odim):
        super(VGG_xy, self).__init__()
        self.depth = len(idim)

        self.blocks_x = nn.ModuleList([
            conv33_2d(idim[i], odim[i]) for i in range(self.depth)
        ])
        self.out_x = nn.Sequential(conv33_2d(odim[-1], 2 * odim[-1], down=False))

        self.blocks_y = nn.ModuleList([
            conv33_2d(idim[i], odim[i]) for i in range(self.depth)
        ])
        self.out_y = nn.Sequential(conv33_2d(odim[-1], 2 * odim[-1], down=False))

        self.pos_emb = True
        self.conv3d = nn.Conv3d(259 if self.pos_emb else 256, 256, 1, 1, 0, bias=True)

    def forward(self, x, y, r):
        x = F.interpolate(x, size=[8 * r, 8 * r], mode='bilinear')
        for i in range(self.depth):
            x = self.blocks_x[i](x)
        x = self.out_x(x)

        y = F.interpolate(y, size=[8 * r, 8 * r], mode='bilinear')
        for i in range(self.depth):
            y = self.blocks_y[i](y)
        y = self.out_y(y)

        x = x.unsqueeze(dim=2).repeat(1, 1, r, 1, 1)
        y = y.unsqueeze(dim=3).repeat(1, 1, 1, r, 1)

        x = x + y

        if self.pos_emb:
            xx, yy, zz = torch.meshgrid(torch.arange(0, r) / r - 0.5, torch.arange(0, r) / r - 0.5,
                                        torch.arange(0, r) / r - 0.5)
            xx = torch.nn.Parameter(xx, requires_grad=False).unsqueeze(dim=0).unsqueeze(dim=0)
            yy = torch.nn.Parameter(yy, requires_grad=False).unsqueeze(dim=0).unsqueeze(dim=0)
            zz = torch.nn.Parameter(zz, requires_grad=False).unsqueeze(dim=0).unsqueeze(dim=0)

            if x.is_cuda:
                # print(x.device)
                device = x.device
                xx = xx.to(device)
                yy = yy.to(device)
                zz = zz.to(device)
            x = torch.cat([x, xx, yy, zz], dim=1)

        out = self.conv3d(x)
        return out
# self.Generate = UNet([2, c, 2 * c, 4 * c, 6 * c, 4 * c, 2 * c],
#                              [c, 2 * c, 4 * c, 4 * c, 4 * c, 2 * c, c],
#                              ['start', 'down', 'down', 'mid', 'up', 'up', 'end'],
#                              depth=7)
