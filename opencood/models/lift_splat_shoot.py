"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
from icecream import ic

from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum, depth_discretization
from opencood.models.sub_modules.lss_submodule import Up, CamEncode, CamEncodeGTDepth, BevEncode
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from matplotlib import pyplot as plt


class LiftSplatShoot(nn.Module):
    def __init__(self, args): 
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = args['grid_conf']  
        self.data_aug_conf = args['data_aug_conf']   
        self.outC = args['outC']
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              ) 

        self.dx = dx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [0.4,0.4,20]
        self.bx = bx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [-49.8,-49.8,0]
        self.nx = nx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [250,250,1]
        
        self.downsample = args['img_downsample'] 
        self.camC = args['img_features']  
        self.frustum = self.create_frustum().clone().detach().requires_grad_(False).to(torch.device("cuda"))  # frustum: DxfHxfWx3

        self.D, _, _, _ = self.frustum.shape  # D: 41

        self.camencode = CamEncode(self.D, self.camC, self.downsample, \
            self.grid_conf['ddiscr'], self.grid_conf['mode'], args['use_depth_gt'], args['depth_supervision'])

        self.bevencode = BevEncode(inC=self.camC, outC=self.outC)
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        self.cls_head = nn.Conv2d(self.outC, args['anchor_number'],
                                  kernel_size=1)                 
        self.reg_head = nn.Conv2d(self.outC, 7 * args['anchor_number'],
                                  kernel_size=1)
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.outC, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        else:
            self.use_dir = False

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim'] 
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  
        ds = torch.tensor(depth_discretization(*self.grid_conf['ddiscr'], self.grid_conf['mode']), dtype=torch.float).view(-1,1,1).expand(-1, fH, fW)

        D, _, _ = ds.shape # D: 41 表示深度方向上网格的数量
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW) 

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1) 
        return frustum

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  # B:4(batchsize)    N: 4(相机数目)
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],  # points[:, :, :, :, :, 2:3] ranges from [4, 45) meters
                            points[:, :, :, :, :, 2:3]
                            ), 5)  
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3) 


        return points  # B x N x D x H x W x 3 (4 x 4 x 41 x 16 x 22 x 3) 

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape  # B: 4  N: 4  C: 3  imH: 256  imW: 352

        x = x.view(B*N, C, imH, imW)  
        depth_items, _, _, _, x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample) 
        x = x.permute(0, 1, 3, 4, 5, 2)  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)

        return x, depth_items

    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 16 x 22 x 3), D is discretization in "UD" or "LID"
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64), D is num_bins

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 16  W: 22  C: 64
        Nprime = B*N*D*H*W  # Nprime

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices

        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()  #
        geom_feats = geom_feats.view(Nprime, 3) 
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])  
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  

        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept] 
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks) 

        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x Z x X x Y
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x 

        # modify griddify (B x C x Z x Y x X) by Yifan Lu 2022.10.7
        # ------> x
        # |
        # |
        # y
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)  # final: 4 x 64 x Z x Y x X
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x 

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  # 消除掉z维

        return final  # final: 4 x 64 x 240 x 240  # B, C, H, W

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans) 
        x, depth_items = self.get_cam_feats(x) 
        x = self.voxel_pooling(geom, x)  # x: 4 x 64 x 240 x 240
 
        return x, depth_items

    def forward(self, data_dict):
        # x: [4,4,3,256, 352]
        # rots: [4,4,3,3]
        # trans: [4,4,3]
        # intrins: [4,4,3,3]
        # post_rots: [4,4,3,3]
        # post_trans: [4,4,3]
        x, rots, trans, intrins, post_rots, post_trans = \
            data_dict['imgs'], data_dict['rots'], data_dict['trans'], data_dict['intrins'], data_dict['post_rots'], data_dict['post_trans']
        x, depth_items = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)  # 将图像转换到BEV下，x: B x C x 240 x 240 (4 x 64 x 240 x 240)
        x = self.bevencode(x)  # 用resnet18提取特征  x: 4 x C x 240 x 240
        if self.shrink_flag:
            x = self.shrink_conv(x)
        # 4 x C x 120 x 120
        psm = self.cls_head(x)
        rm = self.reg_head(x)
        output_dict = {'psm': psm,
                       'rm': rm,
                       'depth_items': depth_items}

        if self.use_dir:
            dm = self.dir_head(x)
            output_dict.update({"dm": dm})

        return output_dict


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)