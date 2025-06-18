import torch.nn as nn
import torch
import yaml
import time
from opencood.models.sparse4D_utils.grid_mask import GridMask
from mmdet3d.models.backbones import ResNet
from mmdet3d.models.necks import FPN
from opencood.models.sparse4D_utils.sparse4Dhead import Sparse4DHead
from .sparse4D_utils.block import DenseDepthNet
from .sparse4D_utils.decoder import SparseBox3DDecoder
from .fuse_modules.fuse_utils import count_parameters

try:
    from .ops import DeformableAggregationFunction as DAF
except:
    DAF = None
import torch.nn.functional as F


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x



class Sparse4D(nn.Module):
    def __init__(
        self,
        args,
    ):
        # # init_cfg['Sparse4D']  provide Sparse4D parameters.
        super(Sparse4D, self).__init__()
        init_cfg = args
        if init_cfg['img_backbone']['depth']==50:
            self.img_backbone = ResNet(
                depth=50,
                num_stages=4,
                frozen_stages=-1,
                norm_eval=False,
                style='pytorch',
                with_cp=True,
                out_indices=(0, 1, 2, 3),
                norm_cfg=dict(type="BN", requires_grad=True),
            )
        if init_cfg['img_backbone']['depth'] == 101:  #TODO: hard code 
            self.img_backbone = ResNet(
                depth=101,
                num_stages=4,
                frozen_stages=-1,
                norm_eval=False,
                style='caffe',
                with_cp=True,
                out_indices=(0, 1, 2, 3),
                norm_cfg=dict(type='BN2d', requires_grad=False),
                # pretrained="/mnt/sdb/public/data/yangk/pretrained/sparse4D/fcos3d.pth",
                stage_with_dcn=[False, False, True, True],
                dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            )

        self.img_neck = FPN(
            num_outs= 4,
            start_level= 1,
            out_channels= 256,
            add_extra_convs="on_output",
            relu_before_extra_convs=True,
            in_channels=[256, 512, 1024, 2048],
        )

        head_init = init_cfg['head_args']
        self.head = Sparse4DHead(head_init)

        self.use_grid_mask = init_cfg['use_grid_mask']
        self.use_deformable_func = init_cfg['use_deformable_func'] and DAF is not None
        self.use_depth = init_cfg['use_depth']
        if self.use_depth:
            self.downsample = init_cfg["downsample"]
        if init_cfg['depth_branch'] is not None:
            self.depth_branch = DenseDepthNet(**init_cfg['depth_branch'])
        else:
            self.depth_branch = None
        if init_cfg['use_grid_mask']:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )
        self.decoder = SparseBox3DDecoder()

        self.time_dict = {}

    def extract_feat(self,img):
        # extract img features.
        # img.shape: bs*num_agent,,4,3,H,W
        assert len(img.shape) == 5
        bs,num_cam,C,H,W = img.shape
        img = img.reshape(bs*num_cam,C,H,W)
        if self.use_depth:
            gt_depths = []
            gt_depth = img[:,-1:,:,:]
            for downsample in self.downsample:
                downsampled = F.interpolate(gt_depth, scale_factor=1 / downsample, mode='bilinear', align_corners=False)
                gt_depths.append(downsampled)

            img = img[:,:-1,:,:]
        if self.use_grid_mask:         #  data_augment
            img = self.grid_mask(img)
        feature_maps = self.img_backbone(img)
        feature_maps =list(self.img_neck(feature_maps))

        for i,feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs,num_cam) + feat.shape[1:]
            )
        # depth model
        if self.use_depth and self.depth_branch is not None:
            depth_loss = self.depth_branch(feature_maps, gt_depths=gt_depths)

        else:
            depth_loss = None
        if self.use_deformable_func:
            feature_maps = DAF.feature_maps_format(feature_maps)

        '''
        depths: list: 3 layers, N*4,1,H,W
        feature_maps:  [list]: 4 lyaers, N,4,256,H,W
        '''
        return feature_maps, depth_loss


    def forward(self,data_dict,mode='train'):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.foward_test(data_dict, mode)

    def forward_train(self,data_dict):
        image_inputs_dict = data_dict['image_inputs']
        x, rots, trans, intrins, post_rots, post_trans = \
            (image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'],
             image_inputs_dict['post_rots'], image_inputs_dict['post_trans'])
        feature_maps, depth_loss = self.extract_feat(x)
        feature_queue = None
        meta_queue = None
        cls_scores, reg_preds,  *rest = self.head(
            feature_maps, data_dict, feature_queue, meta_queue
        )
        
        record_len = data_dict['record_len']
        layer_num = len(cls_scores)

        cls_scores_layer = []
        reg_preds_layer = []
        dir_layer = []
        # only need ego  :
        for layer in range(layer_num):
            cls_b = regroup(cls_scores[layer], record_len)
            box_b = regroup(reg_preds[layer], record_len)
            dir_b = regroup(rest[0][layer], record_len)
            cls_ego = []
            box_ego = []
            dir_ego = []
            for k in range(len(cls_b)):
                cls_ego.append(cls_b[k][0:1]) # only ego
                box_ego.append(box_b[k][0:1]) # only ego
                dir_ego.append(dir_b[k][0:1]) # only ego
        
            cls_scores_layer.append(torch.cat(cls_ego, 0))
            reg_preds_layer.append(torch.cat(box_ego, 0))
            dir_layer.append(torch.cat(dir_ego, 0))
        cls_scores = cls_scores_layer
        reg_preds = reg_preds_layer
        dir_m = dir_layer

        if self.use_deformable_func:
            feature_maps = DAF.feature_maps_format(feature_maps, inverse=True)
        results_dict = {
            'cls_scores': cls_scores,
            'reg_preds': reg_preds,
            'feature_maps': feature_maps,
        }
        if rest:
            results_dict.update({"dm":dir_m})
            if len(rest) != 1:
                diversity_loss = rest[1]
                results_dict.update({"diversity_loss": diversity_loss})
        if self.use_depth:
            results_dict.update({'depth_loss':depth_loss})
        return results_dict

    def foward_test(self,data_dict,mode='val'):
        image_inputs_dict = data_dict['image_inputs']
        x, rots, trans, intrins, post_rots, post_trans = \
            (image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'],
             image_inputs_dict['post_rots'], image_inputs_dict['post_trans'])

        feature_maps, depth_loss = self.extract_feat(x)
        feature_queue = None
        meta_queue = None
        cls_scores, reg_preds,  *rest = self.head(
            feature_maps, data_dict, feature_queue, meta_queue, mode
        )


        record_len = data_dict['record_len']
        layer_num = len(cls_scores)
        cls_scores_layer = []
        reg_preds_layer = []
        dir_layer = []
        # only need ego  :
        for layer in range(layer_num):
            if cls_scores[layer] == None:
                cls_scores_layer.append(None)
                reg_preds_layer.append(None)
                dir_layer.append(None)
                continue

            cls_ego = []
            box_ego = []
            dir_ego = []
            cls_b = regroup(cls_scores[layer], record_len)
            box_b = regroup(reg_preds[layer], record_len)
            dir_b = regroup(rest[0][layer], record_len)
            for k in range(len(cls_b)):
                cls_ego.append(cls_b[k][0:1]) # only ego
                box_ego.append(box_b[k][0:1]) # only ego
                dir_ego.append(dir_b[k][0:1]) # only ego
        
            cls_scores_layer.append(torch.cat(cls_ego, 0))
            reg_preds_layer.append(torch.cat(box_ego, 0))
            dir_layer.append(torch.cat(dir_ego, 0))

        cls_scores = cls_scores_layer
        reg_preds = reg_preds_layer
        dir_m = dir_layer

        
        if self.use_deformable_func:
            feature_maps = DAF.feature_maps_format(feature_maps, inverse=True)
        pred_results_dict = {
            'cls_scores': cls_scores,
            'reg_preds': reg_preds,
            'feature_maps': feature_maps,
        }
        results = self.decoder.decode(cls_scores,reg_preds)
        output = [{"img_bbox": result} for result in results]
        pred_results_dict.update(
            {
                'output': output[0]['img_bbox'] # in test, only need ego
            }
        )
        if rest:
            pred_results_dict.update({"dm":dir_m})
            if len(rest) != 1:
                diversity_loss = rest[1]
                pred_results_dict.update({"diversity_loss": diversity_loss})
        if self.use_depth:
            pred_results_dict.update({'depth_loss':depth_loss})
        return pred_results_dict

def compile_model(grid_conf, data_aug_conf, outC):
    return Sparse4D(grid_conf, data_aug_conf, outC)