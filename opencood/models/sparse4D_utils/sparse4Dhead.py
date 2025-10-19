# Copyright (c) Horizon Robotics. All rights reserved.
import time
from typing import List, Optional, Tuple, Union
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from opencood.models.sparse4D_utils.instance_bank import InstanceBank
from opencood.models.sparse4D_utils.detection3d_block import SparseBox3DKeyPointsGenerator, SparseBox3DRefinementModule, \
      SparseBox3DEncoder, PointEncoderV6
from .MHA import MultiheadAttention as mha
from .anchor_transform import regroup_anchor
from .anchor_utils import generate_anchor_heatmap
from .block import AsymmetricFFN,DeformableFeatureAggregation
from .detection3d_block import X,Y,Z,W,L,H,SIN_YAW,COS_YAW,VX,VY,VZ
from ..fuse_modules.fuse_utils import count_parameters


class ClsHead(nn.Module):
    def __init__(self, per_point_features=256, hidden_dim=128):
        super(ClsHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv1d(per_point_features, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1)
        self.fc = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (bs, 900, 256)
        bs, points, _ = x.shape
        x = x.transpose(1, 2)  # Now shape: (bs, 256, 900)
        x = self.relu(self.conv1(x))  # (bs, hidden_dim, 900)
        x = self.relu(self.conv2(x))  # (bs, hidden_dim // 2, 900)
        x = x.transpose(1, 2).contiguous().view(-1, self.hidden_dim // 2)
        x = self.fc(x)
        x = x.view(bs, points, -1).sigmoid()
        return x


class Sparse4DHead(nn.Module):
    def __init__(
        self,
        init_para,
    ):
        self.feature_embed_dims = init_para['embed_dims']
        super(Sparse4DHead, self).__init__()
        if "use_dir" in init_para:
            self.use_dir = init_para['use_dir']
            if self.use_dir:
                self.dir_head = nn.Conv2d(self.feature_embed_dims, init_para['dir_args']['num_bins'] * init_para['dir_args']['anchor_number'],
                                      kernel_size=1) # BIN_NUM = 2
        else:
            self.use_dir = False
        if "fuse_args" in init_para:
            self.use_fuse = init_para['fuse_args']['fuse']
            # ROI
            self.cls_head_1d = ClsHead(init_para['embed_dims'])
            self.fuse_method = init_para['fuse_args']['method']
            # if self.fuse_method == 'kmeans':
            #     self.multifuse = Kmeansfusion() # need embed_dims, output_dims and normalize_yaw
            # elif self.fuse_method == 'crossattention':
            #     self.multifuse = CrossVehicleAttentionfuse()
            # elif self.fuse_method == 'pointfuse':
            #     self.multifuse = Pointfuse(**init_para['refine_layer'])
            if self.fuse_method == 'PointEncoderV6':
                self.topK = init_para['fuse_args']['topk']
                self.fusion_layer = init_para['fuse_args']['fusion_layer']
                self.multifuse = PointEncoderV6(self.feature_embed_dims,init_para['graph_model'],
                                                init_para['norm_layer'], init_para['ffn'])
            else:
                print('Not implement!')

        if 'cav_lidar_range' in init_para:
            self.cav_lidar_range = init_para['cav_lidar_range']
        self.num_decoder = init_para['num_decoder']
        self.num_single_frame_decoder = init_para['num_single_frame_decoder']
        if 'reg_weights' not in init_para:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = init_para['reg_weights']

        if 'operation_order' not in init_para:
            self.operation_order = [
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * init_para['num_decoder']
        else:
            self.operation_order = init_para['operation_order']

        if 'use_spatial' in init_para:
            self.use_spatial = init_para['use_spatial']
        else:
            self.use_spatial = False

        instance_bank_para = init_para['instance_bank']
        self.instance_bank = InstanceBank(instance_bank_para)
        self.anchor_encoder = SparseBox3DEncoder(**init_para['anchor_encoder'])
        temp_gnn_config = init_para['temp_graph_model']
        gnn_config = init_para['graph_model']
        norm_layer_config = init_para['norm_layer']
        ffn_config = init_para['ffn']
        deformable_model_config = init_para['deformable_model']
        refine_layer_config = init_para['refine_layer']
        # op order
        op_config_dict = {
            'temp_gnn': temp_gnn_config,
            'gnn': gnn_config,
            'norm': norm_layer_config,
            'ffn': ffn_config,
            'deformable': deformable_model_config,
            'refine': refine_layer_config,
        }
        self.layers = nn.ModuleList()
        for layer in self.operation_order:
            for op in layer:
                if op == 'temp_gnn':
                    module_instance = mha(op_config_dict[op])
                elif op == 'gnn':
                    module_instance = mha(op_config_dict[op])
                elif op == 'norm':
                    module_instance = nn.LayerNorm(**op_config_dict[op])
                elif op == 'ffn':
                    module_instance = AsymmetricFFN(**op_config_dict[op])
                elif op == 'deformable':
                    module_instance = DeformableFeatureAggregation(**op_config_dict[op])
                elif op == 'refine':
                    module_instance = SparseBox3DRefinementModule(**op_config_dict[op])
                else:
                    raise ValueError(f"Unsupported operation: {op}")
                self.layers.append(module_instance)
        if 'depth_module' not in init_para:
            self.depth_module = None
        else:
            if init_para['depth_module']['ready']:
                from .block import DepthReweightModule
                self.depth_module = DepthReweightModule(init_para['depth_module']['input_dims'])
            else:
                self.depth_module = None
        self.anchor_num = instance_bank_para['num_anchor']
        self.Lambda = nn.Linear(self.feature_embed_dims, op_config_dict['gnn']['num_heads'])


    def init_weights(self):
        i = 0
        for layer in self.operation_order:
            for op in layer:
                if self.layers[i] is None:
                    continue
                elif op != "refine":
                    for p in self.layers[i].parameters():
                        if p.dim() > 1:
                            nn.init.xavier_uniform_(p)
                i = i + 1
            for m in self.modules():
                if hasattr(m, "init_weight"):
                    m.init_weight()

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
        feature_queue=None,
        meta_queue=None,
        mode='train',
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        #TODO: need N_agent init anchor?
        batch_size = feature_maps[0].shape[0]
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(batch_size, metas)

        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        _feature_queue = self.instance_bank.feature_queue
        _meta_queue = self.instance_bank.meta_queue
        if feature_queue is not None and _feature_queue is not None:
            feature_queue = feature_queue + _feature_queue
            meta_queue = meta_queue + _meta_queue
        elif feature_queue is None:
            feature_queue = _feature_queue
            meta_queue = _meta_queue

        prediction = []
        classification = []
        diversity_losses = []
        dir_m = []
        temp_cls = []
        i = 0
        record_len = metas['record_len'] 
        N_max = max(tensor for tensor in record_len)
        for lay_num,layer in enumerate(self.operation_order):
            for op in layer:
                # useless
                if op == "temp_gnn":
                    instance_feature = self.layers[i](
                        instance_feature,
                        temp_instance_feature,
                        temp_instance_feature,
                        query_pos=anchor_embed,
                        key_pos=temp_anchor_embed,
                    )
                elif op == "gnn":
                    if self.use_spatial:
                        dist_matrix = torch.cdist(anchor[..., :2], anchor[..., :2]) # B,300,300
                        Lm = self.Lambda(instance_feature)  # [B, Q, 8]
                        Lm = Lm.permute(0,2,1)
                        attn_mask = - torch.log(1+dist_matrix[:, None, :, :]) * Lm[..., None]  # [B, 8, Q, Q]
                        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
                        instance_feature = self.layers[i](
                            instance_feature,
                            query_pos=anchor_embed,
                            attn_mask=attn_mask
                        )
                    else:
                        instance_feature = self.layers[i](
                            instance_feature,
                            query_pos=anchor_embed,
                        )
                elif op == "norm" or op == "ffn":
                    instance_feature = self.layers[i](instance_feature)
                # elif op == "identity":
                #     identity = instance_feature
                # elif op == "add":
                #     instance_feature = instance_feature + identity
                elif op == "deformable":
                    instance_feature = self.layers[i](
                        instance_feature,
                        anchor,
                        anchor_embed,    # an embedding .  from 11 --> 256
                        feature_maps,
                        metas,
                        feature_queue=feature_queue,
                        meta_queue=meta_queue,
                        depth_module=self.depth_module,
                        anchor_encoder=self.anchor_encoder, # embedding
                    )

                elif op == "refine":
                    if hasattr(self, 'use_fuse') and self.use_fuse and N_max != 1:
                        c_instance_feature = instance_feature.clone()
                        c_anchor = anchor.clone()
                        from .anchor_transform import anchor_trans
                        if self.fuse_method == 'PointEncoderV6':
                            if lay_num in self.fusion_layer:
                                if lay_num != len(self.operation_order) - 1:
                                    step = 1
                                    return_cls = (
                                            self.training
                                            or len(prediction) == self.num_single_frame_decoder - 1
                                            or lay_num == len(self.operation_order) - 1
                                    )
                                    _, _, _, anchor, instance_feature = anchor_trans(
                                        metas, c_anchor, c_instance_feature, self.fuse_method, self.multifuse
                                        , self.cls_head_1d, self.cav_lidar_range, self.layers[i], self.anchor_encoder,
                                        anchor_embed, return_cls,self.topK, step,lay_num)
                                    # anchor_embed = self.anchor_encoder(instance_feature)
                            else:
                                pass
                    # result prediction
                    if (hasattr(self, 'use_fuse') and self.use_fuse):
                        if lay_num != len(self.operation_order) - 1:
                            anchor, cls, dir = self.layers[i](
                                instance_feature,
                                anchor,
                                anchor_embed,
                                time_interval=time_interval,
                                return_cls=True,
                            )
                            # here we obtain anchor with b,900,8
                            # we need trans 8 -> 7
                            yaw = torch.atan2(anchor[:,:, SIN_YAW], anchor[:,:, COS_YAW])
                            box = torch.cat(
                                    [
                                        anchor[:,:, [X, Y, Z]],
                                        anchor[:,:, [W, L, H]].exp(),
                                        yaw[:,:, None],
                                    ],
                                    dim=-1,
                                )
                    else:
                        anchor, cls, dir = self.layers[i](
                            instance_feature,
                            anchor,
                            anchor_embed,
                            time_interval=time_interval,
                            return_cls=True,
                        )
                        # here we obtain anchor with b,900,8
                        # we need trans 8 -> 7
                        yaw = torch.atan2(anchor[:, :, SIN_YAW], anchor[:, :, COS_YAW])
                        box = torch.cat(
                            [
                                anchor[:, :, [X, Y, Z]],
                                anchor[:, :, [W, L, H]].exp(),
                                yaw[:, :, None],
                            ],
                            dim=-1,
                        )

                    total_time = 0 
                    if hasattr(self, 'use_fuse') and self.use_fuse and N_max != 1:
                        c_instance_feature = instance_feature.clone()
                        c_anchor = anchor.clone()
                        from .anchor_transform import anchor_trans
                        if self.fuse_method in ('PointEncoderV5', 'PointEncoderV6'):
                            if lay_num == len(self.operation_order) - 1 :
                                step=2
                                return_cls = (
                                        self.training
                                        or len(prediction) == self.num_single_frame_decoder - 1
                                        or lay_num == len(self.operation_order) - 1
                                )
                                cls, box, dir, anchor, instance_feature = anchor_trans(
                                    metas, c_anchor, c_instance_feature, self.fuse_method, self.multifuse
                                    , self.cls_head_1d, self.cav_lidar_range, self.layers[i], self.anchor_encoder,
                                    anchor_embed, return_cls,self.topK, step,lay_num)
                                # if return_cls and step==2:
                                #     cls = [F.pad(tensor, (0, 0, 0,
                                #                           (N_max-1) * self.topK + self.anchor_num - tensor.shape[1]),
                                #                       value=-float('inf'))
                                #                 for tensor in cls]
                                #     box = [F.pad(tensor, (0, 0, 0,
                                #                           (N_max-1) * self.topK + self.anchor_num - tensor.shape[1]))
                                #                 for tensor in box]
                                #     dir = [F.pad(tensor, (0, 0, 0,
                                #                           (N_max-1) * self.topK + self.anchor_num - tensor.shape[1]))
                                #             for tensor in dir]
                                cls = torch.cat(cls, dim=0)
                                box = torch.cat(box, dim=0)
                                dir = torch.cat(dir, dim=0)
                            else:
                                pass


                    # generate_anchor_heatmap(anchor[0,:,0],anchor[0,:,1],cls[0,:,0],lay_num)
                    if dir is not None:
                        dir_m.append(dir)
                    prediction.append(box)
                    classification.append(cls)

                    if mode == 'val':
                        if lay_num != self.num_single_frame_decoder - 1 and lay_num != len(self.operation_order) - 1:
                            temp_cls.append(None)
                        else:
                            temp_cls.append(cls)
                    if len(prediction) == self.num_single_frame_decoder:
                        instance_feature, anchor = self.instance_bank.update(
                            instance_feature, anchor, cls
                        )
                    if lay_num != len(self.operation_order) - 1:
                        anchor_embed = self.anchor_encoder(anchor)
                    if (
                        len(prediction) > self.num_single_frame_decoder
                        and temp_anchor_embed is not None
                    ):
                        temp_anchor_embed = anchor_embed[
                            :, : self.instance_bank.num_temp_instances
                        ]
                else:
                    raise NotImplementedError(f"{op} is not supported.")
                i = i + 1

        if mode == 'val':
            classification = temp_cls
    
    
        if len(dir_m) !=0:
            if len(diversity_losses) != 0:
                return classification, prediction, dir_m, diversity_losses
            else:
                return classification, prediction, dir_m
        else:
            return classification, prediction


def visualize_anchor(point):
    import matplotlib.pyplot as plt
    anchor = point.detach().cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.scatter(anchor[:, 0], anchor[:, 1], alpha=0.6) 
    plt.title("Scatter Plot of a 900x2 Tensor") 
    plt.xlabel("x") 
    plt.ylabel("y")  
    plt.grid(True) 
    plt.show() 

