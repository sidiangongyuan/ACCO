import torch.nn as nn
import torch
from typing import List, Optional, Tuple

from matplotlib import pyplot as plt
from mmcv.cnn import Linear, Scale, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import constant_init, xavier_init, BaseModule
from torch import autocast
from torch.nn import Sequential
import numpy as np
from opencood.models.sparse4D_utils.detection3d_block import SparseBox3DKeyPointsGenerator,linear_relu_ln
from mmcv.cnn.bricks.transformer import FFN

try:
    from ..ops import DeformableAggregationFunction as DAF
except:
    DAF = None




class AsymmetricFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=dict(type="LN"),
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels * 4 # 256 * 4
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)



class LidarDeformableFeatureAggregation(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
        cav_lidar_range=None,
    ):
        super(LidarDeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        self.use_deformable_func = use_deformable_func and DAF is not None
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        # self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        self.kps_generator = SparseBox3DKeyPointsGenerator(**kps_generator)
        self.num_pts = self.kps_generator.num_pts
        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
        else:
            self.temp_module = None
        self.output_proj = Linear(embed_dims, embed_dims)

        self.camera_encoder = None
        self.weights_fc = Linear(
            embed_dims, num_groups * num_cams * num_levels * self.num_pts
        )
        self.cav_lidar_range = cav_lidar_range

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],  # feature_maps [0] 4层展平的feature  维度是B,N,F,C 换成Lidar的话N=1 , [1]  4×2 ，每层特征图的H,W  [2] 展平的区间
        metas: dict,
        feature_queue=None,
        meta_queue=None,
        depth_module=None,
        anchor_encoder=None,
        **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        temp_key_points_list = (
            feature_queue
        ) = meta_queue = temp_anchor_embeds = temp_anchors = []
        time_intervals = [instance_feature.new_tensor([0])]

        if self.temp_module is not None and len(feature_queue) == 0:
            features = instance_feature.new_zeros(
                [bs, num_anchor, self.num_pts, self.embed_dims]
            )
        else:
            features = None

        if not self.use_temporal_anchor_embed or anchor_encoder is None:
            weights = self._get_weights(instance_feature, anchor_embed, metas)

        for (
            temp_feature_maps,
            temp_metas,
            temp_key_points,
            temp_anchor_embed,
            temp_anchor,
            time_interval,
        ) in zip(
            feature_queue[::-1] + [feature_maps],
            meta_queue[::-1] + [metas],
            temp_key_points_list[::-1] + [key_points],
            temp_anchor_embeds[::-1] + [anchor_embed],
            temp_anchors[::-1] + [anchor],
            time_intervals[::-1],
        ):
            if self.use_temporal_anchor_embed and anchor_encoder is not None:
                weights = self._get_weights(
                    instance_feature, temp_anchor_embed, metas
                )
            if self.use_deformable_func:
                weights = (
                    weights.permute(0, 1, 4, 2, 3, 5)
                    .contiguous()
                    .reshape(
                        bs,
                        num_anchor * self.num_pts,
                        self.num_cams,
                        self.num_levels,
                        self.num_groups,
                    )
                )
                #TODO: whether need to normalize 
                points_2d = temp_key_points[...,:2].reshape(bs, num_anchor * self.num_pts, self.num_cams, 2)
                # normalize points_2d:
                points_2d[...,:1] = (points_2d[...,:1]-self.cav_lidar_range[0]) / (self.cav_lidar_range[3]-self.cav_lidar_range[0])
                points_2d[...,1:2] = (points_2d[...,1:2]-self.cav_lidar_range[1]) / (self.cav_lidar_range[4]-self.cav_lidar_range[1])
                
                temp_features_next = DAF.apply(
                    *temp_feature_maps, points_2d, weights
                ).reshape(bs, num_anchor, self.num_pts, self.embed_dims)
            else:
                temp_features_next = self.feature_sampling(
                    temp_feature_maps,
                    temp_key_points,
                    temp_metas["projection_mat"],
                    temp_metas.get("image_wh"),
                )
                temp_features_next = self.multi_view_level_fusion(
                    temp_features_next, weights
                )
            if depth_module is not None:
                temp_features_next = depth_module(
                    temp_features_next, temp_anchor[:, :, None]
                )

            if features is None:
                features = temp_features_next
            elif self.temp_module is not None:
                features = self.temp_module(
                    features, temp_features_next, time_interval
                )
            else:
                features = features + temp_features_next

        features = features.sum(dim=2)  # fuse multi-point features
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            if metas['image_inputs']['camera_to_lidar'].shape == 3:
                camera_embed = self.camera_encoder(metas['image_inputs']['camera_to_lidar'][:,:,:3].reshape(bs,self.num_cams,-1))
            else:
                camera_embed = self.camera_encoder(
                    metas['image_inputs']['camera_to_lidar'][:, :,:,:3].reshape(bs, self.num_cams, -1))
            feature = feature[:, :, None] + camera_embed[:, None]
        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, data_dict, visualize_point=False):
        bs, num_anchor, num_pts = key_points.shape[:3]
        x = data_dict['lidar_pts'] # pts_features
        B,N,H,W = x.shape # 
        # deal with post trans
        # key_points: bs,900,13,3       post_trans:4,4,3
        points = key_points.reshape(bs,1,num_anchor,num_pts,-1).repeat(1,N,1,1,1)  # + post_trans.view(bs,N,1,1,-1)

        # visualize point
        if visualize_point:
            imgs = []
            points_ =[]
            img_pts_ = []
            for i in range(N):
                imgs.append(data_dict['image_inputs']['imgs'][0,i,:3].permute(1,2,0))
                points_.append(points_2d[0][i].reshape(-1,2))
            visualize_projected_points(imgs,points_)

        image_wh = torch.tensor([W, H]).repeat(B, N, 1).cuda()

        points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features
        



class DeformableFeatureAggregation(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        self.use_deformable_func = use_deformable_func and DAF is not None
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        # self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        self.kps_generator = SparseBox3DKeyPointsGenerator(**kps_generator)
        self.num_pts = self.kps_generator.num_pts
        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            # self.temp_module = build_from_cfg(
            #     temporal_fusion_module, PLUGIN_LAYERS
            # )
        else:
            self.temp_module = None
        self.output_proj = Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        feature_queue=None,
        meta_queue=None,
        depth_module=None,
        anchor_encoder=None,
        **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        if feature_queue is not None and len(feature_queue) > 0:
            T_cur2temp_list = []
            for meta in meta_queue:
                T_cur2temp_list.append(
                    # xxx.new_tensor(yyy) can generate a new tensor as the same device and type as xxx and the same dimension as yyy
                    instance_feature.new_tensor(
                        [
                            x["T_global_inv"]
                            @ metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(meta["img_metas"])   # from one camera coord --> another camera coord?
                        ]
                    )
                )
            # here we get keypoints of all time
            key_points, temp_key_points_list = self.kps_generator(
                anchor,
                instance_feature,
                T_cur2temp_list,
                metas["timestamp"],
                [meta["timestamp"] for meta in meta_queue],
            )

            temp_anchors = self.kps_generator.anchor_projection(
                anchor,
                T_cur2temp_list,
                metas["timestamp"],
                [meta["timestamp"] for meta in meta_queue],
            )
            temp_anchor_embeds = [
                anchor_encoder(x)
                if self.use_temporal_anchor_embed
                and anchor_encoder is not None
                else None
                for x in temp_anchors
            ]
            time_intervals = [
                (metas["timestamp"] - x["timestamp"]).to(
                    dtype=instance_feature.dtype
                )
                for x in [metas] + meta_queue
            ]
        else:
            key_points = self.kps_generator(anchor, instance_feature)
            temp_key_points_list = (
                feature_queue
            ) = meta_queue = temp_anchor_embeds = temp_anchors = []
            time_intervals = [instance_feature.new_tensor([0])]

        if self.temp_module is not None and len(feature_queue) == 0:
            features = instance_feature.new_zeros(
                [bs, num_anchor, self.num_pts, self.embed_dims]
            )
        else:
            features = None

        if not self.use_temporal_anchor_embed or anchor_encoder is None:
            weights = self._get_weights(instance_feature, anchor_embed, metas)

        for (
            temp_feature_maps,
            temp_metas,
            temp_key_points,
            temp_anchor_embed,
            temp_anchor,
            time_interval,
        ) in zip(
            feature_queue[::-1] + [feature_maps],
            meta_queue[::-1] + [metas],
            temp_key_points_list[::-1] + [key_points],
            temp_anchor_embeds[::-1] + [anchor_embed],
            temp_anchors[::-1] + [anchor],
            time_intervals[::-1],
        ):
            if self.use_temporal_anchor_embed and anchor_encoder is not None:
                weights = self._get_weights(
                    instance_feature, temp_anchor_embed, metas
                )
            if self.use_deformable_func:
                weights = (
                    weights.permute(0, 1, 4, 2, 3, 5)
                    .contiguous()
                    .reshape(
                        bs,
                        num_anchor * self.num_pts,
                        self.num_cams,
                        self.num_levels,
                        self.num_groups,
                    )
                )
                points_2d = (
                    self.project_points(
                        temp_key_points,
                        metas,
                    ).permute(0, 2, 3, 1, 4).reshape(bs, num_anchor * self.num_pts, self.num_cams, 2)
                )   # project 3D keypoints into 2D image feature ,  bs,900*13,6,2
                temp_features_next = DAF.apply(
                    *temp_feature_maps, points_2d, weights
                ).reshape(bs, num_anchor, self.num_pts, self.embed_dims)
            else:
                temp_features_next = self.feature_sampling(
                    temp_feature_maps,
                    temp_key_points,
                    temp_metas["projection_mat"],
                    temp_metas.get("image_wh"),
                )
                temp_features_next = self.multi_view_level_fusion(
                    temp_features_next, weights
                )
            if depth_module is not None:
                temp_features_next = depth_module(
                    temp_features_next, temp_anchor[:, :, None]
                )

            if features is None:
                features = temp_features_next
            elif self.temp_module is not None:
                features = self.temp_module(
                    features, temp_features_next, time_interval
                )
            else:
                features = features + temp_features_next

        features = features.sum(dim=2)  # fuse multi-point features
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            if metas['image_inputs']['camera_to_lidar'].shape == 3:
                camera_embed = self.camera_encoder(metas['image_inputs']['camera_to_lidar'][:,:,:3].reshape(bs,self.num_cams,-1))
            else:
                camera_embed = self.camera_encoder(
                    metas['image_inputs']['camera_to_lidar'][:, :,:,:3].reshape(bs, self.num_cams, -1))
            feature = feature[:, :, None] + camera_embed[:, None]
        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, data_dict, visualize_point=False):
        bs, num_anchor, num_pts = key_points.shape[:3]
        image_inputs_dict = data_dict['image_inputs']
        x, rots, trans, intrins, post_rots, post_trans = \
            (image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'],
             image_inputs_dict['post_rots'], image_inputs_dict['post_trans'])
        _,_,_,H,W = x.shape
        B,N,_ = trans.shape
        # deal with post trans
        # key_points: bs,900,13,3       post_trans:4,4,3
        points = key_points.reshape(bs,1,num_anchor,num_pts,-1).repeat(1,N,1,1,1)  # + post_trans.view(bs,N,1,1,-1)

        # post_rots = post_rots.reshape(bs,N,1,1,3,3).repeat(1,1,num_anchor,num_pts,1,1)
        # points = post_rots.matmul(points.unsqueeze(-1))
        points = points.unsqueeze(-1)

        # inv_rot = torch.inverse(rots).view(B,N,1,1,3,3)
        # inv_trans = -trans.view(B,N,1,1,-1,1)
        # points_cam = torch.matmul(inv_rot,points + inv_trans) # p = R^{-1}(P-t)

        ones = torch.ones_like(points[..., :1, :])
        homogeneous_points = torch.cat([points, ones], dim=4)
        camera_to_lidar = image_inputs_dict["camera_to_lidar"]
        ego_to_camera = torch.inverse(camera_to_lidar).reshape(bs,N,1,1,4,4)
        points_cam = torch.matmul(ego_to_camera,homogeneous_points)
        points_cam =  points_cam[...,:3,:] / points_cam[...,3:4,:]

        intrins = intrins.reshape(B,N,1,1,3,3)
        points_2d = torch.matmul(intrins,points_cam).squeeze(-1)  # must point -1 otherwise, will eliminate some other dims
        # points_2d = []

        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )

        # another method
        # points = key_points.reshape(bs,1,num_anchor,num_pts,-1).repeat(1,N,1,1,1)  # + post_trans.view(bs,N,1,1,-1)
        # points = points.unsqueeze(-1)
        # ones = torch.ones_like(points[..., :1, :])
        # homogeneous_points = torch.cat([points, ones], dim=4)
        #
        # ext_matrix = torch.inverse(camera_to_lidar)[...,:3, :4].reshape(bs,N,1,1,3,4)
        # img_pts = (intrins @ ext_matrix @ homogeneous_points).squeeze()
        # # b,4,900,17,3
        # depth = img_pts[...,2:3]
        # img_pts = img_pts[...,:2] / torch.clamp(
        #     depth, min=1e-6
        # )
        # ones = torch.ones_like(img_pts[..., :1])
        # img_pts = torch.cat([img_pts,ones],dim=-1)
        # post_rots = post_rots.reshape(bs,N,1,1,3,3).repeat(1,1,num_anchor,num_pts,1,1)
        # img_pts = post_rots.matmul(img_pts.unsqueeze(-1)).squeeze()
        # img_pts = img_pts + post_trans.view(bs,N,1,1,-1)
        # img_pts = img_pts[...,:2]


        # for postprocess in image
        ones = torch.ones_like(points_2d[..., :1])
        points_2d = torch.cat([points_2d,ones],dim=-1)
        post_rots = post_rots.reshape(bs,N,1,1,3,3).repeat(1,1,num_anchor,num_pts,1,1)
        points_2d = post_rots.matmul(points_2d.unsqueeze(-1)).squeeze(-1)  # must point -1 otherwise, will eliminate some other dims
        points_2d = points_2d + post_trans.view(bs,N,1,1,-1)
        points_2d = points_2d[...,:2]

        # visualize point
        if visualize_point:
            imgs = []
            points_ =[]
            img_pts_ = []
            for i in range(N):
                imgs.append(data_dict['image_inputs']['imgs'][0,i,:3].permute(1,2,0))
                points_.append(points_2d[0][i].reshape(-1,2))
            visualize_projected_points(imgs,points_)

        image_wh = torch.tensor([W, H]).repeat(B, N, 1).cuda()

        points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features

class DenseDepthNet(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            depth = (depth.T * focal / self.equal_focal).T
            depths.append(depth)
        if gt_depths is not None:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(device_type='cuda'):
                error = torch.abs(pred - gt).sum()
                _loss = (
                    error
                    / max(1.0, len(gt) * len(depth_preds))
                    * self.loss_weight
                )
            loss = loss + _loss
        return loss

class DepthReweightModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        min_depth=1,
        max_depth=81,   # sqrt(70^2 + 40^2)
        depth_interval=5,
        ffn_layers=2,
    ):
        super(DepthReweightModule, self).__init__()
        self.embed_dims = embed_dims
        self.min_depth = min_depth
        self.depth_interval = depth_interval # maybe need to change
        self.depths = np.arange(min_depth, max_depth + 1e-5, depth_interval)
        self.max_depth = max(self.depths)

        layers = []
        for i in range(ffn_layers):
            layers.append(
                FFN(
                    embed_dims=embed_dims,
                    feedforward_channels=embed_dims,
                    num_fcs=2,
                    act_cfg=dict(type="ReLU", inplace=True),
                    dropout=0.0,
                    add_residual=True,
                )
            )
        layers.append(nn.Linear(embed_dims, len(self.depths)))
        self.depth_fc = Sequential(*layers)

    def forward(self, features, points_3d, output_conf=False):
        '''
        input:
            points_3d: anchor
            features: instance feature

        return :
            instance feature.
        '''
        # 
        reference_depths = torch.norm(
            points_3d[..., :2], dim=-1, p=2, keepdim=True
        )
        reference_depths = torch.clip(
            reference_depths,
            max=self.max_depth - 1e-5,
            min=self.min_depth + 1e-5,
        )
        weights = (
            1
            - torch.abs(reference_depths - points_3d.new_tensor(self.depths))
            / self.depth_interval
        )

        top2 = weights.topk(2, dim=-1)[0] # 
        weights = torch.where(
            weights >= top2[..., 1:2], weights, weights.new_tensor(0.0)
        )
        scale = torch.pow(top2[..., 0:1], 2) + torch.pow(top2[..., 1:2], 2)
        confidence = self.depth_fc(features).softmax(dim=-1)
        # 
        confidence = torch.sum(weights * confidence, dim=-1, keepdim=True)
        confidence = confidence / scale

        if output_conf:
            return confidence
        return features * confidence


def visualize_projected_points(images, points_2d):
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    H, W, C = images[0].shape
    save_dir = ""
    os.makedirs(save_dir, exist_ok=True)

    for img_idx, img_tensor in enumerate(images):

        img = img_tensor.cpu().numpy()

        if img.shape[0] in [1, 3]: 
            img = np.transpose(img, (1, 2, 0))


        if img.shape[2] == 1:  # 
            img = np.squeeze(img, axis=2)
        elif img.shape[2] == 3:  # RGB图
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # 
        original_img_path = os.path.join(save_dir, f"original_image_{img_idx}.png")
        cv2.imwrite(original_img_path, img)

        for pt in points_2d[img_idx].detach().cpu().numpy():
            x, y = int(pt[0]), int(pt[1])
            if x > W or x < 0 or y > H or y < 0:
                continue
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # 

        points_img_path = os.path.join(save_dir, f"points_image_{img_idx}.png")
        cv2.imwrite(points_img_path, img)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Image {img_idx}')
        plt.axis('off')
        plt.show()


def visualize_points3d(points_3d, save_path=None):
    points_3d = points_3d.reshape(-1,3)
    kp = points_3d.detach().cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.scatter(kp[:, 0], kp[:, 1], alpha=0.6)  
    plt.title("Scatter Plot of a 900x2 Tensor")  
    plt.xlabel("Dimension 1")  
    plt.ylabel("Dimension 2")  
    plt.grid(True)  
    plt.show()  