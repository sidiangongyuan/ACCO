# -*- coding: utf-8 -*-
# Author: Kang Yang

import torch.nn.functional as F
import torch

from opencood.models.sparse4D_utils.detection3d_block import SIN_YAW, COS_YAW, X, Y, Z, W, H, L, get_anchor_box


def regroup_anchor(x,x_list,N):

    combined_box = [
        torch.cat((x[i], x_list[i]), dim=0).unsqueeze(0) for i in range(N)
    ]
    xx = torch.cat(combined_box, dim=0)

    return xx

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x





def anchor_trans(metas, c_anchor, c_instance_feature , fuse_method, multifuse, cls_head_1d, lidar_range, refinement=None,
                 anchorEncoder=None, anchor_embed=None, return_cls=True, K=100, step=0, lay_num=0, roi_th=None):
    record_len = metas['record_len']
    pairwise_t_matrix = metas['pairwise_t_matrix']  # pairwise_t_matrix[i, j] is Tji, i_to_j
    B, _ = pairwise_t_matrix.shape[:2]  # L is max
    split_x = regroup(c_anchor, record_len)
    split_feature = regroup(c_instance_feature,record_len)
    if anchor_embed is not None:
        split_anchor_embed = regroup(anchor_embed,record_len)
    if roi_th is not None:
        split_roi_th = regroup(roi_th,record_len)
    cls_neighbor = []
    box_neighbor = []
    dir_neighbor = []
    mask_neighbor_for_point = []
    mask_neighbor_for_fps = []
    device = c_anchor.device
    new_anchors = []
    new_features = []
    #TOdo: change
    K_top=20
    for b in range(B):
        N = record_len[b]
        b_anchor = split_x[b]  # bs,900,8
        b_feature = split_feature[b]
        if anchor_embed is not None:
            b_anchor_embed = split_anchor_embed[b]
        if roi_th is not None:
            b_roi_th = split_roi_th[b]
        anchor_num = b_anchor.shape[1]
        # 创建一个空张量，形状为 [0, 900, 8]

        #####################    ROI       ####################################
        if fuse_method == 'queue' or fuse_method == 'pointfuse+topk':
            roi_th = cls_head_1d(b_feature)  # 用来表征每个anchor是否代表物体
            _, top_indices = torch.topk(roi_th, K_top, dim=1)
            _, bot_indices = torch.topk(-roi_th, N * K_top, dim=1)
        #####################    ROI       ####################################

        if N==1 and fuse_method != 'Pointfuse_batch':
            # N是1的话，不需要融合
            if fuse_method == 'pointfuse':
                cls = torch.zeros(anchor_num,1,device=device)
                box = torch.zeros(anchor_num,7,device=device)
                dir = torch.zeros(anchor_num,2,device=device)
                mask_for_point = torch.zeros(anchor_num,1,dtype=torch.bool,device=device) # 这里应该是zero，因为补充的只是为了补齐长度，而不是真正的参与计算。 新建的都是0向量
                cls_neighbor.append(cls)
                box_neighbor.append(box)
                dir_neighbor.append(dir)
                mask_neighbor_for_point.append(mask_for_point)
            elif fuse_method == 'pointfuse+fps':
                mask_for_fps = torch.zeros(1,anchor_num,1,dtype=torch.bool,device=device) # pointfuse+fps 中 用来标记当前这个anchor是否参与loss计算
                mask_neighbor_for_fps.append(mask_for_fps)
            elif fuse_method == 'pointfuse+topk':
                cls = torch.zeros(K_top,1,device=device)
                box = torch.zeros(K_top,7,device=device)
                dir = torch.zeros(K_top,2,device=device)
                mask_for_point = torch.zeros(K_top,1,dtype=torch.bool,device=device) # pointfuse 中 用来标记当前这个anchor是否参与loss计算
                cls_neighbor.append(cls)
                box_neighbor.append(box)
                dir_neighbor.append(dir)
                mask_neighbor_for_point.append(mask_for_point)
            elif fuse_method in ('PointEncoderV2', 'PointEncoderV3', 'PointEncoderV4', 'PointEncoderV5', 'PointEncoderV6'):
                if step == 2:
                    output, cls, dir = refinement(b_feature,b_anchor,b_anchor_embed,return_cls=return_cls)
                    box = get_anchor_box(output)
                    cls_neighbor.append(cls)
                    box_neighbor.append(box)
                    dir_neighbor.append(dir)
            new_anchors.append(b_anchor)
            new_features.append(b_feature)
            continue


        #####################   transform       ###############################

        xyz = b_anchor[:, :, :3]  # get x,y,z

        # Todo: angle transform
        sin_yaw = b_anchor[..., SIN_YAW]
        cos_yaw = b_anchor[...,COS_YAW]
        zeros = torch.zeros_like(sin_yaw)
        ones = torch.ones_like(sin_yaw)
        # 使用torch.cat和广播来构造旋转矩阵
        R_yaw = torch.cat([
            torch.cat([cos_yaw, zeros, -sin_yaw], dim=-1),  # [cos_yaw, 0, -sin_yaw]
            torch.cat([zeros, ones, zeros], dim=-1),  # [0, 1, 0]
            torch.cat([sin_yaw, zeros, cos_yaw], dim=-1)  # [sin_yaw, 0, cos_yaw]
        ], dim=1).view(N, anchor_num, 3, 3)
        exp_b_anchor = b_anchor.unsqueeze(1).expand(-1, N, -1, -1)[:, :, :, 3:6]

        ones = torch.ones(N, anchor_num, 1, device=xyz.device, dtype=xyz.dtype)
        homogeneous_xyz = torch.cat([xyz, ones], dim=2)  # N x 900 x 4
        expanded_xyz = homogeneous_xyz.unsqueeze(1).expand(-1, N, -1, -1).unsqueeze(-1)  # N x N x 900 x 4
        # N,N,4,4
        t_matrix = pairwise_t_matrix[b][:N, :N, :, :].to(dtype=torch.float32)  # only ego to others.
        t_matrix_expand = t_matrix.unsqueeze(2)  # N,N,1,4,4
        transformed_xyz = torch.matmul(t_matrix_expand, expanded_xyz).squeeze(-1)  # N*N,900,4
        transformed_xyz = transformed_xyz[..., :3].reshape(N, N, anchor_num, -1)  # N,N,900,3

        # angle transform
        Rotate = t_matrix_expand[...,:3,:3] # N,N,1,3,3
        R_yaw = R_yaw.unsqueeze(1).expand(-1,N,-1,-1,-1)  # N,N,anchor_num,3,3
        transformed_yaw = torch.matmul(Rotate,R_yaw) # N,N,anchor_num,3,3
        r_sin_yaw = transformed_yaw[...,0,0]  # sin yaw
        r_cos_yaw = transformed_yaw[...,2,0]  # cos yaw
        r_sin_yaw = r_sin_yaw.unsqueeze(-1)
        r_cos_yaw = r_cos_yaw.unsqueeze(-1)

        trans_anchor = torch.cat((transformed_xyz, exp_b_anchor,r_sin_yaw,r_cos_yaw), dim=-1)  # N,N,900,8

        if fuse_method == 'pointfuse+fps':
            i2j_anchor = trans_anchor.permute(1,0,2,3)
            b_feature_expand = b_feature.unsqueeze(0).repeat(N, 1, 1, 1).reshape(N, N * anchor_num, -1)
            new_anchor, new_feature,mask = multifuse(i2j_anchor, b_feature_expand,lidar_range)
            new_anchors.append(new_anchor)
            new_features.append(new_feature)
            mask_neighbor_for_fps.append(mask)
        # elif fuse_method == 'PointSAF':
        #     i2j_anchor = trans_anchor.permute(1,0,2,3)
        #     new_anchor, new_feature = multifuse(b_anchor, i2j_anchor, b_feature)
        #     new_anchors.append(new_anchor)
        #     new_features.append(new_feature)
        elif fuse_method == 'Pointfuse_batch':
            i2j_anchor = trans_anchor.permute(1,0,2,3)
            i2j_t_matrix = t_matrix.permute(1,0,2,3)
            cls, box, mask_for_point, *rest = multifuse(i2j_anchor,b_feature, i2j_t_matrix, lidar_range)
            cls_neighbor.append(cls)
            box_neighbor.append(box)
            if rest:
                dir_neighbor.append(rest[0])
        elif fuse_method == 'PointEncoder':
            i2j_anchor = trans_anchor.permute(1,0,2,3)
            i2j_t_matrix = t_matrix.permute(1,0,2,3)
            cls, box, dir, output, res = multifuse(i2j_anchor, b_anchor, b_feature, i2j_t_matrix, lidar_range,
                                                        refinement,anchorEncoder,b_anchor_embed,return_cls)
            cls_neighbor.append(cls)
            box_neighbor.append(box)
            dir_neighbor.append(dir)
            new_anchors.append(output)
            new_features.append(res)
        elif fuse_method in ('PointEncoderV2', 'PointEncoderV3', 'PointEncoderV4', 'PointEncoderV5', 'PointEncoderV6'):
            i2j_anchor = trans_anchor.permute(1,0,2,3)
            i2j_t_matrix = t_matrix.permute(1,0,2,3)
            cls, box, dir, output, res = multifuse(i2j_anchor, b_anchor, b_feature,
                                                   i2j_t_matrix, lidar_range, refinement,anchorEncoder,
                                                   b_anchor_embed,return_cls,K, step, lay_num)
            cls_neighbor.append(cls)
            box_neighbor.append(box)
            dir_neighbor.append(dir)
            new_anchors.append(output)
            new_features.append(res)
        else:
            for k in range(0,N):
                ego_feature = b_feature[k]
                ego_anchor = b_anchor[k]
                # visualize anchor
                # for kk in range(N):
                #     visualize_anchor(trans_anchor[kk][k])
                if fuse_method == 'queue':
                    new_anchor, new_feature = multifuse(ego_anchor, trans_anchor[:, k],
                                                             ego_feature, b_feature, bot_indices[k], top_indices)
                    new_anchors.append(new_anchor)
                    new_features.append(new_feature)
                    # need mask
                elif fuse_method == 'pointfuse':
                    cls, box, mask_for_point, *rest = multifuse(ego_anchor, trans_anchor[:, k], ego_feature, b_feature, t_matrix[:,k],lidar_range)
                    cls_neighbor.append(cls)
                    box_neighbor.append(box)
                    mask_neighbor_for_point.append(mask_for_point)
                    if rest:
                        dir_neighbor.append(rest[0])
                elif fuse_method == 'pointfuse+topk':
                    cls, box, mask_for_point, *rest = multifuse(ego_anchor, trans_anchor[:, k], ego_feature, b_feature, t_matrix[:,k], top_indices, lidar_range)
                    cls_neighbor.append(cls)
                    box_neighbor.append(box)
                    mask_neighbor_for_point.append(mask_for_point)
                    if rest:
                        dir_neighbor.append(rest[0])
                # elif fuse_method == 'PointEncoder':
                #     cls, box, mask_for_point, *rest = multifuse(b_anchor[k:k+1],
                #                                                 trans_anchor[:, k], b_feature[k:k+1], b_feature, t_matrix[:,k],
                #                                                 lidar_range,refinement,anchorEncoder,anchor_embed[k:k+1,...])
                #     cls_neighbor.append(cls)
                #     box_neighbor.append(box)
                #     mask_neighbor_for_point.append(mask_for_point)
                #     if rest:
                #         dir_neighbor.append(rest[0])
                else:
                    new_anchor, new_feature = multifuse(ego_anchor, trans_anchor[:, k], ego_feature, b_feature)
                    new_anchors.append(new_anchor)
                    new_features.append(new_feature)
                    # need mask
        #####################    fuse multi-agent anchor and featur############

    if fuse_method == 'pointfuse' or fuse_method == 'pointfuse+topk' or fuse_method == 'Pointfuse_batch':
        return cls_neighbor, box_neighbor, mask_neighbor_for_point, dir_neighbor
    elif fuse_method == 'PointSAF':
        new_anchors = torch.stack(new_anchors).squeeze(0)
        new_features = torch.stack(new_features).squeeze(0)
        return new_anchors, new_features
    elif fuse_method == 'PointEncoder':
        box = torch.cat(box_neighbor)
        dir = torch.cat(dir_neighbor)
        new_anchors = torch.cat(new_anchors)
        new_features = torch.cat(new_features)
        if return_cls:
            cls = torch.cat(cls_neighbor)
            return cls, box, dir,new_anchors, new_features
        else:
            return None,box,dir,new_anchors,new_features
    elif fuse_method in ('PointEncoderV2', 'PointEncoderV3', 'PointEncoderV4', 'PointEncoderV5', 'PointEncoderV6'):
        new_anchors = torch.cat(new_anchors)
        new_features = torch.cat(new_features)
        return cls_neighbor, box_neighbor, dir_neighbor, new_anchors, new_features

    else:
        new_anchors = torch.cat(new_anchors,dim=0)
        new_features = torch.cat(new_features,dim=0)
        mask_neighbor_for_fps = torch.cat(mask_neighbor_for_fps,dim=0)
        return new_anchors, new_features, mask_neighbor_for_fps
