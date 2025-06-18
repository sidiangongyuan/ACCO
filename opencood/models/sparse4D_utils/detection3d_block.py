import torch
import torch.nn as nn
import numpy as np
from mmengine.model import xavier_init, bias_init_with_prob
from torch.nn import functional as F
from .anchor_utils import calculate_bbox_corners_with_orientation, compute_features_and_mask_efficient, \
    generate_anchor_heatmap

X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))
YAW = 6

from mmcv.cnn import Linear,Scale
from sklearn.cluster import KMeans

from ..ops.sample import furthest_point_sample
from .MHA import MultiheadAttention




def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers



def get_anchor_box(output):
    yaw = torch.atan2(output[:, :, SIN_YAW], output[:, :, COS_YAW])
    box = torch.cat(
        [
            output[..., [X, Y, Z]],
            output[..., [W, L, H]].exp(),
            yaw[..., None],
        ],
        dim=-1,
    )

    return box

class SparseBox3DEncoder(nn.Module):
    def __init__(self, embed_dims: int = 256, vel_dims: int = 3):
        super().__init__()
        self.embed_dims = embed_dims
        self.vel_dims = vel_dims

        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))

        self.pos_fc = embedding_layer(3)
        self.size_fc = embedding_layer(3)
        self.yaw_fc = embedding_layer(2)
        if vel_dims > 0:
            self.vel_fc = embedding_layer(self.vel_dims)
        self.output_fc = embedding_layer(self.embed_dims)

    def forward(self, box_3d: torch.Tensor):
        pos_feat = self.pos_fc(box_3d[..., [X, Y, Z]])
        size_feat = self.size_fc(box_3d[..., [W, L, H]])
        yaw_feat = self.yaw_fc(box_3d[..., [SIN_YAW, COS_YAW]])
        output = pos_feat + size_feat + yaw_feat # why use add ?
        if self.vel_dims > 0:
            vel_feat = self.vel_fc(box_3d[..., VX : VX + self.vel_dims])
            output = output + vel_feat
        output = self.output_fc(output)
        return output




class Kmeansfusion(nn.Module):
    def __init__(
            self,
            embed_dims=256,
            output_dim=8,
            refine_yaw=False
                 ):
        super(Kmeansfusion,self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim # 仍然是8
        self.anchor_encoder = SparseBox3DEncoder(embed_dims,0)
        self.refine_state = [X, Y, Z, W, H, L]   # you mean only refine position and size of anchors ?
        if refine_yaw:
            self.refine_state += [SIN_YAW, COS_YAW]
        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )

    def forward(self,
                ego_anchor, # 900,8
                trans_anchor, #  N,900,8
                ego_feature, # 900,256
                instance_feature, # N,900,256
                ):
        N, anchor_num, anchor_dim = trans_anchor.shape
        trans_anchor = trans_anchor.reshape(N*anchor_num,-1)
        instance_feature = instance_feature.reshape(N*anchor_num,-1)
        # 使用K-Means找到prototypes（这一步仍在CPU上执行）
        kmeans = KMeans(n_clusters=anchor_num, n_init='auto', random_state=0).fit(trans_anchor[:, :3].detach().cpu().numpy())
        prototypes = torch.from_numpy(kmeans.cluster_centers_).float().to(trans_anchor.device)
        # 计算所有点到每个prototype的距离
        distances = torch.cdist(trans_anchor[:, :3], prototypes)  # [N*900, 900]
        _, nearest_indices = torch.min(distances, dim=0)  # [900]
        prototypes_full = trans_anchor[nearest_indices]  # [900, 8]
        _, indices = torch.topk(distances, N, largest=False)
        # 初始化融合后的特征张量
        fused_features = torch.zeros(anchor_num, 256, device=trans_anchor.device)

        # 使用gather和scatter_add进行特征融合
        for i in range(anchor_num):
            nearest_features = instance_feature[indices[i]].view(-1, 256)  # 获取最近点的特征
            fused_features[i, :] = nearest_features.sum(dim=0)  # 累加特征
        return prototypes_full, fused_features


class CrossVehicleAttentionfuse(nn.Module):
    def __init__(self, feature_dim=256, context_dim=8, num_heads=8):
        super(CrossVehicleAttentionfuse, self).__init__()
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.context_proj = nn.Linear(context_dim, feature_dim)
        self.info_gain_proj = nn.Linear(feature_dim, 1)

        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, current_anchor, neighbor_anchor, current_features, neighbor_features):
        # current_features: [900, 256]
        # neighbor_features: [N, 900, 256]
        # current_anchor: [900, 8]
        # neighbor_anchor: [N, 900, 8]
        residual = current_features
        N, num_anchors, feature_dim = neighbor_features.size()
        expanded_current_features = current_features.unsqueeze(0).expand(N, -1, -1)  # [N, 900, 256]
        query = self.query_proj(expanded_current_features)  # [N, 900, 256]
        key = self.key_proj(neighbor_features) + self.context_proj(neighbor_anchor)  # [N, 900, 256]
        value = self.value_proj(neighbor_features)  # [N, 900, 256]
        attention_scores = torch.einsum('nqc,nkc->nqk', query, key) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)  # [N, 900, 900]
        attention_weights = self.dropout(attention_weights)

        # info_gain = torch.sigmoid(self.info_gain_proj(key))  # [N, 900, 256]
        # attention_weights *= info_gain  # [N, 900, 900]

        fused_features = torch.einsum('nqk,nkv->nqv', attention_weights, value)  # [N, 900, 256]
        fused_features = fused_features.mean(dim=0)  # 取均值以聚合所有邻居的信息 [900, 256]
        fused_features = self.out_proj(fused_features)  # [900, 256]
        fused_features = self.dropout(fused_features)
        fused_features = self.layer_norm(residual + fused_features)  # [900, 256]

        return current_anchor,fused_features


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv1d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv1d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x = x.permute(0,2,1)
        x_se = x_se.permute(0,2,1)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        out = x * self.gate(x_se)
        out = out.permute(0,2,1)
        return out



class PointEncoder(nn.Module):
    def __init__(self,
                 feature_embedding=256,
                 cfg=None
                 ):
        super(PointEncoder,self).__init__()
        # self.bn = nn.InstanceNorm1d(12)  
        self.transformencoding = Mlp(12, feature_embedding, feature_embedding)
        self.context_se = SELayer(feature_embedding)  # NOTE: add camera-aware
        self.att = MultiheadAttention(cfg)
    def forward(self, i2j_anchor,b_anchor,b_feature,
                transform_matrix,lidar_range, refinement, anchorencoder,anchor_embed,return_cls):

        N,num_anchor,anchor_dim = b_anchor.shape
        _,_,feature_dim = b_feature.shape
        # neighbor_anchor = neighbor_anchor.reshape(-1,anchor_dim)
        # neighbor_features = neighbor_features.reshape(-1,feature_dim)

        mask = ~torch.eye(N, dtype=torch.bool)
        i2j_anchor_remove = i2j_anchor[mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_anchor, 8)].reshape(N,-1,anchor_dim) # N,-1,8
        transform_matrix_remove = transform_matrix[mask.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,4,4)].reshape(N,N-1,4,4)
        neighbor_feature = b_feature.unsqueeze(0).repeat(N,1,1,1)
        neighbor_feature_remove = neighbor_feature[mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_anchor, 256)].reshape(N,-1,feature_dim)
        key = neighbor_feature_remove  # N, -1, 256

        mlp_input =  torch.stack(
                    [
                        transform_matrix_remove[..., 0, 0],
                        transform_matrix_remove[..., 0, 1],
                        transform_matrix_remove[..., 0, 2],
                        transform_matrix_remove[..., 0, 3],
                        transform_matrix_remove[..., 1, 0],
                        transform_matrix_remove[..., 1, 1],
                        transform_matrix_remove[..., 1, 2],
                        transform_matrix_remove[..., 1, 3],
                        transform_matrix_remove[..., 2, 0],
                        transform_matrix_remove[..., 2, 1],
                        transform_matrix_remove[..., 2, 2],
                        transform_matrix_remove[..., 2, 3],
                    ],
                    dim=-1,
                )
        #  N-1,12
        # trans_input = self.bn(mlp_input)
        trans_encoding = self.transformencoding(mlp_input)[:,:,None,:].repeat(1,1,num_anchor,1)  # N-1,256
        trans_encoding = trans_encoding.reshape(N,-1,feature_dim)
        anchor_embeding = anchorencoder(i2j_anchor_remove) # not including self
        anchor_embeding = anchor_embeding + trans_encoding

        key_pos = anchor_embeding  # N,-1,256
        # cross-attention
        res = self.att(b_feature,key=key,query_pos=anchor_embed,key_pos=key_pos)

        # output = self.context_se(neighbor_features_remove, anchor_embeding)

        output, cls, dir = refinement(res,b_anchor,anchor_embed,return_cls=return_cls)

        yaw = torch.atan2(output[:, :, SIN_YAW], output[:, :, COS_YAW])
        box = torch.cat(
            [
                output[..., [X, Y, Z]],
                output[..., [W, L, H]].exp(),
                yaw[..., None],
            ],
            dim=-1,
        )

        return cls, box, dir, output, res


class PointEncoderV6(nn.Module):
    def __init__(self,
                 feature_embedding=256,
                 cfg=None,
                 norm_cfg=None,
                 ffn_cfg=None
                 ):
        super(PointEncoderV6, self).__init__()
        self.bn = nn.BatchNorm1d(12)  
        self.transformencoding = Mlp(12, feature_embedding, feature_embedding)
        # self.context_se = SELayer(feature_embedding)  # NOTE: add camera-aware
        self.att = MultiheadAttention(cfg)
        self.rp = Linear(feature_embedding, 1)  # 2 points and 3 offset
        self.norm = nn.LayerNorm(**norm_cfg)
        self.cls_layers = nn.Sequential(
            *linear_relu_ln(feature_embedding, 1, 2),
            Linear(feature_embedding, 1),
        )
        self.Lambda = nn.Linear(feature_embedding, cfg['num_heads'])

    def forward(self, i2j_anchor, b_anchor, b_feature,
                transform_matrix, lidar_range, refinement, anchorencoder,
                anchor_embed, return_cls, K, step, lay_num, roi_th=None):
        N, num_anchor, anchor_dim = b_anchor.shape
        _, _, feature_dim = b_feature.shape

        ####################### for adpative fusion #######################
        confidence = refinement.cls_layers(b_feature)
        roi_th = torch.sigmoid(confidence)
        top_scores, top_indices = torch.topk(roi_th, K, dim=1)
        confidence_mask = top_scores > 0.7  

        top_indices_feature = top_indices.expand(-1, -1, feature_dim)  # N,K,256
        neighbor_feature = torch.gather(b_feature, 1, top_indices_feature)  # N,K,256

        top_indices_anchor = top_indices.unsqueeze(0).expand(N, -1, -1, anchor_dim)
        i2j_anchor = torch.gather(i2j_anchor, 2, top_indices_anchor)  # N,N,K,8

        min_points, max_points = calculate_bbox_corners_with_orientation(b_anchor)
        # file the neighbor points
        ####################### for adpative fusion #######################

        ####################### remove ego point #######################
        mask = ~torch.eye(N, dtype=torch.bool)
        i2j_anchor_remove = i2j_anchor[
            mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, i2j_anchor.shape[2], anchor_dim)].\
            reshape(N, -1,anchor_dim)  # N,-1,8
        transform_matrix_remove = transform_matrix[mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 4)]\
            .reshape(N,N - 1,4,4)
        neighbor_feature = neighbor_feature.unsqueeze(0).repeat(N, 1, 1, 1)
        neighbor_feature_remove = neighbor_feature[
            mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, neighbor_feature.shape[2], feature_dim)].\
            reshape(N, -1,feature_dim)

        confidence_mask = confidence_mask.unsqueeze(0).expand(N,-1,-1,1)
        confidence_mask_remove = confidence_mask[
            mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, i2j_anchor.shape[2], 1)].\
            reshape(N, -1, 1)  # N,-1,8

        key = neighbor_feature_remove  # N, -1, 256

        mlp_input = torch.stack(
            [
                transform_matrix_remove[..., 0, 0],
                transform_matrix_remove[..., 0, 1],
                transform_matrix_remove[..., 0, 2],
                transform_matrix_remove[..., 0, 3],
                transform_matrix_remove[..., 1, 0],
                transform_matrix_remove[..., 1, 1],
                transform_matrix_remove[..., 1, 2],
                transform_matrix_remove[..., 1, 3],
                transform_matrix_remove[..., 2, 0],
                transform_matrix_remove[..., 2, 1],
                transform_matrix_remove[..., 2, 2],
                transform_matrix_remove[..., 2, 3],
            ],
            dim=-1,
        )

        # ####################### CASA #######################
        # #  N-1,12
        trans_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        trans_encoding = self.transformencoding(trans_input)
        trans_encoding = trans_encoding.reshape(N, -1, feature_dim)[:, :, None, :]. \
            repeat(1, 1, K, 1)  # N-1,256
        trans_encoding = trans_encoding.reshape(N, -1, feature_dim)

        anchor_embeding = anchorencoder(i2j_anchor_remove)  # not including self
        anchor_embeding = anchor_embeding + trans_encoding
        key_pos = anchor_embeding  # N,-1,256
        query_pos = anchor_embed

        # select_dist_matrxi = torch.cdist(b_anchor[..., :2], i2j_anchor_remove[..., :2]) # N,300,100*(N-1)
        # Lm = self.Lambda(b_feature)  # [B, Q, 8]
        # Lm = Lm.permute(0, 2, 1)
        # attn_mask = - torch.log(1 + select_dist_matrxi[:, None, :, :]) * Lm[..., None]  # [B, 8, Q, Q]
        # attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]


        # only ego ?
        b_ego_feature = b_feature[0:1].clone()
        b_neighbor_feature = b_feature[1:].clone()
        b_ego_feature = b_ego_feature.expand(N-1,-1,-1)
        query_pos = query_pos[0:1].expand(N-1,-1,-1)
        


        res = self.att(b_ego_feature, key=b_neighbor_feature, query_pos=query_pos)
        res = torch.sum(res, dim=0,keepdim=True)
        b_ego_feature = self.norm(res)
        b_feature = torch.cat([b_ego_feature,b_neighbor_feature])

        b_feature = res
        ####################### CASA #######################


        ####################### remove ego point #######################

        ####################### spatial-aware for fusion #######################
        neighbor_feature_remove = neighbor_feature_remove * confidence_mask_remove
        aggfeature, is_greater_than_range = compute_features_and_mask_efficient\
            (min_points,max_points,i2j_anchor_remove[..., [X, Y, Z]],neighbor_feature_remove)
        b_feature = b_feature + aggfeature
        is_greater_than_range = is_greater_than_range.unsqueeze(-1)
        ####################### spatial-aware for fusion #######################


        ####################### encoding #######################
        if step == 2:
            ####################### encoding #######################
            # output = self.context_se(neighbor_features_remove, anchor_embeding)
            # output, cls, dir = refinement(all_feature, all_anchor, all_anchor_embed, return_cls=return_cls)
            output, cls, dir = refinement(b_feature, b_anchor, anchor_embed, return_cls=return_cls)
            box = get_anchor_box(output)


            return cls, box, dir, output, b_feature
        else:
            return None, None, None, b_anchor, b_feature




class PointEncoderV5(nn.Module):
    def __init__(self,
                 feature_embedding=256,
                 cfg=None,
                 norm_cfg=None,
                 ):
        super(PointEncoderV5, self).__init__()
        self.bn = nn.BatchNorm1d(12)  #
        self.transformencoding = Mlp(12, feature_embedding, feature_embedding)
        # self.context_se = SELayer(feature_embedding)  # NOTE: add camera-aware
        self.att = MultiheadAttention(cfg)
        self.rp = Linear(feature_embedding, 1)  # 2 points and 3 offset
        self.norm =  nn.LayerNorm(**norm_cfg)
        self.cls_layers = nn.Sequential(
            *linear_relu_ln(feature_embedding, 1, 2),
            Linear(feature_embedding, 1),
        )
        self.dir_layers =  nn.Sequential(
                *linear_relu_ln(feature_embedding, 1, 2),
                Linear(feature_embedding, 2),
            )
        self.Lambda = nn.Linear(feature_embedding, cfg['num_heads'])

    def forward(self, i2j_anchor, b_anchor, b_feature,
                transform_matrix, lidar_range, refinement, anchorencoder,
                anchor_embed, return_cls, K, step, roi_th=None):
        N, num_anchor, anchor_dim = b_anchor.shape
        _, _, feature_dim = b_feature.shape

        ####################### for adpative fusion #######################
        # self.cls_layers = refinement.cls_layers
        # self.dir_layers = refinement.dir_layers
        roi_th = refinement.cls_layers(b_feature)  # 
        roi_th = torch.sigmoid(roi_th)
        top_scores, top_indices = torch.topk(roi_th, K, dim=1) # 

        confidence_mask = top_scores > 0.5   # 
        # need i2janchor --> N,N,900,8 --> N,K*N-1,8

        top_indices_feature = top_indices.expand(-1, -1, feature_dim)  # N,N,K,256
        neighbor_feature = torch.gather(b_feature, 1, top_indices_feature)  # N,K,256

        # b_select_anchor = torch.gather(b_anchor,1,top_indices.expand(-1, -1, anchor_dim))
        # b_select_feature = neighbor_feature
        # b_select_anchorembed = torch.gather(anchor_embed,1,top_indices_feature)

        top_indices_anchor = top_indices.unsqueeze(0).expand(N, -1, -1, anchor_dim)
        i2j_anchor = torch.gather(i2j_anchor, 2, top_indices_anchor)  # N,N,K,8


        # 需要再想想
        min_points, max_points = calculate_bbox_corners_with_orientation(b_anchor)


        # file the neighbor points
        ####################### for adpative fusion #######################

        ####################### remove ego point #######################
        mask = ~torch.eye(N, dtype=torch.bool)
        i2j_anchor_remove = i2j_anchor[
            mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, i2j_anchor.shape[2], anchor_dim)].\
            reshape(N, -1,anchor_dim)  # N,-1,8
        transform_matrix_remove = transform_matrix[mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 4)]\
            .reshape(N,N - 1,4,4)
        neighbor_feature = neighbor_feature.unsqueeze(0).repeat(N, 1, 1, 1)
        neighbor_feature_remove = neighbor_feature[
            mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, neighbor_feature.shape[2], feature_dim)].\
            reshape(N, -1,feature_dim)

        confidence_mask = confidence_mask.unsqueeze(0).expand(N,-1,-1,1)
        confidence_mask_remove = confidence_mask[
            mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, i2j_anchor.shape[2], 1)].\
            reshape(N, -1, 1)  # N,-1,8


        key = neighbor_feature_remove  # N, -1, 256
        ####################### remove ego point #######################

        ####################### spatial-aware for fusion #######################
        # 
        neighbor_feature_remove = neighbor_feature_remove * confidence_mask_remove
        aggfeature, is_greater_than_range = compute_features_and_mask_efficient\
            (min_points,max_points,i2j_anchor_remove[..., [X, Y, Z]],neighbor_feature_remove)
        b_feature = b_feature + aggfeature
        is_greater_than_range = is_greater_than_range.unsqueeze(-1)
        ####################### spatial-aware for fusion #######################

        ####################### encoding #######################
        mlp_input = torch.stack(
            [
                transform_matrix_remove[..., 0, 0],
                transform_matrix_remove[..., 0, 1],
                transform_matrix_remove[..., 0, 2],
                transform_matrix_remove[..., 0, 3],
                transform_matrix_remove[..., 1, 0],
                transform_matrix_remove[..., 1, 1],
                transform_matrix_remove[..., 1, 2],
                transform_matrix_remove[..., 1, 3],
                transform_matrix_remove[..., 2, 0],
                transform_matrix_remove[..., 2, 1],
                transform_matrix_remove[..., 2, 2],
                transform_matrix_remove[..., 2, 3],
            ],
            dim=-1,
        )
        #  N-1,12
        trans_input = self.bn(mlp_input.reshape(-1,mlp_input.shape[-1]))
        trans_encoding = self.transformencoding(trans_input)
        trans_encoding = trans_encoding.reshape(N, -1, feature_dim)[:, :, None, :].\
                                                                repeat(1, 1, K,1)  # N-1,256
        trans_encoding = trans_encoding.reshape(N, -1, feature_dim)

        anchor_embeding = anchorencoder(i2j_anchor_remove)  # not including self
        anchor_embeding = anchor_embeding + trans_encoding
        key_pos = anchor_embeding  # N,-1,256
        ####################### encoding #######################

        # cross-attention
        # ?
        select_dist_matrxi = torch.cdist(b_anchor[..., :2], i2j_anchor_remove[..., :2]) # N,300,100*(N-1)
        Lm = self.Lambda(b_feature)  # [B, Q, 8]
        Lm = Lm.permute(0, 2, 1)
        attn_mask = - torch.log(1 + select_dist_matrxi[:, None, :, :]) * Lm[..., None]  # [B, 8, Q, Q]
        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
        key_padding_mask = ~confidence_mask_remove.squeeze(-1)

        if step == 2:
            res = self.att(b_feature, key=key, key_pos=key_pos, attn_mask=attn_mask)
            res = self.norm(res)
            all_feature = torch.cat([res,key],dim=1)
            all_anchor = torch.cat([b_anchor,i2j_anchor_remove],dim=1)
            all_anchor_embed = torch.cat([anchor_embed,anchor_embeding],dim=1)
            output, cls, dir = refinement(all_feature, all_anchor, all_anchor_embed, return_cls=return_cls)
            box = get_anchor_box(output)
            output = output[:, :num_anchor, :]
            cls[:,-K*(N-1):,:][~confidence_mask_remove] = float('-inf')
            box[:,-K*(N-1):,:][~confidence_mask_remove.expand(-1,-1,anchor_dim-1)] = 0
            # Todo: check this. In test phrase, need to do ?
            if return_cls:
                cls[:,-K*(N-1):,:][is_greater_than_range] = float('-inf')
                box[:,-K*(N-1):,:][is_greater_than_range.expand(-1,-1,anchor_dim-1)] = 0
            # points = box[..., :3]
            return cls, box, dir, output, b_feature

        else:
            res = self.att(b_feature, key=key,query_pos=anchor_embed, key_pos=key_pos, attn_mask=attn_mask)
            res = self.norm(res)
            b_feature = res
            return None, None, None, b_anchor, b_feature



class SparseBox3DRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        output_dim=11,
        num_cls=1,
        normalize_yaw=True,
        refine_yaw=False,
        with_cls_branch=True,
        return_dir = True
    ):
        super(SparseBox3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.normalize_yaw = normalize_yaw
        self.refine_yaw = refine_yaw
        self.return_dir = return_dir
        self.refine_state = [X, Y, Z, W, L, H]   # you mean only refine position and size of anchors ?
        if self.refine_yaw:
            self.refine_state += [SIN_YAW, COS_YAW]

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )
        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, self.num_cls),
            )
        if self.return_dir:
            self.dir_layers =  nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, 2),
            )
    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,   # bs,900,256
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        output = self.layers(instance_feature + anchor_embed)  # from 256 --> 1
        output[..., self.refine_state] = (
            output[..., self.refine_state] + anchor[..., self.refine_state].detach()
        ) # refine anchor position
        # TODO: what's the meaning of this normalize step ?
        if self.normalize_yaw:
            output[..., [SIN_YAW, COS_YAW]] = torch.nn.functional.normalize(
                output[..., [SIN_YAW, COS_YAW]], dim=-1
            )

        # TODO: why translation is torch.transpose(output[..., VX:], 0, -1)?
        # don't need use this
        if self.output_dim > 8:
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            translation = torch.transpose(output[..., VX:], 0, -1) # 切片后的部分,第一个维度和最后一个维度交换
            velocity = torch.transpose(translation / time_interval, 0, -1)
            output[..., VX:] = velocity + anchor[..., VX:]

        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(instance_feature)
        else:
            cls = None
        if self.return_dir:
            dir = self.dir_layers(instance_feature)
        else:
            dir = None
        return output, cls, dir  # output is refined anchor




class SparseBox3DKeyPointsGenerator(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=6,  # 6
        fix_scale=None,
    ):
        super(SparseBox3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = np.array(fix_scale)
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, num_learnable_pts * 3)

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def visualize_key_point(self, bs, kp, anchor):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        kp = kp.reshape(bs, -1, 13, 3)
        key_points = kp[0].detach().cpu().numpy()
        anchors = anchor.detach().cpu().numpy()

        # 创建图形，增加图表的尺寸
        plt.figure(figsize=(16, 12))

        for i in range(key_points.shape[0]):
            key_point = key_points[i]
            anchor_box = anchors[i]

            # 提取中心点
            center_point = key_point[0, :2]

            # 提取固定点和随机点
            fixed_points = key_point[1:7, :2]
            random_points = key_point[7:13, :2]

            # 提取box参数 (x, y, w, h)
            x, y, _, w, h, _, _, _ = anchor_box

            # 绘制散点图，调整点的大小
            plt.scatter(fixed_points[:, 0], fixed_points[:, 1], color='blue', alpha=0.6, s=10,
                        label='Fixed Points' if i == 0 else "")
            plt.scatter(random_points[:, 0], random_points[:, 1], color='red', alpha=0.6, s=10,
                        label='Random Points' if i == 0 else "")
            plt.scatter(center_point[0], center_point[1], color='green', alpha=0.6, s=10,
                        label='Center Point' if i == 0 else "")

            # 绘制统一颜色的box，调整线条的粗细
            rect = patches.Rectangle((x - w / 2, y - h / 2), w, h, linewidth=0.5, edgecolor='black', facecolor='none')
            plt.gca().add_patch(rect)

        # 添加标签和网格
        plt.xlabel("X", fontsize=16)
        plt.ylabel("Y", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 添加图例并显示
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3)

        # 显示图表
        plt.show()

    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        bs, num_anchor = anchor.shape[:2] # bs, 900

        fix_scale = anchor.new_tensor(self.fix_scale)
        scale = fix_scale[None, None].tile([bs, num_anchor, 1, 1]) # fix --> bs,900,N,3, what is N ?
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            )  # get learnable keypoints with bs,900,6,3
            scale = torch.cat([scale, learnable_scale], dim=-2)  # two type keypoints --> bs,900,13(7+6),3
        key_points = scale * anchor[..., None, [W, L, H]].exp()
        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])
        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 2, 2] = 1
        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1)
        key_points = key_points + anchor[..., None, [X, Y, Z]] # here, we get keypoints


        # self.visualize_key_point(1,key_points[0],anchor[0])


        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ):
            return key_points

        temp_key_points_list = []
        velocity = anchor[..., VX:]  # X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))
        for i, t_time in enumerate(temp_timestamps):
            time_interval = cur_timestamp - t_time
            translation = (
                velocity
                * time_interval.to(dtype=velocity.dtype)[:, None, None]
            )
            temp_key_points = key_points - translation[:, :, None]  # still in current coordinate
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
            temp_key_points = (
                T_cur2temp[:, None, None, :3]
                @ torch.cat(
                    [
                        temp_key_points,
                        torch.ones_like(temp_key_points[..., :1]),
                    ],
                    dim=-1,
                ).unsqueeze(-1)
            )   # transform to previous time coordinate
            temp_key_points = temp_key_points.squeeze(-1)
            temp_key_points_list.append(temp_key_points)
        return key_points, temp_key_points_list  # obtain current and previous key points

    @staticmethod
    def anchor_projection(
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
    ):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            dst_anchor = anchor.clone()
            vel = anchor[..., VX:]
            vel_dim = vel.shape[-1]
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
            )

            center = dst_anchor[..., [X, Y, Z]]
            if src_timestamp is not None and dst_timestamps is not None:
                translation = vel.transpose(0, -1) * (
                    src_timestamp - dst_timestamps[i]
                ).to(dtype=vel.dtype)
                translation = translation.transpose(0, -1)
                center = center - translation
            dst_anchor[..., [X, Y, Z]] = (
                torch.matmul(
                    T_src2dst[..., :3, :3], center[..., None]
                ).squeeze(dim=-1)
                + T_src2dst[..., :3, 3]
            )

            dst_anchor[..., [COS_YAW, SIN_YAW]] = torch.matmul(
                T_src2dst[..., :2, :2], dst_anchor[..., [COS_YAW, SIN_YAW], None]
            ).squeeze(-1)

            dst_anchor[..., VX:] = torch.matmul(
                T_src2dst[..., :vel_dim, :vel_dim], vel[..., None]
            ).squeeze(-1)

            dst_anchors.append(dst_anchor)
        return dst_anchors

    @staticmethod
    def distance(anchor):
        return torch.norm(anchor[..., :2], p=2, dim=-1)
