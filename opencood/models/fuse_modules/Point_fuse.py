import torch
import torch.nn as nn
from ..sparse4D_utils.detection3d_block import X, Y, Z, H, W, L, SIN_YAW, COS_YAW, Mlp, SELayer

class PointEncoder(nn.Module):
    def __init__(self,
                 feature_embedding=256,
                 cfg=None
                 ):
        super(PointEncoder,self).__init__()
        # self.bn = nn.InstanceNorm1d(12)  # 总共需要编码的维度
        self.transformencoding = Mlp(12, feature_embedding, feature_embedding)
        self.context_se = SELayer(feature_embedding)  # NOTE: add camera-aware
        self.att = MultiheadAttention(cfg)
    def forward(self, i2j_anchor,b_anchor,b_feature,
                transform_matrix,lidar_range, refinement, anchorencoder,ego_anchor_embed):
        N,num_anchor,anchor_dim = b_anchor.shape
        _,_,feature_dim = b_feature.shape
        # neighbor_anchor = neighbor_anchor.reshape(-1,anchor_dim)
        # neighbor_features = neighbor_features.reshape(-1,feature_dim)

        mask = ~torch.eye(N, dtype=torch.bool)

        # 使用这个mask来选择需要的元素
        # 通过unsqueeze扩展mask的维度以匹配原tensor的形状
        i2j_anchor_remove = i2j_anchor[mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 900, 8)] # N-1,N-1,900,8
        i2j_anchor_remove = i2j_anchor_remove.reshape(N,-1,anchor_dim)
        transform_matrix_remove = transform_matrix[mask.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,4,4)]
        neighbor_feature = b_anchor.unsqueeze(0).repeat(N,1,1,1)
        neighbor_feature_remove = neighbor_feature[mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 900, 256)]


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
        trans_encoding = self.transformencoding(mlp_input)[:,None,:].repeat(1,num_anchor,1)  # N-1,256

        anchor_embeding = anchorencoder(i2j_anchor_remove)
        anchor_embeding = anchor_embeding + trans_encoding

        key_pos = anchor_embeding.reshape(N, -1,feature_dim)
        key = neighbor_feature_remove.reshape(N, -1,feature_dim)
        # cross-attention
        res = self.att(b_feature,key=key,query_pos=ego_anchor_embed,key_pos=key_pos)

        # output = self.context_se(neighbor_features_remove, anchor_embeding)

        output, cls, dir = refinement(neighbor_feature_remove,i2j_anchor_remove,anchor_embeding)

        yaw = torch.atan2(output[:, :, SIN_YAW], output[:, :, COS_YAW])
        box = torch.cat(
            [
                output[..., [X, Y, Z]],
                output[..., [W, L, H]].exp(),
                yaw[..., None],
            ],
            dim=-1,
        )
        cls = cls.reshape(-1,cls.shape[-1])
        box = box.reshape(-1,box.shape[-1])

        # neighbor_points = neighbor_anchor_remove.reshape(-1,neighbor_anchor_remove.shape[-1])
        points = box[:,:3]
        # 将range_list转换为min和max范围
        min_range = torch.tensor(lidar_range[:3]).cuda()  # 取前三个值为最小范围
        max_range = torch.tensor(lidar_range[3:]).cuda()  # 取后三个值为最大范围

        # 创建掩码，检查每个点是否在范围内
        mask = (points >= min_range) & (points <= max_range)  # 检查每个维度是否在范围内
        mask = mask.all(dim=1)  # 只有当点在所有维度上都在范围内时，才算在范围内
        mask = mask.unsqueeze(-1)
        cls = torch.where(mask, cls, torch.tensor(float('-inf')).cuda())
        box = torch.where(mask.expand(-1, box.shape[-1]), box, torch.tensor(0.0).cuda())
        return cls, box, mask, dir





