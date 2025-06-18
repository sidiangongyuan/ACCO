import torch
import torch.nn as nn

class Diversity(nn.Module):
    def __init__(self,feature_embedding):
        super().__init__()

        self.query = nn.Linear(feature_embedding, feature_embedding)
        self.key = nn.Linear(feature_embedding, feature_embedding)

    def forward(self,anchor_embed,feature):
        feature = feature + anchor_embed

        batch_size, N, feature_dim = feature.shape

        # 计算特征的L2范数，结果形状为 [batch_size, N, 1]
        norm = torch.norm(feature, p=2, dim=2, keepdim=True)

        # 标准化特征，使其在最后一个维度上具有单位范数
        normalized_feature = feature / norm

        # 计算归一化特征的点积，结果形状为 [batch_size, N, N]
        cos_similarity = torch.bmm(normalized_feature, normalized_feature.transpose(1, 2))

        # 创建一个上三角掩码，掩码形状为 [N, N]
        mask = torch.triu(torch.ones(N, N, device=feature.device), diagonal=0).unsqueeze(0) == 1

        # 应用掩码，只保留上三角部分的值，其余部分设为0
        upper_triangular_cos_sim = torch.where(mask, cos_similarity, torch.zeros_like(cos_similarity))

        # 计算每个批次中上三角余弦相似度矩阵的Frobenius范数
        div_loss = torch.norm(upper_triangular_cos_sim, p='fro', dim=(1, 2)) / N

        div_loss = torch.mean(div_loss)

        return div_loss