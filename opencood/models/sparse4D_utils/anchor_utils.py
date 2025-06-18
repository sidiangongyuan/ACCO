# 重新导入torch库
import numpy as np
import torch

# 定义常量，便于理解和使用
X, Y, Z, W, H, L, SIN_YAW, COS_YAW = range(8)

def calculate_bbox_corners_with_orientation(anchors, scale=1.0):
    bs, num_anchor, _ = anchors.shape
    key_points = scale * torch.exp(anchors[..., [W, L, H]])

    rotation_mat = anchors.new_zeros((bs, num_anchor, 3, 3))
    rotation_mat[..., 0, 0] = anchors[..., COS_YAW]
    rotation_mat[..., 0, 1] = -anchors[..., SIN_YAW]
    rotation_mat[..., 1, 0] = anchors[..., SIN_YAW]
    rotation_mat[..., 1, 1] = anchors[..., COS_YAW]
    rotation_mat[..., 2, 2] = 1

    rotated_key_points = torch.matmul(rotation_mat.view(-1, 3, 3), key_points.view(-1, 3, 1)).view(bs, num_anchor, 3)

    center = anchors[..., [X, Y, Z]]
    min_points = center - rotated_key_points / 2
    max_points = center + rotated_key_points / 2

    return min_points, max_points


def compute_features_and_mask_efficient(min_points, max_points, neighbor_anchor_xyz, features, weight_net=None):
    # min_points: B, N, 3
    # max_points: B, N, 3
    # neighbor_anchor: B, K, 3
    # features: B, K, feature_dim
    inside_min = (neighbor_anchor_xyz.unsqueeze(1) >= min_points.unsqueeze(2)).all(dim=-1)
    inside_max = (neighbor_anchor_xyz.unsqueeze(1) <= max_points.unsqueeze(2)).all(dim=-1)
    inside_bbox = inside_min & inside_max

    expanded_features = features.unsqueeze(1).expand(-1, min_points.size(1), -1, -1)

    if not weight_net:
        aggregated_features = torch.where(inside_bbox.unsqueeze(-1), expanded_features,
                                          torch.zeros_like(expanded_features)).sum(dim=2)
    else:
        raw_weights = weight_net(expanded_features)  # B, N, K, 1
        adjusted_weights = torch.where(inside_bbox.unsqueeze(-1), raw_weights, torch.zeros_like(raw_weights))
        weights = torch.softmax(adjusted_weights, dim=2)
        weighted_features = expanded_features * weights
        aggregated_features = weighted_features.sum(dim=2)
        has_valid_points = inside_bbox.any(dim=2).unsqueeze(-1)  # B, N, 1
        aggregated_features *= has_valid_points.float()

    final_mask = inside_bbox.any(dim=1)

    return aggregated_features, final_mask



def generate_anchor_heatmap(x, y, scores, layer=None, title='Heatmap', figsize=(3.84,2.4), Issigmoid=False, save_path=None):
    import matplotlib.pyplot as plt
    """
    """

    if x.is_cuda:
        x = x.cpu()
    if y.is_cuda:
        y = y.cpu()
    if scores.is_cuda:
        scores = scores.cpu()

    if not Issigmoid:
        scores = torch.sigmoid(scores)


    x = x.numpy()
    y = y.numpy()
    scores = scores.numpy()


    heatmap, xedges, yedges = np.histogram2d(x, y, bins=100, weights=scores, range=[[-76.8, 76.8], [-48, 48]])

    plt.figure(figsize=figsize)
    X, Y = np.meshgrid(xedges, yedges)
    pcm = plt.pcolormesh(X, Y, heatmap.T, cmap='inferno', shading='auto')
    pcm.set_clim(0, 1) 


    cbar = plt.colorbar(pcm, pad=0.01)
    cbar.ax.tick_params(labelsize=12)


    plt.axis('off')


    plt.tight_layout()


    if save_path:
        if layer:
            plt.savefig(f'{save_path}/heatmap_layer_{layer}_fusion.png', dpi=400, transparent=False)
        else:
            plt.savefig(save_path, dpi=400, transparent=False)

    plt.show()


if __name__ == '__main__':

    bs, num_anchor = 2, 3
    anchors = torch.randn(bs, num_anchor, 8) 


    min_points, max_points = calculate_bbox_corners_with_orientation(anchors)
    print("最小点:", min_points)
    print("最大点:", max_points)

    B, N, K, feature_dim = 2, 3, 10, 4  
    # min_points = torch.rand(B, N, 3)
    # max_points = torch.rand(B, N, 3)
    neighbor_anchor = torch.rand(B, K, 3)
    features = torch.rand(B, K, feature_dim)
    aggregated_features, final_mask = compute_features_and_mask_efficient(min_points, max_points, neighbor_anchor, features)
