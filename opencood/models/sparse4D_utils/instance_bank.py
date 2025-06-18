import numpy as np
import torch
from torch import nn
import numpy as np
from .detection3d_block import SparseBox3DKeyPointsGenerator


class InstanceBank(nn.Module):
    def __init__(
        self,
        config
    ):
        super(InstanceBank, self).__init__()
        self.embed_dims = config['embed_dims']
        self.num_temp_instances = config['num_temp_instances']
        self.default_time_interval = config['default_time_interval']
        self.max_queue_length = config['max_queue_length']
        self.confidence_decay = config['confidence_decay']
        self.max_time_interval = config['max_time_interval']
        self.anchor_handler = SparseBox3DKeyPointsGenerator()
        if isinstance(config['anchor'], str):
            anchor = np.load(config['anchor'])
        elif isinstance(config['anchor'], (list, tuple)):
            anchor = np.array(config['anchor'])
        self.num_anchor = min(len(anchor), config['num_anchor'])
        anchor = anchor[:self.num_anchor] # both 900
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=config['anchor_grad'],
        )
        # visualize anchor
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=config['feat_grad'],
        )   # 900 x embed_dims
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None
        self.feature_queue = [] if config['max_queue_length'] > 0 else None
        self.meta_queue = [] if config['max_queue_length'] > 0 else None

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def visualize_anchor(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # Detach and convert tensor to numpy array
        anchor = self.anchor.detach().cpu().numpy()

        # Set the style and color palette
        sns.set(style="whitegrid")

        # Create the plot
        plt.figure(figsize=(6, 4))  # Adjust the figure size for single column

        # Use Seaborn scatter plot for better aesthetics
        sns.scatterplot(x=anchor[:, 0], y=anchor[:, 1],
                        alpha=0.7, s=50, edgecolor='w', linewidth=0.5, palette='viridis')

        # Add labels
        plt.xlabel("X", fontsize=14)
        plt.ylabel("Y", fontsize=14)

        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)

        # Set tighter layout to better fit the content
        plt.tight_layout()

        # Save the plot
        plt.savefig('/home/yangk/coSparse4D/coSparse4D/images/anchor_dis_orginal.png', bbox_inches='tight', dpi=300)

        # Show plot
        plt.show()

    def get(self, batch_size, metas=None):
        # bs,900,256
        # self.visualize_anchor()
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1)) # bs,900,11

        if self.cached_anchor is not None:
            instance_feature_ = torch.tile(
                self.cached_feature, (batch_size, 1, 1)   # torch.tile --> the same as repeat ? yes
            )

            anchor_ = torch.tile(
                self.cached_anchor, (batch_size, 1, 1)   # torch.tile --> the same as repeat ? yes
            )
        else:
            instance_feature_ = None
            anchor_ = None
        time_interval = 1
        return (
            instance_feature,
            anchor,
            instance_feature_,
            anchor_,
            time_interval
        )

    def update(self, instance_feature, anchor, confidence):
        if self.cached_feature is None:
            return instance_feature, anchor
        bs = instance_feature.shape[0]
        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature[0:1,...]], dim=1
        )
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor[0:1,...]], dim=1
        )

        instance_feature_ = torch.tile(
            selected_feature, (bs, 1, 1)   # torch.tile --> the same as repeat ? yes
        )

        anchor_ = torch.tile(
            selected_anchor, (bs, 1, 1)   # torch.tile --> the same as repeat ? yes
        )
        instance_feature = 0.9 * instance_feature_ + 0.1* instance_feature
        anchor = 0.9 * anchor_ + 0.1 * anchor
        # instance_feature = torch.where(
        #     self.mask[:, None, None], selected_feature, instance_feature[0:1,...]
        # )
        # anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor[0:1,...])
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        if self.feature_queue is not None and not self.training:
            while len(self.feature_queue) > self.max_queue_length - 1:
                self.feature_queue.pop()
                self.meta_queue.pop()
            self.feature_queue.insert(0, feature_maps)
            self.meta_queue.insert(0, metas)

        if self.num_temp_instances > 0:
            instance_feature = instance_feature.detach()
            anchor = anchor.detach()
            confidence = confidence.detach()

            self.metas = metas
            confidence = confidence.max(dim=-1).values.sigmoid()
            if self.confidence is not None:
                confidence[:, : self.num_temp_instances] = torch.maximum(
                    self.confidence * self.confidence_decay,
                    confidence[:, : self.num_temp_instances],
                )

            (
                confidence_,
                outputs_,
            ) = topk(
                confidence, self.num_temp_instances, instance_feature, anchor
            )
            self.confidence = confidence_[0:1,...]
            self.cached_feature = outputs_[0][0:1,...]
            self.cached_anchor = outputs_[1][0:1,...]


def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs



