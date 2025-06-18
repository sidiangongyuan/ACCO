import torch
import numpy as np
from mmdet3d.structures import BaseInstance3DBoxes
from scipy.optimize import linear_sum_assignment

from .decoder import X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ, YAW

class SparseBox3DTarget(object):
    def __init__(
        self,
        cls_weight=2.0,
        alpha=0.25,
        gamma=2,
        eps=1e-12,
        box_weight=0.25,
        reg_weights=None,
        cls_wise_reg_weights=None,
    ):
        super(SparseBox3DTarget, self).__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reg_weights = reg_weights
        if self.reg_weights is None:
            self.reg_weights = [1.0] * 7
        if cls_wise_reg_weights == 'None':
            self.cls_wise_reg_weights = None
        else:
            self.cls_wise_reg_weights = cls_wise_reg_weights

    def encode_reg_target(self, box_target, device=None):
        outputs = []
        for box in box_target:
            if isinstance(box, BaseInstance3DBoxes):
                center = box.gravity_center
                box = box.tensor.clone()
                box[..., [X, Y, Z]] = center

            output = torch.cat(
                [
                    box[..., [X, Y, Z]],
                    box[..., [W, L, H]].log(),
                    torch.sin(box[..., YAW]).unsqueeze(-1),
                    torch.cos(box[..., YAW]).unsqueeze(-1),
                    box[..., YAW + 1 :],
                ],
                dim=-1,
            )
            if device is not None:
                output = output.to(device=device)
            outputs.append(output)
        return outputs

    def sample(
        self,
        cls_pred,
        box_pred,
        cls_target,
        box_target,
    ):
        '''

        Args:
            cls_pred: bs,900,(num class)
            box_pred:bs,900, 11   or 7 ?
            cls_target: [list]  [tensor,tensor,...] --> [8,1,5,...,] in nuscenes, but in OPV2V , only one class --> [1,0,1,1,....]
            box_target: [list]  [LIDARInstance,LiDARInstance,...]
            LIDARInstance:
                bev: 32,5
                botto_center:32,3
                botto_height:32
                center: 32,3
                corners: 32,8,3
                tensor: 32,9   9 means ?
        Returns:

        '''
        bs, num_pred, num_cls = cls_pred.shape # bs, 900, 10
        # cls_cost is bs,900,ojbect ?

        cls_cost = self._cls_cost(cls_pred, cls_target)  # focalloss

        # box_target = self.encode_reg_target(box_target, box_pred.device) # list: bs (N, 10 ) , N is the objects in a scenario.   I need to get  box_target:[list]  8,N,7

        # I don't need weights, only one class
        instance_reg_weights = []
        for i in range(len(box_target)):
            weights  = torch.logical_not(
                box_target[i].isnan()
            ).to(dtype=box_target[i].dtype)
            if self.cls_wise_reg_weights is not None:
                for cls, weight in self.cls_wise_reg_weights.items():
                    weights = torch.where(
                        (cls_target[i] == cls)[:, None],
                        weights.new_tensor(weight),
                        weights
                    )
            instance_reg_weights.append(weights)

        box_cost = self._box_cost(box_pred, box_target, instance_reg_weights)  # so here, I box_pred: B,900,7  and box_target:[list]  8,N,7

        indices = []
        for i in range(bs):
            if cls_cost[i] is not None and box_cost[i] is not None:
                cost = (cls_cost[i] + box_cost[i]).detach().cpu().numpy()
                cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
                indices.append(
                    [
                        cls_pred.new_tensor(x, dtype=torch.int64)
                        for x in linear_sum_assignment(cost)
                    ]
                )
            else:
                indices.append([None, None])

        output_cls_target = cls_target[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        # output_cls_target = torch.ones(cls_target[0], dtype=torch.int64).to(cls_target[0].device)
        output_box_target = box_pred.new_zeros(box_pred.shape)
        output_reg_weights = box_pred.new_zeros(box_pred.shape)
        
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(cls_target[i]) == 0:
                continue
            output_cls_target[i, pred_idx] = cls_target[i][target_idx]
            output_box_target[i, pred_idx] = box_target[i][target_idx]
            output_reg_weights[i, pred_idx] = instance_reg_weights[i][target_idx]

        return output_cls_target, output_box_target, output_reg_weights

    def _cls_cost(self, cls_pred, cls_target):
        bs = cls_pred.shape[0]
        cls_pred = cls_pred.sigmoid() # why not use softmax ?
        cost = []
        for i in range(bs):
            if len(cls_target[i]) > 0:
                neg_cost = (
                    -(1 - cls_pred[i] + self.eps).log()
                    * (1 - self.alpha)
                    * cls_pred[i].pow(self.gamma)
                )
                pos_cost = (
                    -(cls_pred[i] + self.eps).log()
                    * self.alpha
                    * (1 - cls_pred[i]).pow(self.gamma)
                )
                cost.append(
                    (pos_cost[:, cls_target[i]] - neg_cost[:, cls_target[i]])
                    * self.cls_weight
                )
            else:
                cost.append(None)
        return cost

    def _box_cost(self, box_pred, box_target, instance_reg_weights):
        bs = box_pred.shape[0]
        cost = []
        for i in range(bs):
            if len(box_target[i]) > 0:
                cost.append(
                    torch.sum(
                        torch.abs(box_pred[i, :, None] - box_target[i][None])
                        * instance_reg_weights[i][None]
                        * box_pred.new_tensor(self.reg_weights),
                        dim=-1,
                    )
                    * self.box_weight
                )
            else:
                cost.append(None)
        return cost
