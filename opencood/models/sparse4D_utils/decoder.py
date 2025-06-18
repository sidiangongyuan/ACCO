# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Optional
import torch
X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))
YAW = 6

class SparseBox3DDecoder(object):
    def __init__(
        self,
        num_output: int = 100,
        score_threshold: Optional[float] = 0.25,
    ):
        super(SparseBox3DDecoder, self).__init__()
        self.num_output = num_output
        self.score_threshold = score_threshold
        self.anchor_nums = 600

    def decode(self, cls_scores: torch.Tensor, box_preds: torch.Tensor):
        # only used in test
        cls_scores = cls_scores[-1].sigmoid()
        box_preds = box_preds[-1]
        bs, num_pred, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1
        )
        cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold
        output = []
        for i in range(bs):
            #
            vehicle_type = ["ego" if idx < self.anchor_nums else "other" for idx in indices[i]]
            category_ids = cls_ids[i]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
                vehicle_type = [vehicle_type[j] for j, m in enumerate(mask[i]) if m]

            output.append(
                {
                    "boxes_3d": box,
                    "scores_3d": scores,
                    "labels_3d": category_ids,
                    "vehicle_type": vehicle_type,

                }
            )
        return output
