import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from mmdet.models import weight_reduce_loss
from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss

from ..models.sparse4D_utils.decoder import SparseBox3DDecoder
from ..models.sparse4D_utils.target import SparseBox3DTarget
from ..models.sparse4D_utils.block import DeformableFeatureAggregation as DFG

def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A warpper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = sigmoid_focal_loss
            else:
                num_classes = pred.size(1)
                target = F.one_hot(target, num_classes=num_classes + 1)
                target = target[:, :num_classes]
                calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls

class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

            #anchors = H * W * anchor_num

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = torch.mul(loss , weights)

        return loss



class DetrLoss(nn.Module):
    def __init__(self,args):
        super(DetrLoss,self).__init__()
        self.reg_weights = args['reg_weights']
        self.reg_weight = args['reg_weight']
        self.sampler = SparseBox3DTarget(**args['sampler'])
        self.decoder = SparseBox3DDecoder()
        self.loss_cls = FocalLoss(**args['focalloss'])
        self.loss_reg = WeightedSmoothL1Loss()
        self.cls_threshold_to_reg = args['cls_threshold_to_reg']
        self.loss_dict = {}
        self.use_dir = False

    def forward(self,result, gt_dict):
        # here, cls and reg is list, which contain len(self.layer) results.
        # We only need ego . So before this , we need to fuse agent
        '''
        Args:
            cls_scores: [list] : layer of op.(6)    cls_scores[0] :  B,900,1
            reg_preds:  [list] :  ---               reg_preds[0] : B,900,11
            data: include gt_box_3d, gt_label_3d
            feature_maps:

        Returns:
        '''
        # Todo: direction loss

        cls_scores = result['cls_scores']
        reg_preds = result['reg_preds']
        feature_maps = result['feature_maps']
        # only train ego
        object_target_list = []
        cls_target_list = []
        ego_target = gt_dict['object_target'][0]
        ego_obj_num = len(ego_target)
        object_target_list.append(ego_target)
        cls_target = torch.zeros(ego_obj_num, dtype=torch.int64).to(ego_target.device)
        cls_target_list.append(cls_target)

        layer_num = len(cls_scores)

        ego_cls_scores = []
        ego_reg_preds = []
        for i in range(layer_num):
            if cls_scores[i] is None:
                continue
            ego_cls_scores.append(cls_scores[i][0:1])
            ego_reg_preds.append(reg_preds[i][0:1])

        # object_target_list.append(object_target)
        # cls_target_list.append(cls_target)
        output = {}
        total_loss = 0
        for decoder_idx, (cls, reg) in enumerate(zip(ego_cls_scores, ego_reg_preds)):
            if cls is None:
                continue
            reg = reg[..., : len(self.reg_weights)]  # self.reg_weights = 2,2,2,1,...,1    10 length the same as classification number.
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                cls_target_list,
                object_target_list
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )

            box_preds_sin, reg_targets_sin = add_sin_difference(reg, reg_target)

            reg_loss = self.loss_reg(
                box_preds_sin, reg_targets_sin, reg_weights
            )
            reg_loss = torch.sum(reg_loss) / reg_loss.shape[0]
            reg_loss = reg_loss * self.reg_weight
            # 检查mask中是否有有效的预测框
            if mask.sum() == 0:
                # 如果没有有效预测框，添加一个固定的惩罚值
                penalty = reg.new_tensor(25)  # YOUR_PENALTY_VALUE是您设定的惩罚值
                reg_loss += penalty
            total_loss = total_loss + cls_loss + reg_loss
            output.update(
                {
                    f"loss_cls_{decoder_idx}": cls_loss,
                    f"loss_reg_{decoder_idx}": reg_loss,
                }
            )
        self.loss_dict = output
        self.loss_dict.update(
            {
                'total_loss': total_loss,
            }
        )

        # if (
        #     self.depth_module is not None
        #     and self.kps_generator is not None
        #     and feature_maps is not None
        # ):
        #     reg_target = self.sampler.encode_reg_target(
        #         data[self.gt_reg_key], reg_preds[0].device
        #     )
        #     loss_depth = []
        #     for i in range(len(reg_target)):
        #         if len(reg_target[i]) == 0:
        #             continue
        #         key_points = self.kps_generator(reg_target[i][None])
        #         features = (
        #             DFG.feature_sampling(
        #                 [f[i : i + 1] for f in feature_maps],
        #                 key_points,
        #                 data["projection_mat"][i : i + 1],
        #                 data["image_wh"][i : i + 1],
        #             )
        #             .mean(2)
        #             .mean(2)
        #         )
        #         depth_confidence = self.depth_module(
        #             features, reg_target[i][None, :, None], output_conf=True
        #         )
        #         loss_depth.append(-torch.log(depth_confidence).sum())
        #     output["loss_depth"] = (
        #         sum(loss_depth) / num_pos / self.kps_generator.num_pts
        #     )
        #
        # if self.dense_depth_module is not None:
        #     output["loss_dense_depth"] = self.dense_depth_module(
        #         feature_maps,
        #         focal=data.get("focal"),
        #         gt_depths=data["gt_depth"],
        #     )
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        conf_loss = self.loss_dict['loss_cls_5']
        reg_loss  = torch.sum(self.loss_dict['loss_reg_5'])

        print_msg = ("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                     " || Loc Loss: %.4f" % (
                         epoch, batch_id + 1, batch_len,
                         total_loss.item(), conf_loss.item(), reg_loss.item()))

        if self.use_dir:
            dir_loss = self.loss_dict['dir_loss']
            print_msg += " || Dir Loss: %.4f" % dir_loss.item()

        print(print_msg)

        if not writer is None:
            writer.add_scalar('Regression_loss', reg_loss.item(),
                              epoch * batch_len + batch_id)
            writer.add_scalar('Confidence_loss', conf_loss.item(),
                              epoch * batch_len + batch_id)

            if self.use_dir:
                writer.add_scalar('dir_loss', dir_loss.item(),
                                  epoch * batch_len + batch_id)


import torch.distributed as dist
def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor



def add_sin_difference(boxes1, boxes2, dim=6):
    assert dim != -1
    rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                        torch.cos(boxes2[..., dim:dim + 1])
    rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                      torch.sin(boxes2[..., dim:dim + 1])

    boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                        boxes1[..., dim + 1:]], dim=-1)
    boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                        boxes2[..., dim + 1:]], dim=-1)
    return boxes1, boxes2