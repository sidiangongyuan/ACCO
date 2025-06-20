# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from collections import OrderedDict

import numpy as np
import torch
from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.utils.transformation_utils import get_relative_transformation
from opencood.utils.box_utils import create_bbx, project_box3d, nms_rotated
from opencood.utils.camera_utils import indices_to_depth
from sklearn.metrics import mean_squared_error
from torch.profiler import profile, record_function, ProfilerActivity
from thop import profile

def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return return_dict



def inference_no_fusion(batch_data, model, dataset, single_gt=False):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    single_gt : bool
        if True, only use ego agent's label.
        else, use all agent's merged labels.
    """
    output_dict_ego = OrderedDict()
    if single_gt:
        batch_data = {'ego': batch_data['ego']}
        
    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_no_fusion(batch_data,  # only for late fusion dataset
                             output_dict_ego)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return_dict.update({'output_dict': output_dict_ego})
    return return_dict

def inference_no_fusion_w_uncertainty(batch_data, model, dataset):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict_ego = OrderedDict()

    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor, uncertainty_tensor = \
        dataset.post_process_no_fusion(batch_data, # only for late fusion dataset
                             output_dict_ego, return_uncertainty=True)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor, \
                    "uncertainty_tensor" : uncertainty_tensor}

    return return_dict


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    output_dict['ego'] = model(cav_content,'test')

    # flops, params = profile(model, inputs=(cav_content,'test'))
    # print('FLOPs: %.2f G, Params: %.2f M' % (flops / 1e9, params / 1e6))

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)
    if 'output' in output_dict['ego']:
        if 'vehicle_type' in output_dict['ego']['output']:
            return_dict = {"pred_box_tensor" : pred_box_tensor, \
                            "pred_score" : pred_score, \
                            "gt_box_tensor" : gt_box_tensor, \
                            "vehicle_type" : output_dict['ego']['output']['vehicle_type']
                           }
    else:
        return_dict = {"pred_box_tensor" : pred_box_tensor,
                        "pred_score" : pred_score,
                        "gt_box_tensor" : gt_box_tensor,
                       }
    
    if hasattr(model, 'time_dict'):
        return_dict.update({'time_dict' : model.time_dict})

    if "depth_items" in output_dict['ego']:
        return_dict.update({"depth_items" : output_dict['ego']['depth_items']})
    return_dict.update({'output_dict': output_dict})
    return return_dict


def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return_dict = inference_early_fusion(batch_data, model, dataset)
    return return_dict


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)


def depth_metric(depth_items, grid_conf):
    # depth logdit: [N, D, H, W]
    # depth gt indices: [N, H, W]
    depth_logit, depth_gt_indices = depth_items
    depth_pred_indices = torch.argmax(depth_logit, 1)
    depth_pred = indices_to_depth(depth_pred_indices, *grid_conf['ddiscr'], mode=grid_conf['mode']).flatten()
    depth_gt = indices_to_depth(depth_gt_indices, *grid_conf['ddiscr'], mode=grid_conf['mode']).flatten()
    rmse = mean_squared_error(depth_gt.cpu(), depth_pred.cpu(), squared=False)
    return rmse


# def fix_cavs_box(pred_box_tensor, gt_box_tensor, pred_score, batch_data):
#     """
#     Fix the missing pred_box and gt_box for ego and cav(s).
#     Args:
#         pred_box_tensor : tensor
#             shape (N1, 8, 3), may or may not include ego agent prediction, but it should include
#         gt_box_tensor : tensor
#             shape (N2, 8, 3), not include ego agent in camera cases, but it should include
#         batch_data : dict
#             batch_data['lidar_pose'] and batch_data['record_len'] for putting ego's pred box and gt box
#     Returns:
#         pred_box_tensor : tensor
#             shape (N1+?, 8, 3)
#         gt_box_tensor : tensor
#             shape (N2+1, 8, 3)
#     """
#     if pred_box_tensor is None or gt_box_tensor is None:
#         return pred_box_tensor, gt_box_tensor, pred_score, 0
#     # prepare cav's boxes
#     lidar_pose = np.zeros((1,6)) \
#                 if 'lidar_pose' not in batch_data['ego'] \
#                 else batch_data['ego']['lidar_pose'].cpu().numpy()
    
#     # assert "cav_id_list" in batch_data['ego']
#     if "cav_id_list" in batch_data['ego']:
#         cav_id_list = batch_data['ego']['cav_id_list']
#     else:
#         cav_id_list = None

#     N = 1 \
#         if 'record_len' not in batch_data['ego']\
#         else batch_data['ego']['record_len']
        
#     extent = [2.45, 1.06, 0.75]
#     ego_box = create_bbx(extent).reshape(1, 8, 3) # [8, 3]
#     ego_box[..., 2] -= 1.2 # hard coded now

#     box_list = [ego_box]
#     relative_t = get_relative_transformation(lidar_pose) # [N, 4, 4], cav_to_ego, T_ego_cav
#     for i in range(1, N):
#         box_list.append(project_box3d(ego_box, relative_t[i]))
#     cav_box_tensor = torch.tensor(np.concatenate(box_list, axis=0), device=pred_box_tensor.device)
    
#     pred_box_tensor_ = torch.cat((cav_box_tensor, pred_box_tensor), dim=0)
#     gt_box_tensor_ = torch.cat((cav_box_tensor, gt_box_tensor), dim=0)

#     pred_score_ = torch.cat((torch.ones(N, device=pred_score.device), pred_score))

#     gt_score_ = torch.ones(gt_box_tensor_.shape[0], device=pred_box_tensor.device)
#     gt_score_[N:] = 0.5

#     keep_index = nms_rotated(pred_box_tensor_,
#                             pred_score_,
#                             0.01)
#     pred_box_tensor = pred_box_tensor_[keep_index]
#     pred_score = pred_score_[keep_index]

#     keep_index = nms_rotated(gt_box_tensor_,
#                             gt_score_,
#                             0.01)
#     gt_box_tensor = gt_box_tensor_[keep_index]

#     return pred_box_tensor, gt_box_tensor, pred_score, N, cav_id_list

def fix_cavs_box(pred_box_tensor, gt_box_tensor, pred_score, batch_data, vehicle_type=None):
    """
    Fix the missing pred_box and gt_box for ego and cav(s).
    Args:
        pred_box_tensor : tensor
            shape (N1, 8, 3), may or may not include ego agent prediction, but it should include
        gt_box_tensor : tensor
            shape (N2, 8, 3), not include ego agent in camera cases, but it should include
        batch_data : dict
            batch_data['lidar_pose'] and batch_data['record_len'] for putting ego's pred box and gt box
    Returns:
        pred_box_tensor : tensor
            shape (N1+?, 8, 3)
        gt_box_tensor : tensor
            shape (N2+1, 8, 3)
    """
    if pred_box_tensor is None or gt_box_tensor is None:
        return pred_box_tensor, gt_box_tensor, pred_score, 0
    # prepare cav's boxes

    # if key only contains "ego", like intermediate fusion
    if 'record_len' in batch_data['ego']:
        lidar_pose =  batch_data['ego']['lidar_pose'].cpu().numpy()
        N = batch_data['ego']['record_len']
        relative_t = get_relative_transformation(lidar_pose) # [N, 4, 4], cav_to_ego, T_ego_cav
    # elif key contains "ego", "641", "649" ..., like late fusion
    else:
        # relative_t = []
        # for cavid, cav_data in batch_data.items():
        #     relative_t.append(cav_data['transformation_matrix'])
        # N = len(relative_t)
        # relative_t = torch.stack(relative_t, dim=0).cpu().numpy()
        N = 1
        
    extent = [2.45, 1.06, 0.75]
    ego_box = create_bbx(extent).reshape(1, 8, 3) # [8, 3]
    ego_box[..., 2] -= 1.2 # hard coded now

    box_list = [ego_box]
    
    for i in range(1, N):
        box_list.append(project_box3d(ego_box, relative_t[i]))
    cav_box_tensor = torch.tensor(np.concatenate(box_list, axis=0), device=pred_box_tensor.device)
    
    pred_box_tensor_ = torch.cat((cav_box_tensor, pred_box_tensor), dim=0)
    gt_box_tensor_ = torch.cat((cav_box_tensor, gt_box_tensor), dim=0)

    pred_score_ = torch.cat((torch.ones(N, device=pred_score.device), pred_score))

    gt_score_ = torch.ones(gt_box_tensor_.shape[0], device=pred_box_tensor.device)
    gt_score_[N:] = 0.5

    keep_index = nms_rotated(pred_box_tensor_,
                            pred_score_,
                            0.01)
    pred_box_tensor = pred_box_tensor_[keep_index]
    pred_score = pred_score_[keep_index]

    # 首先，确定新增加的cav_box_tensor中框的数量

    if vehicle_type is not None:
        num_new_boxes = cav_box_tensor.shape[0]
        new_vehicle_types = ["ego"] * num_new_boxes
        vehicle_type = new_vehicle_types + vehicle_type
        vehicle_type = [vehicle_type[idx] for idx in keep_index.tolist()]

    keep_index = nms_rotated(gt_box_tensor_,
                            gt_score_,
                            0.01)
    gt_box_tensor = gt_box_tensor_[keep_index]

    if vehicle_type is not None:
        return pred_box_tensor, gt_box_tensor, pred_score, N, vehicle_type

    return pred_box_tensor, gt_box_tensor, pred_score, N

