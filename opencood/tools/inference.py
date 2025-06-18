# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import random
import sys
import time
from typing import OrderedDict

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(project_root)

import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
torch.multiprocessing.set_sharing_strategy('file_system')



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def test_parser():
    #
    set_seed(0)
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str,
                        help='Continued training path', default="/mnt/sdb/public/data/yangk/result/cosparse_logs/opv2v_sparse4D_resnet50_multiloss_w_dir_depth_V6_2024_04_21_13_25_49")
    # parser.add_argument('--model_dir', type=str,
    #                     help='Continued training path', default="/mnt/sdb/public/data/yangk/result/cosparse_logs/layer_for_fps/opv2v_sparse4D_resnet50_layer1/")
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--new_dir', default=False, help='New dir for continued training path')
    parser.add_argument('--save_vis_interval', type=int, default=20,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="140.8,40",
                        help="detection range is [-140.8,+140.8m, -40m, +40m]")
    parser.add_argument('--cavnum', type=int, default=5, 
                        help="number of agent in collaboration")
    parser.add_argument('--fix_cavs_box', dest='fcb', action='store_true',
                        help="fix(add) bounding box for cav(s)",default=True)
    parser.add_argument('--depth_metric', '-d', action='store_true',
                        help="evaluate depth estimation performance")
    parser.add_argument('--note', default="[newtest_ego_all_gt]", type=str, help="any other thing?")
    # parser.set_defaults(fcb=True)
    parser.add_argument('--cuda_device', type=int, default=0)
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    torch.cuda.set_device(opt.cuda_device)
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)
    
    hypes['validate_dir'] = hypes['test_dir']  # test replace val

    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if "OPV2V" in hypes['test_dir'] else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    ##############################
    x_max = hypes['preprocess']['cav_lidar_range'][3]
    y_max = hypes['preprocess']['cav_lidar_range'][4]
    ##############################

    print('Creating Model')
    model = train_utils.create_model(hypes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)
    from collections import OrderedDict
    noise_setting = OrderedDict()
    noise_setting['add_noise'] = False
    
    # build dataset for each noise setting
    print('Dataset Building')
    print(f"No Noise Added.")
    hypes.update({"noise_setting": noise_setting})
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            worker_init_fn=seed_worker)
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    
    if opt.depth_metric:
        depth_stat = []
    
    noise_level = "no_noise_"+opt.fusion_method+ f"[_{x_max}m_{y_max}m]" + f"{opt.cavnum}agent" +opt.note

    # import time
    sasa_time = 0
    total_time_afb = 0
    total_batch = len(data_loader) # 2170
    agent_fusion_time = 0
    extract_image_time = 0 
    for i, batch_data in enumerate(data_loader):
        # if i != 240:
        #     continue
        
        print(f"{noise_level}_{i}")
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            start_time = time.time()
            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')

            # if 'time_dict' in infer_result:
            #     agent_fusion_time = agent_fusion_time + infer_result['time_dict']['total_time_fusion']
            #     sasa_time = sasa_time + infer_result['time_dict']['total_time_sasa']
            #     total_time_afb = total_time_afb + infer_result['time_dict']['total_time_afb']
            #     extract_image_time = extract_image_time + infer_result['time_dict']['extract_image_time']

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            # for fusion visualize
            if 'vehicle_type' in infer_result:
                vehicle_type = infer_result['vehicle_type']
            else:
                vehicle_type = None

            if "uncertainty_tensor" in infer_result:
                uncertainty_tensor = infer_result['uncertainty_tensor']
            else:
                uncertainty_tensor = None

            if "depth_items" in infer_result and opt.depth_metric:
                depth_items = infer_result['depth_items']
                depth_stat.append(inference_utils.depth_metric(depth_items, hypes['fusion']['args']['grid_conf']))

            
            cavnum = 0
            # if opt.fix_cavs_box:
            if opt.fcb:
                if vehicle_type is not None:
                    pred_box_tensor, gt_box_tensor, pred_score, cavnum, vehicle_type = inference_utils.fix_cavs_box(pred_box_tensor, gt_box_tensor, pred_score, batch_data, vehicle_type)
                else:
                    pred_box_tensor, gt_box_tensor, pred_score, cavnum = inference_utils.fix_cavs_box(
                        pred_box_tensor, gt_box_tensor, pred_score, batch_data, vehicle_type)
            
            if pred_box_tensor is not None:
                pred_box_tensor = pred_box_tensor[:,:4,:2]
            if gt_box_tensor is not None:
                gt_box_tensor = gt_box_tensor[:,:4,:2]
            
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            # if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None) or (i == 87 or i == 560 or i == 563 or i==564 or i==568 or i==572):
            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{noise_level}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(pred_box_tensor,
                #                     gt_box_tensor,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand,
                #                     uncertainty=uncertainty_tensor)
                
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)

                if 'vehicle_type' in infer_result:
                    mask_ego = torch.tensor([v_type == "ego" for v_type in vehicle_type])
                    mask_neighbor = ~mask_ego
                    pred_boxes_ego_tensor = pred_box_tensor[mask_ego]
                    pred_boxes_neighbor_tensor = pred_box_tensor[mask_neighbor]
                    pred_box_tensor = (pred_boxes_ego_tensor,pred_boxes_neighbor_tensor)
                simple_vis.visualize(pred_box_tensor,
                                    gt_box_tensor,
                                    batch_data['ego'][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand,
                                    uncertainty=uncertainty_tensor,
                                    cavnum=cavnum)

        torch.cuda.empty_cache()

    _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, noise_level)
    if opt.depth_metric:
        depth_rmse = np.mean(depth_stat).tolist()
        print("depth_rmse: ",depth_rmse)
        yaml_utils.save_yaml({'depth_rmse': depth_rmse}, os.path.join(opt.model_dir, f'eval_depth_{noise_level}.yaml'))


if __name__ == '__main__':
    main()
