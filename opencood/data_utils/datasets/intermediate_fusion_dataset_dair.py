# -*- coding: utf-8 -*-
# Author: Quanhao Li

"""
Dataset class for early fusion
"""
import random
import math
from collections import OrderedDict

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.utils import box_utils

from opencood.data_utils.datasets import intermediate_fusion_dataset
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2, json_file_transformation_matrix_load, \
    generate_K_from_json_file, camera_to_world_coords
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.transformation_utils import tfm_to_pose
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import x_to_world

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

class IntermediateFusionDatasetDAIR(intermediate_fusion_dataset.IntermediateFusionDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)
        self.max_cav = 2
        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        assert 'proj_first' in params['fusion']['args']
        if params['fusion']['args']['proj_first']:
            self.proj_first = True
        else:
            self.proj_first = False

        if "kd_flag" in params.keys():
            self.kd_flag = params['kd_flag']
        else:
            self.kd_flag = False

        if "box_align" in params.keys():
            self.box_align = True
            self.stage1_result_path = params['box_align']['train_result'] if train else params['box_align']['val_result']
            self.stage1_result = load_json(self.stage1_result_path)
            self.box_align_args = params['box_align']['args']
        
        else:
            self.box_align = False
            
        assert 'clip_pc' in params['fusion']['args']
        if params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False
        
        if 'select_kp' in params:
            self.select_keypoint = params['select_kp']
        else:
            self.select_keypoint = None

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

        if self.train:
            split_dir = params['root_dir']
        else:
            split_dir = params['validate_dir']

        self.root_dir = params['data_dir']
        self.split_info = load_json(split_dir)
        co_datainfo = load_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        self.co_data = OrderedDict()
        for frame_info in co_datainfo:
            veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
            self.co_data[veh_frame_id] = frame_info

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.
        Parameters
        ----------
        idx : int
            Index given by dataloader.
        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        
        veh_frame_id = self.split_info[idx]
        frame_info = self.co_data[veh_frame_id]
        system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()
        data[0] = OrderedDict() # veh-side
        data[0]['ego'] = True
        data[1] = OrderedDict() # inf-side
        data[1]['ego'] = False

        data[0]['params'] = OrderedDict()
        data[0]['params']['vehicles'] = load_json(os.path.join(self.root_dir,frame_info['cooperative_label_path']))

        lidar_to_novatel_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        novatel_to_world_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))
        lidar_to_camera_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_camera/' + str(veh_frame_id)+'.json'))

        # get camera's instrincs and distort
        camera_instrinc_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/camera_intrinsic/' + str(veh_frame_id)+'.json'))
        K,D = generate_K_from_json_file(camera_instrinc_json_file)
        data[0]['params']['camera_instrincs'] = K
        data[0]['params']['cam_D'] = D

        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file,novatel_to_world_json_file)

        data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix) # lidar to world
        lidar_to_camera_matrix = json_file_transformation_matrix_load(lidar_to_camera_json_file)  # lidar to camera
        camera_to_lidar_matrix = np.linalg.inv(lidar_to_camera_matrix)
        v_camera_coords = camera_to_world_coords(camera_to_lidar_matrix,transformation_matrix) # camera in world coords
        data[0]['params']['camera_coords'] = tfm_to_pose(v_camera_coords)


        data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["vehicle_pointcloud_path"]))
        if self.clip_pc:
            data[0]['lidar_np'] = data[0]['lidar_np'][data[0]['lidar_np'][:,0]>0]

        from PIL import Image
        data[0]['camera_data'] = Image.open(os.path.join(self.root_dir,frame_info['vehicle_image_path']))


        data[1]['params'] = OrderedDict()
        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")

        data[1]['params']['vehicles'] = [] # we only load cooperative label in vehicle side

        virtuallidar_to_world_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))
        virtuallidar_to_camera_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_camera/' + str(inf_frame_id)+'.json'))

        camera_instrinc_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side/calib/camera_intrinsic/' + str(inf_frame_id)+'.json'))
        K,D = generate_K_from_json_file(camera_instrinc_json_file)
        data[1]['params']['camera_instrincs'] = K
        data[1]['params']['cam_D'] = D

        transformation_matrix1 = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world_json_file,system_error_offset)
        data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix1)

        lidar_to_camera_matrix = json_file_transformation_matrix_load(virtuallidar_to_camera_json_file) # lidar to camera
        camera_to_lidar_matrix = np.linalg.inv(lidar_to_camera_matrix)
        i_camera_coords = camera_to_world_coords(camera_to_lidar_matrix,transformation_matrix1) # camera in world coords
        data[1]['params']['camera_coords'] = tfm_to_pose(i_camera_coords)

        data[1]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["infrastructure_pointcloud_path"]))

        data[1]['camera_data'] = Image.open(os.path.join(self.root_dir,frame_info['infrastructure_image_path']))
        return data

    def __len__(self):
        return len(self.split_info)

    ### rewrite generate_object_center ###
    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Notice: it is a wrap of postprocessor function

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """

        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)
