# -*- coding: utf-8 -*-
# Author: Kang Yang
"""
Dair-v2x-C Dataset class for intermediate fusion
"""

import random
import re
import cv2
import math
from collections import OrderedDict
import os
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import json
import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np
from opencood.utils import box_utils

from opencood.data_utils.datasets import intermediate_fusion_dataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.camera_utils import sample_augmentation, img_transform, normalize_img
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.pose_utils import add_noise_data_dict
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

def get_file_numbers(directory):

    numbers_list = []

    for filename in os.listdir(directory):

        match = re.search(r'\d+', filename)
        if match:
            numbers_list.append(match.group())
    return numbers_list

class CameraIntermediateFusionDatasetDAIR(intermediate_fusion_dataset.IntermediateFusionDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]
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
        cop_label = get_file_numbers(os.path.join(self.root_dir, 'cooperative/label_world'))
        self.co_data = OrderedDict()
        veh_frame_idxs = []
        for frame_info in co_datainfo:
            # veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
            veh_frame_idx = frame_info["vehicle_frame"]
            veh_frame_idxs.append(veh_frame_idx)
            if veh_frame_idx in self.split_info and veh_frame_idx in cop_label:
                self.co_data[veh_frame_idx] = frame_info
        split_info_filte = []
        for split_data in self.split_info:
            if split_data in veh_frame_idxs and split_data in cop_label:
                split_info_filte.append(split_data)
        self.split_info = split_info_filte

        self.use_gt_depth = True \
            if ('camera_params' in params and params['camera_params']['use_depth_gt']) \
            else False
        if self.use_gt_depth:
            self.depth_max = params["fusion"]["args"]["depth_max"]

        self.image_mean = params['fusion']['args']['normalize_img']['mean']
        self.image_std = params['fusion']['args']['normalize_img']['std']
        self.normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=self.image_mean,
                                 std=self.image_std)
))

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
        
        _id = self.split_info[idx]
        frame_info = self.co_data[_id]

        system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()
        data[0] = OrderedDict() # veh-side
        veh_frame_id = frame_info['vehicle_frame']
        data[0]['ego'] = True
        data[1] = OrderedDict() # inf-side
        data[1]['ego'] = False

        data[0]['params'] = OrderedDict()
        data[0]['params']['vehicles'] = load_json(os.path.join(self.root_dir,"cooperative/label_world/" + str(veh_frame_id) + '.json')) # "cooperative_label_path": "cooperative/label_world/011362.json"

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


        data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,'vehicle-side/velodyne/' + str(veh_frame_id) + '.pcd'))
        if self.clip_pc:
            data[0]['lidar_np'] = data[0]['lidar_np'][data[0]['lidar_np'][:,0]>0]

        from PIL import Image
        data[0]['camera_data'] = Image.open(os.path.join(self.root_dir,'vehicle-side/image/' + str(veh_frame_id) + '.jpg'))

        data[1]['params'] = OrderedDict()
        # inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
        inf_frame_id = frame_info['infrastructure_frame']
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

        data[1]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,'infrastructure-side/velodyne/' + str(inf_frame_id) + '.pcd'))

        data[1]['camera_data'] = Image.open(os.path.join(self.root_dir,'infrastructure-side/image/' + str(inf_frame_id) + '.jpg'))
        return data

    def __len__(self):
        return len(self.split_info)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)

        base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                break

        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        processed_features = []
        object_stack = []
        object_id_stack = []
        too_far = []
        lidar_pose_list = []
        lidar_pose_clean_list = []
        projected_lidar_clean_list = []
        cav_id_list = []

        if self.visualize:
            projected_lidar_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)

            # if distance is too far, we will just skip this agent
            if distance > self.params['comm_range']:
                too_far.append(cav_id)
                continue

            lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
            lidar_pose_list.append(selected_cav_base['params']['lidar_pose'])  # 6dof pose
            cav_id_list.append(cav_id)

        ########## Added by Yifan Lu 2022.8.14 ##############
        # box align to correct pose.
        if self.box_align and str(idx) in self.stage1_result.keys():
            stage1_content = self.stage1_result[str(idx)]
            if stage1_content is not None:
                cav_id_list_stage1 = stage1_content['cav_id_list']

                pred_corners_list = stage1_content['pred_corner3d_np_list']
                pred_corners_list = [np.array(corners, dtype=np.float64) for corners in pred_corners_list]
                uncertainty_list = stage1_content['uncertainty_np_list']
                uncertainty_list = [np.array(uncertainty, dtype=np.float64) for uncertainty in uncertainty_list]
                stage1_lidar_pose_list = [base_data_dict[cav_id]['params']['lidar_pose'] for cav_id in
                                          cav_id_list_stage1]
                stage1_lidar_pose = np.array(stage1_lidar_pose_list)

                refined_pose = box_alignment_relative_sample_np(pred_corners_list,
                                                                stage1_lidar_pose,
                                                                uncertainty_list=uncertainty_list,
                                                                **self.box_align_args)
                stage1_lidar_pose[:, [0, 1, 4]] = refined_pose
                stage1_lidar_pose_refined_list = stage1_lidar_pose.tolist()  # updated lidar_pose_list
                for cav_id, lidar_pose_refined in zip(cav_id_list_stage1, stage1_lidar_pose_refined_list):
                    if cav_id not in cav_id_list:
                        continue
                    idx_in_list = cav_id_list.index(cav_id)
                    lidar_pose_list[idx_in_list] = lidar_pose_refined
                    base_data_dict[cav_id]['params']['lidar_pose'] = lidar_pose_refined
        agents_image_inputs = []
        for cav_id in cav_id_list:
            selected_cav_base = base_data_dict[cav_id]

            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose,
                ego_lidar_pose_clean)

            object_stack.append(selected_cav_processed['object_bbx_center'])

            object_id_stack += selected_cav_processed['object_ids']

            agents_image_inputs.append(
                selected_cav_processed['image_inputs'])

            if self.kd_flag:
                projected_lidar_clean_list.append(
                    selected_cav_processed['projected_lidar_clean'])

            if self.visualize:
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])

        ########## Added by Yifan Lu 2022.4.5 ################
        # filter those out of communicate range
        # then we can calculate get_pairwise_transformation
        for cav_id in too_far:
            base_data_dict.pop(cav_id)

        pairwise_t_matrix = \
            self.get_pairwise_transformation(base_data_dict,
                                             self.max_cav)

        lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
        lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]
        ######################################################

        # only need object list
        object_target = object_stack
        object_target_temp = []

        for i in range(len(object_target)):
            object_target_temp.append(torch.from_numpy(object_target[i]))


        ############ for disconet ###########
        if self.kd_flag:
            stack_lidar_np = np.vstack(projected_lidar_clean_list)
            stack_lidar_np = mask_points_by_range(stack_lidar_np,
                                                  self.params['preprocess'][
                                                      'cav_lidar_range'])
            stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # merge preprocessed features from different cavs into the same dict

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        label_dict.update(
            {
                'object_target': object_target_temp,
            }
        )
        cav_num = len(agents_image_inputs)
        merged_image_inputs_dict = self.merge_features_to_dict(agents_image_inputs, merge='stack')

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'image_inputs': merged_image_inputs_dict,
             'label_dict': label_dict,
             'cav_num': cav_num,
             'pairwise_t_matrix': pairwise_t_matrix,
             'lidar_poses_clean': lidar_poses_clean,
             'lidar_poses': lidar_poses})

        if self.kd_flag:
            processed_data_dict['ego'].update({'teacher_processed_lidar':
                                                   stack_feature_processed})

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                np.vstack(
                    projected_lidar_stack)})

        processed_data_dict['ego'].update({'sample_idx': idx,
                                           'cav_id_list': cav_id_list})

        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose, ego_pose_clean):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list, length 6
            The ego vehicle lidar pose under world coordinate.
        ego_pose_clean : list, length 6
            only used for gt box generation

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = \
            x1_to_x2(selected_cav_base['params']['lidar_pose'],
                     ego_pose)  # T_ego_cav
        transformation_matrix_clean = \
            x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
                     ego_pose_clean)
        # retrieve objects under ego coordinates
        # this is used to generate accurate GT bounding box.
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
                                                                                     ego_pose_clean)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        # x,y,z in ego space
        projected_lidar = \
            box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                     transformation_matrix)
        if self.kd_flag:
            import copy
            lidar_np_clean = copy.deepcopy(lidar_np)

        if self.proj_first:
            lidar_np[:, :3] = projected_lidar

        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])

        params = selected_cav_base["params"]

        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        camera_to_lidars = []

        img = selected_cav_base['camera_data']
        instrinc = np.array(params['camera_instrincs']).astype(
            np.float32
        )
        intrin = torch.from_numpy(instrinc)
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)

        camera_coords = params['camera_coords']

        camera_to_lidar = x1_to_x2(
            camera_coords, params["lidar_pose_clean"]
        ).astype(np.float32)  # T_LiDAR_camera
        camera_to_lidar_temp = torch.from_numpy(camera_to_lidar)

        rot = torch.from_numpy(
            camera_to_lidar[:3, :3]
        )  # R_wc, we consider world-coord is the lidar-coord
        tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc


        img_src = [img]

        resize, resize_dims, crop, flip, rotate = sample_augmentation(
            self.data_aug_conf, self.train
        )

        img_src, post_rot2, post_tran2 = img_transform(
            img_src,
            post_rot,
            post_tran,
            resize=resize,
            resize_dims=resize_dims,
            crop=crop,
            flip=flip,
            rotate=rotate,
        )

        # for convenience, make augmentation matrices 3x3
        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2

        img_src[0] = self.normalize_img(img_src[0])

        if self.use_gt_depth:
            C, H, W = img_src[0].shape
            point_depth_position, depth = self.get_lidar_depth(camera_to_lidar, instrinc, lidar_np, post_rot,
                                                               post_tran, H, W)

            # U = point_depth_position[:,0].astype(np.int32)
            # V = point_depth_position[:,1].astype(np.int32)
            # depths = depth.reshape(-1)
            # sort_idx = np.argsort(depths)[::-1]
            # V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            #
            # for j,downsample in enumerate(self.downsample):
            #     if len(gt_depth) < j + 1:
            #         gt_depth.append([])
            #     h, w = (int(H / downsample), int(W / downsample))
            #     u = np.floor(U / downsample).astype(np.int32)
            #     v = np.floor(V / downsample).astype(np.int32)
            #     depth_map = np.ones([h, w], dtype=np.float32) * -1
            #     depth_map[v, u] = depths
            #     gt_depth[j].append(depth_map)

            depth_map = torch.zeros((H,W))
            for idx in range(point_depth_position.shape[0]):
                x, y = point_depth_position[idx]
                x = int(x.item())
                y = int(y.item())
                depth_value = depth[idx].item()
                depth_map[y, x] = depth_value

            # data = depth_map
            # sns.heatmap(data, cmap="viridis", cbar=True, vmin=0, vmax=50, xticklabels=False, yticklabels=False)
            # plt.show()

            depth_augmented, procd = fill_in_multiscale(depth_map)  # return tesnor

            # data = depth_augmented
            # sns.heatmap(data, cmap="viridis", cbar=True, vmin=0, vmax=50, xticklabels=False, yticklabels=False)
            # plt.show()

            if len(depth_augmented.shape) == 2:
                depth_map = depth_map.unsqueeze(0)

            img_src.append(depth_map)

            # self.visualize_projected_points(img_src[0].permute(1,2,0),point_depth_position)
            # point_depth_augmented, procd = fill_in_multiscale(point_depth)
            # from opencood.utils.common_utils import check_numpy_to_torch
            # point_depth_augmented,*rest = check_numpy_to_torch(point_depth_augmented)
            # self.visualize_projected_points(img_src[0].permute(1, 2, 0), point_depth_augmented)


        imgs.append(torch.cat(img_src, dim=0))
        intrins.append(intrin)
        rots.append(rot)
        trans.append(tran)
        post_rots.append(post_rot)
        post_trans.append(post_tran)
        camera_to_lidars.append(camera_to_lidar_temp)


        selected_cav_processed.update(
            {
                "image_inputs":
                    {
                        "camera_to_lidar": torch.stack(camera_to_lidars),
                        "imgs": torch.stack(imgs),
                        "intrins": torch.stack(intrins),
                        "rots": torch.stack(rots),
                        "trans": torch.stack(trans),
                        "post_rots": torch.stack(post_rots),
                        "post_trans": torch.stack(post_trans)
                    }
            }
        )

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': projected_lidar,
             'transformation_matrix': transformation_matrix,
             'transformation_matrix_clean': transformation_matrix_clean,
             }
        )

        if self.kd_flag:
            projected_lidar_clean = \
                box_utils.project_points_by_matrix_torch(lidar_np_clean[:, :3],
                                                         transformation_matrix_clean)
            lidar_np_clean[:, :3] = projected_lidar_clean
            lidar_np_clean = mask_points_by_range(lidar_np_clean,
                                                  self.params['preprocess'][
                                                      'cav_lidar_range'])
            selected_cav_processed.update(
                {"projected_lidar_clean": lidar_np_clean}
            )

        return selected_cav_processed


    def get_lidar_depth(self, extrins, intrins, lidar_np,post_rot,post_tran,H,W):
        '''
        rot:3,3
        tran:3
        intrins:3,3
        img: PngImageFile
        '''
        lidar_np = lidar_np[:,:3]
        n_points, _ = lidar_np.shape
        # trans from lidar to camera coord
        xyz_hom = np.concatenate(
            [lidar_np, np.ones((lidar_np.shape[0], 1), dtype=np.float32)], axis=1)

        ext_matrix = np.linalg.inv(extrins)[:3, :4]
        img_pts = (intrins @ ext_matrix @ xyz_hom.T).T

        depth = img_pts[:, 2]
        # Todo:不能除0，跟key_point一样 处理
        # uv = img_pts[:, :2] / depth[:, None]
        uv = img_pts[:, :2] / np.clip(depth[:, None],1e-5,None)

        # for postprocess in image
        from opencood.utils.common_utils import check_numpy_to_torch
        points_2d, *rest = check_numpy_to_torch(uv) # N,2
        ones = torch.ones_like(points_2d[..., :1])
        points_2d = torch.cat([points_2d,ones],dim=-1)
        points_2d = post_rot.matmul(points_2d.unsqueeze(-1)).squeeze()
        points_2d = points_2d + post_tran
        points_2d = points_2d[...,:2]


        uv = points_2d
        uv = uv.round().long()
        if (torch.is_tensor(uv)):
            uv = uv.numpy()

        valid_mask1 = ((uv[:, 0] >= 0) & (uv[:, 0] < W) &
                       (uv[:, 1] >= 0) & (uv[:, 1] < H)).reshape(-1,1)

        valid_mask2 = ((depth > 0.5) & (depth < self.depth_max)).reshape(-1,1)

        gt_depth_mask = valid_mask1.any(axis=1) & valid_mask2.all(axis=1)  # [N, ]

        points_2d = uv[gt_depth_mask]

        coloring = depth[:, None]
        coloring = coloring[gt_depth_mask]


        # to tensor
        # points_2d, *rest = check_numpy_to_torch(points_2d)
        # coloring,*rest1 = check_numpy_to_torch(coloring)


        return points_2d,coloring


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

    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        # used to record different scenario
        record_len = []
        label_dict_list = []
        lidar_pose_list = []
        lidar_pose_clean_list = []
        image_inputs_list = []
        # pairwise transformation matrix
        pairwise_t_matrix_list = []
        object_target = []
        if self.kd_flag:
            teacher_processed_lidar_list = []
        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])
            lidar_pose_list.append(ego_dict['lidar_poses'])  # ego_dict['lidar_pose'] is np.ndarray [N,6]
            lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])

            object_target.append(ego_dict['label_dict']['object_target'])

            # processed_lidar_list.append(ego_dict['processed_lidar']) # different cav_num, ego_dict['processed_lidar'] is list.
            image_inputs_list.append(ego_dict['image_inputs'])  # different cav_num, ego_dict['image_inputs'] is dict.
            record_len.append(ego_dict['cav_num'])

            label_dict_list.append(ego_dict['label_dict'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

            if self.kd_flag:
                teacher_processed_lidar_list.append(ego_dict['teacher_processed_lidar'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        # [[N1, 6], [N2, 6]...] -> [[N1+N2+...], 6]
        lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
        lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)

        # (B, max_cav)
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))
        merged_image_inputs_dict = self.merge_features_to_dict(image_inputs_list, merge='cat')
        # add pairwise_t_matrix to label dict
        label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
        label_torch_dict['record_len'] = record_len

        flattened_list = [tensor for sublist in object_target for tensor in sublist]
        label_torch_dict['object_target'] = flattened_list

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'image_inputs': merged_image_inputs_dict,
                                   'record_len': record_len,
                                   'label_dict': label_torch_dict,
                                   'object_ids': object_ids[0],
                                   'pairwise_t_matrix': pairwise_t_matrix,
                                   'lidar_pose_clean': lidar_pose_clean,
                                   'lidar_pose': lidar_pose})

        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        if self.kd_flag:
            teacher_processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(teacher_processed_lidar_list)
            output_dict['ego'].update({'teacher_processed_lidar': teacher_processed_lidar_torch_dict})

        return output_dict







# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

def fill_in_multiscale(depth_map, max_depth=50,
                       dilation_kernel_far=CROSS_KERNEL_3,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_7,
                       extrapolate=False,
                       blur_type='bilateral',
                       show_process=False):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med: dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict

    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    # Convert to float32
    if (torch.is_tensor(depth_map)):
        depth_map = depth_map.numpy()
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > 0.1) & (depths_in <= 15.0)
    valid_pixels_med = (depths_in > 15.0) & (depths_in <= 30.0)
    valid_pixels_far = (depths_in > 30.0)

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.1)
    s1_inverted_depths[valid_pixels] = \
        max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far),
        dilation_kernel_far)
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med),
        dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)

    # Find valid pixels for each binned dilation
    valid_pixels_near = (dilated_near > 0.1)
    valid_pixels_med = (dilated_med > 0.1)
    valid_pixels_far = (dilated_far > 0.1)

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.1)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.1)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = (s4_blurred_depths > 0.1)
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=bool)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.1, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
                               pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.1) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.1) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == 'gaussian':
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.1) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == 'bilateral':
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.1)
    s8_inverted_depths[valid_pixels] = \
        max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None

    depths_out = torch.from_numpy(depths_out)
    return depths_out, process_dict