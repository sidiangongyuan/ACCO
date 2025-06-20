"""
3D Anchor Generator for Camera.
Only reserve visible GT boxes 
"""
import math
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn.functional import sigmoid
import torch.nn.functional as F

from opencood.data_utils.post_processor.base_postprocessor \
    import BasePostprocessor
from opencood.utils import box_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.visualization import vis_utils
from opencood.utils.common_utils import limit_period


class CameraPostprocessor(BasePostprocessor):
    def __init__(self, anchor_params, train):
        super(CameraPostprocessor, self).__init__(anchor_params, train)
        self.anchor_num = self.params['anchor_args']['num']  # 2

    def generate_anchor_box(self):
        W = self.params['anchor_args']['W']
        H = self.params['anchor_args']['H']

        l = self.params['anchor_args']['l']
        w = self.params['anchor_args']['w']
        h = self.params['anchor_args']['h']
        r = self.params['anchor_args']['r']

        assert self.anchor_num == len(r)
        r = [math.radians(ele) for ele in r]

        vh = self.params['anchor_args']['vh'] # voxel_size
        vw = self.params['anchor_args']['vw']

        xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                  self.params['anchor_args']['cav_lidar_range'][3]]
        yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                  self.params['anchor_args']['cav_lidar_range'][4]]

        if 'feature_stride' in self.params['anchor_args']:
            feature_stride = self.params['anchor_args']['feature_stride']
        else:
            feature_stride = 2


        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride)
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride)


        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], self.anchor_num) # center
        cy = np.tile(cy[..., np.newaxis], self.anchor_num)
        cz = np.ones_like(cx) * -1.0

        w = np.ones_like(cx) * w
        l = np.ones_like(cx) * l
        h = np.ones_like(cx) * h

        r_ = np.ones_like(cx)
        for i in range(self.anchor_num):
            r_[..., i] = r[i]

        if self.params['order'] == 'hwl': # pointpillar
            anchors = np.stack([cx, cy, cz, h, w, l, r_], axis=-1) # (50, 176, 2, 7)

        elif self.params['order'] == 'lhw':
            anchors = np.stack([cx, cy, cz, l, h, w, r_], axis=-1)
        else:
            sys.exit('Unknown bbx order.')

        return anchors


    def generate_label(self, **kwargs):
        """
        Generate targets for training.

        Parameters
        ----------
        argv : list
            gt_box_center:(max_num, 7), anchor:(H, W, anchor_num, 7)

        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """
        assert self.params['order'] == 'hwl', 'Currently Voxel only support' \
                                              'hwl bbx order.'
        # (max_num, 7)
        gt_box_center = kwargs['gt_box_center']
        # (H, W, anchor_num, 7)
        anchors = kwargs['anchors']
        # (max_num)
        masks = kwargs['mask']

        # (H, W)
        feature_map_shape = anchors.shape[:2]

        # (H*W*anchor_num, 7)
        anchors = anchors.reshape(-1, 7)
        # normalization factor, (H * W * anchor_num)
        anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2) # all the same

        # (H, W, 2)
        pos_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        neg_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        # (H, W, self.anchor_num * 7)
        targets = np.zeros((*feature_map_shape, self.anchor_num * 7))

        # (n, 7)  here, 7 means x,y,z,h,w,l,\theta
        gt_box_center_valid = gt_box_center[masks == 1]
        # (n, 8, 3) ,  here 8 means 8 points in a 3D-bbox
        gt_box_corner_valid = \
            box_utils.boxes_to_corners_3d(gt_box_center_valid,
                                          self.params['order'])
        # (H*W*anchor_num, 8, 3)
        anchors_corner = \
            box_utils.boxes_to_corners_3d(anchors,
                                          order=self.params['order'])
        # (H*W*anchor_num, 4)
        anchors_standup_2d = \
            box_utils.corner2d_to_standup_box(anchors_corner)
        # (n, 4)
        gt_standup_2d = \
            box_utils.corner2d_to_standup_box(gt_box_corner_valid)

        # (H*W*anchor_n)
        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )

        # the anchor boxes has the largest iou across
        # shape: (n)
        id_highest = np.argmax(iou.T, axis=1)
        # [0, 1, 2, ..., n-1]
        id_highest_gt = np.arange(iou.T.shape[0])
        # make sure all highest iou is larger than 0
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]


        # find anchors iou > params['pos_iou']
        id_pos, id_pos_gt = \
            np.where(iou >
                     self.params['target_args']['pos_threshold'])
        #  find anchors iou  params['neg_iou']
        id_neg = np.where(np.sum(iou <
                                 self.params['target_args']['neg_threshold'],
                                 axis=1) == iou.shape[1])[0]
        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*feature_map_shape, self.anchor_num))
        pos_equal_one[index_x, index_y, index_z] = 1
        
        # calculate the targets
        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_box_center[id_pos_gt, 0] - anchors[id_pos, 0]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_box_center[id_pos_gt, 1] - anchors[id_pos, 1]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_box_center[id_pos_gt, 2] - anchors[id_pos, 2]) / anchors[
                id_pos, 3]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            gt_box_center[id_pos_gt, 3] / anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            gt_box_center[id_pos_gt, 4] / anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            gt_box_center[id_pos_gt, 5] / anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
                gt_box_center[id_pos_gt, 6] - anchors[id_pos, 6])

        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 1

        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 0


        label_dict = {'pos_equal_one': pos_equal_one,
                      'neg_equal_one': neg_equal_one,
                      'targets': targets}

        return label_dict


    def generate_gt_bbx(self, data_dict):
        """
        The base postprocessor will generate 3d groundtruth bounding box.
        ONLY visible boxes.

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.
            To generate gt boxes, transformation_matrix should be clean

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        Returns
        -------
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor, shape (N, 8, 3).
        """
        gt_box3d_list = []
        # used to avoid repetitive bounding box
        object_id_list = []

        for cav_id, cav_content in data_dict.items():
            # used to project gt bounding box to ego space
            # object_bbx_center is clean.
            transformation_matrix = cav_content['transformation_matrix_clean']

            object_bbx_center = cav_content['object_bbx_center']
            object_bbx_mask = cav_content['object_bbx_mask']
            object_ids = cav_content['object_ids']
            object_bbx_center = object_bbx_center[object_bbx_mask == 1]

            # convert center to corner
            object_bbx_corner = \
                box_utils.boxes_to_corners_3d(object_bbx_center,
                                              self.params['order'])
            projected_object_bbx_corner = \
                box_utils.project_box3d(object_bbx_corner.float(),
                                        transformation_matrix)
            gt_box3d_list.append(projected_object_bbx_corner)
            # append the corresponding ids
            object_id_list += object_ids

        # gt bbx 3d
        gt_box3d_list = torch.vstack(gt_box3d_list)
        # some of the bbx may be repetitive, use the id list to filter
        gt_box3d_selected_indices = \
            [object_id_list.index(x) for x in set(object_id_list)]
        gt_box3d_tensor = gt_box3d_list[gt_box3d_selected_indices]

        # filter the gt_box to make sure all bbx are in the range
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor, self.params['gt_range'])
        gt_box3d_tensor = gt_box3d_tensor[mask, :, :]

        # we need project gtbox to image
        # for cav_id, cav_content in data_dict.items():
        #     transformation_matrix = cav_content['transformation_matrix_clean']
        #     ego_obj = cav_content['label_dict']['object_target'][0]
        #     object_bbx_corner = \
        #         box_utils.boxes_to_corners_3d(ego_obj,
        #                                       self.params['order'])
        #     projected_object_bbx_corner = \
        #         box_utils.project_box3d(object_bbx_corner.float(),
        #                                 transformation_matrix)
        #     N,_,_ = projected_object_bbx_corner.shape
        #
        #     points = projected_object_bbx_corner.unsqueeze(-1).repeat(4,1,1,1,1)
        #     ones = torch.ones_like(points[..., :1, :])
        #     homogeneous_points = torch.cat([points, ones], dim=3)
        #     camera_to_lidar = cav_content['image_inputs']["camera_to_lidar"][0] # ego     4,4,4
        #     # ego_to_camera = torch.inverse(camera_to_lidar).reshape(  4, 1,1,4, 4)
        #
        #     ego_to_camera = cav_content['image_inputs']["extrins"][0].reshape(4,1,1,4,4)
        #
        #     points_cam = torch.matmul(ego_to_camera, homogeneous_points)
        #     points_cam = points_cam[..., :3, :] / points_cam[..., 3:4, :]
        #
        #     intrins = cav_content['image_inputs']["intrins"][0]
        #     intrins = intrins.reshape(4, 1, 1, 3, 3)
        #     points_2d = torch.matmul(intrins, points_cam).squeeze()
        #     # points_2d = []
        #
        #
        #     #
        #
        #     depth = points_2d[..., 2:3]
        #     uv = points_2d[..., :2] / depth
        #     uv_int = uv.round()  # [N*8, 2]
        #     valid_mask1 = ((uv_int[..., 0:1] >= 0) & (uv_int[..., 0:1] < 480) &
        #                    (uv_int[..., 1:2] >= 0) & (uv_int[..., 1:2] < 320))
        #
        #     valid_mask2 = ((depth > 0.5) & (depth < 40))
        #     gt_box2d_mask = valid_mask1.any(axis=1) & valid_mask2.all(axis=1)  # [N, ]
        #     uv = uv[gt_box2d_mask]
        #
        #     points_2d = points_2d[..., :2] / torch.clamp(
        #         points_2d[..., 2:3], min=1e-5
        #     )
        #
        #     # for postprocess in image
        #     post_trans = cav_content['image_inputs']["post_trans"][0]
        #     post_rots = cav_content['image_inputs']["post_rots"][0]
        #
        #     # post_T = torch.zeros(4, 3, 3).cuda()
        #     # # 设置旋转矩阵的部分
        #     # post_T[:, :2, :2] = post_rots[:,:2,:2]
        #     # trans_expanded = torch.cat((post_trans[:,:2], torch.zeros(4, 1).cuda()), dim=1).unsqueeze(2)
        #     # post_T[:, :3, 2] = trans_expanded.squeeze(2)
        #     # post_T[:, 2, 2] = 1.0
        #
        #     ones = torch.ones_like(points_2d[..., :1])
        #     points_2d = torch.cat([points_2d, ones], dim=-1)
        #
        #     # post_T = post_T.reshape(4,1,1,3,3)
        #     # points_2d = (points_2d.unsqueeze(-1)).matmul(post_T).squeeze()
        #
        #     post_rots = post_rots.reshape(4,1,1,3,3)
        #
        #     # points_2d = torch.matmul(post_rots, points_2d.unsqueeze(-1)).squeeze()
        #     # points_2d = points_2d + post_trans.view(4, 1, 1, -1)
        #
        #     points_2d = torch.matmul(post_rots, points_2d.unsqueeze(-1)).squeeze()
        #     points_2d = points_2d + post_trans.view(4, 1, 1, -1)
        #
        #
        #     points_2d = points_2d[..., :2]
        #
        #
        #     # visualize point
        #     imgs = []
        #     points_ =[]
        #     for i in range(4):
        #         imgs.append(cav_content['image_inputs']['imgs'][0][i].permute(1,2,0))
        #         points_.append(points_2d[i].reshape(-1,2))
        #     self.visualize_projected_points(imgs,points_)
        return gt_box3d_tensor

    def visualize_projected_points(self,images, points_2d, save_path=None):
        import cv2
        H, W, C = images[0].shape
        for img_idx, img_tensor in enumerate(images):
            # 确保图像是 NumPy 数组
            img = img_tensor.cpu().numpy()
            # 如果图像是 (C, H, W)，转换为 (H, W, C)
            if img.shape[0] in [1, 3]:  # 灰度图或RGB图
                img = np.transpose(img, (1, 2, 0))
            # 处理彩色通道
            if img.shape[2] == 1:  # 灰度图
                img = np.squeeze(img, axis=2)
            elif img.shape[2] == 3:  # RGB图
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 画点
            for pt in points_2d[img_idx].detach().cpu().numpy():
                x, y = int(pt[0]), int(pt[1])
                if x > W or x < 0 or y > H or y < 0:
                    continue
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # 绿色点

            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f'Image {img_idx}')
            plt.axis('off')

            if save_path:
                plt.savefig(f'{save_path}_img_{img_idx}.png')
            plt.show()

    @staticmethod
    def collate_batch(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        targets = []
        object_target = []
        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]['pos_equal_one'])
            neg_equal_one.append(label_batch_list[i]['neg_equal_one'])
            targets.append(label_batch_list[i]['targets'])

        pos_equal_one = \
            torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = \
            torch.from_numpy(np.array(neg_equal_one))
        targets = \
            torch.from_numpy(np.array(targets))


        return {'targets': targets,
                'pos_equal_one': pos_equal_one,
                'neg_equal_one': neg_equal_one,
                }

                

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.


        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []
        for cav_id, cav_content in data_dict.items():
            assert cav_id in output_dict
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix'] # no clean

            # (H, W, anchor_num, 7)
            anchor_box = cav_content['anchor_box']

            # classification probability
            prob = output_dict[cav_id]['psm']
            prob = F.sigmoid(prob.permute(0, 2, 3, 1))
            prob = prob.reshape(1, -1)

            # regression map
            reg = output_dict[cav_id]['rm']

            # convert regression map back to bounding box
            batch_box3d = self.delta_to_boxes3d(reg, anchor_box)
            mask = \
                torch.gt(prob, self.params['target_args']['score_threshold'])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0],
                                          mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])

            # adding dir classifier
            if 'dm' in output_dict[cav_id].keys() and len(boxes3d) !=0:
                dir_offset = self.params['dir_args']['dir_offset']
                num_bins = self.params['dir_args']['num_bins']


                dm  = output_dict[cav_id]['dm'] # [N, H, W, 4]
                dir_cls_preds = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins) # [1, N*H*W*2, 2]
                dir_cls_preds = dir_cls_preds[mask]
                # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0
                
                period = (2 * np.pi / num_bins) # pi
                dir_rot = limit_period(
                    boxes3d[..., 6] - dir_offset, 0, period
                ) # 限制在0到pi之间
                boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype) # 转化0.25pi到2.5pi
                boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi) # limit to [-pi, pi]

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = \
                    box_utils.boxes_to_corners_3d(boxes3d,
                                                  order=self.params['order'])
                
                # STEP 2
                # (N, 8, 3)
                projected_boxes3d = \
                    box_utils.project_box3d(boxes3d_corner,
                                            transformation_matrix)
                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)

        if len(pred_box2d_list) ==0 or len(pred_box3d_list) == 0:
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # remove large bbx
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]

        # STEP3
        # nms
        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                           scores,
                                           self.params['nms_thresh']
                                           )

        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        # select cooresponding score
        scores = scores[keep_index]
        
        # filter out the prediction out of the range.
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor, self.params['gt_range'])
        pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        scores = scores[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        return pred_box3d_tensor, scores


    def post_process_detr(self,data_dict,output_dict):
        pred_box3d_list = []
        vehicle_type_list = []
        pred_score_tensor = output_dict['ego']['output']['scores_3d']
        for cav_id, cav_content in data_dict.items():
            transformation_matrix = cav_content['transformation_matrix']
            boxes3d = output_dict['ego']['output']['boxes_3d']
            boxes3d_corner = \
                box_utils.boxes_to_corners_3d(boxes3d,
                                              order=self.params['order'])
            projected_boxes3d = \
                box_utils.project_box3d(boxes3d_corner,
                                        transformation_matrix)
            pred_box3d_list.append(projected_boxes3d)
        pred_box3d_tensor = torch.vstack(pred_box3d_list)

        return pred_box3d_tensor,pred_score_tensor

    @staticmethod
    def delta_to_boxes3d(deltas, anchors):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, W, L, 14)?? should be (N, 14, H, W)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        boxes3d = torch.zeros_like(deltas)

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                      anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d

    @staticmethod
    def visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        """
        Visualize the prediction, ground truth with point cloud together.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        show_vis : bool
            Whether to show visualization.

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        """
        vis_utils.visualize_single_sample_output_gt(pred_box_tensor,
                                                    gt_tensor,
                                                    pcd,
                                                    show_vis,
                                                    save_path)
