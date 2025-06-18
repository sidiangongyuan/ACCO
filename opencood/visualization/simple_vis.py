from matplotlib import pyplot as plt
import numpy as np

import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev

def visualize(pred_box_tensor, gt_tensor, pcd, pc_range, save_path, method='3d', vis_gt_box=True, vis_pred_box=True, left_hand=False, uncertainty=None, **kwargs):
        """
        Visualize the prediction, ground truth with point cloud together.
        They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        pc_range : list
            [xmin, ymin, zmin, xmax, ymax, zmax]

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        method: str, 'bev' or '3d'

        """
        plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
        pc_range = [int(i) for i in pc_range]
        pcd_np = pcd.cpu().numpy()
        if vis_gt_box:
            gt_box_np = gt_tensor.cpu().numpy()
            gt_name = ['gt'] * gt_box_np.shape[0]
            # gt_name = [''] * gt_box_np.shape[0]
        if vis_pred_box:
            if isinstance(pred_box_tensor, tuple):
                pred_boxes_ego, pred_boxes_neighbor = pred_box_tensor

                pred_boxes_ego_np = pred_boxes_ego.cpu().numpy()
                pred_boxes_neighbor_np = pred_boxes_neighbor.cpu().numpy()

                pred_name_ego = ['pred'] * pred_boxes_ego_np.shape[0]
                pred_name_neighbor = ['pred'] * pred_boxes_neighbor_np.shape[0]

                pred_box_np = np.concatenate([pred_boxes_ego_np, pred_boxes_neighbor_np], axis=0)
                pred_name = pred_name_ego + pred_name_neighbor

            else:
                # 如果pred_box_tensor不是元组，按原始方式处理
                pred_box_np = pred_box_tensor.cpu().numpy()
                pred_name = ['pred'] * pred_box_np.shape[0]



        if method == 'bev':
            canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=left_hand) 

            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords
            canvas.draw_canvas_points(canvas_xy[valid_mask]) # Only draw valid points
            if vis_gt_box:
                canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
                # canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name, box_line_thickness=6)
            if vis_pred_box:
                if isinstance(pred_box_tensor, tuple):
                    canvas.draw_boxes(pred_boxes_ego_np, colors=(255, 0, 0), texts=pred_name_ego)
                    canvas.draw_boxes(pred_boxes_neighbor_np, colors=(0, 0, 255), texts=pred_name_neighbor)
                else:
                    canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)
                # canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name, box_line_thickness=6)
                if 'cavnum' in kwargs:
                    canvas.draw_boxes(pred_box_np[:kwargs['cavnum']], colors=(0,191,255), texts=['']*kwargs['cavnum'])  # something wrong
                    # canvas.draw_boxes(pred_box_np[:kwargs['cavnum']], colors=(0,191,255), texts=['']*kwargs['cavnum'], box_line_thickness=6)


        elif method == '3d':
            canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
            canvas.draw_canvas_points(canvas_xy[valid_mask])
            if vis_pred_box:
                canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)
            if vis_gt_box:
                canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
        else:
            raise(f"Not Completed for f{method} visualization.")

        plt.axis("off")

        plt.imshow(canvas.canvas)

        plt.tight_layout()
        plt.savefig(save_path, transparent=True, dpi=400)
        plt.clf()
        plt.close()