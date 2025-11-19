'''
Extract pedestrian point clouds, bbox, and keypoints from Waymo Perception dataset
Saves the extracted data into pickle files for further processing.

Author: Guangxun Zhu
Date: 2025-11-19
'''

import dask.dataframe as dd
from waymo_open_dataset import v2
import numpy as np
import os
from matplotlib import pylab as plt
import io
import PIL.Image
import math
import open3d as o3d
import cv2
import pickle
import tqdm
import tensorflow as tf
import time

def create_point_cloud(points, color, translation=(0, 0, 0)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    if translation != (0, 0, 0):  
        pcd.translate(translation)
    return pcd

def read(waymo_dir: str, split: str, tag: str, context_name: str) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    paths = tf.io.gfile.glob(f'{waymo_dir}/{split}/{tag}/{context_name}.parquet')
    return dd.read_parquet(paths)

def read_dataframes(waymo_dir, split, context_name):
    # join all dataframes of original waymo dataset
    lidar_df = read(waymo_dir, split, 'lidar', context_name)
    lidar_box_df = read(waymo_dir, split, 'lidar_box', context_name)
    lidar_hkp_df = read(waymo_dir, split, 'lidar_hkp', context_name)
    lidar_cali_df = read(waymo_dir, split, 'lidar_calibration', context_name)
    lidar_pose_df = read(waymo_dir, split, 'lidar_pose', context_name)
    vehicle_pose_df = read(waymo_dir, split, 'vehicle_pose', context_name)

    # filter lidar_box_df ：type == 2 is pedestrian
    lidar_box_filtered = lidar_box_df[lidar_box_df['[LiDARBoxComponent].type'] == 2]

    merged_lidar_pose = dd.merge(lidar_df, lidar_pose_df, on=['key.segment_context_name', 'key.frame_timestamp_micros','key.laser_name'], how='left')
    merged_lidar_pose_vehicle = dd.merge(merged_lidar_pose, vehicle_pose_df, on=['key.segment_context_name', 'key.frame_timestamp_micros'], how='left')
    merged_all = dd.merge(merged_lidar_pose_vehicle, lidar_box_filtered, on=['key.segment_context_name', 'key.frame_timestamp_micros'], how='left')
    merged_all_hkp = dd.merge(merged_all, lidar_hkp_df, on=['key.segment_context_name', 'key.frame_timestamp_micros', 'key.laser_object_id'], how='left')
    final_merged_df = dd.merge(merged_all_hkp, lidar_cali_df, on=['key.segment_context_name','key.laser_name'], how='left')

    return final_merged_df

if __name__ == '__main__':


    waymo_dir = '/dataset/guangxun/dataset'
    split_list = ['training', 'validation']

    save_dir = '/home/guangxun/test/waymo_extracted'


    for split in split_list:
        # obtain the context names
        context_names = []
        for file_name in os.listdir(os.path.join(waymo_dir, split, 'lidar')):
            if os.path.isfile(os.path.join(waymo_dir, split, 'lidar', file_name)) and file_name.endswith('.parquet'):
                file_name_without_extension = os.path.splitext(file_name)[0]
                context_names.append(file_name_without_extension)
        # context_names = ['8582923946352460474_2360_000_2380_000']
        # Read the original waymo dataset
        for context_name in tqdm.tqdm(context_names):

            extracted_data = []

            waymo_data = read_dataframes(waymo_dir, split, context_name)

            # iterate over each row in hkp_box_df

            for i, (_, row) in enumerate(waymo_data.iterrows()):

                # obtain the lidar box and lidar human keypoints
                lidar = v2.LiDARComponent.from_dict(row)
                lidar_box = v2.LiDARBoxComponent.from_dict(row)
                lidar_hkp = v2.LiDARHumanKeypointsComponent.from_dict(row)
                lidar_cali = v2.LiDARCalibrationComponent.from_dict(row)
                lidar_pose = v2.LiDARPoseComponent.from_dict(row)
                vehicle_pose = v2.VehiclePoseComponent.from_dict(row)

                object_id = lidar_box.key.laser_object_id
                time_stamp = lidar_box.key.frame_timestamp_micros
                laser_name = lidar.key.laser_name
                vehicle_to_world = vehicle_pose.world_from_vehicle.transform

                if not isinstance(object_id, str):
                    continue

                # convert range image to point cloud
                if laser_name == 1:
                    point_cloud = v2.convert_range_image_to_point_cloud(lidar.range_image_return1, lidar_cali, lidar_pose.range_image_return1, vehicle_pose)
                else:   
                    point_cloud = v2.convert_range_image_to_point_cloud(lidar.range_image_return1, lidar_cali)

                point_cloud = np.array(point_cloud)  # (N, 3)
                # get keypoints if exist
                if not np.isnan(lidar_hkp.lidar_keypoints.type).any():
                    keypoints = np.stack((lidar_hkp.lidar_keypoints.keypoint_3d.location_m.x,
                                        lidar_hkp.lidar_keypoints.keypoint_3d.location_m.y,
                                        lidar_hkp.lidar_keypoints.keypoint_3d.location_m.z), axis=1)
                    keypoint_is_occluded = lidar_hkp.lidar_keypoints.keypoint_3d.visibility.is_occluded

                    keypoint_type = lidar_hkp.lidar_keypoints.type

                else:
                    keypoints = None
                    keypoint_is_occluded = None
                    keypoint_type = None

                # get transformation matrix
                transform = np.reshape(np.array(vehicle_to_world), [4, 4])


                # transform keypoints to world frame
                keypoints_world = None
                if keypoints is not None:

                    keypoints_homogeneous = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])  # (17, 4)
                    keypoints_world = np.transpose(np.matmul(transform, np.transpose(keypoints_homogeneous)))[:, :3]  # transform to world frame

                # obtain bbox parameters
                center = np.array([lidar_box.box.center.x,
                                lidar_box.box.center.y,
                                lidar_box.box.center.z])
                size = np.array([lidar_box.box.size.x,
                                lidar_box.box.size.y,
                                lidar_box.box.size.z])
                heading = -lidar_box.box.heading  # convert to right-hand rule

                speed  = np.array([lidar_box.speed.x, lidar_box.speed.y, lidar_box.speed.z])
                acceleration = np.array([lidar_box.acceleration.x, lidar_box.acceleration.y, lidar_box.acceleration.z])

                # compute 8 corners of the bbox
                half = size / 2.0
                signs = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                                [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
                local_vertices = half * signs

                # rotate around z-axis
                c, s = np.cos(heading), np.sin(heading)
                Rz = np.array([[c, -s, 0],
                            [s,  c, 0],
                            [0,  0, 1]])
                rotated_vertices = local_vertices @ Rz.T

                # translate to center
                rotated_vertices += center

                # transform to world frame
                transform = np.reshape(np.array(vehicle_to_world), [4,4])
                ones = np.ones((8,1))
                hom_vertices = np.hstack([rotated_vertices, ones])
                world_vertices = (transform @ hom_vertices.T).T[:, :3]

                # recompute center in world frame
                center_hom = np.hstack([center, 1.0])
                center_world = (transform @ center_hom)[:3]

                # compute heading direction in world frame
                heading_dir_local = np.array([1, 0, 0])
                heading_dir_rotated = Rz @ heading_dir_local  # rotate according to heading
                heading_dir_world = (transform[:3, :3] @ heading_dir_rotated)  # transform to world frame
                heading_dir_world = heading_dir_world / np.linalg.norm(heading_dir_world)  # normalize

                size_world = size  # size remains the same in world frame


                # Extract points within the bbox
                pc_centered = point_cloud - center  # (N, 3)
                pc_local = (Rz.T @ pc_centered.T).T  # (N, 3)
                half = size / 2.0
                eps = 1e-6
                mask = np.all(np.abs(pc_local) <= half + eps, axis=1)

                points_in_box = point_cloud[mask]

                points_homogeneous = np.hstack(
                    [points_in_box, np.ones((points_in_box.shape[0], 1))]
                )  # (M, 4)
                points_in_box_world = (transform @ points_homogeneous.T).T[:, :3]

                # pcd = create_point_cloud(points_in_box_world, [0, 1, 0])
                # o3d.visualization.draw_geometries([pcd])


                # Check if the object_id and time_stamp already exist in extracted_data
                existing_entry = next((item for item in extracted_data 
                                    if item['basic_info']['context_name'] == context_name
                                    and item['basic_info']['object_id'] == object_id 
                                    and item['basic_info']['time_stamp'] == time_stamp), None)

                # create new entry or merge with existing one
                if existing_entry:
                    # Merge the pedestrian point clouds
                    existing_entry['vehicle_frame']['pedestrian_pcd'] = np.vstack((existing_entry['vehicle_frame']['pedestrian_pcd'], points_in_box))
                    existing_entry['world_frame']['pedestrian_pcd'] = np.vstack((existing_entry['world_frame']['pedestrian_pcd'], points_in_box_world))
                else:
                    extracted_data.append({
                        'basic_info': {
                            'object_id': object_id,
                            'time_stamp': time_stamp,
                            'context_name': context_name,
                            'split': split
                        },
                        'vehicle_frame': {
                            'keypoints': keypoints,
                            'keypoint_type': keypoint_type,
                            'keypoint_is_occluded': keypoint_is_occluded,
                            'center': center,
                            'size': size,
                            'heading': heading,
                            'pedestrian_pcd': points_in_box,
                        },
                        'world_frame': {
                            'keypoints': keypoints_world,
                            'keypoint_type': keypoint_type,
                            'keypoint_is_occluded': keypoint_is_occluded,
                            'center': center_world,
                            'size': size_world,
                            'heading': heading_dir_world,
                            'pedestrian_pcd': points_in_box_world,
                        }
                    })

            save_path = os.path.join(save_dir, split)
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, '{}.pkl'.format(context_name)), 'wb') as f:
                pickle.dump(extracted_data, f)
        print('Extracted data saved for split: {}'.format(split))

