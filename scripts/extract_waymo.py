"""Extract pedestrian point clouds, bounding boxes, and keypoints from Waymo.

This script reads the raw Waymo perception parquet files, filters pedestrian boxes,
reconstructs point clouds / keypoints in both vehicle and world frames, and stores
per-object trajectories as pickle files for later processing.
"""
from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Optional, Tuple

import dask.dataframe as dd
import numpy as np
import tensorflow as tf
import tqdm
from waymo_open_dataset import v2

# Force CPU execution for TensorFlow to avoid needless GPU allocation.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _read_parquet_component(waymo_dir: str, split: str, tag: str, context_name: str) -> dd.DataFrame:
    """Create a Dask DataFrame for a specific Waymo component."""
    pattern = f"{waymo_dir}/{split}/{tag}/{context_name}.parquet"
    paths = tf.io.gfile.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No parquet files for pattern: {pattern}")
    return dd.read_parquet(paths)


def load_waymo_dataframe(waymo_dir: str, split: str, context_name: str) -> dd.DataFrame:
    """Join the components required to recover pedestrian trajectories."""
    lidar_df = _read_parquet_component(waymo_dir, split, "lidar", context_name)
    lidar_box_df = _read_parquet_component(waymo_dir, split, "lidar_box", context_name)
    lidar_hkp_df = _read_parquet_component(waymo_dir, split, "lidar_hkp", context_name)
    lidar_cali_df = _read_parquet_component(waymo_dir, split, "lidar_calibration", context_name)
    lidar_pose_df = _read_parquet_component(waymo_dir, split, "lidar_pose", context_name)
    vehicle_pose_df = _read_parquet_component(waymo_dir, split, "vehicle_pose", context_name)

    # Keep only pedestrian boxes.
    lidar_box_filtered = lidar_box_df[lidar_box_df['[LiDARBoxComponent].type'] == 2]

    merged_lidar_pose = dd.merge(lidar_df, lidar_pose_df, on=['key.segment_context_name', 'key.frame_timestamp_micros','key.laser_name'], how='left')
    merged_lidar_pose_vehicle = dd.merge(merged_lidar_pose, vehicle_pose_df, on=['key.segment_context_name', 'key.frame_timestamp_micros'], how='left')
    merged_all = dd.merge(merged_lidar_pose_vehicle, lidar_box_filtered, on=['key.segment_context_name', 'key.frame_timestamp_micros'], how='left')
    merged_all_hkp = dd.merge(merged_all, lidar_hkp_df, on=['key.segment_context_name', 'key.frame_timestamp_micros', 'key.laser_object_id'], how='left')
    final_merged_df = dd.merge(merged_all_hkp, lidar_cali_df, on=['key.segment_context_name','key.laser_name'], how='left')

    return final_merged_df


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _convert_range_image_to_point_cloud(
    lidar: v2.LiDARComponent,
    lidar_cali: v2.LiDARCalibrationComponent,
    lidar_pose: v2.LiDARPoseComponent,
    vehicle_pose: v2.VehiclePoseComponent,
) -> np.ndarray:
    """Decode the LiDAR range image into a point cloud."""
    if lidar.key.laser_name == 1:
        points = v2.convert_range_image_to_point_cloud(
            lidar.range_image_return1,
            lidar_cali,
            lidar_pose.range_image_return1,
            vehicle_pose,
        )
    else:
        points = v2.convert_range_image_to_point_cloud(
            lidar.range_image_return1,
            lidar_cali,
        )
    return np.asarray(points, dtype=np.float32)


def _compute_bbox_vertices(
    lidar_box: v2.LiDARBoxComponent,
    vehicle_pose: v2.VehiclePoseComponent,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (center, size, heading, rotated_vertices, world_vertices)."""
    center = np.array([
        lidar_box.box.center.x,
        lidar_box.box.center.y,
        lidar_box.box.center.z,
    ], dtype=np.float32)
    size = np.array([
        lidar_box.box.size.x,
        lidar_box.box.size.y,
        lidar_box.box.size.z,
    ], dtype=np.float32)
    heading = -float(lidar_box.box.heading)

    half = size / 2.0
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    local_vertices = half * signs

    c, s = np.cos(heading), np.sin(heading)
    rot_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    rotated_vertices = (local_vertices @ rot_z.T) + center

    transform = np.asarray(vehicle_pose.world_from_vehicle.transform, dtype=np.float32).reshape(4, 4)
    hom_vertices = np.hstack([rotated_vertices, np.ones((8, 1), dtype=np.float32)])
    world_vertices = (transform @ hom_vertices.T).T[:, :3]

    return center, size, np.array([heading], dtype=np.float32), rotated_vertices, world_vertices


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply the homogeneous transform (4x4) to a set of points."""
    if points is None or len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    hom_points = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])
    return (transform @ hom_points.T).T[:, :3]


# ---------------------------------------------------------------------------
# Row processing
# ---------------------------------------------------------------------------
def _decode_row(row, split: str, context_name: str) -> Optional[Dict]:
    """Convert a merged dataframe row into a dictionary entry."""
    lidar = v2.LiDARComponent.from_dict(row)
    lidar_box = v2.LiDARBoxComponent.from_dict(row)
    lidar_hkp = v2.LiDARHumanKeypointsComponent.from_dict(row)
    lidar_cali = v2.LiDARCalibrationComponent.from_dict(row)
    lidar_pose = v2.LiDARPoseComponent.from_dict(row)
    vehicle_pose = v2.VehiclePoseComponent.from_dict(row)

    object_id = lidar_box.key.laser_object_id
    if not isinstance(object_id, str):
        return None
    time_stamp = int(lidar_box.key.frame_timestamp_micros)

    point_cloud = _convert_range_image_to_point_cloud(lidar, lidar_cali, lidar_pose, vehicle_pose)

    keypoints = None
    keypoint_type = None
    keypoint_is_occluded = None
    if not np.isnan(lidar_hkp.lidar_keypoints.type).any():
        keypoints = np.stack(
            (
                lidar_hkp.lidar_keypoints.keypoint_3d.location_m.x,
                lidar_hkp.lidar_keypoints.keypoint_3d.location_m.y,
                lidar_hkp.lidar_keypoints.keypoint_3d.location_m.z,
            ),
            axis=1,
        )
        keypoint_type = lidar_hkp.lidar_keypoints.type
        keypoint_is_occluded = lidar_hkp.lidar_keypoints.keypoint_3d.visibility.is_occluded

    transform = np.asarray(vehicle_pose.world_from_vehicle.transform, dtype=np.float32).reshape(4, 4)
    keypoints_world = _transform_points(keypoints, transform) if keypoints is not None else None

    center, size, heading, rotated_vertices, world_vertices = _compute_bbox_vertices(lidar_box, vehicle_pose)

    half = size / 2.0
    heading = float(heading)  # ensure scalar

    c = np.cos(heading)
    s = np.sin(heading)

    rot_z = np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pc_local = (rot_z.T @ (point_cloud - center).T).T
    mask = np.all(np.abs(pc_local) <= half + 1e-6, axis=1)
    points_vehicle = point_cloud[mask]
    points_world = _transform_points(points_vehicle, transform)

    heading_dir_local = np.array([1, 0, 0], dtype=np.float32)
    heading_dir_world = (transform[:3, :3] @ (rot_z @ heading_dir_local))
    heading_dir_world /= np.linalg.norm(heading_dir_world) + 1e-8

    return {
        'basic_info': {
            'object_id': object_id,
            'time_stamp': time_stamp,
            'context_name': context_name,
            'split': split,
        },
        'vehicle_frame': {
            'keypoints': keypoints,
            'keypoint_type': keypoint_type,
            'keypoint_is_occluded': keypoint_is_occluded,
            'center': center,
            'size': size,
            'heading': heading,
            'pedestrian_pcd': points_vehicle,
            'bbox_vertices': rotated_vertices,

        },
        'world_frame': {
            'keypoints': keypoints_world,
            'keypoint_type': keypoint_type,
            'keypoint_is_occluded': keypoint_is_occluded,
            'center': _transform_points(center[None, :], transform)[0],
            'size': size,
            'heading': heading_dir_world,
            'pedestrian_pcd': points_world,
            'bbox_vertices': world_vertices,

        },
    }


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------
def list_contexts(waymo_dir: str, split: str) -> List[str]:
    """Return all context names in the given split."""
    lidar_dir = os.path.join(waymo_dir, split, 'lidar_box')
    if not os.path.isdir(lidar_dir):
        return []
    return [
        os.path.splitext(fn)[0]
        for fn in os.listdir(lidar_dir)
        if fn.endswith('.parquet')
    ]


def process_context(waymo_dir: str, split: str, context_name: str, output_dir: str) -> None:
    """Extract one context and persist its pickle file."""
    output_path = os.path.join(output_dir, split, f'{context_name}.pkl')
    if os.path.exists(output_path):
        return

    df = load_waymo_dataframe(waymo_dir, split, context_name)
    entries: Dict[Tuple[str, int], Dict] = {}

    for _, row in df.iterrows():
        entry = _decode_row(row, split, context_name)
        if entry is None:
            continue
        key = (entry['basic_info']['object_id'], entry['basic_info']['time_stamp'])
        if key in entries:
            existing = entries[key]
            existing['vehicle_frame']['pedestrian_pcd'] = np.vstack(
                (existing['vehicle_frame']['pedestrian_pcd'], entry['vehicle_frame']['pedestrian_pcd'])
            )
            existing['world_frame']['pedestrian_pcd'] = np.vstack(
                (existing['world_frame']['pedestrian_pcd'], entry['world_frame']['pedestrian_pcd'])
            )
        else:
            entries[key] = entry

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(list(entries.values()), f)


def main(waymo_dir: str, output_dir: str, splits: List[str]):
    for split in splits:
        contexts = list_contexts(waymo_dir, split)
        for context_name in tqdm.tqdm(contexts, desc=f'Extracting {split}'):
            process_context(waymo_dir, split, context_name, output_dir)
        print(f'Finished split: {split}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract pedestrian data from Waymo")
    parser.add_argument("--waymo-dir", default="/dataset/guangxun/dataset",help="Root directory containing the split subfolders")
    parser.add_argument("--output-dir", default="/dataset/guangxun/test/waymo_extracted",help="Directory to write extracted pickle files")
    parser.add_argument( "--splits", nargs="+",default=["training", "validation"], help="Dataset splits to process", required=False)
    cli_args = parser.parse_args()

    main(cli_args.waymo_dir, cli_args.output_dir, cli_args.splits)
