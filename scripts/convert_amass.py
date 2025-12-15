"""Convert LiDAR-HMR outputs into AMASS-style sequences and metadata bundles."""
from __future__ import annotations

import argparse
import glob
import os
import pickle as pkl
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - activates 3D plotting backend
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

# Use GPU when available for the SMPL forward pass used during visualization only.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


SMPL_BONE_LINK = np.array([
    [0, 1], [0, 2],
    [1, 4], [2, 5],
    [4, 7], [5, 8],
    [0, 3],
    [3, 6], [6, 9],
    [9, 12],
    [12, 15],
    [9, 13], [9, 14],
    [13, 16], [14, 17],
    [16, 18], [17, 19],
    [18, 20], [19, 21],
])


# ---------------------------------------------------------------------------
# SMPL orientation correction helpers
# ---------------------------------------------------------------------------
def correct_smpl_global_orient(
    trans: np.ndarray,
    root_orient: np.ndarray,
    smooth_sigma: float = 1.0,
    angle_threshold_deg: float = 90.0,
    epsilon: float = 1e-6,
    motion_threshold: float = 2e-2,
    motion_smooth_sigma: float = 2.0,
) -> np.ndarray:
    """Align SMPL root orientation with the motion Frenet frame when moving."""
    N = trans.shape[0]
    traj_smooth = gaussian_filter1d(trans, sigma=smooth_sigma, axis=0)

    tangents = np.zeros((N, 3))
    for i in range(N):
        if 0 < i < N - 1:
            delta = traj_smooth[i + 1] - traj_smooth[i - 1]
        elif i == 0:
            delta = traj_smooth[1] - traj_smooth[0]
        else:
            delta = traj_smooth[-1] - traj_smooth[-2]
        tangents[i] = delta / (np.linalg.norm(delta) + epsilon)

    planar_dirs = tangents.copy()
    planar_dirs[:, 2] = 0.0
    norms = np.linalg.norm(planar_dirs, axis=1)
    valid = norms > epsilon
    planar_dirs[valid] = (planar_dirs[valid].T / norms[valid]).T
    for i in range(N):
        if not valid[i]:
            planar_dirs[i] = planar_dirs[i - 1] if i > 0 else np.array([1.0, 0.0, 0.0])

    vel_xy = np.linalg.norm(np.diff(traj_smooth[:, :2], axis=0), axis=1)
    vel_xy = np.concatenate([[vel_xy[0]], vel_xy])
    vel_xy = gaussian_filter1d(vel_xy, sigma=motion_smooth_sigma)

    up = np.array([0.0, 0.0, 1.0])
    corrected_rots = []
    raw_rots = Rotation.from_rotvec(root_orient)
    angle_thresh = np.deg2rad(angle_threshold_deg)

    for i in range(N):
        if vel_xy[i] < motion_threshold:
            corrected_rots.append(raw_rots[i])
            continue

        z_axis = planar_dirs[i]
        x_axis = np.cross(up, z_axis)
        if np.linalg.norm(x_axis) < epsilon:
            corrected_rots.append(raw_rots[i])
            continue

        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        r_des = np.stack([x_axis, y_axis, z_axis], axis=1)
        r_raw = raw_rots[i].as_matrix()
        r_align = r_des @ r_raw.T

        angle = np.arccos(np.clip((np.trace(r_align) - 1.0) / 2.0, -1.0, 1.0))
        if angle > angle_thresh:
            corrected_rots.append(Rotation.from_matrix(r_align) * raw_rots[i])
        else:
            corrected_rots.append(raw_rots[i])

    return Rotation.concatenate(corrected_rots).as_rotvec()


# ---------------------------------------------------------------------------
# Interpolation utilities
# ---------------------------------------------------------------------------
def interpolate_axis_angle(axis_angles, t_original, t_target):
    """Interpolate axis-angle rotations using SciPy's Slerp with extrapolation."""
    t_min, t_max = t_original.min(), t_original.max()
    mask_valid = (t_target >= t_min) & (t_target <= t_max)
    t_valid = t_target[mask_valid]
    t_low = t_target[t_target < t_min]
    t_high = t_target[t_target > t_max]

    rots = Rotation.from_rotvec(axis_angles)
    if len(t_valid) > 0:
        slerp = Slerp(t_original, rots)
        interp_rots = slerp(t_valid)
    else:
        interp_rots = Rotation.from_rotvec(np.zeros((0, 3)))

    if len(t_low) > 0:
        rot_low = Rotation.concatenate([rots[0]] * len(t_low))
        interp_rots = Rotation.concatenate([rot_low, interp_rots])
        t_valid = np.concatenate([t_low, t_valid])

    if len(t_high) > 0:
        rot_high = Rotation.concatenate([rots[-1]] * len(t_high))
        interp_rots = Rotation.concatenate([interp_rots, rot_high])
        t_valid = np.concatenate([t_valid, t_high])

    sort_idx = np.argsort(t_valid)
    return interp_rots[sort_idx].as_rotvec()


def interpolate_world_params_linear(
    original_timestamps: np.ndarray,
    data_timestamps: np.ndarray,
    world_center: Sequence[np.ndarray],
    world_size: Sequence[np.ndarray],
    heading_world: Sequence[np.ndarray],
    world_vertices: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate world-frame parameters with linear and spherical schemes."""
    t_min = np.min(original_timestamps)
    t_original = (original_timestamps - t_min).astype(np.float64) / 1e6
    t_data = (data_timestamps - t_min).astype(np.float64) / 1e6

    sort_idx = np.argsort(t_data)
    t_data_sorted = t_data[sort_idx]
    world_center_sorted = np.asarray(world_center)[sort_idx]
    world_size_sorted = np.asarray(world_size)[sort_idx]
    heading_sorted = np.asarray(heading_world)[sort_idx]
    world_vertices_sorted = np.asarray(world_vertices)[sort_idx]

    interp_center = np.zeros((len(t_original), 3))
    interp_size = np.zeros_like(interp_center)
    for axis in range(3):
        fn_center = interp1d(t_data_sorted, world_center_sorted[:, axis], kind='linear', fill_value='extrapolate')
        fn_size = interp1d(t_data_sorted, world_size_sorted[:, axis], kind='linear', fill_value='extrapolate')
        interp_center[:, axis] = fn_center(t_original)
        interp_size[:, axis] = fn_size(t_original)

    interp_heading = np.zeros((len(t_original), 3))
    for idx, t in enumerate(t_original):
        if t <= t_data_sorted[0]:
            interp_heading[idx] = heading_sorted[0]
        elif t >= t_data_sorted[-1]:
            interp_heading[idx] = heading_sorted[-1]
        else:
            left = np.searchsorted(t_data_sorted, t) - 1
            t0, t1 = t_data_sorted[left], t_data_sorted[left + 1]
            ratio = (t - t0) / (t1 - t0)
            interp_heading[idx] = slerp_unit(heading_sorted[left], heading_sorted[left + 1], ratio)
    interp_heading /= np.linalg.norm(interp_heading, axis=1, keepdims=True) + 1e-8

    interp_vertices = np.zeros((len(t_original), 8, 3))
    for corner in range(8):
        for axis in range(3):
            fn_vertex = interp1d(
                t_data_sorted,
                world_vertices_sorted[:, corner, axis],
                kind='linear',
                fill_value='extrapolate',
            )
            interp_vertices[:, corner, axis] = fn_vertex(t_original)

    return interp_center, interp_size, interp_heading, interp_vertices


def slerp_unit(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit vectors."""
    v0 = v0 / (np.linalg.norm(v0) + 1e-8)
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    theta = np.arccos(dot)
    if abs(theta) < 1e-6:
        return v0
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * v0 + (np.sin(t * theta) / sin_theta) * v1


def interpolate_smpl_params(
    original_timestamps: np.ndarray,
    data_timestamps: np.ndarray,
    data_smpl: Dict[str, np.ndarray],
    data_global_dis: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, np.ndarray]:
    """Interpolate SMPL parameters to the provided timestamp grid."""
    if len(data_timestamps) == 1:
        repeat = len(original_timestamps)
        return {
            'betas': np.repeat(data_smpl['betas'][0][np.newaxis, :], repeat, axis=0),
            'pose_body': np.repeat(data_smpl['pose_body'][0][np.newaxis, :], repeat, axis=0),
            'root_orient': np.repeat(data_smpl['root_orient'][0][np.newaxis, :], repeat, axis=0),
            'trans': np.repeat(data_smpl['trans'][0][np.newaxis, :], repeat, axis=0),
        }

    t_min = np.min(original_timestamps)
    t_original = (original_timestamps - t_min).astype(np.float64) / 1e6
    t_data = (data_timestamps - t_min).astype(np.float64) / 1e6
    sort_idx = np.argsort(t_data)
    t_data_sorted = t_data[sort_idx]

    pose_body_sorted = data_smpl['pose_body'][sort_idx]
    root_orient_sorted = data_smpl['root_orient'][sort_idx]
    trans_sorted = data_smpl['trans'][sort_idx]

    full = {
        'betas': data_smpl['betas'],
        'pose_body': np.zeros((len(t_original), 69)),
        'root_orient': np.zeros((len(t_original), 3)),
        'trans': np.zeros((len(t_original), 3)),
    }

    for axis in range(3):
        t_min_data, t_max_data = t_data_sorted.min(), t_data_sorted.max()
        mask_low = t_original < t_min_data
        mask_high = t_original > t_max_data
        mask_valid = ~(mask_low | mask_high)
        values = np.zeros_like(t_original)

        if data_global_dis is not None:
            trans_data = np.array(data_global_dis)[:, axis]
            values[mask_low] = trans_data[mask_low]
            values[mask_high] = trans_data[mask_high]
        else:
            values[mask_low] = trans_sorted[0, axis]
            values[mask_high] = trans_sorted[-1, axis]

        if np.any(mask_valid):
            fn = interp1d(t_data_sorted, trans_sorted[:, axis], kind='linear', fill_value='extrapolate')
            values[mask_valid] = fn(t_original[mask_valid])
        full['trans'][:, axis] = values

    full['root_orient'] = interpolate_axis_angle(root_orient_sorted, t_data_sorted, t_original)
    for joint in range(23):
        joint_rot = pose_body_sorted[:, joint * 3:(joint + 1) * 3]
        full['pose_body'][:, joint * 3:(joint + 1) * 3] = interpolate_axis_angle(
            joint_rot,
            t_data_sorted,
            t_original,
        )
    return full


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def adjust_range(min_val: float, max_val: float, target: float = 4.0) -> Tuple[float, float]:
    """Expand or shrink a range toward the requested target span."""
    diff = max_val - min_val
    if diff <= target or diff >= target + 2:
        delta = (target - diff) / 2.0
        min_val -= delta
        max_val += delta
    return min_val, max_val


def plot_keypoints_animation(
    object_id: str,
    data: Dict[str, np.ndarray],
    bone_links: np.ndarray,
    smpl_model,
    video_dir: str,
    fps: int = 10,
) -> None:
    """Render and save a SMPL skeleton animation for visual inspection."""
    trans = torch.as_tensor(data['trans'], device=DEVICE)
    trans = trans - trans[0, :].unsqueeze(0)
    betas = torch.as_tensor(np.repeat(data['betas'][0][np.newaxis, :], len(trans), axis=0), device=DEVICE)
    root_orient = torch.as_tensor(data['root_orient'], device=DEVICE)
    pose_body = torch.as_tensor(data['pose_body'], device=DEVICE)

    smpl_output = smpl_model(
        betas=betas,
        body_pose=pose_body,
        global_orient=root_orient,
        transl=trans,
        return_verts=True,
    )
    joints = smpl_output.joints.detach().cpu().numpy().squeeze()[:, :22, :]

    x_min, x_max = joints[:, :, 0].min(), joints[:, :, 0].max()
    y_min, y_max = joints[:, :, 1].min(), joints[:, :, 1].max()
    z_min, z_max = joints[:, :, 2].min(), joints[:, :, 2].max()
    z_max = z_min + 3
    x_min, x_max = adjust_range(x_min, x_max, target=8)
    y_min, y_max = adjust_range(y_min, y_max, target=8)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('motion')

    lines = [ax.plot([], [], [], lw=2, c='b')[0] for _ in range(len(bone_links))]
    traj_line, = ax.plot([], [], [], c='b', lw=2, linestyle='--', label='Trajectory')
    ax.legend()

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        traj_line.set_data([], [])
        traj_line.set_3d_properties([])
        return lines + [traj_line]

    def update(frame):
        skeleton = joints[frame]
        for idx, (start, end) in enumerate(bone_links):
            lines[idx].set_data([skeleton[start, 0], skeleton[end, 0]], [skeleton[start, 1], skeleton[end, 1]])
            lines[idx].set_3d_properties([skeleton[start, 2], skeleton[end, 2]])
        traj_line.set_data(joints[:frame + 1, 0, 0], joints[:frame + 1, 0, 1])
        traj_line.set_3d_properties(np.full(frame + 1, z_min))
        return lines + [traj_line]

    ani = animation.FuncAnimation(fig, update, frames=len(joints), init_func=init, interval=100, blit=True)
    os.makedirs(video_dir, exist_ok=True)
    ani.save(os.path.join(video_dir, f'{object_id}.mp4'), writer='ffmpeg', fps=fps)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data IO helpers
# ---------------------------------------------------------------------------
def load_waymo_entries(pkl_path: str) -> List[Dict]:
    with open(pkl_path, 'rb') as f:
        return pkl.load(f)


def group_waymo_by_object(entries: Sequence[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for entry in entries:
        obj_id = entry['basic_info']['object_id']
        grouped.setdefault(obj_id, []).append(entry)
    for seq in grouped.values():
        seq.sort(key=lambda item: int(item['basic_info']['time_stamp']))
    return grouped


def load_lidarhmr_sequence(object_dir: str) -> List[Dict]:
    npz_files = glob.glob(os.path.join(object_dir, '*.npz'))
    sequence = []
    for npz_file in npz_files:
        with np.load(npz_file, allow_pickle=True) as archive:
            entry = {key: archive[key].item() for key in archive.files}
            sequence.append(entry)
    sequence.sort(key=lambda item: int(item['basic_info']['time_stamp']))
    return sequence


def save_npz_bundle(base_dir: str, category: str, split: str, context: str, object_id: str, payload: Dict) -> None:
    target_dir = os.path.join(base_dir, category, split, context)
    os.makedirs(target_dir, exist_ok=True)
    np.savez(os.path.join(target_dir, f'{object_id}.npz'), **payload)


def save_npy_bundle(base_dir: str, category: str, split: str, context: str, object_id: str, values: Sequence[int]) -> None:
    if not values:
        return
    target_dir = os.path.join(base_dir, category, split, context)
    os.makedirs(target_dir, exist_ok=True)
    np.save(os.path.join(target_dir, f'{object_id}.npy'), np.asarray(values, dtype=np.int64))


def build_fps30_timestamps(waymo_timestamps: Sequence[int]) -> List[int]:
    if not waymo_timestamps:
        return []
    timestamps: List[int] = []
    for idx, current in enumerate(waymo_timestamps):
        current = int(current)
        if idx == 0:
            timestamps.append(current - 33333)
            timestamps.append(current)
            continue
        previous = int(waymo_timestamps[idx - 1])
        delta = max(1, int(round(current - previous, -4) / 33333))
        interpolated = np.linspace(previous, current, delta, endpoint=False)[1:]
        timestamps.extend(interpolated.astype(int))
        timestamps.append(current)
    timestamps.append(int(waymo_timestamps[-1]) + 33333)
    return timestamps


def compute_gt_indices(waymo_obj_data: Sequence[Dict], fps30_timestamps: Sequence[int]) -> List[int]:
    lookup = {ts: idx for idx, ts in enumerate(fps30_timestamps)}
    indices: List[int] = []
    for entry in waymo_obj_data:
        if entry['world_frame']['keypoints'] is None:
            continue
        ts = int(entry['basic_info']['time_stamp'])
        if ts in lookup:
            indices.append(lookup[ts] - 1)
    return indices


def compute_pseudo_gt_indices(waymo_obj_data: Sequence[Dict], fps30_timestamps: Sequence[int]) -> List[int]:
    lookup = {ts: idx for idx, ts in enumerate(fps30_timestamps)}
    indices: List[int] = []
    for entry in waymo_obj_data:
        if entry['world_frame']['keypoints'] is not None:
            continue
        pcd = entry['world_frame']['pedestrian_pcd']
        if pcd is None or len(pcd) == 0:
            continue
        max_point = pcd[np.argmax(pcd[:, 2])]
        min_point = pcd[np.argmin(pcd[:, 2])]
        height = max_point[2] - min_point[2]
        if height > 1.2 and len(pcd) > 100:
            ts = int(entry['basic_info']['time_stamp'])
            if ts in lookup:
                indices.append(lookup[ts] - 1)
    return indices


def select_vertices_key(entry: Dict) -> str:
    if 'bbox_vertices' in entry['world_frame']:
        return 'bbox_vertices'
    return 'world_vertices'


def extract_world_attributes(waymo_obj_data: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    if not waymo_obj_data:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 8, 3)), []
    vertices_key = select_vertices_key(waymo_obj_data[0])
    world_centers = np.array([entry['world_frame']['center'] for entry in waymo_obj_data])
    world_sizes = np.array([entry['world_frame']['size'] for entry in waymo_obj_data])
    heading_world = np.array([entry['world_frame']['heading'] for entry in waymo_obj_data])
    world_vertices = np.array([entry['world_frame'][vertices_key] for entry in waymo_obj_data])
    timestamps = [int(entry['basic_info']['time_stamp']) for entry in waymo_obj_data]
    return world_centers, world_sizes, heading_world, world_vertices, timestamps


def extract_smpl_sequences(lidarhmr_obj_data: Sequence[Dict]) -> Tuple[Dict[str, np.ndarray], List[int], List[np.ndarray]]:
    if not lidarhmr_obj_data:
        return {
            'root_orient': np.empty((0, 3)),
            'pose_body': np.empty((0, 69)),
            'trans': np.empty((0, 3)),
            'betas': np.empty((0, 10)),
        }, [], []
    root_orient = np.array([item['SMPL_lidarhmr']['root_orient'] for item in lidarhmr_obj_data])
    pose_body = np.array([item['SMPL_lidarhmr']['pose_body'] for item in lidarhmr_obj_data])
    trans = np.array([
        item['SMPL_lidarhmr']['trans'] + item['SMPL_lidarhmr']['global_dis']
        for item in lidarhmr_obj_data
    ])
    betas = np.array([item['SMPL_lidarhmr']['betas'] for item in lidarhmr_obj_data])
    timestamps = [int(item['basic_info']['time_stamp']) for item in lidarhmr_obj_data]
    smpl_global = [item['SMPL_lidarhmr']['global_dis'] for item in lidarhmr_obj_data]
    return {
        'root_orient': root_orient,
        'pose_body': pose_body,
        'trans': trans,
        'betas': betas,
    }, timestamps, smpl_global


# ---------------------------------------------------------------------------
# Core processing pipeline
# ---------------------------------------------------------------------------
def process_object(
    split: str,
    context_name: str,
    object_id: str,
    waymo_obj_data: Sequence[Dict],
    lidarhmr_obj_data: Sequence[Dict],
    save_root: str,
    smpl_model,
    video_dir: Optional[str],
) -> None:
    world_centers, world_sizes, heading_world, world_vertices, waymo_timestamps = extract_world_attributes(waymo_obj_data)
    smpl_single, smpl_timestamps, smpl_global_position = extract_smpl_sequences(lidarhmr_obj_data)

    if len(waymo_timestamps) == 0 or len(smpl_timestamps) == 0:
        return

    raw_payload = {
        'betas': smpl_single['betas'],
        'pose_body': smpl_single['pose_body'],
        'root_orient': smpl_single['root_orient'],
        'trans': smpl_single['trans'],
        'smpl_timestamps': smpl_timestamps,
        'waymo_timestamps': waymo_timestamps,
        'world_center': world_centers,
        'world_size': world_sizes,
        'heading_world': heading_world,
        'world_vertices': world_vertices,
    }
    save_npz_bundle(save_root, 'raw', split, context_name, object_id, raw_payload)

    if len(smpl_timestamps) >= 2:
        smpl_single['root_orient'] = correct_smpl_global_orient(
            smpl_single['trans'],
            smpl_single['root_orient'],
        )

    raw_frenet_payload = {
        'betas': smpl_single['betas'],
        'pose_body': smpl_single['pose_body'],
        'root_orient': smpl_single['root_orient'],
        'trans': smpl_single['trans'],
        'smpl_timestamps': smpl_timestamps,
        'waymo_timestamps': waymo_timestamps,
        'world_center': world_centers,
        'world_size': world_sizes,
        'heading_world': heading_world,
        'world_vertices': world_vertices,
    }
    save_npz_bundle(save_root, 'raw_frenet', split, context_name, object_id, raw_frenet_payload)

    first_interp = interpolate_smpl_params(
        np.array(waymo_timestamps),
        np.array(smpl_timestamps),
        smpl_single,
        world_centers,
    )

    interpolated_payload = {
        'betas': first_interp['betas'],
        'pose_body': first_interp['pose_body'],
        'root_orient': first_interp['root_orient'],
        'trans': first_interp['trans'],
        'smpl_timestamps': smpl_timestamps,
        'waymo_timestamps': waymo_timestamps,
        'world_center': world_centers,
        'world_size': world_sizes,
        'heading_world': heading_world,
        'world_vertices': world_vertices,
    }
    save_npz_bundle(save_root, 'interpolated_empty', split, context_name, object_id, interpolated_payload)

    fps30_timestamps = build_fps30_timestamps(waymo_timestamps)
    if not fps30_timestamps:
        return

    interp_30fps = interpolate_smpl_params(
        np.array(fps30_timestamps),
        np.array(waymo_timestamps),
        first_interp,
    )

    new_center, new_size, new_heading, new_vertices = interpolate_world_params_linear(
        np.array(fps30_timestamps),
        np.array(waymo_timestamps),
        world_centers,
        world_sizes,
        heading_world,
        world_vertices,
    )

    interp_30fps['smpl_timestamps'] = smpl_timestamps
    interp_30fps['waymo_timestamps'] = waymo_timestamps
    interp_30fps['fps30_timestamps'] = fps30_timestamps
    interp_30fps['waymo_global_position'] = list(world_centers)
    interp_30fps['smpl_global_position'] = smpl_global_position
    interp_30fps['first_frame_trans'] = interp_30fps['trans'][0]
    interp_30fps['world_center'] = new_center
    interp_30fps['world_size'] = new_size
    interp_30fps['world_vertices'] = new_vertices
    interp_30fps['heading_world'] = new_heading

    save_npz_bundle(save_root, '30fps', split, context_name, object_id, interp_30fps)

    gt_indices = compute_gt_indices(waymo_obj_data, fps30_timestamps)
    save_npy_bundle(save_root, '30fps_gt_index', split, context_name, object_id, gt_indices)

    pseudo_indices = compute_pseudo_gt_indices(waymo_obj_data, fps30_timestamps)
    save_npy_bundle(save_root, '30fps_pesudo_gt_index', split, context_name, object_id, pseudo_indices)

    if video_dir:
        plot_keypoints_animation(object_id, raw_payload, SMPL_BONE_LINK, smpl_model, video_dir)


def process_context(
    split: str,
    context_path: str,
    waymo_split_dir: str,
    save_root: str,
    smpl_model,
    video_dir: Optional[str],
) -> None:
    context_name = os.path.basename(context_path)
    waymo_pkl = os.path.join(waymo_split_dir, f'{context_name}.pkl')
    if not os.path.exists(waymo_pkl):
        print(f"[WARN] Missing Waymo pickle for context {context_name} in split {split}")
        return

    waymo_entries = load_waymo_entries(waymo_pkl)
    waymo_by_object = group_waymo_by_object(waymo_entries)
    object_dirs = glob.glob(os.path.join(context_path, '*'))

    for obj_dir in object_dirs:
        object_id = os.path.basename(obj_dir)
        if object_id not in waymo_by_object:
            continue
        lidar_sequence = load_lidarhmr_sequence(obj_dir)
        if not lidar_sequence:
            continue
        process_object(
            split,
            context_name,
            object_id,
            waymo_by_object[object_id],
            lidar_sequence,
            save_root,
            smpl_model,
            video_dir,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert LiDAR-HMR outputs into AMASS format')
    parser.add_argument('--waymo-path', default='/dataset/guangxun/test/waymo_extracted', help='Directory containing extracted Waymo pickles')
    parser.add_argument('--lidarhmr-path', default='/dataset/guangxun/test/lidar_hmr_results', help='Directory containing LiDAR-HMR npz outputs')
    parser.add_argument('--save-path', default='/dataset/guangxun/test/amass_format', help='Destination root for AMASS-style artifacts')
    parser.add_argument('--splits', nargs='+', default=['training', 'validation'], help='Dataset splits to process')
    parser.add_argument('--smpl-model-dir', default='LiDAR-HMR/smplx_models', help='Directory with SMPL model files')
    parser.add_argument('--gender', default='neutral', choices=['neutral', 'male', 'female'], help='SMPL gender to instantiate')
    parser.add_argument('--video-dir', default=None, help='Optional directory to dump MP4 visualizations')
    parser.add_argument('--video-fps', type=int, default=10, help='FPS for saved visualization videos')
    return parser.parse_args()


def build_smpl_model(model_dir: str, gender: str):
    return smplx.create(model_dir, model_type='smpl', gender=gender, use_pca=False).to(DEVICE)


def main() -> None:
    args = parse_args()
    smpl_model = build_smpl_model(args.smpl_model_dir, args.gender)

    for split in args.splits:
        print(f'Processing split: {split}')
        waymo_split_dir = os.path.join(args.waymo_path, split)
        lidar_split_dir = os.path.join(args.lidarhmr_path, split)
        context_dirs = glob.glob(os.path.join(lidar_split_dir, '*'))
        for context_path in tqdm(context_dirs, desc=f'Split {split}'):
            process_context(split, context_path, waymo_split_dir, args.save_path, smpl_model, args.video_dir)
        print(f'Finished split: {split}')


if __name__ == '__main__':
    main()
