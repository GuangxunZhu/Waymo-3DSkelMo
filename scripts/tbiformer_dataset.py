"""Dataset builder for TBIFormer experiments."""
from __future__ import annotations

import argparse
import copy
import os
from bisect import bisect_left
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import smplx
import torch
from scipy.interpolate import interp1d
from sklearn.neighbors import KDTree
from tqdm import tqdm

def make_groups(pedestrians: np.ndarray, num_persons: int) -> List[Tuple[int, ...]]:
        """Group nearby pedestrians using k-nearest neighbors."""
        tree = KDTree(pedestrians, leaf_size=40)
        _, indices = tree.query(pedestrians, k=num_persons)
        return [tuple(idx_list) for idx_list in indices]

def max_pairwise_distance(group_data: Sequence[np.ndarray]) -> float:
    """Return the maximum pelvis distance across all people and frames."""
    max_dist = 0.0
    num_people = len(group_data)
    if num_people < 2:
        return max_dist
    num_frames = group_data[0].shape[0]
    for t in range(num_frames):
        roots = np.array([person[t, 0] for person in group_data])
        diff = roots[:, np.newaxis] - roots
        dist_matrix = np.linalg.norm(diff, axis=2)
        upper_triangle = dist_matrix[np.triu_indices(num_people, k=1)]
        if upper_triangle.size > 0:
            max_dist = max(max_dist, float(np.max(upper_triangle)))
    return max_dist

def downsample_joints(original_times: np.ndarray, joints: np.ndarray, target_fps: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample joints to the target fps using linear interpolation."""
    interval = int(1e6 / target_fps)
    new_times = np.arange(original_times[0], original_times[-1] + interval, interval).astype(int)
    new_joints = []
    for nt in new_times:
        pos = bisect_left(original_times, nt)
        if pos == 0:
            new_joints.append(joints[0])
        elif pos >= len(original_times):
            new_joints.append(joints[-1])
        else:
            t_prev = original_times[pos - 1]
            t_next = original_times[pos]
            alpha = (nt - t_prev) / (t_next - t_prev)
            interp_joints = (1 - alpha) * joints[pos - 1] + alpha * joints[pos]
            new_joints.append(interp_joints)
    return np.array(new_joints), new_times

def approximate_intersect(ts_list: Sequence[np.ndarray], tolerance: int = 10) -> np.ndarray:
    """Return fuzzy intersection of timestamp arrays (values from first array)."""
    if not ts_list:
        return np.array([])
    overlap = ts_list[0]
    for ts in ts_list[1:]:
        diff = np.abs(overlap[:, np.newaxis] - ts)
        mask = np.any(diff <= tolerance, axis=1)
        overlap = overlap[mask]
    return overlap

def augment_scene(
    scene: np.ndarray,
    angles: np.ndarray = np.arange(0, 360, 10),
    scales: Sequence[float] = (0.9, 1.0, 1.1),
    noise_std: float = 0.01,
) -> List[np.ndarray]:
    """Augment one scene with rotations, scales, and Gaussian noise."""
    P, T, D = scene.shape
    J = D // 3
    xyz = scene.reshape(P, T, J, 3)
    augmented: List[np.ndarray] = []
    for angle in angles:
        theta = np.deg2rad(angle)
        cos, sin = np.cos(theta), np.sin(theta)
        Ry = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
        rotated = xyz @ Ry.T
        for scale in scales:
            scaled = rotated * scale
            noise = np.random.normal(scale=noise_std, size=scaled.shape)
            augmented.append((scaled + noise).reshape(P, T, D))
    return augmented

def augment_dataset(data: np.ndarray) -> np.ndarray:
    """Apply augmentations scene by scene (keeps original + aug versions)."""
    out = []
    for scene in data:
        out.append(scene)
        out.extend(augment_scene(scene))
    return np.stack(out, axis=0)

def load_scene_joints(
    split_dir: str,
    smpl_model: smplx.SMPL,
    device: torch.device,
    keep_joints: Sequence[int],
    joint_reorder: Sequence[int],
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    multi_person = defaultdict(list)
    timestamps = defaultdict(list)
    for context_name in tqdm(os.listdir(split_dir), desc='Contexts'):
        context_path = os.path.join(split_dir, context_name)
        for file_name in os.listdir(context_path):
            if not file_name.endswith('.npz'):
                continue
            file_path = os.path.join(context_path, file_name)
            try:
                data = np.load(file_path, allow_pickle=True)
                seq_len = len(data['trans'])
                params = {
                    'betas': torch.zeros((seq_len, 10), device=device, dtype=torch.float32),
                    'body_pose': torch.tensor(data['pose_body'], device=device, dtype=torch.float32),
                    'global_orient': torch.tensor(data['root_orient'], device=device, dtype=torch.float32),
                    'transl': torch.tensor(data['trans'], device=device, dtype=torch.float32),
                }
                with torch.no_grad():
                    joints = smpl_model(**params).joints.detach().cpu().numpy().squeeze()[:, :22, :]
                subset = joints[:, keep_joints][:, joint_reorder]
                multi_person[context_name].append(subset)
                timestamps[context_name].append(np.array(data['waymo_timestamps']))
            except Exception as exc:  # pylint: disable=broad-except
                print(f'Error processing {file_path}: {exc}')
    return multi_person, timestamps

def window_single_person(person_data: np.ndarray, data_length: int, step: int) -> List[np.ndarray]:
    windows = []
    num_frames = len(person_data)
    for start in range(0, num_frames - data_length + 1, step):
        window = copy.deepcopy(person_data[start : start + data_length])
        mean_xyz = np.mean(window[:, 0, :], axis=0)
        window -= mean_xyz
        window[:, :, 2] += 0.9
        windows.append(window)
    return windows


def interpolate_window(window: np.ndarray, target_frames: int = 75) -> np.ndarray:
    t_old = np.arange(window.shape[0])
    t_new = np.linspace(0, window.shape[0] - 1, target_frames)
    upsampled = np.zeros((target_frames, window.shape[1], 3), dtype=window.dtype)
    for joint_idx in range(window.shape[1]):
        for axis in range(3):
            f_interp = interp1d(t_old, window[:, joint_idx, axis], kind='cubic')
            upsampled[:, joint_idx, axis] = f_interp(t_new)
    return upsampled


def process_scene_windows(
    multi_person: Dict[str, List[np.ndarray]],
    timestamps: Dict[str, List[np.ndarray]],
    num_person: int,
    data_length: int,
    step: int,
    max_distance: float,
) -> List[List[np.ndarray]]:
    final_data: List[List[np.ndarray]] = []
    for context_name, persons_data in tqdm(multi_person.items(), desc='Scenes'):
        persons_time = timestamps[context_name]
        if len(persons_data) != num_person:
            continue
        if num_person == 1:
            for person_joints in persons_data:
                if len(person_joints) < data_length:
                    continue
                for window in window_single_person(person_joints, data_length, step):
                    final_data.append([window])
            continue
        overlap = persons_time[0]
        for ts in persons_time[1:]:
            overlap = np.intersect1d(overlap, ts)
        if len(overlap) < data_length:
            continue
        for start in range(0, len(overlap) - data_length + 1, step):
            window_ts = overlap[start : start + data_length]
            if not np.all(np.diff(window_ts) <= np.min(np.diff(window_ts)) + 5e4):
                continue
            window_data = []
            for person_idx in range(num_person):
                person_ts = persons_time[person_idx]
                idx = [j for j, t in enumerate(person_ts) if t in window_ts]
                window_data.append(persons_data[person_idx][idx])
            group_data = np.concatenate(window_data, axis=0)
            mean_xyz = np.mean(group_data[:, 0, :], axis=0)
            window_data = [data - mean_xyz + np.array([0, 0, 0.9]) for data in window_data]
            if max_pairwise_distance(window_data) > max_distance:
                continue
            upsampled = [interpolate_window(person_window) for person_window in window_data]
            final_data.append(upsampled)
    return final_data


def generate_dataset(
    motion_data_path: str,
    output_root: str,
    splits: Sequence[str],
    num_person: int,
    data_length: int,
    step: int,
    keep_joints: Sequence[int],
    joint_reorder: Sequence[int],
    max_distance: float,
) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl_model = smplx.create(model_path='./LiDAR-HMR/smplx_models', model_type='smpl').to(device)
    for split in splits:
        split_dir = os.path.join(motion_data_path, split)
        multi_person, timestamps = load_scene_joints(split_dir, smpl_model, device, keep_joints, joint_reorder)
        windows = process_scene_windows(multi_person, timestamps, num_person, data_length, step, max_distance)
        if not windows:
            print(f'[WARN] No usable windows for {split}.')
            continue
        final_array = np.array(windows, dtype=np.float32)
        final_array = final_array[..., [0, 2, 1]].reshape(len(final_array), num_person, 75, -1)
        augmented = augment_dataset(final_array)
        split_output = os.path.join(output_root, f'{split}_{num_person}_{75}_{len(keep_joints)}.npy')
        os.makedirs(os.path.dirname(split_output), exist_ok=True)
        np.save(split_output, augmented)
        print(f'[INFO] Saved {split} dataset → {split_output}, shape={augmented.shape}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build TBIFormer dataset from SMPL sequences.')
    parser.add_argument('--input-root', default='/dataset/guangxun/test/generated_1/final_10fps', help='Directory with generated sequences')
    parser.add_argument('--output-root', default='/dataset/guangxun/test/joints_10fps', help='Directory to store joint datasets')
    parser.add_argument('--splits', nargs='+', default=['training', 'validation'], help='Dataset splits to process')
    parser.add_argument('--num-person', type=int, default=30, help='Number of persons per scene')
    parser.add_argument('--data-length', type=int, default=30, help='Sliding window length (frames)')
    parser.add_argument('--step', type=int, default=20, help='Sliding window stride')
    parser.add_argument('--max-distance', type=float, default=18.0, help='Maximum pelvis distance allowed in a scene')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_joints = [0, 1, 2, 4, 5, 7, 8, 9, 12, 16, 17, 18, 19, 20, 21]
    joint_reorder = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13]
    generate_dataset(
        motion_data_path=args.input_root,
        output_root=args.output_root,
        splits=args.splits,
        num_person=args.num_person,
        data_length=args.data_length,
        step=args.step,
        keep_joints=keep_joints,
        joint_reorder=joint_reorder,
        max_distance=args.max_distance,
    )


if __name__ == '__main__':
    main()




