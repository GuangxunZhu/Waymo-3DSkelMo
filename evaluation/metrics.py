"""Unified evaluation metrics for generated Waymo human motions."""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pickle as pkl
import torch
import smplx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WAYMO_KEYPOINT_NAME_TO_ID = {
    "UNSPECIFIED": 0,
    "NOSE": 1,
    "LEFT_SHOULDER": 5,
    "LEFT_ELBOW": 6,
    "LEFT_WRIST": 7,
    "LEFT_HIP": 8,
    "LEFT_KNEE": 9,
    "LEFT_ANKLE": 10,
    "RIGHT_SHOULDER": 13,
    "RIGHT_ELBOW": 14,
    "RIGHT_WRIST": 15,
    "RIGHT_HIP": 16,
    "RIGHT_KNEE": 17,
    "RIGHT_ANKLE": 18,
    "FOREHEAD": 19,
    "HEAD": 20,
}

SMPL_JOINT_NAME_TO_ID = {
    "PELVIS": 0,
    "LEFT_HIP": 1,
    "RIGHT_HIP": 2,
    "SPINE1": 3,
    "LEFT_KNEE": 4,
    "RIGHT_KNEE": 5,
    "SPINE2": 6,
    "LEFT_ANKLE": 7,
    "RIGHT_ANKLE": 8,
    "SPINE3": 9,
    "LEFT_FOOT": 10,
    "RIGHT_FOOT": 11,
    "NECK": 12,
    "LEFT_COLLAR": 13,
    "RIGHT_COLLAR": 14,
    "HEAD": 15,
    "LEFT_SHOULDER": 16,
    "RIGHT_SHOULDER": 17,
    "LEFT_ELBOW": 18,
    "RIGHT_ELBOW": 19,
    "LEFT_WRIST": 20,
    "RIGHT_WRIST": 21,
}

COMMON_JOINTS = [
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
    "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
    "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
    "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", "HEAD",
]

WAYMO_IDS = {name: WAYMO_KEYPOINT_NAME_TO_ID[name] for name in COMMON_JOINTS}
SMPL_IDS = {name: SMPL_JOINT_NAME_TO_ID[name] for name in COMMON_JOINTS}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_device(user_choice: Optional[str] = None) -> torch.device:
    if user_choice:
        return torch.device(user_choice)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_smpl_model(model_root: str, device: torch.device) -> smplx.SMPL:
    model = smplx.create(model_path=model_root, model_type='smpl')
    return model.to(device)


def zero_betas(num_frames: int) -> np.ndarray:
    return np.zeros((num_frames, 10), dtype=np.float32)


def run_smpl_sequence(
    model: smplx.SMPL,
    device: torch.device,
    trans: np.ndarray,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    betas: Optional[np.ndarray] = None,
) -> np.ndarray:
    betas = betas if betas is not None else zero_betas(len(trans))
    params = {
        'betas': torch.tensor(betas, device=device, dtype=torch.float32),
        'body_pose': torch.tensor(pose_body, device=device, dtype=torch.float32),
        'global_orient': torch.tensor(root_orient, device=device, dtype=torch.float32),
        'transl': torch.tensor(trans, device=device, dtype=torch.float32),
    }
    with torch.no_grad():
        output = model(**params)
    joints = output.joints.detach().cpu().numpy()
    if joints.ndim == 2:
        joints = joints[None, :, :]
    return joints[:, :22, :]


def load_waymo_tracks(waymo_root: str, split: str, context_name: str) -> Dict[str, List[dict]]:
    file_path = os.path.join(waymo_root, split, f'{context_name}.pkl')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Missing Waymo pickle: {file_path}')
    data = pkl.load(open(file_path, 'rb'))
    mapping: Dict[str, List[dict]] = defaultdict(list)
    for entry in data:
        obj_id = entry['basic_info']['object_id']
        mapping[obj_id].append(entry)
    for track in mapping.values():
        track.sort(key=lambda x: int(x['basic_info']['time_stamp']))
    return mapping


def iter_generated_files(root: str, splits: Sequence[str]) -> Iterable[Tuple[str, str, str]]:
    for split in splits:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for dirpath, _, filenames in os.walk(split_dir):
            if not filenames:
                continue
            context = os.path.basename(dirpath)
            for filename in filenames:
                if filename.endswith('.npz'):
                    yield split, context, os.path.join(dirpath, filename)


def joints_in_obb_ratio(joints: np.ndarray, corners: np.ndarray, eps: float = 1e-6) -> float:
    joints = np.asarray(joints, dtype=np.float64)
    corners = np.asarray(corners, dtype=np.float64)
    center = corners.mean(axis=0)
    c0 = corners[0]
    d = np.linalg.norm(corners - c0, axis=1)
    nn = np.argsort(d)[1:4]
    e1, e2, e3 = corners[nn[0]] - c0, corners[nn[1]] - c0, corners[nn[2]] - c0
    a1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2_ = e2 - np.dot(e2, a1) * a1
    a2 = e2_ / (np.linalg.norm(e2_) + 1e-12)
    a3 = np.cross(a1, a2)
    a3 = a3 / (np.linalg.norm(a3) + 1e-12)
    if np.dot(e3, a3) < 0:
        a3 = -a3
    X = corners - center
    h1 = np.max(np.abs(X @ a1))
    h2 = np.max(np.abs(X @ a2))
    h3 = np.max(np.abs(X @ a3))
    Y = joints - center
    p1 = Y @ a1
    p2 = Y @ a2
    p3 = Y @ a3
    inside = (np.abs(p1) <= h1 + eps) & (np.abs(p2) <= h2 + eps) & (np.abs(p3) <= h3 + eps)
    return inside.mean()


def compute_motion_jitter(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    F, J, _ = pts.shape
    mask = ~np.isnan(pts).any(axis=(1, 2))
    clean_pts = pts[mask]
    if clean_pts.shape[0] < 3:
        return np.zeros((0, J)), 0.0
    vel = np.diff(clean_pts, axis=0)
    acc = np.diff(vel, axis=0)
    jitter_per_frame = np.linalg.norm(acc, axis=2)
    overall_jitter = float(jitter_per_frame.mean()) if jitter_per_frame.size else 0.0
    return jitter_per_frame, overall_jitter

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate_mpjpe(
    generated_root: str,
    waymo_root: str,
    model: smplx.SMPL,
    device: torch.device,
    splits: Sequence[str],
) -> Tuple[float, Dict[str, float]]:
    sum_error: MutableMapping[str, float] = defaultdict(float)
    count: MutableMapping[str, int] = defaultdict(int)
    frame_errors: List[float] = []

    for split in splits:
        split_dir = os.path.join(generated_root, split)
        if not os.path.isdir(split_dir):
            continue
        for dirpath, _, filenames in os.walk(split_dir):
            if not filenames:
                continue
            context = os.path.basename(dirpath)
            try:
                waymo_tracks = load_waymo_tracks(waymo_root, split, context)
            except FileNotFoundError:
                continue
            for filename in filenames:
                if not filename.endswith('.npz'):
                    continue
                file_path = os.path.join(dirpath, filename)
                data = np.load(file_path, allow_pickle=True)
                object_id = os.path.splitext(filename)[0]
                track = waymo_tracks.get(object_id)
                if not track:
                    continue
                trans = data['trans']
                root_orient = data['root_orient']
                pose_body = data['pose_body']
                joints = run_smpl_sequence(model, device, trans, root_orient, pose_body)
                keypoints = [entry['world_frame']['keypoints'] for entry in track]
                keypoint_types = [entry['world_frame']['keypoint_type'] for entry in track]
                for idx, kp in enumerate(keypoints):
                    if kp is None or idx >= len(joints):
                        continue
                    kp_type = keypoint_types[idx]
                    if kp_type is None:
                        continue
                    kp_type_list = kp_type.tolist()
                    joint_errors: List[float] = []
                    for name in COMMON_JOINTS:
                        wid = WAYMO_IDS[name]
                        sid = SMPL_IDS[name]
                        if wid not in kp_type_list:
                            continue
                        w_idx = kp_type_list.index(wid)
                        waymo_joint = kp[w_idx]
                        smpl_joint = joints[idx, sid]
                        err = float(np.linalg.norm(waymo_joint - smpl_joint))
                        sum_error[name] += err
                        count[name] += 1
                        joint_errors.append(err)
                    if joint_errors:
                        frame_errors.append(float(np.mean(joint_errors)))

    per_joint_mpjpe = {name: (sum_error[name] / count[name]) if count[name] else None for name in COMMON_JOINTS}
    overall_mpjpe = float(np.mean(frame_errors)) if frame_errors else float('nan')
    return overall_mpjpe, per_joint_mpjpe


def evaluate_motion_jitter(
    generated_root: str,
    model: smplx.SMPL,
    device: torch.device,
    splits: Sequence[str],
) -> float:
    jitter_values: List[float] = []
    for split, context, file_path in iter_generated_files(generated_root, splits):
        data = np.load(file_path, allow_pickle=True)
        trans = data['trans']
        root_orient = data['root_orient']
        pose_body = data['pose_body']
        joints = run_smpl_sequence(model, device, trans, root_orient, pose_body)
        _, jitter_val = compute_motion_jitter(joints[:, :22, :])
        if jitter_val:
            jitter_values.append(jitter_val)
    return float(np.mean(jitter_values)) if jitter_values else float('nan')


def evaluate_keypoint_bbox_overlap(
    generated_root: str,
    waymo_root: str,
    model: smplx.SMPL,
    device: torch.device,
    splits: Sequence[str],
) -> float:
    ratios: List[float] = []
    for split in splits:
        split_dir = os.path.join(generated_root, split)
        if not os.path.isdir(split_dir):
            continue
        for context in os.listdir(split_dir):
            context_path = os.path.join(split_dir, context)
            if not os.path.isdir(context_path):
                continue
            try:
                waymo_tracks = load_waymo_tracks(waymo_root, split, context)
            except FileNotFoundError:
                continue
            for object_id, track in waymo_tracks.items():
                keypoints_file = os.path.join(context_path, f'{object_id}.npz')
                if not os.path.exists(keypoints_file):
                    continue
                keypoints_data = np.load(keypoints_file, allow_pickle=True)
                trans = keypoints_data['trans']
                root_orient = keypoints_data['root_orient']
                pose_body = keypoints_data['pose_body']
                joints = run_smpl_sequence(model, device, trans, root_orient, pose_body)
                waymo_ts = np.array([entry['basic_info']['time_stamp'] for entry in track], dtype=np.int64)
                idx_lookup = {ts: idx for idx, ts in enumerate(waymo_ts)}
                data_ts = keypoints_data['waymo_timestamps']
                world_centers = [entry['world_frame']['center'] for entry in track]
                world_vertices = [entry['world_frame']['bbox_vertices'] for entry in track]
                for frame_idx, ts in enumerate(data_ts):
                    if frame_idx >= len(joints):
                        break
                    idx = idx_lookup.get(int(ts))
                    if idx is None:
                        continue
                    ratio = joints_in_obb_ratio(joints[frame_idx], world_vertices[idx])
                    ratios.append(ratio)
    return float(np.mean(ratios)) if ratios else float('nan')

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run evaluation metrics for generated motions.')
    parser.add_argument('--generated-root', default='/dataset/guangxun/test/generated/final_10fps', help='Directory containing generated motion npz files')
    parser.add_argument('--waymo-root', default='/dataset/guangxun/test/waymo_extracted', help='Directory containing extracted Waymo pickles')
    parser.add_argument('--model-root', default='./LiDAR-HMR/smplx_models', help='SMPL model directory')
    parser.add_argument('--device', default=None, help='Torch device override, e.g. cuda:0 or cpu')
    parser.add_argument('--splits', nargs='+', default=['training', 'validation'], help='Dataset splits to evaluate')
    parser.add_argument('--metrics', nargs='+', choices=['mpjpe', 'jitter', 'kp-box'], default=['mpjpe', 'jitter', 'kp-box'], help='Metrics to run')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f'[INFO] Using device: {device}')
    model = build_smpl_model(args.model_root, device)

    if 'mpjpe' in args.metrics:
        overall, per_joint = evaluate_mpjpe(args.generated_root, args.waymo_root, model, device, args.splits)
        print(f'[MPJPE] Overall: {overall:.4f} m' if not np.isnan(overall) else '[MPJPE] Overall: NaN')
        for name in COMMON_JOINTS:
            value = per_joint[name]
            if value is None:
                print(f'    {name:15s}: no detections')
            else:
                print(f'    {name:15s}: {value:.4f} m')

    if 'jitter' in args.metrics:
        jitter = evaluate_motion_jitter(args.generated_root, model, device, args.splits)
        if np.isnan(jitter):
            print('[Jitter] Not enough valid sequences to compute jitter.')
        else:
            print(f'[Jitter] Mean motion jitter: {jitter:.6f}')

    if 'kp-box' in args.metrics:
        ratio = evaluate_keypoint_bbox_overlap(args.generated_root, args.waymo_root, model, device, args.splits)
        if np.isnan(ratio):
            print('[KP-Box] Not enough data to compute keypoint bbox ratio.')
        else:
            print(f'[KP-Box] Mean joint-in-bbox ratio: {ratio:.4f}')


if __name__ == '__main__':
    main()
