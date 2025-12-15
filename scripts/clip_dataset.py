"""Clip AMASS-format sequences into fixed-length chunks and optionally regroup them."""
from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def make_symlink(src: str, dst: str, overwrite: bool = True, use_relative: bool = True) -> None:
    """Create (optionally relative) symlinks, overwriting existing targets."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if overwrite and os.path.lexists(dst):
        os.unlink(dst)
    link_src = os.path.relpath(src, start=os.path.dirname(dst)) if use_relative else src
    os.symlink(link_src, dst)


def parse_path_components(npz_path: str) -> Tuple[str, str, str]:
    """Return (split_name, context_name, object_id) for a dataset npz path."""
    object_id = os.path.splitext(os.path.basename(npz_path))[0]
    context_name = os.path.basename(os.path.dirname(npz_path))
    split_name = os.path.basename(os.path.dirname(os.path.dirname(npz_path)))
    return split_name, context_name, object_id


# ---------------------------------------------------------------------------
# Index utilities
# ---------------------------------------------------------------------------
def load_index_file(base_dir: str, split: str, context: str, object_id: str) -> Optional[np.ndarray]:
    path = os.path.join(base_dir, split, context, f'{object_id}.npy')
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def collect_local_indices(global_indices: Optional[Sequence[int]], start: int, end: int, valid_length: int) -> np.ndarray:
    if global_indices is None:
        return np.empty((0,), dtype=np.int32)
    collected: List[int] = []
    for idx in global_indices:
        idx_int = int(idx)
        if start <= idx_int < end:
            local_idx = idx_int - start
            if local_idx < valid_length:
                collected.append(local_idx)
    return np.asarray(collected, dtype=np.int32)


# ---------------------------------------------------------------------------
# Clipping helpers
# ---------------------------------------------------------------------------
def build_clip_indices(start: int, clip_frames: int, total_frames: int) -> Tuple[np.ndarray, int, int]:
    """Return (indices_with_padding, valid_length, exclusive_end)."""
    end = start + clip_frames
    if end <= total_frames:
        indices = np.arange(start, end)
        valid_length = clip_frames
    else:
        valid_part = np.arange(start, total_frames)
        padding = np.full(end - total_frames, total_frames - 1, dtype=int)
        indices = np.concatenate([valid_part, padding])
        valid_length = len(valid_part)
    return indices, valid_length, end


def clip_sequence(
    dataset_dir: str,
    data_splits: Sequence[str],
    save_dir: str,
    gt_pose_index_dir: str,
    pseudo_gt_index_dir: str,
    clip_gt_index_dir: str,
    clip_pseudo_index_dir: str,
    clip_frames: int,
    overlap: int,
) -> None:
    step = max(1, clip_frames - overlap)
    for split in data_splits:
        pattern = os.path.join(dataset_dir, split, '*/*.npz')
        npz_files = glob.glob(pattern)
        for npz_path in tqdm(npz_files, desc=f'Clipping {split}'):
            split_name, context_name, object_name = parse_path_components(npz_path)
            gt_indices = load_index_file(gt_pose_index_dir, split_name, context_name, object_name)
            pseudo_indices = load_index_file(pseudo_gt_index_dir, split_name, context_name, object_name)

            data = np.load(npz_path)
            root_orient = data['root_orient']
            pose_body = data['pose_body']
            trans = data['trans']
            contacts = data['contacts']
            betas = data['betas']
            world_center = data['world_center']
            world_size = data['world_size']
            world_vertices = data['world_vertices']

            poses = np.concatenate((root_orient, pose_body, np.zeros((len(pose_body), 6))), axis=1)
            clip_idx = 0

            for start in range(0, len(poses), step):
                clip_indices, valid_length, end = build_clip_indices(start, clip_frames, len(poses))
                valid_indices = np.arange(valid_length, dtype=np.int32)

                clip_payload = {
                    'poses': poses[clip_indices],
                    'trans': trans[clip_indices],
                    'contacts': contacts[clip_indices],
                    'betas': betas,
                    'valid_indices': valid_indices,
                    'world_center': world_center[clip_indices],
                    'world_size': world_size[clip_indices],
                    'world_vertices': world_vertices[clip_indices],
                }

                output_dir = os.path.join(save_dir, split_name, context_name)
                os.makedirs(output_dir, exist_ok=True)
                clip_name = f'{object_name}-{clip_idx}'
                np.savez(os.path.join(output_dir, f'{clip_name}.npz'), **clip_payload)

                clip_gt = collect_local_indices(gt_indices, start, end, valid_length)
                if clip_gt.size:
                    target = os.path.join(clip_gt_index_dir, split_name, context_name)
                    os.makedirs(target, exist_ok=True)
                    np.save(os.path.join(target, f'{clip_name}.npy'), clip_gt)

                clip_pseudo = collect_local_indices(pseudo_indices, start, end, valid_length)
                if clip_pseudo.size:
                    target = os.path.join(clip_pseudo_index_dir, split_name, context_name)
                    os.makedirs(target, exist_ok=True)
                    np.save(os.path.join(target, f'{clip_name}.npy'), clip_pseudo)

                clip_idx += 1


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------
def distribute_evenly(items: Sequence[str], num_groups: int) -> List[List[str]]:
    groups = [[] for _ in range(num_groups)]
    for idx, item in enumerate(items):
        groups[idx % num_groups].append(item)
    return groups


def group_dataset(
    clip_dir: str,
    num_groups: int,
    output_base_dir: str,
    clip_gt_index_dir: str,
    clip_pseudo_index_dir: str,
    grouped_gt_index_dir: str,
    grouped_pseudo_index_dir: str,
) -> None:
    all_npz: List[str] = []
    for dirpath, _, filenames in os.walk(clip_dir):
        for filename in filenames:
            if filename.endswith('.npz'):
                all_npz.append(os.path.join(dirpath, filename))
    if not all_npz:
        print(f'[WARN] No clips found under {clip_dir}; skipping grouping.')
        return
    if num_groups <= 0:
        print('[INFO] num_groups <= 0, skip grouping step.')
        return

    all_npz.sort()
    groups = distribute_evenly(all_npz, num_groups)

    for group_idx, files in enumerate(groups):
        for src_path in files:
            split_name, context_name, clip_name = parse_path_components(src_path)
            dst_dir = os.path.join(output_base_dir, f'group_{group_idx}', context_name)
            make_symlink(src_path, os.path.join(dst_dir, f'{clip_name}.npz'))

            src_gt = os.path.join(clip_gt_index_dir, split_name, context_name, f'{clip_name}.npy')
            if os.path.exists(src_gt):
                dst_gt = os.path.join(grouped_gt_index_dir, f'group_{group_idx}', context_name, f'{clip_name}.npy')
                make_symlink(src_gt, dst_gt)

            src_pseudo = os.path.join(clip_pseudo_index_dir, split_name, context_name, f'{clip_name}.npy')
            if os.path.exists(src_pseudo):
                dst_pseudo = os.path.join(grouped_pseudo_index_dir, f'group_{group_idx}', context_name, f'{clip_name}.npy')
                make_symlink(src_pseudo, dst_pseudo)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Clip AMASS-format sequences into fixed windows')
    parser.add_argument('--dataset-dir', default='/dataset/guangxun/test/humor', help='Directory containing per-object npz files')
    parser.add_argument('--gt-index-dir', default='/dataset/guangxun/test/amass_format/30fps_gt_index', help='Directory with global GT indices')
    parser.add_argument('--pseudo-gt-index-dir', default='/dataset/guangxun/test/amass_format/30fps_pesudo_gt_index', help='Directory with pseudo GT indices')
    parser.add_argument('--clip-save-dir', default='/dataset/guangxun/test/humor/clip', help='Destination root for clipped npz files (index folders derived automatically)')
    parser.add_argument('--group-save-dir', default='/dataset/guangxun/test/humor/grouped', help='Destination root for grouped symlinks (index folders derived automatically)')
    parser.add_argument('--splits', nargs='+', default=['training', 'validation'], help='Dataset splits to visit')
    parser.add_argument('--clip-frames', type=int, default=128, help='Number of frames per clip')
    parser.add_argument('--overlap', type=int, default=2, help='Frame overlap between consecutive clips')
    parser.add_argument('--num-groups', type=int, default=12, help='Number of groups for dataset splitting (<=0 disables grouping)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip_gt_index_dir = os.path.join(args.clip_save_dir, '30fps_gt_index_clip')
    clip_pseudo_index_dir = os.path.join(args.clip_save_dir, '30fps_pesudo_gt_index_clip')
    grouped_gt_dir = os.path.join(args.group_save_dir, '30fps_gt_index_grouped')
    grouped_pseudo_dir = os.path.join(args.group_save_dir, '30fps_pesudo_gt_index_grouped')

    clip_sequence(
        dataset_dir=args.dataset_dir,
        data_splits=args.splits,
        save_dir=args.clip_save_dir,
        gt_pose_index_dir=args.gt_index_dir,
        pseudo_gt_index_dir=args.pseudo_gt_index_dir,
        clip_gt_index_dir=clip_gt_index_dir,
        clip_pseudo_index_dir=clip_pseudo_index_dir,
        clip_frames=args.clip_frames,
        overlap=args.overlap,
    )

    group_dataset(
        clip_dir=args.clip_save_dir,
        num_groups=args.num_groups,
        output_base_dir=args.group_save_dir,
        clip_gt_index_dir=clip_gt_index_dir,
        clip_pseudo_index_dir=clip_pseudo_index_dir,
        grouped_gt_index_dir=grouped_gt_dir,
        grouped_pseudo_index_dir=grouped_pseudo_dir,
    )


if __name__ == '__main__':
    main()
