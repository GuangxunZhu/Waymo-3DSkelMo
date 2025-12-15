# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# AMASS: Archive of Motion Capture as Surface Shapes <https://arxiv.org/abs/1904.03278>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2019.08.09
import os
import sys

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '..'))

import glob
from datetime import datetime

import holden.BVH as BVH
import numpy as np
import torch
from arguments import Arguments
from holden.Animation import Animation
from holden.Quaternions import Quaternions
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import log2file, makepath
from nemf.fk import ForwardKinematicsLayer
from rotations import axis_angle_to_matrix, axis_angle_to_quaternion, matrix_to_rotation_6d, rotation_6d_to_matrix
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import CONTACTS_IDX, align_joints, build_canonical_frame, estimate_angular_velocity, estimate_linear_velocity, normalize


def dump_amass2pytroch(datasets, amass_dir, out_posepath, logger=None):
    makepath(out_posepath, isfile=True)

    if logger is None:
        starttime = datetime.now().replace(microsecond=0)
        log_name = datetime.strftime(starttime, '%Y%m%d_%H%M')
        logger = log2file(out_posepath.replace(f'pose-{args.data.gender}-{args.data.clip_length}-{args.data.fps}fps.pt', '%s.log' % (log_name)))
        logger('Creating pytorch dataset at %s' % out_posepath)

    data_poses = []
    data_trans = []
    data_contacts = []

    clip_frames = args.data.clip_length
    # clip_frames = 104

    data_metas = []  # 元数据格式: {filename, clip_idx, valid_indices}
    overlap = 2  # 重叠帧数

    import gc

    # 新增优化点2：内存监控（可选）
    def log_memory():
        import psutil
        mem = psutil.virtual_memory()
        logger(f"Memory used: {mem.used / 1024**3:.2f} GB / {mem.total / 1024**3:.2f} GB")

    for ds_name in datasets:
        npz_fnames = glob.glob(os.path.join(amass_dir, ds_name, '*/*.npz'))
        logger('processing data points from %s.' % (ds_name))
        for npz_fname in tqdm(npz_fnames):
            try:
                cdata = np.load(npz_fname)
            except:
                logger('Could not read %s! skipping..' % npz_fname)
                continue
            N_raw = len(cdata['pose_body'])
            gender = str(cdata['gender'])
            if gender.startswith('b'):
                gender = str(cdata['gender'], encoding='utf-8')

            # Only process data with the specific gender.
            # if gender != args.data.gender:
            #     continue

            root_orient = cdata['root_orient']
            pose_body = cdata['pose_body']
            poses = np.concatenate((root_orient, pose_body, np.zeros((N_raw, 6))), axis=1)
            # Chop the data into evenly splitted sequence clips.
            trans = cdata['trans']
            contacts = cdata['contacts']

            # --- 合并补长逻辑到分割过程 ---
            step = clip_frames - overlap
            clip_idx = 0  # 当前文件的片段计数器
            
            for i in range(0, N_raw, step):
                end = i + clip_frames
                # 生成片段索引（可能超出原始数据长度）
                if end <= N_raw:
                    clip_indices = np.arange(i, end)
                    valid_indices = np.arange(len(clip_indices))
                else:
                    # 实际有效部分
                    valid_part = np.arange(i, N_raw)
                    valid_indices = np.arange(len(valid_part))
                    # 计算需要补长的帧数
                    padding_length = end - N_raw
                    # 用最后一帧填充
                    padding = np.full(padding_length, N_raw - 1)
                    clip_indices = np.concatenate([valid_part, padding])
                
                # 记录有效帧索引（未被补长的部分）
                valid_indices.tolist()
                
                # 补长后的数据（直接从原始数据中提取，不需要预补长）
                clip_poses = poses[clip_indices]
                clip_trans = trans[clip_indices]
                clip_contacts = contacts[clip_indices]
                
                # 保存数据
                data_poses.append(clip_poses)
                data_trans.append(clip_trans)
                data_contacts.append(clip_contacts)
                
                # 记录元数据
                data_metas.append({
                    'filename': npz_fname,
                    'clip_idx': clip_idx,
                    'valid_indices': valid_indices
                })
                
                clip_idx += 1  # 更新片段计数器


    assert len(data_poses) != 0
    np.savez(out_posepath.replace(f'pose-{args.data.gender}-{args.data.clip_length}-{args.data.fps}fps.pt','data_metas.npz'), data_metas=data_metas)

    del data_metas, cdata, root_orient, pose_body, trans, contacts, poses
    gc.collect()  # 强制垃圾回收


    # Choose the device to run the body model on.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")  # when GPU memory is limited
    pose = torch.from_numpy(np.asarray(data_poses, np.float32)).to(device)
    pose = pose.view(-1, clip_frames, 24, 3)  # axis-angle (N, T, J, 3)
    trans = torch.from_numpy(np.asarray(data_trans, np.float32)).to(device)  # global translation (N, T, 3)

    # Compute necessary data for model training.
    rotmat = axis_angle_to_matrix(pose)  # rotation matrix (N, T, J, 3, 3)
    root_orient = rotmat[:, :, 0].clone()
    root_orient = matrix_to_rotation_6d(root_orient)  # root orientation (N, T, 6)
    if args.unified_orientation:
        identity = torch.zeros_like(rotmat[:, :, 0])  # (N, T, 3, 3)
        identity[:, :, 0, 0] = 1
        identity[:, :, 1, 1] = 1
        identity[:, :, 2, 2] = 1
        rotmat[:, :, 0] = identity

    rot6d = matrix_to_rotation_6d(rotmat)  # 6D rotation representation (N, T, J, 6)
    rot_seq = rotmat.clone()
    angular = estimate_angular_velocity(rot_seq, dt=1.0 / args.data.fps)  # angular velocity of all the joints (N, T, J, 3)

    fk = ForwardKinematicsLayer(args, device=device)
    pos, global_xform = fk(rot6d.view(-1, 24, 6))  # 位移+全局变换矩阵
    pos = pos.contiguous().view(-1, clip_frames, 24, 3)  # local joint positions (N, T, J, 3)
    global_xform = global_xform.view(-1, clip_frames, 24, 4, 4)  # global transformation matrix for each joint (N, T, J, 4, 4)
    velocity = estimate_linear_velocity(pos, dt=1.0 / args.data.fps)  # linear velocity of all the joints (N, T, J, 3)

    contacts = torch.from_numpy(np.asarray(data_contacts, np.float32)).to(device)
    contacts = contacts[:, :, CONTACTS_IDX]  # contacts information (N, T, 8)

    if args.unified_orientation:
        root_rotation = rotation_6d_to_matrix(root_orient)  # (N, T, 3, 3)
        root_rotation = root_rotation.unsqueeze(2).repeat(1, 1, args.smpl.joint_num, 1, 1)  # (N, T, J, 3, 3)
        global_pos = torch.matmul(root_rotation, pos.unsqueeze(-1)).squeeze(-1)
        height = global_pos + trans.unsqueeze(2)
    else:
        height = pos + trans.unsqueeze(2)
    height = height[:, :, :, 'xyz'.index(args.data.up)]  # (N, T, J)
    root_vel = estimate_linear_velocity(trans, dt=1.0 / args.data.fps)  # linear velocity of the root joint (N, T, 3)

    _, rotmat_mean, rotmat_std = normalize(rotmat)
    _, pos_mean, pos_std = normalize(pos)
    _, trans_mean, trans_std = normalize(trans)
    _, root_vel_mean, root_vel_std = normalize(root_vel)
    _, height_mean, height_std = normalize(height)
    _, contacts_mean, contacts_std = normalize(contacts)

    mean = {
        'rotmat': rotmat_mean.detach().cpu(),
        'pos': pos_mean.detach().cpu(),
        'trans': trans_mean.detach().cpu(),
        'root_vel': root_vel_mean.detach().cpu(),
        'height': height_mean.detach().cpu(),
        'contacts': contacts_mean.detach().cpu()
    }
    std = {
        'rotmat': rotmat_std.detach().cpu(),
        'pos': pos_std.detach().cpu(),
        'trans': trans_std.detach().cpu(),
        'root_vel': root_vel_std.detach().cpu(),
        'height': height_std.detach().cpu(),
        'contacts': contacts_std.detach().cpu()
    }

    torch.save(rotmat.detach().cpu(), out_posepath.replace('pose', 'rotmat'))  # (N, T, J, 3, 3)
    torch.save(pos.detach().cpu(), out_posepath.replace('pose', 'pos'))  # (N, T, J, 3)
    torch.save(trans.detach().cpu(), out_posepath.replace('pose', 'trans'))  # (N, T, 3)
    torch.save(root_vel.detach().cpu(), out_posepath.replace('pose', 'root_vel'))  # (N, T, 3)
    torch.save(height.detach().cpu(), out_posepath.replace('pose', 'height'))  # (N, T, J)
    torch.save(contacts.detach().cpu(), out_posepath.replace('pose', 'contacts'))  # (N, T, J)

    if args.canonical:
        forward = rotmat[:, :, 0, :, 2].clone()
        canonical_frame = build_canonical_frame(forward, up_axis=args.data.up)
        root_rotation = canonical_frame.transpose(-2, -1)  # (N, T, 3, 3)
        root_rotation = root_rotation.unsqueeze(2).repeat(1, 1, args.smpl.joint_num, 1, 1)  # (N, T, J, 3, 3)

        theta = torch.atan2(forward[..., 1], forward[..., 0])
        dt = 1.0 / args.data.fps
        forward_ang = (theta[:, 1:] - theta[:, :-1]) / dt
        forward_ang = torch.cat((forward_ang, forward_ang[..., -1:]), dim=-1)  # (N, T)

        local_pos = torch.matmul(root_rotation, pos.unsqueeze(-1)).squeeze(-1)
        local_vel = torch.matmul(root_rotation, velocity.unsqueeze(-1)).squeeze(-1)
        local_rot = torch.matmul(root_rotation, global_xform[:, :, :, :3, :3])
        local_rot = matrix_to_rotation_6d(local_rot)
        local_ang = torch.matmul(root_rotation, angular.unsqueeze(-1)).squeeze(-1)

        _, forward_mean, forward_std = normalize(forward)
        _, forward_ang_mean, forward_ang_std = normalize(forward_ang)
        _, local_pos_mean, local_pos_std = normalize(local_pos)
        _, local_vel_mean, local_vel_std = normalize(local_vel)
        _, local_rot_mean, local_rot_std = normalize(local_rot)
        _, local_ang_mean, local_ang_std = normalize(local_ang)

        mean['forward'] = forward_mean.detach().cpu()
        mean['forward_ang'] = forward_ang_mean.detach().cpu()
        mean['local_pos'] = local_pos_mean.detach().cpu()
        mean['local_vel'] = local_vel_mean.detach().cpu()
        mean['local_rot'] = local_rot_mean.detach().cpu()
        mean['local_ang'] = local_ang_mean.detach().cpu()

        std['forward'] = forward_std.detach().cpu()
        std['forward_ang'] = forward_ang_std.detach().cpu()
        std['local_pos'] = local_pos_std.detach().cpu()
        std['local_vel'] = local_vel_std.detach().cpu()
        std['local_rot'] = local_rot_std.detach().cpu()
        std['local_ang'] = local_ang_std.detach().cpu()

        torch.save(forward.detach().cpu(), out_posepath.replace('pose', 'forward'))  # (N, T, 3)
        torch.save(local_pos.detach().cpu(), out_posepath.replace('pose', 'local_pos'))  # (N, T, J, 3)
        torch.save(local_vel.detach().cpu(), out_posepath.replace('pose', 'local_vel'))  # (N, T, J, 3)
        torch.save(local_rot.detach().cpu(), out_posepath.replace('pose', 'local_rot'))  # (N, T, J, 6)
        torch.save(local_ang.detach().cpu(), out_posepath.replace('pose', 'local_ang'))  # (N, T, J, 3)
    else:
        global_xform = global_xform[:, :, :, :3, :3]  # (N, T, J, 3, 3)
        global_xform = matrix_to_rotation_6d(global_xform)  # (N, T, J, 6)

        _, rot6d_mean, rot6d_std = normalize(rot6d)
        _, angular_mean, angular_std = normalize(angular)
        _, global_xform_mean, global_xform_std = normalize(global_xform)
        _, velocity_mean, velocity_std = normalize(velocity)
        _, orientation_mean, orientation_std = normalize(root_orient)

        mean['rot6d'] = rot6d_mean.detach().cpu()
        mean['angular'] = angular_mean.detach().cpu()
        mean['global_xform'] = global_xform_mean.detach().cpu()
        mean['velocity'] = velocity_mean.detach().cpu()
        mean['root_orient'] = orientation_mean.detach().cpu()

        std['rot6d'] = rot6d_std.detach().cpu()
        std['angular'] = angular_std.detach().cpu()
        std['global_xform'] = global_xform_std.detach().cpu()
        std['velocity'] = velocity_std.detach().cpu()
        std['root_orient'] = orientation_std.detach().cpu()

        torch.save(rot6d.detach().cpu(), out_posepath.replace('pose', 'rot6d'))  # (N, T, J, 6)
        torch.save(angular.detach().cpu(), out_posepath.replace('pose', 'angular'))  # (N, T, J, 3)
        torch.save(global_xform.detach().cpu(), out_posepath.replace('pose', 'global_xform'))  # (N, T, J, 6)
        torch.save(velocity.detach().cpu(), out_posepath.replace('pose', 'velocity'))  # (N, T, J, 3)
        torch.save(root_orient.detach().cpu(), out_posepath.replace('pose', 'root_orient'))  # (N, T, 3, 3)

    if args.normalize:
        torch.save(mean, out_posepath.replace('pose', 'mean'))
        torch.save(std, out_posepath.replace('pose', 'std'))

    return len(data_poses)


def dump_amass2single(datasets, amass_dir, split_name, logger=None, bvh_viz=False):
    # Choose the device to run the body model on.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fk = ForwardKinematicsLayer(args)

    index = 0
    for ds_name in datasets:
        npz_fnames = sorted(glob.glob(os.path.join(amass_dir, ds_name, '*/*_poses_*.npz')))
        logger('processing data points from %s.' % (ds_name))
        for npz_fname in tqdm(npz_fnames):
            try:
                cdata = np.load(npz_fname)
            except:
                logger('Could not read %s! skipping..' % npz_fname)
                continue

            gender = str(cdata['gender'])
            if gender.startswith('b'):
                gender = str(cdata['gender'], encoding='utf-8')

            # Only process data with the specific gender.
            if gender != args.data.gender:
                continue

            N = len(cdata['pose_body'])
            logger(f'Sequence {index} has {N} frames')

            root_orient = cdata['root_orient']
            pose_body = cdata['pose_body']
            data_poses = np.concatenate((root_orient, pose_body, np.zeros((N, 6))), axis=1)
            pose = torch.from_numpy(np.asarray(data_poses, np.float32)).to(device)
            pose = pose.view(-1, 24, 3)  # axis-angle (T, J, 3)
            trans = torch.from_numpy(np.asarray(cdata['trans'], np.float32)).to(device)  # global translation (T, 3)

            # Compute necessary data for model training.
            rotmat = axis_angle_to_matrix(pose)  # rotation matrix (T, J, 3, 3)
            root_orient = rotmat[:, 0].clone()
            root_orient = matrix_to_rotation_6d(root_orient)  # root orientation (T, 6)
            if args.unified_orientation:
                identity = torch.eye(3).cuda()
                identity = identity.view(1, 3, 3).repeat(rotmat.shape[0], 1, 1)
                rotmat[:, 0] = identity
            rot6d = matrix_to_rotation_6d(rotmat)  # 6D rotation representation (T, J, 6)

            rot_seq = rotmat.clone()
            angular = estimate_angular_velocity(rot_seq.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)  # angular velocity of all the joints (T, J, 3)

            pos, global_xform = fk(rot6d)  # local joint positions (T, J, 3), global transformation matrix for each joint (T, J, 4, 4)
            pos = pos.contiguous()
            global_xform = global_xform.contiguous()
            velocity = estimate_linear_velocity(pos.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)  # linear velocity of all the joints (T, J, 3)

            contacts = torch.from_numpy(np.asarray(cdata['contacts'], np.float32)).to(device)
            contacts = contacts[:, CONTACTS_IDX]  # contacts information (T, 8)

            if args.unified_orientation:
                root_rotation = rotation_6d_to_matrix(root_orient)  # (T, 3, 3)
                root_rotation = root_rotation.unsqueeze(1).repeat(1, args.smpl.joint_num, 1, 1)  # (T, J, 3, 3)
                global_pos = torch.matmul(root_rotation, pos.unsqueeze(-1)).squeeze(-1)
                height = global_pos + trans.unsqueeze(1)
            else:
                height = pos + trans.unsqueeze(1)
            height = height[..., 'xyz'.index(args.data.up)]  # (T, J)
            root_vel = estimate_linear_velocity(trans.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)  # linear velocity of the root joint (T, 3)

            out_posepath = os.path.join(work_dir, split_name, f'trans_{index}.pt')
            torch.save(trans.detach().cpu(), out_posepath)  # (T, 3)
            torch.save(root_vel.detach().cpu(), out_posepath.replace('trans', 'root_vel'))  # (T, 3)
            torch.save(pos.detach().cpu(), out_posepath.replace('trans', 'pos'))  # (T, J, 3)
            torch.save(rotmat.detach().cpu(), out_posepath.replace('trans', 'rotmat'))  # (T, J, 3, 3)
            torch.save(height.detach().cpu(), out_posepath.replace('trans', 'height'))  # (T, J)
            torch.save(contacts.detach().cpu(), out_posepath.replace('trans', 'contacts'))  # (T, J)

            if args.canonical:
                forward = rotmat[:, 0, :, 2].clone()
                canonical_frame = build_canonical_frame(forward, up_axis=args.data.up)
                root_rotation = canonical_frame.transpose(-2, -1)  # (T, 3, 3)
                root_rotation = root_rotation.unsqueeze(1).repeat(1, args.smpl.joint_num, 1, 1)  # (T, J, 3, 3)

                if args.data.up == 'x':
                    theta = torch.atan2(forward[..., 2], forward[..., 1])
                elif args.data.up == 'y':
                    theta = torch.atan2(forward[..., 0], forward[..., 2])
                else:
                    theta = torch.atan2(forward[..., 1], forward[..., 0])
                dt = 1.0 / args.data.fps
                forward_ang = (theta[1:] - theta[:-1]) / dt
                forward_ang = torch.cat((forward_ang, forward_ang[-1:]), dim=-1)

                local_pos = torch.matmul(root_rotation, pos.unsqueeze(-1)).squeeze(-1)
                local_vel = torch.matmul(root_rotation, velocity.unsqueeze(-1)).squeeze(-1)
                local_rot = torch.matmul(root_rotation, global_xform[:, :, :3, :3])
                local_rot = matrix_to_rotation_6d(local_rot)
                local_ang = torch.matmul(root_rotation, angular.unsqueeze(-1)).squeeze(-1)

                torch.save(forward.detach().cpu(), out_posepath.replace('trans', 'forward'))
                torch.save(forward_ang.detach().cpu(), out_posepath.replace('trans', 'forward_ang'))
                torch.save(local_pos.detach().cpu(), out_posepath.replace('trans', 'local_pos'))
                torch.save(local_vel.detach().cpu(), out_posepath.replace('trans', 'local_vel'))
                torch.save(local_rot.detach().cpu(), out_posepath.replace('trans', 'local_rot'))
                torch.save(local_ang.detach().cpu(), out_posepath.replace('trans', 'local_ang'))
            else:
                global_xform = global_xform[:, :, :3, :3]  # (T, J, 3, 3)
                global_xform = matrix_to_rotation_6d(global_xform)  # (T, J, 6)

                torch.save(rot6d.detach().cpu(), out_posepath.replace('trans', 'rot6d'))  # (N, T, J, 6)
                torch.save(angular.detach().cpu(), out_posepath.replace('trans', 'angular'))  # (N, T, J, 3)
                torch.save(global_xform.detach().cpu(), out_posepath.replace('trans', 'global_xform'))  # (N, T, J, 6)
                torch.save(velocity.detach().cpu(), out_posepath.replace('trans', 'velocity'))  # (N, T, J, 3)
                torch.save(root_orient.detach().cpu(), out_posepath.replace('trans', 'root_orient'))  # (N, T, 3, 3)

            if bvh_viz:
                bvh_fname = os.path.join(work_dir, 'bvh', split_name, f'pose_{index}.bvh')
                makepath(bvh_fname, isfile=True)
                rotation = axis_angle_to_quaternion(pose)
                rotation = align_joints(rotation)
                position = trans.unsqueeze(1)
                # Export the motion data to .bvh file with SMPL skeleton.
                anim = Animation(Quaternions(c2c(rotation)), c2c(position), None, offsets=args.smpl.offsets[gender], parents=args.smpl.parents)
                BVH.save(bvh_fname, anim, names=args.smpl.joint_names, frametime=1 / args.data.fps)

            index += 1

    return index


def prepare_amass(amass_splits, amass_dir, work_dir, logger=None):

    if logger is None:
        starttime = datetime.now().replace(microsecond=0)
        log_name = datetime.strftime(starttime, '%Y%m%d_%H%M')
        logger = log2file(os.path.join(work_dir, '%s.log' % (log_name)))
        logger('Creating pytorch dataset at %s' % work_dir)

    logger('Fetch data from waymo npz files, augment the data and dump every data field as final pytorch pt files')

    for split_name, datasets in amass_splits.items():
        outpath = makepath(os.path.join(work_dir, split_name, f'pose-{args.data.gender}-{args.data.clip_length}-{args.data.fps}fps.pt'), isfile=True)
        if os.path.exists(outpath):
            continue
        if args.single_motion:
            ndatapoints = dump_amass2single(datasets, amass_dir, split_name, logger=logger, bvh_viz=True)
        else:
            ndatapoints = dump_amass2pytroch(datasets, amass_dir, outpath, logger=logger)
        logger('%s has %d data points!' % (split_name, ndatapoints))


def dump_amass2bvh(amass_splits, amass_dir, out_bvhpath, logger=None):
    from utils import export_bvh_animation

    if logger is None:
        log_name = os.path.join(out_bvhpath, 'bvh_viz.log')
        if os.path.exists(log_name):
            os.remove(log_name)
        logger = log2file(log_name)
        logger('Creating bvh visualization at %s' % out_bvhpath)

    for split_name, datasets in amass_splits.items():
        outpath = makepath(os.path.join(out_bvhpath, split_name))
        if os.path.exists(outpath) and len(os.listdir(outpath)) != 0:
            continue

        for ds_name in datasets:
            npz_fnames = glob.glob(os.path.join(amass_dir, ds_name, '*/*_poses_*.npz'))
            logger('visualizing data points at %s.' % (ds_name))
            for npz_fname in tqdm(npz_fnames):
                bvh_fname = os.path.split(npz_fname)[1].replace('.npz', '.bvh')
                try:
                    export_bvh_animation(npz_fname, os.path.join(outpath, ds_name + '_' + bvh_fname))
                except AssertionError:
                    logger('Could not export %s because its gender is not supported! skipping..' % npz_fname)
                    continue


def collect_amass_stats(amass_splits, amass_dir, logger=None):
    import matplotlib.pyplot as plt

    if logger is None:
        log_name = os.path.join('./data/amass', 'amass_stats.log')
        if os.path.exists(log_name):
            os.remove(log_name)
        logger = log2file(log_name)

    logger('Collecting stats for AMASS datasets:')

    gender_stats = {}
    fps_stats = {}
    durations = []
    for split_name, datasets in amass_splits.items():
        for ds_name in datasets:
            logger(f'\t{ds_name} dataset')
            npz_fnames = glob.glob(os.path.join(amass_dir, ds_name, '*/*_poses_*.npz'))
            for npz_fname in tqdm(npz_fnames):
                try:
                    cdata = np.load(npz_fname)
                except:
                    logger('Could not read %s! skipping..' % npz_fname)
                    continue

                gender = str(cdata['gender'])
                if gender.startswith('b'):
                    gender = str(cdata['gender'], encoding='utf-8')
                fps = 30
                duration = len(cdata['pose_body']) / fps

                durations.append(duration)
                if gender in gender_stats.keys():
                    gender_stats[gender].append(duration)
                else:
                    gender_stats[gender] = [duration]
                if fps in fps_stats.keys():
                    fps_stats[fps] += 1
                else:
                    fps_stats[fps] = 1

    logger('\n')
    logger('Total motion sequences: {:,}'.format(len(durations)))
    logger('\tTotal Duration: {:,.2f}s'.format(sum(durations)))
    logger('\tMin Duration: {:,.2f}s'.format(min(durations)))
    logger('\tMax Duration: {:,.2f}s'.format(max(durations)))
    logger('\tAverage Duration: {:,.2f}s'.format(sum(durations) / len(durations)))
    logger('\tSequences longer than 5s: {:,} ({:.2f}%)'.format(sum(i > 5 for i in durations), sum(i > 5 for i in durations) / len(durations) * 100))
    logger('\tSequences longer than 10s: {:,} ({:.2f}%)'.format(sum(i > 10 for i in durations), sum(i > 10 for i in durations) / len(durations) * 100))
    logger('\n')
    logger('Gender:')
    for key, value in gender_stats.items():
        logger('\t{}: {:,} (Duration: {:,.2f}s)'.format(key, len(value), sum(value)))
    logger('\n')
    logger('FPS:')
    for key, value in fps_stats.items():
        logger('\t{}: {:,}'.format(key, value))

    # Plot histograms for duration distributions.
    fig, axes = plt.subplots(3, sharex=True)
    fig.tight_layout(pad=3.0)
    fig.suptitle('AMASS Data Duration Distribution')
    axes[0].hist(durations, density=False, bins=100)
    axes[0].set_title('total')
    for key, value in gender_stats.items():
        if key == 'female':
            f = axes[1]
        else:
            f = axes[2]
        f.hist(value, density=False, bins=100)
        f.set_title(f'{key}')
    # Add a big axis, hide frame.
    fig.add_subplot(111, frameon=False)
    # Hide tick and tick label of the big axis.
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Duration (second)")
    plt.ylabel("Counts", labelpad=10)
    fig.savefig('./data/amass/duration_dist.pdf')


class AMASS(Dataset):
    """AMASS: a pytorch loader for unified human motion capture dataset. http://amass.is.tue.mpg.de/"""

    def __init__(self, dataset_dir):
        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).split('-')[0]
            self.ds[k] = torch.load(data_fname)

    def __len__(self):
        return len(self.ds['trans'])

    def __getitem__(self, idx):
        data = {k: self.ds[k][idx] for k in self.ds.keys() if k not in ['mean', 'std']}

        return data


if __name__ == '__main__':
    msg = ''' Using standard AMASS dataset preparation pipeline: 
    0) Donwload all npz files from https://amass.is.tue.mpg.de/ 
    1) Convert npz files to pytorch readable pt files. '''

    # amass_dir = '/home/guangxun/dataset/waymo/waymo_amassformat_30fps_processed'
    amass_dir = '/home/guangxun/dataset/waymo/waymo_amassformat_30fps_processed_clustered'


    args = Arguments('./configs', filename='waymo.yaml')
    work_dir = makepath(args.dataset_dir)

    log_name = os.path.join(work_dir, 'waymo.log')
    if os.path.exists(log_name):
        os.remove(log_name)
    logger = log2file(log_name)
    logger('waymo Data Preparation Began.')
    logger(msg)

    if args.single_motion:
        amass_splits = {
            'train': ['DanceDB']
        }
    else:
        amass_splits = {
            # 'valid': ['MPI_HDM05', 'SFU', 'MPI_mosh'],
            'cluster_0': ['cluster_0'],
            'cluster_1': ['cluster_1'],
            'cluster_2': ['cluster_2'],
            'cluster_3': ['cluster_3'],
            'cluster_4': ['cluster_4'],
            'cluster_5': ['cluster_5'],
            'cluster_6': ['cluster_6'],
            'cluster_7': ['cluster_7'],
            # 'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BioMotionLab_NTroje', 'BMLmovi', 'EKUT', 'ACCAD', 'BMLhandball', 'DanceDB', 'DFaust_67', 'SSM_synced']
        }
        # amass_splits = {
        #     'no_split': ['MPI_HDM05', 'SFU', 'MPI_mosh', 'Transitions_mocap', 'HumanEva', 'CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BioMotionLab_NTroje', 'BMLmovi', 'EKUT', 'ACCAD', 'BMLhandball', 'DanceDB', 'DFaust_67', 'SSM_synced']
        # }

    # collect_amass_stats(amass_splits, amass_dir)
    prepare_amass(amass_splits, amass_dir, work_dir, logger=logger)
