import glob
import os
import sys

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '..'))

import holden.BVH as BVH
import numpy as np
import torch
from arguments import Arguments
from human_body_prior.tools.omni_tools import log2file, makepath
from nemf.fk import ForwardKinematicsLayer
from rotations import matrix_to_rotation_6d, quaternion_to_matrix, rotation_6d_to_matrix
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import build_canonical_frame, estimate_angular_velocity, estimate_linear_velocity


def dump_animal2single(animal_data_dir, logger):
    fk = ForwardKinematicsLayer(parents=args.animal.parents, positions=args.animal.offsets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bvh_fnames = glob.glob(os.path.join(animal_data_dir, '*.bvh'))
    index = 0
    for bvh_fname in tqdm(bvh_fnames):
        try:
            anim, _, ftime = BVH.load(bvh_fname)
        except:
            logger('Could not read %s! skipping..' % bvh_fname)
            continue

        N = len(anim.rotations)

        fps = np.around(1.0 / ftime).astype(np.int32)
        logger(f'{bvh_fname} frame: {N}\tFPS: {fps}')

        data_rotation = anim.rotations.qs
        data_translation = anim.positions[:, 0] / 100.0

        # insert identity quaternion
        data_rotation = np.insert(data_rotation, [5, 9, 13, 16, 19, 21], [1, 0, 0, 0], axis=1)
        poses = torch.from_numpy(np.asarray(data_rotation, np.float32)).to(device)  # quaternion (T, J, 4)
        trans = torch.from_numpy(np.asarray(data_translation, np.float32)).to(device)  # global translation (T, 3)

        # Compute necessary data for model training.
        rotmat = quaternion_to_matrix(poses)  # rotation matrix (T, J, 3, 3)
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

        if args.unified_orientation:
            root_rotation = rotation_6d_to_matrix(root_orient)  # (T, 3, 3)
            root_rotation = root_rotation.unsqueeze(1).repeat(1, args.animal.joint_num, 1, 1)  # (T, J, 3, 3)
            global_pos = torch.matmul(root_rotation, pos.unsqueeze(-1)).squeeze(-1)
            height = global_pos + trans.unsqueeze(1)
        else:
            height = pos + trans.unsqueeze(1)
        height = height[..., 'xyz'.index(args.data.up)]  # (T, J)
        root_vel = estimate_linear_velocity(trans.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)  # linear velocity of the root joint (T, 3)

        out_posepath = makepath(os.path.join(work_dir, 'train', f'trans_{index}.pt'), isfile=True)
        torch.save(trans.detach().cpu(), out_posepath)  # (T, 3)
        torch.save(root_vel.detach().cpu(), out_posepath.replace('trans', 'root_vel'))  # (T, 3)
        torch.save(pos.detach().cpu(), out_posepath.replace('trans', 'pos'))  # (T, J, 3)
        torch.save(rotmat.detach().cpu(), out_posepath.replace('trans', 'rotmat'))  # (T, J, 3, 3)
        torch.save(height.detach().cpu(), out_posepath.replace('trans', 'height'))  # (T, J)

        if args.canonical:
            forward = rotmat[:, 0, :, 2].clone()
            canonical_frame = build_canonical_frame(forward, up_axis=args.data.up)
            root_rotation = canonical_frame.transpose(-2, -1)  # (T, 3, 3)
            root_rotation = root_rotation.unsqueeze(1).repeat(1, args.animal.joint_num, 1, 1)  # (T, J, 3, 3)

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

        index += 1


class Animal(Dataset):
    def __init__(self, dataset_dir):
        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).split('-')[0]
            self.ds[k] = torch.load(data_fname)

    def __len__(self):
        return len(self.ds['trans'])

    def __getitem__(self, idx):
        data = {k: self.ds[k][idx] for k in self.ds.keys()}

        return data


if __name__ == '__main__':
    animal_data_dir = '../AnimalData/'

    args = Arguments('./configs', filename='dog_mocap.yaml')
    work_dir = makepath(args.dataset_dir)

    log_name = os.path.join(work_dir, 'animal.log')
    if os.path.exists(log_name):
        os.remove(log_name)
    logger = log2file(log_name)

    logger('Start processing the animal mocap data ...')
    dump_animal2single(animal_data_dir, logger)
