import faulthandler
faulthandler.enable()
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from datasets.waymo_v2 import WAYMOV2_Dataset, WAYMOV2_Dataset_Gen

# from models.graphormer.graphormer_model import graphormer_model
# from models.unsupervised.Network import point_net_ssg, smpl_model
from models.pose_mesh_net import LiDAR_HMR
# from models.v2v_posenet import V2VPoseNet
from tqdm import tqdm
import torch.optim as optim
import logging
import torch.nn.functional as F
import argparse
from models.pmg_config import config, update_config
from scripts.eval_utils import mean_per_vertex_error, setup_seed, mean_per_edge_error, smpl_model, J_regressor, JOINTS_IDX, get_mesh
from models.graphormer.data.config import H36M_J17_NAME, H36M_J17_TO_J14, J24_NAME, J24_TO_J14
from fvcore.nn import FlopCountAnalysis, parameter_count_table
# from thop import profile, clever_format
from models.pct_config import pct_config, update_pct_config
import smplx

# for visualizing
import open3d as o3d
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.optimize import curve_fit

import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--dataset', default='waymov2', required=False, type=str)
    parser.add_argument('--state_dict', default='LiDAR-HMR/lidar_hmr_ckpt/waymov2/lidar_hmr_mesh.pth', required=False, type=str)
    parser.add_argument('--cfg', default='LiDAR-HMR/configs/mesh/waymo.yaml', required=False, type=str)
    parser.add_argument('--dataset-path', default='/dataset/guangxun/test/waymo_extracted', required=False, type=str)
    parser.add_argument('--save-path', default='/dataset/guangxun/test/lidar_hmr_results', required=False, type=str)
    parser.add_argument('--batch-size', default=32, required=False, type=int)
    
    args, rest = parser.parse_known_args()
    return args

logger = logging.getLogger(__name__)

# create PointCloud object function
def create_point_cloud(points, color, translation=(0, 0, 0)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    if translation != (0, 0, 0):  # default no translation
        pcd.translate(translation)
    return pcd

# create LineSet object function
def create_lineset(points, bone_links, color, translation=(0, 0, 0)):
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(bone_links)
    lineset.colors = o3d.utility.Vector3dVector([color] * len(bone_links))
    if translation != (0, 0, 0):  # default no translation
        lineset.translate(translation)
    return lineset

def test(model, dataloader, dataset_path, split, save = False, save_path=None):
    model.eval()
    
    print('============Testing============')
    # save = False
    if save:
        saver_dict = {}
        saver_dict['pcd'] = []
        saver_dict['mesh_o'] = []
        saver_dict['mesh_r'] = []
    
        saver_dict['number_list'] = []
        saver_dict['pose_theta'] = []
        saver_dict['pose_beta'] = []
        saver_dict['pose_trans'] = []

    J_r = smpl_model.J_regressor.cuda()

    for sample in tqdm(dataloader):

        for key in sample:
            if type(sample[key]) is not dict and type(sample[key]) is not list:
                sample[key] = sample[key].cuda()
        pcd = sample['human_points_local']

        root = sample['global_trans'].cpu().numpy()

        with torch.no_grad():
            ret_dict = model(pcd)#， sample['betas'])

        root_orient = ret_dict['pose_theta'][:, :3].cpu().numpy()
        pose_body = ret_dict['pose_theta'][:, 3:].cpu().numpy()
        trans = ret_dict['trans'].cpu().numpy()
        beta = ret_dict['pose_beta'].cpu().numpy()
        # vertices = ret_dict['mesh_refine'].cpu().numpy()

        # process each item in the batch
        for idx in range(len(sample['basic_info']['context_name'])):

            context_name = sample['basic_info']['context_name'][idx]
            object_id = sample['basic_info']['object_id'][idx]
            time_stamp = sample['basic_info']['time_stamp'][idx].item()

            new_data = {
                'basic_info': {
                    'context_name': context_name,
                    'object_id': object_id,
                    'time_stamp': time_stamp,
                    'split': split
                },
                'SMPL_lidarhmr' : {
                    'global_dis': root[idx],
                    'root_orient': root_orient[idx],
                    'pose_body': pose_body[idx],
                    'trans':trans[idx],
                    'betas': beta[idx],
                }
            }
            # extracted_data.append(new_data)

            obj_path = os.path.join(save_path, split, context_name, str(object_id))
            os.makedirs(obj_path, exist_ok=True)
            # save new_data as a npz file
            np.savez_compressed(os.path.join(obj_path, f'{time_stamp}.npz'), **new_data)


    print('Completed for split: {}'.format(split))
        # sample_info = [frame, time, id]



if __name__ == '__main__':
    args = parse_args()
    setup_seed(10)
    dataset_task = args.dataset #'sloper4d', 'waymov2', 'collect'
    dataset_path = args.dataset_path
    split_list = ['training', 'validation']
    save_path = args.save_path

    num_keypoints = 15
    # model = pose_meshformer().cuda()
    config_name = args.cfg.split('/')[-1].split('.')[0]
    update_config(args.cfg)
    model = LiDAR_HMR(pmg_cfg = config, train_pmg = True).cuda()
    # model = LiDAR_HMR(pct_config).cuda()
    total = sum([param.nelement() for param in model.parameters()])
    bs = args.batch_size

    for split in split_list:
        
        test_dataset = WAYMOV2_Dataset_Gen( dataset_path = dataset_path, split = split, is_train = False,
                                    return_torch=True, device = 'cuda',
                                    fix_pts_num=True)

        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers = 8)

        optimizer = torch.optim.Adam(params=list(model.parameters()),
                                                lr=5e-4,
                                                betas=(0.9, 0.999),
                                                weight_decay=0)
        epoch_now = -1

        if args.state_dict != '':
            state_dict = torch.load(args.state_dict)
            model.load_state_dict(state_dict['net'])
            
        save = True
        test(model, test_loader, dataset_path, split, save, save_path)
    
    print('All done!')