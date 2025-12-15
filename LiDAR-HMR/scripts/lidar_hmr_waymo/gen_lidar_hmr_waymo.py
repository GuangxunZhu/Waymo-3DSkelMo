import faulthandler
faulthandler.enable()
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets.sloper4d import SLOPER4D_Dataset
from datasets.waymo_v2 import WAYMOV2_Dataset, WAYMOV2_Dataset_Gen
from datasets.humanm3 import HumanM3_Dataset
from datasets.cimi4d import CIMI4D_Dataset
from datasets.lidarh26m import LiDARH26M_Dataset
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
    parser.add_argument(
        '--dataset', default='waymov2', required=False, type=str)
    parser.add_argument(
        '--state_dict', default='lidar_hmr_ckpt/waymov2/lidar_hmr_mesh.pth', required=False, type=str)
    parser.add_argument(
        '--cfg', default='configs/mesh/waymo.yaml', required=False, type=str)
    args, rest = parser.parse_known_args()
    return args

logger = logging.getLogger(__name__)

# 创建 PointCloud 对象的函数
def create_point_cloud(points, color, translation=(0, 0, 0)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    if translation != (0, 0, 0):  # 默认不平移
        pcd.translate(translation)
    return pcd

# 创建 LineSet 对象的函数
def create_lineset(points, bone_links, color, translation=(0, 0, 0)):
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(bone_links)
    lineset.colors = o3d.utility.Vector3dVector([color] * len(bone_links))
    if translation != (0, 0, 0):  # 默认不平移
        lineset.translate(translation)
    return lineset

def test(model, dataloader, dataset_path, split, save = False):
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

    data_dic = {}
    width, height= 1536, 801,
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(height=height, width = width)
    # vis.get_render_option().point_size = 4.0
    bone_link = np.array([[0,1], [0,2], [1,3], [2,4], [3,5], [4,6], [0,7], [7,8], [7,9], [7,10], [9,11], [10,12], [11,13], [12,14]])

    all_geometries = []
    extracted_data = []


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

        # 批量处理每个样本
        for idx in range(len(sample['basic_info']['context_name'])):
            
            new_data = {
                'basic_info': {
                    'context_name': sample['basic_info']['context_name'][idx],
                    'object_id': sample['basic_info']['object_id'][idx],
                    'time_stamp': sample['basic_info']['time_stamp'][idx].item(),
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
            extracted_data.append(new_data)


    save_path = os.path.join(dataset_path, 'SMPL-data', split)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, '{}.pkl'.format(split)), 'wb') as f:
        pickle.dump(extracted_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # extracted_data = []

    print('Completed for split: {}'.format(split))
        # sample_info = [frame, time, id]

        # pcd = pcd.cpu().numpy().squeeze(axis=0) 

        # pct_pose = ret_dict['pct_pose'].cpu().numpy().squeeze(axis=0)

        # pred_vertices = ret_dict['mesh_refine']
        # pose = torch.einsum('bnk,mn->bmk', [pred_vertices, J_regressor])[:,JOINTS_IDX]

        # pred_vertices = pred_vertices.cpu().numpy().squeeze(axis=0)

        # pose = pose.cpu().numpy().squeeze(axis=0)
        
        # origin_data = sample['origin_data']


        # pcd += root
        # # pred_vertices += root
        # pct_pose += root
        # pose += root


        # # 创建 PointClouds
        # pcd_k = create_point_cloud(pcd, [0.5, 0.5, 0.5], (0, 0, 0))
        # pcd_m = create_point_cloud(pcd, [0.5, 0.5, 0.5], (0, -1, 0))
        # pct_pose_ = create_point_cloud(pct_pose, [1, 0, 0], (0, 1, 0))
        # pose_ = create_point_cloud(pose, [0, 0, 1], (0, 0, 0))

        # # 创建 LineSets
        # pct_lineset = create_lineset(pct_pose, bone_link, [1, 0, 0], (0, 1, 0))
        # lineset = create_lineset(pose, bone_link, [0, 0, 1], (0, 0, 0))

        # center_in = [torch.tensor(root)]

        # for ind_v, p_v in enumerate(np.expand_dims(pred_vertices, axis=0)):
        #     p_k = p_v + center_in[ind_v].numpy()
            # mesh = get_mesh(p_k, model.pmg.smpl.faces)
        #     mesh.translate((0, 0, 0))  # 将 pcd2 移动到坐标 (2, 0, 0)
        #     all_geometries.append(mesh)
        #     # vis.add_geometry(mesh)

        # all_geometries.extend([pcd_k, pose_, lineset])

        # all_geometries.extend([pcd_k])

        # o3d.visualization.draw_geometries([pcd_k, pcd_m, pct_pose_, pct_lineset, pose_, lineset, mesh])


    # 在遍历结束后一次性显示所有几何对象
    # for geometry in all_geometries:
    #     vis.add_geometry(geometry)

    # o3d.visualization.draw_geometries(all_geometries)


    # print('============Testing Done============')


if __name__ == '__main__':
    args = parse_args()
    setup_seed(10)
    dataset_task = args.dataset #'sloper4d', 'waymov2', 'collect'
    dataset_path = '/home/guangxun/dataset/waymo'
    split_list = ['training', 'validation']

    num_keypoints = 15
    # model = pose_meshformer().cuda()
    config_name = args.cfg.split('/')[-1].split('.')[0]
    update_config(args.cfg)
    model = LiDAR_HMR(pmg_cfg = config, train_pmg = True).cuda()
    # model = LiDAR_HMR(pct_config).cuda()
    total = sum([param.nelement() for param in model.parameters()])
    bs = 8

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
        test(model, test_loader, dataset_path, split, save)