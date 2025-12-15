import faulthandler
faulthandler.enable()
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets.sloper4d import SLOPER4D_Dataset
from datasets.waymo_v2 import WAYMOV2_Dataset
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

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, dataloader, optimizer, epoch):
    model.train()
    loss_all = AverageMeter()
    loss_shift = AverageMeter()
    loss_joint = AverageMeter()
    loss_vert = AverageMeter()
    PRINT_FREQ = 500
    for i, sample in enumerate(dataloader):
        for key in sample:
            if type(sample[key]) is not dict and type(sample[key]) is not list:
                sample[key] = sample[key].cuda()
        pcd = sample['human_points_local']
        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()

        # flops = FlopCountAnalysis(model, pcd)
        # print("FLOPs: ", flops.total() / 1e9)

        # flops, params = profile(model, inputs = (pcd,))
        # flops, params = clever_format([flops, params], "%.3f")
        # print(flops, params)
        # import pdb; pdb.set_trace()
        
        ret_dict = model(pcd)
        loss_dict = model.all_loss(ret_dict, sample)
        all_loss = loss_dict['loss']
        if epoch >= 7:
            all_loss += loss_dict['edge_loss']
            loss_dict['loss_vert'] += loss_dict['edge_loss']
            # all_loss += loss_dict['pen_loss']
            # loss_dict['loss_vert'] += loss_dict['pen_loss']
            
        all_loss.backward()
        optimizer.step()
        loss_all.update(loss_dict['loss'].item())
        loss_joint.update(loss_dict['loss_joint'].item())
        loss_shift.update(loss_dict['loss_shift'].item())
        # if 'loss_vert' in loss_dict.keys():
        # import pdb; pdb.set_trace()
        loss_vert.update(loss_dict['loss_vert'].item())
        # print(loss_dict['loss_vert'])
        if i % PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Loss_all: {loss_all.val:.3f} ({loss_all.avg:.3f})\t' \
                  'loss_shift: {loss_shift.val:.3f} ({loss_shift.avg:.3f})\t' \
                    'loss_joint: {loss_joint.val:.3f} ({loss_joint.avg:.3f})\t' \
                    'loss_vert: {loss_vert.val:.3f} ({loss_vert.avg:.3f})'.format(
                    epoch, i, len(dataloader), loss_all=loss_all, loss_shift=loss_shift, \
                        loss_joint = loss_joint, loss_vert = loss_vert)
            print(msg)
            # logger.info(msg)



def interpolate_world_trajectory(pose_list, interval_num, intrinsic):
    n = len(pose_list)
    rot_list = [np.array(pose)[:3, :3] for pose in pose_list]
    t_list = [np.array(pose)[:3, 3] for pose in pose_list]

    # Interpolate rotation
    slerp = Slerp(range(n), R.from_matrix(rot_list))
    rot_interp_list = slerp(np.linspace(0, n-1, interval_num*(n-1)+1)).as_matrix()
    # Interpolate translation
    interp = interp1d(range(n), t_list, axis=0, kind='linear')
    t_interp_list = interp(np.linspace(0, n-1, interval_num*(n-1)+1))
    # import pdb; pdb.set_trace()
    # Generate trajectory
    param_list = []
    for rot, t in zip(rot_interp_list, t_interp_list):
        pose = compose(rot, t)
        extrinsic = pose #convert_coord_world_to_camera(pose)
        param = combine_camera_param(extrinsic, intrinsic)
        param_list.append(param)
    # trajectory = []#o3d.camera.PinholeCameraTrajectory()
    # trajectory.parameters = param_list
    return param_list

def combine_camera_param(extrinsic, intrinsic):
    param = o3d.camera.PinholeCameraParameters()
    param.extrinsic = extrinsic
    param.intrinsic = intrinsic
    return param

def compose(rotmat, t):
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotmat, t
    return pose

extrinsic_1=np.array([[
		-0.89708720579826495,
		-0.034173150294390997,
		0.4405300682042243,
		0.0,
		-0.41269527900787267,
		0.42098143273050526,
		-0.80774825284910723,
		0.0,
		-0.15785167682883725,
		-0.9064253025457355,
		-0.39176015497648087,
		0.0,
		3.5227150901381403,
		7.3423157880753296,
		0.31605670414597631,
		1.0
	]]).reshape(4,4).swapaxes(0,1)
extrinsic_2=np.array([[
		0.3068401243404672,
		-0.046503514730810921,
		0.95062430076892535,
		0.0,
		-0.94447387814632044,
		-0.13823896455238247,
		0.29809240543619142,
		0.0,
		0.11755097444928402,
		-0.98930653075635089,
		-0.086338615977194,
		0.0,
		-7.5635313088926299,
		4.7498668313466261,
		1.9980358781411716,
		1.0
	]]).reshape(4,4).swapaxes(0,1)
extrinsic_3=np.array([[
	    0.35160559611087217,
		-0.31755877731870036,
		0.88064177037622415,
		0.0,
		-0.93485037486949363,
		-0.1686233561512252,
		0.3124434994789686,
		0.0,
		0.049277595212078251,
		-0.93312517204721757,
		-0.35615885767725247,
		0.0,
		3.2870677718442085,
		8.5524152202898165,
		-3.0020802096379078,
		1.0
]]).reshape(4,4).swapaxes(0,1)

intrinsic = np.array([[
		693.68634843133543,
        0.0,
        0.0,
        0.0,
        693.68634843133543,
        0.0,
        767.5,
        400.0,
        1.0
	]]).reshape(3,3).swapaxes(0,1)

# extrinsic_1=np.array([[
# 		0.93100651629130216,
# 		0.24561441972756587,
# 		-0.27000078415631384,
# 		0.0,
# 		0.34652035344491439,
# 		-0.36235791559573605,
# 		0.86522851643570164,
# 		0.0,
# 		0.11467567864000139,
# 		-0.89909415403894422,
# 		-0.42246797618452758,
# 		0.0,
# 		-11.177958062075026,
# 		1.5570957759269806,
# 		-0.88254431059294824,
# 		1.0
# 	]]).reshape(4,4).swapaxes(0,1)
# extrinsic_2=np.array([[
# 		0.84918243938706095,
# 		0.28221468665174504,
# 		-0.44636762346153425,
# 		0.0,
# 		0.52768035324775042,
# 		-0.48711299033032768,
# 		0.69589825366053004,
# 		0.0,
# 		-0.021038760252686615,
# 		-0.82648400183518578,
# 		-0.56256694292992948,
# 		0.0,
# 		-3.3430918379474472,
# 		7.4437619347796913,
# 		0.50700492661491259,
# 		1.0
# 	]]).reshape(4,4).swapaxes(0,1)
# extrinsic_3=np.array([[
# 		0.83963636802678321,
# 		0.42229656870808535,
# 		-0.34157924050528848,
# 		0.0,
# 		0.52494801399343594,
# 		-0.46950081839482316,
# 		0.7099285626955284,
# 		0.0,
# 		0.13942866309026944,
# 		-0.77539318386478528,
# 		-0.6158937070021836,
# 		0.0,
# 		10.35573298652376,
# 		15.697113601967315,
# 		-8.4537068787341312,
# 		1.0
# 	]]).reshape(4,4).swapaxes(0,1)
# extrinsic_4=np.array([[
# 		-0.95144869591262793,
# 		-0.10209318241354115,
# 		0.29038312821311646,
# 		0.0,
# 		-0.30432785406727836,
# 		0.45345598513800262,
# 		-0.83771249649347412,
# 		0.0,
# 		-0.046151232756722442,
# 		-0.88541213660486884,
# 		-0.46250990483214072,
# 		0.0,
# 		-13.323414642167835,
# 		3.8699147029076375,
# 		2.3730658768694548,
# 		1.0
# 	]]).reshape(4,4).swapaxes(0,1)


width, height, fx, fy, cx, cy = 1536, 801, 693.68634843, 693.68634843, 768, 400.5
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
# import pdb; pdb.set_trace()
in_len = 1024
use_gt = True

def visualize_mesh(model, sample_info, data):

    if len(sample_info) == 3:
        frame, time, id = sample_info
        new_ = None
    else:
        frame, time, id, new_ = sample_info

    root, pcd, pred_vertices, pct_pose, pose, gt_pose = data

    pcd += root
    pred_vertices += root
    pct_pose += root
    pose += root
    gt_pose += root


    vis = o3d.visualization.Visualizer()
    vis.create_window(height=height, width = width)
    vis.get_render_option().point_size = 4.0
    center_in = [torch.tensor([0,0,0]).unsqueeze(0)]

    meshes = []
    pcd_k = o3d.open3d.geometry.PointCloud()
    pcd_k.points= o3d.open3d.utility.Vector3dVector(pcd)
    pcd_k.paint_uniform_color([0.5,0.5,0.5])

    pcd_k.translate((0, 1, 0))  # 将 pcd2 移动到坐标 (2, 0, 0)

    pcd_m = o3d.open3d.geometry.PointCloud()
    pcd_m.points= o3d.open3d.utility.Vector3dVector(pcd)
    pcd_m.paint_uniform_color([0.5,0.5,0.5])
    pcd_m.translate((0, -1, 0))  # 将 pcd2 移动到坐标 (2, 0, 0)

    pct_pose_ = o3d.open3d.geometry.PointCloud()
    pct_pose_.points= o3d.open3d.utility.Vector3dVector(pct_pose)
    pct_pose_.paint_uniform_color([1,0,0])

    pct_pose_.translate((0, 1, 0))  # 将 pcd2 移动到坐标 (2, 0, 0)

    pose_ = o3d.open3d.geometry.PointCloud()
    pose_.points= o3d.open3d.utility.Vector3dVector(pose)
    pose_.paint_uniform_color([0,0,1])

    pose_.translate((0, 1, 0))  # 将 pcd2 移动到坐标 (2, 0, 0)

    gt_pose_ = o3d.open3d.geometry.PointCloud()
    gt_pose_.points= o3d.open3d.utility.Vector3dVector(gt_pose)
    gt_pose_.paint_uniform_color([0,1,0])

    gt_pose_.translate((0, 1, 0))  # 将 pcd2 移动到坐标 (2, 0, 0)

    # 创建一个 LineSet 对象来表示骨骼
    bone_link = np.array([[0,1], [0,2], [1,3], [2,4], [3,5], [4,6], [0,7], [7,8], [7,9], [7,10], [9,11], [10,12], [11,13], [12,14]])

    pct_lineset = o3d.geometry.LineSet()
    pct_lineset.points = o3d.open3d.utility.Vector3dVector(pct_pose)
    pct_lineset.lines = o3d.utility.Vector2iVector(bone_link)
    pct_lineset.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(bone_link))])  # 设置线为绿色
    pct_lineset.translate((0, 1, 0))  # 将 pcd2 移动到坐标 (2, 0, 0)

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.open3d.utility.Vector3dVector(pose)
    lineset.lines = o3d.utility.Vector2iVector(bone_link)
    lineset.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(bone_link))])  # 设置线为绿色
    lineset.translate((0, 1, 0))  # 将 pcd2 移动到坐标 (2, 0, 0)

    gt_lineset = o3d.geometry.LineSet()
    gt_lineset.points = o3d.open3d.utility.Vector3dVector(gt_pose)
    gt_lineset.lines = o3d.utility.Vector2iVector(bone_link)
    gt_lineset.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(bone_link))])  # 设置线为绿色
    gt_lineset.translate((0, 1, 0))  # 将 pcd2 移动到坐标 (2, 0, 0)
    
    # meshes.append(pcd_)
    # for ind_v, p_v in enumerate(pred_vertices.cpu().numpy()):
    #     p_k = p_v + center_in[ind_v].numpy()
    #     meshes.append(get_mesh(p_k, model.pmg.smpl.faces))

    # o3d.visualization.draw_geometries([meshes[0]])

    # o3d.visualization.draw_geometries(meshes)

    # material = o3d.visualization.rendering.MaterialRecord()
    # material.base_color = [0.5, 0.5, 0.5, 0.3]  # 设置透明度（最后一个值0.3代表30%的不透明）
    # renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    for ind_v, p_v in enumerate(np.expand_dims(pred_vertices, axis=0)):
        p_k = p_v + center_in[ind_v].numpy()
        mesh = get_mesh(p_k, model.pmg.smpl.faces)
        mesh.translate((0, -1, 0))  # 将 pcd2 移动到坐标 (2, 0, 0)
        vis.add_geometry(mesh)

    # 添加点云数据
    vis.add_geometry(pcd_k) 
    vis.add_geometry(pcd_m) 
    vis.add_geometry(pct_pose_)
    vis.add_geometry(pct_lineset)

    vis.add_geometry(pose_)
    vis.add_geometry(lineset)
    vis.add_geometry(gt_pose_)
    vis.add_geometry(gt_lineset)


    # o3d.visualization.draw_geometries([pcd_k, pcd_m, pct_pose_, pct_lineset, pose_, lineset, gt_pose_, gt_lineset, mesh])

    # 获取视角控制对象
    ctr = vis.get_view_control()

    # # 设置视角的旋转角度（通过设置目标点的位置来实现）
    # ctr.set_lookat([0, 0, 0])  # 视角指向坐标 [0, 0, 0]
    ctr.set_up([0, 0, 1])      # 设置上方向的向量（如Z轴向上）
    ctr.set_front([-1, 0, 0])   # 设置视角前方的方向（如X轴方向）
    save_folder = 'demo_waymo/annular2'
    save_folder += f'/{id}'
    os.makedirs(save_folder, exist_ok = True)
    if new_ is not None:
        vis.capture_screen_image(os.path.join(save_folder, f"{frame}_{time}_{new_}.png"),do_render=True)
    else:
        vis.capture_screen_image(os.path.join(save_folder, f"{frame}_{time}.png"),do_render=True)
    vis.clear_geometries()  
        # 显示可视化窗口
        # vis.run()
        # vis.destroy_window()

        # vis.poll_events()
        # vis.update_renderer()
        # save_folder = 'demo_waymo/annular2'
        # os.makedirs(save_folder, exist_ok = True)
        # vis.capture_screen_image(os.path.join(save_folder, f'{1:04d}.png'),do_render=True)
        # vis.clear_geometries()   

        # vis.add_geometry(pcd_)
        
        # # vis.add_geometry(get_mesh(p_k, model.smpl.faces))
        # for ind_v, p_v in enumerate(pred_vertices.cpu().numpy()):
        #     p_k = p_v + center_in[ind_v].numpy()
        #     vis.add_geometry(get_mesh(p_k, model.pmg.smpl.faces))
            
        # param1 = interpolate_world_trajectory([extrinsic_1, extrinsic_2], 60, intrinsic)
        # param2 = interpolate_world_trajectory([extrinsic_2, extrinsic_3], 60, intrinsic)
        # # param3 = interpolate_world_trajectory([extrinsic_3, extrinsic_4], 60, intrinsic)
        # param_list = param1 + param2 #+ param3
        # for indp, param in enumerate(param_list):
        #     vis.get_view_control().convert_from_pinhole_camera_parameters(param)
        #     vis.poll_events()
        #     vis.update_renderer()
        #     save_folder = 'demo_waymo/annular2'
        #     os.makedirs(save_folder, exist_ok = True)
        #     vis.capture_screen_image(os.path.join(save_folder, f"{frame}_{time}_{indp:04d}.png"),do_render=True)
        # vis.clear_geometries()   


# 定义多项式拟合函数
def poly_func(t, a, b):
    return a * t + b 

# 定义一个简单的正弦函数作为拟合模型
def sine_model(t, a, b, c, d):
    return a * np.sin(b * t + c) + d

# 定义二次函数模型
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

# 曲线拟合函数
def fit_curve(t, uniform_relative_timestamps, data):
    """根据数据数量选择拟合方法"""
    n, m, k = data.shape  # 获取数据的形状
    fitted_data = np.zeros((len(uniform_relative_timestamps), m, k))  # 存储拟合后的数据
    
    # for idx, item in enumerate(data):
    for i in range(k):
        for j in range(m):

            if n < 3:  # 如果数据点少于3个，使用简单的线性拟合
                popt, _ = curve_fit(poly_func, t, data[:,j,i])  # 线性拟合

                fitted_data[:, j, i] = poly_func(uniform_relative_timestamps , *popt)  # 使用拟合的参数生成拟合结果

            else:  # 否则，使用正弦波拟合
                popt, _ = curve_fit(quadratic_model, t, data[:,j,i])

                fitted_data[:, j, i] = quadratic_model(uniform_relative_timestamps , *popt)  # 使用拟合的参数生成拟合结果

    return fitted_data

def test(model, dataloader, save = False):
    model.eval()
    mpjpe = 0
    mpvpe = 0
    mpee = 0
    mpere = 0
    number = 0
    precision = 0
    print('============Testing============')
    mesh_face = model.smpl_.faces
    # save = False
    if save:
        saver_dict = {}
        saver_dict['pcd'] = []
        saver_dict['mesh_o'] = []
        saver_dict['mesh_r'] = []
        saver_dict['mesh_gt'] = []
        saver_dict['mpvpe_o'] = []
        saver_dict['mpere_o'] = []
        saver_dict['mpvpe_r'] = []
        saver_dict['mpere_r'] = []
        saver_dict['number_list'] = []
        saver_dict['pose_theta'] = []
        saver_dict['pose_beta'] = []
        saver_dict['pose_trans'] = []

    J_r = smpl_model.J_regressor.cuda()

    data_dic = {}
    
    for sample in tqdm(dataloader):
        for key in sample:
            if type(sample[key]) is not dict and type(sample[key]) is not list:
                sample[key] = sample[key].cuda()
        pcd = sample['human_points_local']
        gt_pose = sample['smpl_joints_local']

        root = sample['global_trans']

        with torch.no_grad():
            ret_dict = model(pcd)#， sample['betas'])
        
        seg = ret_dict['seg']
        gt_seg = sample['seg_label'].to(seg.device)
        pct_pose = ret_dict['pct_pose']

        precision += (seg.argmax(dim = 1) == gt_seg).sum() / gt_seg.numel() 
        pred_vertices = ret_dict['mesh_refine']
        pose = torch.einsum('bnk,mn->bmk', [pred_vertices, J_regressor])[:,JOINTS_IDX]
        gt_vertices = sample['smpl_verts_local'].to(pred_vertices.device)
        if args.dataset == 'lidarh26m':
            pred_vertices -= pose[:,[0]]
            gt_vertices -= gt_pose[:,[0]]
            pose -= pose[:,[0]]
            gt_pose -= gt_pose[:,[0]]

        # # 计算差值
        # difference = pct_pose - pose

        # # 求取差值的绝对值（逐元素差值的绝对值）
        # absolute_difference = torch.abs(difference)

        # # 求取差值的均值（可作为整体差异的度量）
        # mean_difference = torch.mean(absolute_difference)

        # # 输出结果
        # print("逐元素差值：", difference)
        # print("差值的绝对值：", absolute_difference)
        # print("差值的均值：", mean_difference.item())
        # bone_link = np.array([[0,1], [0,2], [1,3], [2,4], [3,5], [4,6], [0,7], [7,8], [7,9], [7,10], [9,11], [10,12], [11,13], [12,14]])
        # pose_show = pct_pose[0].detach().cpu().numpy()

        frame = sample['location_dict']['frame'][0]
        time = sample['location_dict']['time'][0]
        id = sample['location_dict']['id'][0]

        sample_info = [frame, time, id]


        if id in data_dic.keys():
            data_dic[id].append([sample_info,[root.cpu().numpy(), pcd.cpu().numpy().squeeze(axis=0), pred_vertices.cpu().numpy().squeeze(axis=0),
                pct_pose.cpu().numpy().squeeze(axis=0), pose.cpu().numpy().squeeze(axis=0), gt_pose.cpu().numpy().squeeze(axis=0)]])
        else:
            data_dic[id] = []
            data_dic[id].append([sample_info,[root.cpu().numpy(), pcd.cpu().numpy().squeeze(axis=0), pred_vertices.cpu().numpy().squeeze(axis=0), 
                pct_pose.cpu().numpy().squeeze(axis=0), pose.cpu().numpy().squeeze(axis=0), gt_pose.cpu().numpy().squeeze(axis=0)]])


        # visualize_mesh(model,sample_info, pcd, pred_vertices, pct_pose, pose, gt_pose)

        
        num_person = pose.shape[0]
        mpjpe += (pose - gt_pose.to(pose.device)).norm(dim = -1).mean() * num_person
        # gt_root = torch.einsum('ji,bik->bjk', [J_r, gt_vertices])[:,[0]]
        # pred_root = torch.einsum('ji,bik->bjk', [J_r, pred_vertices])[:,[0]]
        # pred_vertices -= pred_root
        # gt_vertices -= gt_root

        error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices)
        error_edges, error_relative_edges = mean_per_edge_error(pred_vertices, gt_vertices, mesh_face)
        # import pdb; pdb.set_trace()
        mpvpe += error_vertices.mean() * num_person
        mpee += error_edges * num_person
        mpere += error_relative_edges.mean() * num_person
        number += num_person
        if save:
            saver_dict['pcd'].append(pcd.detach().cpu().numpy())
            saver_dict['mesh_o'].append(ret_dict['mesh_out'].detach().cpu().numpy())
            saver_dict['mesh_r'].append(ret_dict['mesh_refine'].detach().cpu().numpy())
            saver_dict['mesh_gt'].append(gt_vertices.detach().cpu().numpy())
            saver_dict['mpvpe_r'].append(error_vertices)
            saver_dict['mpere_r'].append(error_relative_edges.mean(dim = -1).numpy())

            error_vertices = mean_per_vertex_error(ret_dict['mesh_out'], gt_vertices)
            error_edges, error_relative_edges = mean_per_edge_error(ret_dict['mesh_out'], gt_vertices, mesh_face)
            saver_dict['mpvpe_o'].append(error_vertices)
            saver_dict['mpere_o'].append(error_relative_edges.mean(dim = -1).numpy())
            saver_dict['number_list'].append(sample['num_points'].cpu().numpy())
            saver_dict['pose_theta'].append(ret_dict['pose_theta'].cpu().numpy())
            saver_dict['pose_beta'].append(ret_dict['pose_beta'].cpu().numpy())
            saver_dict['pose_trans'].append(ret_dict['trans'].cpu().numpy())
        # beta_, theta_, trans_ = ret_dict['pose_beta'], ret_dict['pose_theta'], ret_dict['trans']
        # import pdb; pdb.set_trace()
        # vertices = smpl_model(transl = trans_, betas = beta_, global_orient = theta_[:,:3], body_pose = theta_[:,3:]).vertices
        # import pdb; pdb.set_trace()

    for key, value in data_dic.items():
        id = key
        data_dic[key] = sorted(value, key=lambda x: int( x[0][1] ))

    # 每秒 60 帧
    target_fps = 60
    target_interval = 1 / target_fps

    # 遍历每个 id 的数据
    for id, data_list in data_dic.items():
        if len(data_list) > 1:

            # 转换时间戳为相对时间
            original_time_stamps = np.array([int(entry[0][1]) for entry in data_list])
            time_stamps = np.array([int(entry[0][1]) for entry in data_list]) / 1e6

            root = np.array([entry[1][0] for entry in data_list])  # 提取 pose 数据
            pcd = np.array([entry[1][1] for entry in data_list])  # 提取 pose 数据
            pred_vertices = np.array([entry[1][2] for entry in data_list])  # 提取 pose 数据
            pct_pose= np.array([entry[1][3] for entry in data_list])  # 提取 pose 数据
            pose= np.array([entry[1][4] for entry in data_list])  # 提取 pose 数据
            gt_pose = np.array([entry[1][5] for entry in data_list])  # 提取 pose 数据




            # 转换为相对时间
            relative_timestamps = time_stamps - time_stamps[0]


            # 生成均匀时间序列（每秒30个点）
            duration = relative_timestamps[-1]
            uniform_relative_timestamps = np.linspace(0, duration, int(duration * 60))

            # 对 pose 和 pct_pose 使用曲线拟合
            fitted_root = fit_curve(relative_timestamps, uniform_relative_timestamps, root)

            fitted_pcd = fit_curve(relative_timestamps, uniform_relative_timestamps, pcd)
            fitted_pred_vertices = fit_curve(relative_timestamps, uniform_relative_timestamps, pred_vertices)
            fitted_pct_pose = fit_curve(relative_timestamps, uniform_relative_timestamps, pct_pose)
            fitted_pose = fit_curve(relative_timestamps, uniform_relative_timestamps, pose)
            fitted_gt_pose= fit_curve(relative_timestamps, uniform_relative_timestamps, gt_pose)

            # 重新构建结果
            uniform_absolute_timestamps = (uniform_relative_timestamps + time_stamps[0]) * 1e6

            # 输出格式
            fitted_data = []
            for i in range(len(uniform_absolute_timestamps)):

                if int(uniform_absolute_timestamps[i]) not in original_time_stamps:
                    fitted_data.append([[data_list[0][0][0], int(uniform_absolute_timestamps[i]), id, 'new'], [fitted_root[i],fitted_pcd[i], fitted_pred_vertices[i], fitted_pct_pose[i], fitted_pose[i], fitted_gt_pose[i]]])
            
            # 更新原数据
            data_dic[id].extend(fitted_data)

            # 按时间排序
            data_dic[id].sort(key=lambda x: int( x[0][1] ))

            for data_ in data_dic[id]:

                visualize_mesh(model, data_[0], data_[1])

    



    
    if save:
        saver_dict['mesh_o'] = np.concatenate(saver_dict['mesh_o'], axis = 0)
        saver_dict['mesh_r'] = np.concatenate(saver_dict['mesh_r'], axis = 0)
        saver_dict['mesh_gt'] = np.concatenate(saver_dict['mesh_gt'], axis = 0)
        saver_dict['mpvpe_r'] = np.concatenate(saver_dict['mpvpe_r'], axis = 0)
        saver_dict['mpere_r'] = np.concatenate(saver_dict['mpere_r'], axis = 0)
        saver_dict['mpvpe_o'] = np.concatenate(saver_dict['mpvpe_o'], axis = 0)
        saver_dict['mpere_o'] = np.concatenate(saver_dict['mpere_o'], axis = 0)
        saver_dict['pose_theta'] = np.concatenate(saver_dict['pose_theta'], axis = 0)
        saver_dict['pose_beta'] = np.concatenate(saver_dict['pose_beta'], axis = 0)
        saver_dict['pose_trans'] = np.concatenate(saver_dict['pose_trans'], axis = 0)
        saver_dict['pcd'] = np.concatenate(saver_dict['pcd'], axis = 0)
        # saver_dict['number_list'] = np.concatenate(saver_dict['number_list'], axis = 0)
        np.save('save_meshik/'+dataset_task+'.npy', np.array(saver_dict))
    mpjpe = mpjpe.item() / number    
    precision = precision.item() / number
    mpvpe = mpvpe.item() / number
    mpee = mpee.item() / number
    mpere = mpere.item() / number
    return mpjpe, precision, mpvpe, mpee, mpere

args = parse_args()
setup_seed(10)
dataset_task = args.dataset #'sloper4d', 'waymov2', 'collect'
if dataset_task == 'sloper4d':
    scene_train = [
            'seq002_football_001',
            'seq003_street_002',
            'seq005_library_002',
            'seq007_garden_001',
            'seq008_running_001'
        ]
    scene_test = ['seq009_running_002']
    dataset_root = '/Extra/fanbohao/posedataset/PointC/sloper4d/'
    train_dataset = SLOPER4D_Dataset(dataset_root, scene_train, is_train = True, dataset_path = './save_data/sloper4d/',
                                return_torch=False, device = 'cuda',
                                fix_pts_num=True, return_smpl = True, augmentation = True, interval = 5)
    test_dataset = SLOPER4D_Dataset(dataset_root, scene_test, is_train = False, dataset_path = './save_data/sloper4d/',
                                return_torch=False, device = 'cuda',
                                fix_pts_num=True, return_smpl = True, interval = 5)
    num_keypoints = 15

elif dataset_task == 'waymov2':
    dataset_root = '/workspace/code/LiDAR-HMR/'
    train_dataset = WAYMOV2_Dataset(dataset_root, is_train = True, dataset_path = './save_data/waymov2/',
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True)
    test_dataset = WAYMOV2_Dataset(dataset_root, is_train = False, dataset_path = './save_data/waymov2/',
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True)
    
    merged_dataset = WAYMOV2_Dataset(dataset_root, is_train = False, dataset_path = './save_data/waymov2/',
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, merged = True)
    num_keypoints = 15

elif dataset_task == 'humanm3':
    train_dataset = HumanM3_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, interval = 5)
    test_dataset = HumanM3_Dataset(is_train = False,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, interval = 5)
    num_keypoints = 15

elif dataset_task == 'cimi4d':
    train_dataset = CIMI4D_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, interval = 5)
    test_dataset = CIMI4D_Dataset(is_train = False,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, interval = 5)
    num_keypoints = 15

elif dataset_task == 'lidarh26m':
    train_dataset = LiDARH26M_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, interval = 5)
    test_dataset = LiDARH26M_Dataset(is_train = False,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, interval = 5)
    num_keypoints = 15

# model = pose_meshformer().cuda()
config_name = args.cfg.split('/')[-1].split('.')[0]
update_config(args.cfg)
model = LiDAR_HMR(pmg_cfg = config, train_pmg = True).cuda()
# model = LiDAR_HMR(pct_config).cuda()

total = sum([param.nelement() for param in model.parameters()])
bs = 1
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers = 8)

test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers = 8)
merged_loader = DataLoader(merged_dataset, batch_size=bs, shuffle=False, num_workers = 8)

optimizer = torch.optim.Adam(params=list(model.parameters()),
                                           lr=5e-4,
                                           betas=(0.9, 0.999),
                                           weight_decay=0)
epoch_now = -1

if args.state_dict != '':
    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict['net'])
    
save = True
mpjpe, precision, mpvpe, mpee, mpere = test(model, merged_loader, save)
print('MPJPE: '+str(mpjpe) + '; Precision:' + str(precision) + '; MPVPE:'+str(mpvpe) + '; MPERE:'+str(mpere))
# save = False
# if save:
#     np.save('statistics/'+str(args.dataset)+'.npy', np.array(sta_dict))