import pickle as pkl
import os
import numpy as np
import smplx
import torch
from tqdm import tqdm
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d, PchipInterpolator, CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def correct_smpl_global_orient(
    trans: np.ndarray,
    root_orient: np.ndarray,
    smooth_sigma: float = 1.0,
    angle_threshold_deg: float = 90.0,
    epsilon: float = 1e-6,
    motion_threshold: float = 2e-2,
    motion_smooth_sigma: float = 2.0  # 对速度平滑的标准差
) -> np.ndarray:
    """
    基于 Frenet 切向量构建目标旋转，将 SMPL 根节点全局朝向对齐至运动方向。
    仅在 XY 平面运动时进行纠正。使用平滑后的速度判断“几乎不动”。

    参数:
    - trans: (N,3) 根节点平移参数
    - root_orient: (N,3) 全局朝向 axis-angle 参数
    - smooth_sigma: 对 trans 做高斯滤波的标准差
    - angle_threshold_deg: 检测是否需要修正的角度阈值（度）
    - epsilon: 数值稳定项
    - motion_threshold: XY 平面判断“几乎不动”的速度阈值
    - motion_smooth_sigma: 对速度做高斯平滑的标准差

    返回:
    - corrected_orient: (N,3) 修正后的全局朝向 axis-angle 参数
    """
    N = trans.shape[0]
    # 1. 平滑轨迹
    P_smooth = gaussian_filter1d(trans, sigma=smooth_sigma, axis=0)

    # 2. 计算三维切向量（中心差分）
    T3 = np.zeros((N, 3))
    for i in range(N):
        if 0 < i < N - 1:
            v = P_smooth[i + 1] - P_smooth[i - 1]
        elif i == 0:
            v = P_smooth[1] - P_smooth[0]
        else:
            v = P_smooth[-1] - P_smooth[-2]
        norm3 = np.linalg.norm(v) + epsilon
        T3[i] = v / norm3

    # 3. 水平投影
    T2 = T3.copy()
    T2[:, 2] = 0
    norms2 = np.linalg.norm(T2, axis=1)
    valid = norms2 > epsilon
    T2[valid] = (T2[valid].T / norms2[valid]).T
    for i in range(N):
        if not valid[i]:
            T2[i] = T2[i - 1] if i > 0 else np.array([1.0, 0.0, 0.0])

    # 4. 计算平滑后的 XY 平面速度
    vel_xy = np.linalg.norm(np.diff(P_smooth[:, :2], axis=0), axis=1)
    vel_xy = np.concatenate([[vel_xy[0]], vel_xy])  # 补齐长度
    vel_xy_smooth = gaussian_filter1d(vel_xy, sigma=motion_smooth_sigma)

    # 5. 定义全局 up 向量
    up = np.array([0.0, 0.0, 1.0])

    # 6. 计算目标旋转矩阵与修正
    corrected_rots = []
    raw_rots = Rotation.from_rotvec(root_orient)
    angle_thresh = np.deg2rad(angle_threshold_deg)

    for i in range(N):
        # 如果平滑速度很小，则认为几乎静止，保留原始朝向
        if vel_xy_smooth[i] < motion_threshold:
            corrected_rots.append(raw_rots[i])
            continue

        z_des = T2[i]
        x_des = np.cross(up, z_des)
        if np.linalg.norm(x_des) < epsilon:
            corrected_rots.append(raw_rots[i])
            continue
        x_des /= np.linalg.norm(x_des)
        y_des = np.cross(z_des, x_des)

        R_des = np.stack([x_des, y_des, z_des], axis=1)
        R_raw = raw_rots[i].as_matrix()
        R_align = R_des @ R_raw.T

        angle = np.arccos(np.clip((np.trace(R_align) - 1) / 2, -1.0, 1.0))
        if angle > angle_thresh:
            q_align = Rotation.from_matrix(R_align)
            q_new = q_align * raw_rots[i]
            corrected_rots.append(q_new)
        else:
            corrected_rots.append(raw_rots[i])

    # 7. 转回 axis-angle
    corrected_rot = Rotation.concatenate(corrected_rots)
    corrected_orient = corrected_rot.as_rotvec()
    return corrected_orient



# def correct_smpl_global_orient(
#     trans: np.ndarray,
#     root_orient: np.ndarray,
#     smooth_sigma: float = 1.0,
#     angle_threshold_deg: float = 90.0,
#     epsilon: float = 1e-6
# ) -> np.ndarray:
#     """
#     基于完整的 Frenet 切向量构建目标旋转，将 SMPL 根节点全局朝向对齐至运动方向。

#     参数:
#     - trans: (N,3) 根节点平移参数
#     - root_orient: (N,3) 全局朝向 axis-angle 参数
#     - smooth_sigma: 对 trans 做高斯滤波的标准差
#     - angle_threshold_deg: 检测是否需要修正的角度阈值（度）
#     - epsilon: 数值稳定项

#     返回:
#     - corrected_orient: (N,3) 修正后的全局朝向 axis-angle 参数
#     """
#     N = trans.shape[0]
#     # 1. 平滑根节点轨迹
#     P_smooth = gaussian_filter1d(trans, sigma=smooth_sigma, axis=0)

#     # 2. 计算三维切向量 T 和其在水平面的投影 T2
#     T3 = np.zeros((N, 3))
#     for i in range(N):
#         if 0 < i < N-1:
#             v = P_smooth[i+1] - P_smooth[i-1]
#         elif i == 0:
#             v = P_smooth[1] - P_smooth[0]
#         else:
#             v = P_smooth[-1] - P_smooth[-2]
#         norm3 = np.linalg.norm(v) + epsilon
#         T3[i] = v / norm3
#     # 水平投影
#     T2 = T3.copy()
#     T2[:, 2] = 0
#     norms2 = np.linalg.norm(T2, axis=1)
#     valid = norms2 > epsilon
#     T2[valid] = (T2[valid].T / norms2[valid]).T
#     for i in range(N):
#         if not valid[i]:
#             T2[i] = T2[i-1] if i>0 else np.array([1.0, 0.0, 0.0])

#     # 3. 定义全局 up 向量
#     up = np.array([0.0, 0.0, 1.0])

#     # 4. 计算目标旋转矩阵和原始旋转
#     corrected_rots = []
#     angle_thresh = np.deg2rad(angle_threshold_deg)
#     raw_rots = Rotation.from_rotvec(root_orient)
#     for i in range(N):
#         z_des = T2[i]
#         # x轴方向: up × z_des
#         x_des = np.cross(up, z_des)
#         if np.linalg.norm(x_des) < epsilon:
#             # 如果运动方向接近 up 或 down，用前一帧结果
#             corrected_rots.append(raw_rots[i])
#             continue
#         x_des /= np.linalg.norm(x_des)
#         y_des = np.cross(z_des, x_des)
#         # 构建目标旋转矩阵 [x_des, y_des, z_des]
#         R_des = np.stack([x_des, y_des, z_des], axis=1)
#         # 原始旋转矩阵
#         R_raw = raw_rots[i].as_matrix()
#         # 计算两个旋转矩阵之间的差: R_align * R_raw = R_des → R_align = R_des * R_raw^T
#         R_align = R_des @ R_raw.T
#         angle = np.arccos(np.clip((np.trace(R_align) - 1) / 2, -1.0, 1.0))
#         if angle > angle_thresh:
#             # 应用对齐旋转
#             q_align = Rotation.from_matrix(R_align)
#             q_new = q_align * raw_rots[i]
#             corrected_rots.append(q_new)
#         else:
#             corrected_rots.append(raw_rots[i])

#     # 5. 转回 axis-angle
#     corrected_rot = Rotation.concatenate(corrected_rots)
#     corrected_orient = corrected_rot.as_rotvec()
#     return corrected_orient


def interpolate_axis_angle(axis_angles, t_original, t_target):
    """处理旋转参数插值（支持外推）"""
    # 分离内插和外推时间点
    t_min, t_max = t_original.min(), t_original.max()
    mask_valid = (t_target >= t_min) & (t_target <= t_max)
    t_valid = t_target[mask_valid]
    t_extrap_low = t_target[t_target < t_min]
    t_extrap_high = t_target[t_target > t_max]
    
    # 内插部分
    rots = Rotation.from_rotvec(axis_angles)
    # interp_rots_valid = Rotation.from_rotvec([])  # 空旋转序列
    
    if len(t_valid) > 0:
        slerp = Slerp(t_original, rots)
        interp_rots_valid = slerp(t_valid)
    
    # 外推部分（复制首尾帧）
    # 修复错误：使用 Rotation.concatenate 合并旋转对象
    if len(t_extrap_low) > 0:
        rot_low = Rotation.concatenate([rots[0]] * len(t_extrap_low))
        interp_rots_valid = Rotation.concatenate([ rot_low, interp_rots_valid])
        t_valid = np.concatenate([t_valid, t_extrap_low])
    
    if len(t_extrap_high) > 0:
        rot_high = Rotation.concatenate([rots[-1]] * len(t_extrap_high))
        interp_rots_valid = Rotation.concatenate([interp_rots_valid, rot_high])
        t_valid = np.concatenate([t_valid, t_extrap_high])
    
    # 按原始顺序重新排列
    sort_idx = np.argsort(t_valid)
    return interp_rots_valid[sort_idx].as_rotvec()


def interpolate_axis_angle_new(axis_angles, t_data_sorted, t_original):
    """
    处理旋转参数插值（支持外推），保留已有数据。
    axis_angles: (N,3) 原始旋转向量
    t_data_sorted: 原始时间戳（秒），已排序
    t_original: 目标时间戳（秒）
    """
    axis_angles = np.array(axis_angles)
    full_rot = np.zeros((len(t_original), 3))

    # 找到原始时间点中已有数据的位置
    mask_exact = np.isin(t_original, t_data_sorted)
    exact_idx_in_data = {t: i for i, t in enumerate(t_data_sorted)}

    # 直接赋值已有数据
    for idx, t_val in enumerate(t_original):
        if mask_exact[idx]:
            full_rot[idx] = axis_angles[exact_idx_in_data[t_val]]

    # 对新增时间点进行插值/外推
    mask_interp = ~mask_exact
    if np.any(mask_interp):
        t_min, t_max = t_data_sorted[0], t_data_sorted[-1]
        rots = Rotation.from_rotvec(axis_angles)
        slerp = Slerp(t_data_sorted, rots)

        for idx in np.where(mask_interp)[0]:
            t = t_original[idx]
            if t < t_min:
                full_rot[idx] = rots[0].as_rotvec()  # 低于最小时间，复制首帧
            elif t > t_max:
                full_rot[idx] = rots[-1].as_rotvec()  # 超过最大时间，复制尾帧
            else:
                full_rot[idx] = slerp([t]).as_rotvec()[0]  # 内插

    return full_rot
# def interpolate_axis_angle(axis_angles, t_original, t_target):
#     """处理旋转参数插值（支持外推）"""
#     # 分离内插和外推时间点
#     t_min, t_max = t_original.min(), t_original.max()
#     mask_valid = (t_target >= t_min) & (t_target <= t_max)
#     t_valid = t_target[mask_valid]
#     t_extrap_low = t_target[t_target < t_min]
#     t_extrap_high = t_target[t_target > t_max]
    
#     # 内插部分
#     rots = Rotation.from_rotvec(axis_angles)
#     # interp_rots_valid = Rotation.from_rotvec([])  # 空旋转序列
    
#     if len(t_valid) > 0:
#         slerp = Slerp(t_original, rots)
#         interp_rots_valid = slerp(t_valid)
    
#     # 外推部分（复制首尾帧）
#     # 修复错误：使用 Rotation.concatenate 合并旋转对象
#     if len(t_extrap_low) > 0:
#         rot_low = Rotation.concatenate([rots[0]] * len(t_extrap_low))
#         interp_rots_valid = Rotation.concatenate([interp_rots_valid, rot_low])
#         t_valid = np.concatenate([t_valid, t_extrap_low])
    
#     if len(t_extrap_high) > 0:
#         rot_high = Rotation.concatenate([rots[-1]] * len(t_extrap_high))
#         interp_rots_valid = Rotation.concatenate([interp_rots_valid, rot_high])
#         t_valid = np.concatenate([t_valid, t_extrap_high])
    
#     # 按原始顺序重新排列
#     sort_idx = np.argsort(t_valid)
#     return interp_rots_valid[sort_idx].as_rotvec()

def slerp(v0, v1, t):
    """对单位向量做球面线性插值 (Spherical Linear intERPolation)"""
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    theta = np.arccos(dot)
    if np.abs(theta) < 1e-6:
        return v0
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * v0 + (np.sin(t * theta) / sin_theta) * v1

def interpolate_world_params_linear(original_timestamps, data_timestamps, 
                                    world_center, world_size, speed, heading_world):
    """
    对 world_center, world_size, speed 使用线性插值
    heading_world 使用球面插值保持单位向量
    """
    # 时间预处理
    t_min_global = np.min(original_timestamps)
    t_original = (original_timestamps - t_min_global).astype(np.float64) / 1e6
    t_data = (data_timestamps - t_min_global).astype(np.float64) / 1e6

    # 排序
    sort_idx = np.argsort(t_data)
    t_data_sorted = t_data[sort_idx]
    world_center_sorted = np.array(world_center)[sort_idx]
    world_size_sorted = np.array(world_size)[sort_idx]
    speed_sorted = np.array(speed)[sort_idx]
    heading_sorted = np.array(heading_world)[sort_idx]

    

    # 1️⃣ world_center (Nx3) 线性插值
    interp_center = np.zeros((len(t_original), 3))
    for i in range(3):
        fn = interp1d(t_data_sorted, world_center_sorted[:, i], kind='linear', fill_value='extrapolate')
        interp_center[:, i] = fn(t_original)
    new_world_center = interp_center

    # 2️⃣ world_size (Nx3) 线性插值
    interp_size = np.zeros((len(t_original), 3))
    for i in range(3):
        fn = interp1d(t_data_sorted, world_size_sorted[:, i], kind='linear', fill_value='extrapolate')
        interp_size[:, i] = fn(t_original)
    new_world_size = interp_size

    # 3️⃣ speed (Nx3) 线性插值
    interp_speed = np.zeros((len(t_original), 3))
    for i in range(3):
        fn = interp1d(t_data_sorted, speed_sorted[:, i], kind='linear', fill_value='extrapolate')
        interp_speed[:, i] = fn(t_original)
    new_speed = interp_speed

    # 4️⃣ heading_world (Nx3) SLERP 保持单位向量
    interp_heading = np.zeros((len(t_original), 3))
    for ti, t in enumerate(t_original):
        if t <= t_data_sorted[0]:
            interp_heading[ti] = heading_sorted[0]
        elif t >= t_data_sorted[-1]:
            interp_heading[ti] = heading_sorted[-1]
        else:
            idx = np.searchsorted(t_data_sorted, t) - 1
            t0, t1 = t_data_sorted[idx], t_data_sorted[idx + 1]
            ratio = (t - t0) / (t1 - t0)
            interp_heading[ti] = slerp(heading_sorted[idx], heading_sorted[idx + 1], ratio)
    new_heading = interp_heading / np.linalg.norm(interp_heading, axis=1, keepdims=True)

    return new_world_center, new_world_size, new_speed, new_heading

def interpolate_smpl_params(original_timestamps, data_timestamps, data_smpl, data_global_dis = None):
    """支持外推的 SMPL 参数插值"""
    # --- 预处理时间戳 ---
    t_min_global = np.min(original_timestamps)
    t_original = (original_timestamps - t_min_global).astype(np.float64) / 1e6
    t_data = (data_timestamps - t_min_global).astype(np.float64) / 1e6
    
    # 确保数据时间戳单调递增
    sort_idx = np.argsort(t_data)
    t_data_sorted = t_data[sort_idx]
    pose_body_sorted = data_smpl['pose_body'][sort_idx]
    root_orient_sorted = data_smpl['root_orient'][sort_idx]
    trans_sorted = data_smpl['trans'][sort_idx]
    
    # 初始化输出
    full_smpl = {
        'betas': data_smpl['betas'],
        'pose_body': np.zeros((len(t_original), 69)),
        'root_orient': np.zeros((len(t_original), 3)),
        'trans': np.zeros((len(t_original), 3)),
        # 'timestamps': original_timestamps,
    }
    
    # --- 插值逻辑 ---
    # t_min, t_max = t_data_sorted.min(), t_data_sorted.max()
    # mask_valid = (t_original >= t_min) & (t_original <= t_max)

    # t_valid = t_original[mask_valid]
    # t_extrap_low = t_original[t_original < t_min]
    # t_extrap_high = t_original[t_original > t_max]

    # 位移参数（允许外推）
    for i in range(3):
        data = trans_sorted[:, i]
        t_data_min = t_data_sorted.min()
        t_data_max = t_data_sorted.max()
        
        full_values = np.zeros_like(t_original)
        
        mask_low = t_original < t_data_min
        mask_high = t_original > t_data_max
        mask_valid = ~(mask_low | mask_high)

        if data_global_dis is not None:
            trans_data = np.array(data_global_dis)[:, i]
            full_values[mask_low] = trans_data[mask_low]
            full_values[mask_high] = trans_data[mask_high]
        else:
            full_values[mask_low] = data[0]
            full_values[mask_high] = data[-1]
        
        t_valid = t_original[mask_valid]

        
        if len(t_valid) > 0:
            interp_fn = interp1d(t_data_sorted, trans_sorted[:, i], 
                    kind='linear', fill_value="extrapolate")
            
            full_values[mask_valid] = interp_fn(t_valid)

        full_smpl['trans'][:, i] = full_values

    # # 提取插值后的 x, y, z 坐标
    # x_coords = full_smpl['trans'][:, 0]
    # y_coords = full_smpl['trans'][:, 1]
    # z_coords = full_smpl['trans'][:, 2]

    # # 原始（未插值）数据
    # x_data = trans_sorted[:, 0]
    # y_data = trans_sorted[:, 1]
    # z_data = trans_sorted[:, 2]

    # # 创建绘图
    # fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # # --- X 坐标 ---
    # axes[0].plot(t_original, x_coords, label='Interpolated X', color='r')
    # axes[0].scatter(t_data_sorted, x_data, color='k', marker='o', s=40, label='Original Data')
    # axes[0].set_title('X Coordinate Trajectory')
    # axes[0].set_xlabel('Time (s)')
    # axes[0].set_ylabel('X Value')
    # axes[0].grid(True, alpha=0.3)
    # axes[0].legend()

    # # --- Y 坐标 ---
    # axes[1].plot(t_original, y_coords, label='Interpolated Y', color='g')
    # axes[1].scatter(t_data_sorted, y_data, color='k', marker='o', s=40, label='Original Data')
    # axes[1].set_title('Y Coordinate Trajectory')
    # axes[1].set_xlabel('Time (s)')
    # axes[1].set_ylabel('Y Value')
    # axes[1].grid(True, alpha=0.3)
    # axes[1].legend()

    # # --- Z 坐标 ---
    # axes[2].plot(t_original, z_coords, label='Interpolated Z', color='b')
    # axes[2].scatter(t_data_sorted, z_data, color='k', marker='o', s=40, label='Original Data')
    # axes[2].set_title('Z Coordinate Trajectory')
    # axes[2].set_xlabel('Time (s)')
    # axes[2].set_ylabel('Z Value')
    # axes[2].grid(True, alpha=0.3)
    # axes[2].legend()

    # # 调整布局
    # plt.tight_layout()
    # plt.show()
    
    # 根关节旋转（支持外推）
    full_smpl['root_orient'] = interpolate_axis_angle(
        root_orient_sorted, t_data_sorted, t_original
    )
    
    # 身体姿势（逐关节处理）
    for j in range(23):
        joint_rot = pose_body_sorted[:, j*3:(j+1)*3]
        interp_joint = interpolate_axis_angle(joint_rot, t_data_sorted, t_original)
        full_smpl['pose_body'][:, j*3:(j+1)*3] = interp_joint
    
    return full_smpl

def adjust_range(min_val, max_val, target=4):
    """
    调整给定的 min_val 和 max_val，使得范围为 target，
    如果当前范围 diff 不等于 target，则左右对称地扩展或收缩。
    """
    diff = max_val - min_val
    if diff <= target or diff >= target + 2:
        delta = (target - diff) / 2.0
        min_val -= delta
        max_val += delta
    return min_val, max_val


def set_smpl_heading_direct(root_orient: np.ndarray,
                            heading_vecs: np.ndarray,
                            epsilon: float = 1e-6) -> np.ndarray:
    """
    使用世界坐标系下 heading 向量直接修正 SMPL 根节点全局朝向
    不使用角度阈值，所有帧都修改

    Args:
        root_orient: [N,3] 原始 axis-angle
        heading_vecs: [N,3] 世界坐标系下 heading 向量
        epsilon: 数值稳定项

    Returns:
        corrected_orient: [N,3] 修正后的 axis-angle
    """
    N = root_orient.shape[0]
    corrected_rots = []
    raw_rots = R.from_rotvec(root_orient)
    up = np.array([0.0, 0.0, 1.0])

    for i in range(N):
        heading = heading_vecs[i]
        heading = heading / (np.linalg.norm(heading) + epsilon)

        # 水平投影（可选）
        heading_proj = heading.copy()
        heading_proj[2] = 0
        norm = np.linalg.norm(heading_proj)
        if norm < epsilon:
            heading_proj = np.array([1.0, 0.0, 0.0])
        else:
            heading_proj /= norm

        z_des = heading_proj
        x_des = np.cross(up, z_des)
        if np.linalg.norm(x_des) < epsilon:
            x_des = np.array([1.0, 0.0, 0.0])
        else:
            x_des /= np.linalg.norm(x_des)
        y_des = np.cross(z_des, x_des)
        R_des = np.stack([x_des, y_des, z_des], axis=1)

        R_raw = raw_rots[i].as_matrix()
        R_new = R.from_matrix(R_des)  # 直接替换原始旋转
        corrected_rots.append(R_new)

    corrected_rot = R.concatenate(corrected_rots)
    return corrected_rot.as_rotvec()
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

def correct_smpl_orient_by_speed(
    speed: np.ndarray,
    root_orient: np.ndarray,
    smooth_sigma: float = 2.0,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    使用平滑后的 XY 平面速度方向修正 SMPL 全局朝向。

    参数:
    ----------
    speed : np.ndarray
        (N, 3) 每帧速度向量（世界坐标系下）
    root_orient : np.ndarray
        (N, 3) 原始 SMPL axis-angle 旋转向量（全局朝向）
    smooth_sigma : float
        高斯平滑参数，越大越平滑 (1.0~3.0 比较常用)
    epsilon : float
        防止除零的数值稳定项

    返回:
    ----------
    corrected_orient : np.ndarray
        (N, 3) 修正后的全局朝向（axis-angle 格式）
    """
    N = speed.shape[0]
    raw_rots = Rotation.from_rotvec(root_orient)
    corrected_rots = []
    up = np.array([0.0, 0.0, 1.0])

    # 1. 仅使用 XY 平面速度
    V = speed.copy()
    V[:, 2] = 0.0

    # 2. 计算方向并归一化
    V_norm = np.linalg.norm(V, axis=1, keepdims=True) + epsilon
    dirs = V / V_norm

    # 3. 对方向进行高斯平滑（逐维平滑）
    dirs_smooth = gaussian_filter1d(dirs, sigma=smooth_sigma, axis=0)

    # 4. 再次归一化
    dirs_smooth /= np.linalg.norm(dirs_smooth, axis=1, keepdims=True) + epsilon

    # 5. 用平滑后的方向修正旋转
    for i in range(N):
        z_des = dirs_smooth[i]
        if np.linalg.norm(z_des) < epsilon:
            corrected_rots.append(raw_rots[i])
            continue

        # 构造新旋转矩阵
        x_des = np.cross(up, z_des)
        if np.linalg.norm(x_des) < epsilon:
            corrected_rots.append(raw_rots[i])
            continue
        x_des /= np.linalg.norm(x_des)
        y_des = np.cross(z_des, x_des)
        R_des = np.stack([x_des, y_des, z_des], axis=1)

        # 修正当前帧旋转
        R_raw = raw_rots[i].as_matrix()
        R_align = R_des @ R_raw.T
        q_align = Rotation.from_matrix(R_align)
        q_new = q_align * raw_rots[i]
        corrected_rots.append(q_new)

    corrected_orient = Rotation.concatenate(corrected_rots).as_rotvec()
    return corrected_orient




smpl_bone_link = np.array([
    [0, 1], [0, 2],  # Pelvis -> 左臀、右臀
    [1, 4], [2, 5],  # 左臀 -> 左膝，右臀 -> 右膝
    [4, 7], [5, 8],  # 左膝 -> 左脚踝，右膝 -> 右脚踝
    [0, 3],          # Pelvis -> Spine 1
    [3, 6], [6, 9],  # Spine 1 -> Spine 2 -> Spine 3
    [9, 12],         # Spine 3 -> 颈部（Neck）
    [12, 15],        # 颈部（Neck） -> 头部（Head）
    [9, 13], [9, 14],  # Spine 3 -> 左锁骨，右锁骨
    [13, 16], [14, 17], # 左锁骨 -> 左肩，右锁骨 -> 右肩
    [16, 18], [17, 19], # 左肩 -> 左肘，右肩 -> 右肘
    [18, 20], [19, 21]  # 左肘 -> 左手腕，右肘 -> 右手腕
])

if __name__ == "__main__":
    waymo_path = '/home/guangxun/dataset/waymo'
    split_list = ['training', 'validation']

    smpl_file_path = '/home/guangxun/dataset/waymo/SMPL-data/merged.pkl'

    bbox_path = '/home/guangxun/dataset/waymo/waymo_pedestrian_bbox'


    # Load SMPL data
    smpl_data = pkl.load(open(smpl_file_path, 'rb'))

    smpl_grouped_data = {}
    for item in smpl_data:
        key = f"{item['basic_info']['context_name']}+{item['basic_info']['object_id']}"
        smpl_grouped_data.setdefault(key, []).append(item)

    # 预处理所有SMPL分组的时间戳
    for key in list(smpl_grouped_data.keys()):
        items = smpl_grouped_data[key]
        items.sort(key=lambda x: int(x['basic_info']['time_stamp']))
        smpl_grouped_data[key] = items

    waymo_cache = {}

    global_dis_dict = {}

    cache_counter = 0

    smpl_model = smplx.create('./smplx_models', model_type='smpl', gender='neutral', use_pca=False).to(DEVICE)   

    fixed_smpl_grouped_data = {}
    # fixed_smpl_grouped_data['1146261869236413282_1680_000_1700_000+VhhThLiVqRGbyZ4S5fWaoA'] = smpl_grouped_data['1146261869236413282_1680_000_1700_000+VhhThLiVqRGbyZ4S5fWaoA']

    

    # 主处理循环
    for key, items in tqdm(smpl_grouped_data.items()) :


        if len(items) > 1:

            cache_counter += 1
            # 如果处理了100个分组后就清空一次缓存
            if cache_counter % 1000 == 0:
                waymo_cache.clear()

            # 提取上下文信息
            context_name, object_id = key.split('+')
            split = items[0]['basic_info']['split']
            
            # 缓存Waymo数据（带对象索引）
            cache_key = (split, context_name)
            if cache_key not in waymo_cache:
                file_path = os.path.join(waymo_path, split, f"{context_name}.pkl")
                waymo_data = pkl.load(open(file_path, 'rb'))
                
                # 创建对象ID到数据的映射
                obj_id_map = {}
                for data in waymo_data:
                    obj_id = data['basic_info']['object_id']
                    obj_id_map.setdefault(obj_id, []).append(data)
                
            
                for obj_id in obj_id_map:
                    obj_id_map[obj_id].sort(key=lambda x: int(x['basic_info']['time_stamp']))
                waymo_cache[cache_key] = obj_id_map

            # 读取对应的bbox数据
            bbox_file_path = os.path.join(bbox_path, split, context_name, f"{object_id}.npz")
            bbox_data = np.load(bbox_file_path, allow_pickle=True)

            world_centers = bbox_data['world_center']
            world_sizes = bbox_data['world_size']
            
            bbox_time_stamps = bbox_data['time_stamps']
            heading_world = bbox_data['heading_dir_world']
            speed = bbox_data['speed']

            # 获取当前对象的Waymo数据
            waymo_single_person = waymo_cache[cache_key].get(object_id, [])

            smpl_single_person = {
                'root_orient': np.array([item['SMPL_lidarhmr']['root_orient'] for item in items]),
                'pose_body': np.array([item['SMPL_lidarhmr']['pose_body'] for item in items]),
                'trans': np.array([item['SMPL_lidarhmr']['trans'] + item['SMPL_lidarhmr']['global_dis'] for item in items]),
                'betas': np.array([item['SMPL_lidarhmr']['betas'] for item in items])
            }

            smpl_time_stamp = [int(item['basic_info']['time_stamp']) for item in items]
            waymo_time_stamp = [int(data['basic_info']['time_stamp']) for data in waymo_single_person]


            raw_data = {
                'betas':smpl_single_person['betas'],
                'pose_body': smpl_single_person['pose_body'],
                'root_orient': smpl_single_person['root_orient'],
                'trans': smpl_single_person['trans'],
                'smpl_timestamps': smpl_time_stamp,
                'waymo_timestamps': waymo_time_stamp,
                'world_center': world_centers,
                'world_size': world_sizes,
                'speed': speed,
                'heading_world': heading_world,
            }

            raw_data_path = f'/home/guangxun/dataset/waymo/waymo_amassformat/original_30fps_improved/final_version/raw/{split}/{context_name}'
            os.makedirs(raw_data_path, exist_ok=True)
            filename = f"{object_id}.npz"
            np.savez(f'{raw_data_path}/{filename}', **raw_data)

            # 修正全局朝向


            heading_world_filtered = np.array([heading_world[np.where(bbox_time_stamps == ts)[0][0]] for ts in smpl_time_stamp if ts in bbox_time_stamps])

            # smpl_single_person['root_orient'] = set_smpl_heading_direct(smpl_single_person['root_orient'], heading_world)
            speed_filtered = np.array([speed[np.where(bbox_time_stamps == ts)[0][0]] for ts in smpl_time_stamp if ts in bbox_time_stamps])
            # smpl_single_person['root_orient'] = correct_smpl_orient_by_speed(speed_filtered, smpl_single_person['root_orient'])

            smpl_single_person['root_orient'] = correct_smpl_global_orient(
                                                smpl_single_person['trans'],
                                                smpl_single_person['root_orient'],
                                                smooth_sigma=1.0,
                                                angle_threshold_deg=90.0
                                            )

            # 准备时间戳数据

            smpl_index_in_waymo = [waymo_time_stamp.index(smpl_time) for smpl_time in smpl_time_stamp if smpl_time in waymo_time_stamp]

            waymo_global_position = [data['world_frame']['center'] for data in waymo_single_person]
            smpl_global_position = [item['SMPL_lidarhmr']['global_dis'] for item in items]


            raw_data_frenet = {
                'betas':smpl_single_person['betas'],
                'pose_body': smpl_single_person['pose_body'],
                'root_orient': smpl_single_person['root_orient'],
                'trans': smpl_single_person['trans'],
                'smpl_timestamps': smpl_time_stamp,
                'waymo_timestamps': waymo_time_stamp,
                'world_center': world_centers,
                'world_size': world_sizes,
                'speed': speed,
                'heading_world': heading_world,
            }

            raw_data_frenet_path = f'/home/guangxun/dataset/waymo/waymo_amassformat/frenet_30fps_improved/final_version/raw/{split}/{context_name}'
            os.makedirs(raw_data_frenet_path, exist_ok=True)
            filename = f"{object_id}.npz"
            np.savez(f'{raw_data_frenet_path}/{filename}', **raw_data_frenet)



            # 插值空白帧
            first_interpolated_smpl = interpolate_smpl_params(
                np.array(waymo_time_stamp), 
                np.array(smpl_time_stamp), 
                smpl_single_person,
                waymo_global_position
            )


            interpolated_empty_data_ = {
                'betas':first_interpolated_smpl['betas'],
                'pose_body': first_interpolated_smpl['pose_body'],
                'root_orient': first_interpolated_smpl['root_orient'],
                'trans': first_interpolated_smpl['trans'],
                'smpl_timestamps': smpl_time_stamp,
                'waymo_timestamps': waymo_time_stamp,
                'world_center': world_centers,
                'world_size': world_sizes,
                'speed': speed,
                'heading_world': heading_world,
            }

            interpolated_empty_data_path = f'/home/guangxun/dataset/waymo/waymo_amassformat/frenet_30fps_improved/final_version/interpolated_empty/{split}/{context_name}'

            os.makedirs(interpolated_empty_data_path, exist_ok=True)
            filename = f"{object_id}.npz"
            np.savez(f'{interpolated_empty_data_path}/{filename}', **interpolated_empty_data_)


            fps30_timestamps = []

            for i in range(len(waymo_time_stamp)):
                if i == 0:
                    frame_before_first = int(waymo_time_stamp[i] - 33333) 
                    fps30_timestamps.append(frame_before_first)
                    fps30_timestamps.append(waymo_time_stamp[i])

                else:
                    previous_frame = waymo_time_stamp[i-1]
                    current_frame = waymo_time_stamp[i]

                    # 计算两帧之间的时间差（单位：微秒）
                    time_diff = current_frame - previous_frame

                    # 计算需要插值的帧数（包括起点，不包括终点）
                    num_interpolated_frames = max(1, int(round(time_diff, -4)/33333))  # 每帧间隔约 33,333 微秒

                    # 在两帧之间生成插值时间戳
                    interpolated_frames = np.linspace(previous_frame, current_frame, num_interpolated_frames, endpoint=False)[1:]
                    fps30_timestamps.extend(interpolated_frames.astype(int))

                    # 添加当前帧时间戳
                    fps30_timestamps.append(current_frame)

                    # first_interpolated = int(previous_frame + (current_frame - previous_frame) / 3) 
                    # second_interpolated = int(previous_frame + 2 * (current_frame - previous_frame) / 3)
                    # fps30_timestamps.append(first_interpolated)
                    # fps30_timestamps.append(second_interpolated)
                    # fps30_timestamps.append(current_frame)
                    if i == len(waymo_time_stamp) - 1:
                        frame_after_last = int(waymo_time_stamp[i] + 33333)
                        fps30_timestamps.append(frame_after_last)

            smpl_index_in_30fps = [fps30_timestamps.index(smpl_time) for smpl_time in smpl_time_stamp if smpl_time in fps30_timestamps]

            waymo_index_in_30fps = [fps30_timestamps.index(waymo_time) for waymo_time in waymo_time_stamp if waymo_time in fps30_timestamps]

            # 30fps interpolation

            interpolation_30fps = interpolate_smpl_params(
                np.array(fps30_timestamps),
                np.array(waymo_time_stamp),
                first_interpolated_smpl,

            )

            new_world_center, new_world_size, new_speed, new_heading = interpolate_world_params_linear(
                np.array(fps30_timestamps),
                np.array(waymo_time_stamp),
                world_centers,
                world_sizes,
                speed,
                heading_world
            )

            interpolation_30fps['smpl_timestamps'] = smpl_time_stamp
            interpolation_30fps['waymo_timestamps'] = waymo_time_stamp
            interpolation_30fps['fps30_timestamps'] = fps30_timestamps

            interpolation_30fps['waymo_global_position'] = waymo_global_position
            interpolation_30fps['smpl_global_position'] = smpl_global_position
            interpolation_30fps['first_frame_trans'] = interpolation_30fps['trans'][0]

            interpolation_30fps['world_center'] = new_world_center
            interpolation_30fps['world_size'] = new_world_size
            interpolation_30fps['speed'] = new_speed
            interpolation_30fps['heading_world'] = new_heading

            # 保存插值后的数据
            save_dir = f'/home/guangxun/dataset/waymo/waymo_amassformat/frenet_30fps_improved/final_version/30fps/{split}/{context_name}'
            os.makedirs(save_dir, exist_ok=True)

            filename = f"{object_id}.npz"
            np.savez(f'{save_dir}/{filename}', **interpolation_30fps)

            # load gt pose
            gt_pose_timestamp = []
            for way_data in waymo_single_person:
                if way_data['world_frame']['keypoints'] is None:
                    continue
                gt_pose_timestamp.append(int(way_data['basic_info']['time_stamp']))

            gt_index_in_30fps = [fps30_timestamps.index(gt_time) -1 for gt_time in gt_pose_timestamp if gt_time in fps30_timestamps]

            if len(gt_index_in_30fps) != 0:
                save_dir = f'/home/guangxun/dataset/waymo/waymo_amassformat/frenet_30fps_improved/final_version/30fps_gt_index/{split}/{context_name}'
                os.makedirs(save_dir, exist_ok=True)
                filename = f"{object_id}.npy"
                np.save(f'{save_dir}/{filename}', gt_index_in_30fps)

            # save pesudo gt pose
            pesudo_gt_timestamp = []
            for way_data in waymo_single_person:
                if way_data['world_frame']['keypoints'] is not None:
                    continue
                num_point_cloud = way_data['world_frame']['pedestrain_pcd'].shape[0]
                if num_point_cloud == 0:
                    continue
                # 找到z坐标最大的点（最高点）
                max_point = way_data['world_frame']['pedestrain_pcd'][np.argmax(way_data['world_frame']['pedestrain_pcd'][:, 2])]
                # 找到z坐标最小的点（最低点）
                min_point = way_data['world_frame']['pedestrain_pcd'][np.argmin(way_data['world_frame']['pedestrain_pcd'][:, 2])]
                height = max_point[2] - min_point[2]
                if height > 1.2 and num_point_cloud > 200:
                    pesudo_gt_timestamp.append(int(way_data['basic_info']['time_stamp']))
            pesudo_gt_index_in_30fps = [fps30_timestamps.index(ts) - 1 for ts in pesudo_gt_timestamp if ts in fps30_timestamps]
            if len(pesudo_gt_index_in_30fps) != 0:
                save_dir = f'/home/guangxun/dataset/waymo/waymo_amassformat/frenet_30fps_improved/final_version/30fps_pesudo_gt_index/{split}/{context_name}'
                os.makedirs(save_dir, exist_ok=True)
                filename = f"{object_id}.npy"
                np.save(f'{save_dir}/{filename}', pesudo_gt_index_in_30fps)
            ################################################################################################################################################

            # trans = raw_data_frenet['trans']

            # betas = raw_data_frenet['betas'][0]
            # betas = np.repeat(betas[np.newaxis], len(trans), axis=0)

            # root_orient = raw_data_frenet['root_orient']
            # pose_body = raw_data_frenet['pose_body']

            # trans = torch.Tensor(trans).to('cuda')
            # trans = trans - trans[0,:].unsqueeze(0)  # 平移到原点
            # betas = torch.Tensor(betas).to('cuda')
            # root_orient = torch.Tensor(root_orient).to('cuda')
            # pose_body = torch.Tensor(pose_body).to('cuda')

            # # 生成SMPL输出
            # smpl_output = smpl_model(betas=betas,
            #                             body_pose=pose_body,
            #                             global_orient=root_orient,
            #                             transl=trans,
            #                             return_verts=True)

            # # 获取顶点和面
            # joints = smpl_output.joints.detach().cpu().numpy().squeeze()[:,:22, :]


            # # 根据数据计算坐标轴范围
            # x_min, x_max = joints[:, :, 0].min(), joints[:, :, 0].max()
            # y_min, y_max = joints[:, :, 1].min(), joints[:, :, 1].max()
            # z_min, z_max = joints[:, :, 2].min(), joints[:, :, 2].max()

            # z_max = z_min + 3

            # # 对每个轴进行调整
            # x_min, x_max = adjust_range(x_min, x_max, target=8)
            # y_min, y_max = adjust_range(y_min, y_max, target=8)
            # # z_min, z_max = adjust_range(z_min, z_max, target=4)

            # # 创建3D画布
            # fig = plt.figure(figsize=(16, 12))
            # ax = fig.add_subplot(111, projection='3d')
            # ax.set_xlim(x_min, x_max)
            # ax.set_ylim(y_min, y_max)
            # ax.set_zlim(z_min, z_max)
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # ax.set_title('motion')
            # # ax.view_init(elev=0, azim=0)

            # # 为每条骨骼连接创建线条对象
            # lines = []
            # for _ in range(len(smpl_bone_link)):
            #     line, = ax.plot([], [], [], lw=2, c='b')
            #     lines.append(line)

            # # 创建轨迹线对象：显示第一个关节在 xy 平面的轨迹（z 固定为 z_min，作为“地面”）
            # traj_line, = ax.plot([], [], [], c='b', lw=2, linestyle='--', label='Trajectory')
            # ax.legend()

            # def init():
            #     """初始化所有线条对象"""
            #     for line in lines:
            #         line.set_data([], [])
            #         line.set_3d_properties([])
            #         traj_line.set_data([], [])
            #         traj_line.set_3d_properties([])
            #     return lines + [traj_line]

            # def update(frame):
            #     """更新当前帧骨骼的连线和轨迹"""

            #     # 如果当前帧为特殊帧，则采用红色，否则蓝色
            #     color = 'r'

            #     skeleton = joints[frame]  # 当前帧 15 个关节点的坐标，shape 为 (15, 3)
            #     for i, (start, end) in enumerate(smpl_bone_link):
            #         # 获取每条连线的起点和终点坐标
            #         x_vals = [skeleton[start, 0], skeleton[end, 0]]
            #         y_vals = [skeleton[start, 1], skeleton[end, 1]]
            #         z_vals = [skeleton[start, 2], skeleton[end, 2]]
            #         lines[i].set_color(color)
            #         lines[i].set_data(x_vals, y_vals)
            #         lines[i].set_3d_properties(z_vals)

            #         # 更新轨迹：取前 frame+1 帧第一个关节的 xy 坐标，
            #         # 将轨迹绘制在“地面”，即 z 轴固定为 z_min
            #         traj_x = joints[:frame+1, 0, 0]
            #         traj_y = joints[:frame+1, 0, 1]
            #         traj_z = np.full_like(traj_x, z_min)
            #         traj_line.set_data(traj_x, traj_y)
            #         traj_line.set_3d_properties(traj_z)

            #     return lines + [traj_line]

            # # 创建动画，interval 控制每帧间隔（单位毫秒）
            # ani = animation.FuncAnimation(fig, update, frames=len(joints), init_func=init,
            #                     interval=100, blit=True)
            
            # # plt.show()

            # save_path = f'demo_waymo/raw_frenet'
            # os.makedirs(save_path, exist_ok=True)

            # ani.save(f'{save_path}/{object_id}_motion.mp4', writer='ffmpeg', fps=10)


            # plt.close()
