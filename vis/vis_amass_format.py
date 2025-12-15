import os
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import smplx
except Exception:
    smplx = None

from scripts.convert_amass import corners_from_center_size_heading_batch

import numpy as np

from matplotlib.ticker import ScalarFormatter




def corners_from_center_size_headingdir_batch(
    centers_world, sizes_world, heading_dirs_world,
    up_world=(0.0, 0.0, 1.0), eps=1e-12, deg_thresh=1e-6
):
    """
    centers_world:      (N,3) or (3,)
    sizes_world:        (N,3) or (3,)   full lengths (dx,dy,dz)
    heading_dirs_world: (N,3) or (3,)   bbox local +X axis direction in world (need not be unit)

    return:
      corners_world: (N,8,3)  if batch input
                   or (8,3)   if single input
    """
    C = np.asarray(centers_world, dtype=np.float64)
    S = np.asarray(sizes_world, dtype=np.float64)
    H = np.asarray(heading_dirs_world, dtype=np.float64)

    single = (C.ndim == 1)
    if C.ndim == 1: C = C[None, :]
    if S.ndim == 1: S = S[None, :]
    if H.ndim == 1: H = H[None, :]

    N = C.shape[0]

    # up: (N,3)
    up = np.asarray(up_world, dtype=np.float64)
    if up.ndim == 1:
        up = np.broadcast_to(up[None, :], (N, 3)).copy()
    else:
        up = up.reshape(N, 3)

    up = up / (np.linalg.norm(up, axis=1, keepdims=True) + eps)

    # heading normalize
    h = H / (np.linalg.norm(H, axis=1, keepdims=True) + eps)

    # x = heading projected onto plane orthogonal to up
    dot_h_up = np.sum(h * up, axis=1, keepdims=True)
    x = h - up * dot_h_up

    x_norm = np.linalg.norm(x, axis=1, keepdims=True)

    # handle degenerate (heading nearly parallel to up) per-sample
    tmp1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    tmp2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    use_tmp1 = (np.abs(up[:, 0]) < 0.9)[:, None]              # (N,1)
    tmp = np.where(use_tmp1, tmp1[None, :], tmp2[None, :])    # (N,3)
    dot_tmp_up = np.sum(tmp * up, axis=1, keepdims=True)
    x_alt = tmp - up * dot_tmp_up

    deg = (x_norm[:, 0] < deg_thresh)[:, None]               # (N,1)
    x = np.where(deg, x_alt, x)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

    # y = up × x
    y = np.cross(up, x)
    y = y / (np.linalg.norm(y, axis=1, keepdims=True) + eps)

    # z = up (assume vertical)
    z = up

    # R: local->world, columns are x,y,z
    R = np.stack([x, y, z], axis=2)  # (N,3,3)

    signs = np.array([[-1,-1,-1],[-1,-1, 1],[-1, 1,-1],[-1, 1, 1],
                      [ 1,-1,-1],[ 1,-1, 1],[ 1, 1,-1],[ 1, 1, 1]], dtype=np.float64)  # (8,3)

    half = 0.5 * S                                  # (N,3)
    local = signs[None, :, :] * half[:, None, :]     # (N,8,3)

    corners_world = local @ R.transpose(0, 2, 1) + C[:, None, :]  # (N,8,3)

    return corners_world[0] if single else corners_world


SMPL_BONE_LINK = np.array([
    [0, 1], [0, 2],  # Pelvis -> left/right hip
    [1, 4], [2, 5],  # hip -> knee
    [4, 7], [5, 8],  # knee -> ankle
    [0, 3],          # pelvis -> spine1
    [3, 6], [6, 9],  # spine chain
    [9, 12],         # spine3 -> neck
    [12, 15],        # neck -> head
    [9, 13], [9, 14],  # clavicles
    [13, 16], [14, 17], # shoulders
    [16, 18], [17, 19], # elbows
    [18, 20], [19, 21]  # wrists
])


def load_npz(path, allow_pickle=True):
    d = np.load(path, allow_pickle=allow_pickle)
    out = {}
    # common keys from convert_amass / lidar_hmr outputs
    for k in d.files:
        v = d[k]
        # unwrap 0-d object arrays
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            out[k] = v.item()
        else:
            out[k] = v
    return out


def setup_smpl(model_folder='./LiDAR-HMR/smplx_models', device='cpu'):
    if smplx is None:
        raise RuntimeError('smplx not installed; please pip install smplx')
    model = smplx.create(model_folder, model_type='smpl', gender='neutral', use_face_contour=False, ext='npz')
    if device == 'cuda':
        model = model.to('cuda')
    return model


def params_to_joints(smpl_model, betas, pose_body, root_orient, trans, device='cpu'):
    """Convert SMPL params to joint locations.

    Inputs may be for multiple frames:
      betas: (B, n_betas) or (n_betas,)
      pose_body: (B, 69) or (69,)
      root_orient: (B, 3) or (3,)
      trans: (B, 3) or (3,)

    Returns: joints np array shape (B, J, 3)
    """
    single = False
    if betas is None or pose_body is None or root_orient is None or trans is None:
        raise ValueError('Missing SMPL params')

    betas = np.asarray(betas)
    if betas.shape[0] != pose_body.shape[0]:
        betas = np.repeat(betas[0,:][None, :], pose_body.shape[0], axis=0)
    pose_body = np.asarray(pose_body)
    root_orient = np.asarray(root_orient)
    trans = np.asarray(trans)

    # ensure batch dim
    if betas.ndim == 1:
        betas = betas[None]
        pose_body = pose_body[None]
        root_orient = root_orient[None]
        trans = trans[None]
        single = True

    device_torch = torch.device(device)
    betas_t = torch.from_numpy(betas).float().to(device_torch)
    body_t = torch.from_numpy(pose_body).float().to(device_torch)
    root_t = torch.from_numpy(root_orient).float().to(device_torch)
    trans_t = torch.from_numpy(trans).float().to(device_torch)

    with torch.no_grad():
        try:
            out = smpl_model(betas=betas_t, body_pose=body_t, global_orient=root_t, transl=trans_t)
        except TypeError:
            out = smpl_model(betas=betas_t, body_pose=body_t, global_orient=root_t, trans=trans_t)

    if hasattr(out, 'joints'):
        joints = out.joints.cpu().numpy()
    else:
        # fallback: compute joints using regressor if available
        verts = out.vertices.cpu().numpy()
        J_reg = getattr(smpl_model, 'J_regressor', None)
        if J_reg is None:
            raise RuntimeError('SMPL output has no joints and model has no J_regressor')
        if isinstance(J_reg, np.ndarray):
            joints = np.einsum('ij,bjk->bik', J_reg, verts)
        else:
            try:
                J = J_reg.cpu().numpy()
                joints = np.einsum('ij,bjk->bik', J, verts)
            except Exception:
                raise

    # squeeze batch if originally single
    if single:
        joints = joints[0]
    return joints


def bbox_corners_from_center_size(center, size):
    # center: (3,), size: (3,) full lengths
    c = np.asarray(center).reshape(3)
    s = np.asarray(size).reshape(3) / 2.0
    # corners: combinations of +/- s
    corners = np.array([[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
    corners = corners + c[None, :]
    # edges as pairs of indices
    edges = [
        (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    ]
    return corners, edges


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_frame(joints, bone_links=SMPL_BONE_LINK, bbox=None, ax=None, title=None, frame_idx=0):
    if ax is None:
        fig = plt.figure(figsize=(8,10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    joints = np.asarray(joints)
    if joints.ndim == 3:
        # multiple frames, take first
        joints = joints[frame_idx]

    ax.scatter(joints[:,0], joints[:,1], joints[:,2], c='b', s=20)
    for a,b in bone_links:
        if a < len(joints) and b < len(joints):
            ax.plot([joints[a,0], joints[b,0]], [joints[a,1], joints[b,1]], [joints[a,2], joints[b,2]], c='k', lw=2)

    if bbox is not None:
        corners, edges = bbox
        for i,j in edges:
            ax.plot([corners[i,0], corners[j,0]], [corners[i,1], corners[j,1]], [corners[i,2], corners[j,2]], c='r', lw=1.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    # ax.yaxis.set_major_formatter(fmt)
    # ax.xaxis.set_major_formatter(fmt)
    # ax.zaxis.set_major_formatter(fmt)
    if title:
        ax.set_title(title)
    set_axes_equal(ax)
    
    plt.savefig('temp1.png')


def visualize_npz(npz_path, model_folder=None, frame_idx=0, device='cpu'):
    data = load_npz(npz_path)

    # find smpl params
    # convert_amass files may store keys directly (betas, pose_body, root_orient, trans)
    if 'SMPL_lidarhmr' in data:
        smpld = data['SMPL_lidarhmr']
        betas = smpld.get('betas')
        pose_body = smpld.get('pose_body')
        root_orient = smpld.get('root_orient')
        trans = smpld.get('trans')
    else:
        betas = data.get('betas')
        pose_body = data.get('pose_body')
        root_orient = data.get('root_orient')
        trans = data.get('trans')

    # smpl_time_stamp = data.get('smpl_timestamps').tolist()
    # waymo_time_stamp = data.get('waymo_timestamps').tolist()
    # time_stamp.sort()

    # bbox_frame_idx = waymo_time_stamp.index(1550088183498178) if waymo_time_stamp is not None else frame_idx
    # frame_idx = smpl_time_stamp.index(1550088183498178) if smpl_time_stamp is not None else frame_idx
    joints = None
    # if joints included directly
    if 'joints' in data:
        joints = data['joints']

    if joints is None:
        if smplx is None:
            raise RuntimeError('smplx not available and joints not present in npz')
        smpl_model = setup_smpl(model_folder or './LiDAR-HMR/smplx_models', device=device)
        joints = params_to_joints(smpl_model, betas, pose_body, root_orient, trans, device=device)

    # bbox: prefer world_center/world_size else bbox vertices if present
    bbox = None
    bbox_center = data.get('world_center')
    bbox_size = data.get('world_size')
    bbox_heading = data.get('heading_world')

    print('bbox center',bbox_center[frame_idx])
    print('joint root',joints[frame_idx][0])

    if 'world_vertices' in data:
        verts = np.asarray(data['world_vertices'])
        # verts= corners_from_center_size_headingdir_batch(bbox_center, bbox_size, bbox_heading)
        # assume (8,3) or (T,8,3)
        if verts.ndim == 3:
            corners = verts[frame_idx]
        else:
            corners = verts
        # standard edges
        print('bbox corners center',np.mean(corners, axis=0))
        edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
        bbox = (corners, edges)
    elif 'world_center' in data and 'world_size' in data:
        center = data['world_center']
        size = data['world_size']
        # if arrays for sequences, pick frame_idx
        if np.asarray(center).ndim == 2:
            c = center[frame_idx]
        else:
            c = center
        if np.asarray(size).ndim == 2:
            s = size[frame_idx]
        else:
            s = size
        bbox = bbox_corners_from_center_size(c, s)

    title = os.path.basename(npz_path)
    plot_frame(joints if joints is not None else [], SMPL_BONE_LINK, bbox=bbox, title=title, frame_idx=frame_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz',default='/dataset/guangxun/test/amass_format/30fps/training/4292360793125812833_3080_000_3100_000/TmlAqAVLLVTTm_v1m3Q6cQ.npz', help='path to .npz file to visualize')
    parser.add_argument('--model_folder', default='./LiDAR-HMR/smplx_models', help='smplx model folder')
    parser.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    parser.add_argument('--frame', type=int, default=30, help='frame index to plot when data is a sequence')
    args = parser.parse_args()
    visualize_npz(args.npz, model_folder=args.model_folder, frame_idx=args.frame, device=args.device)


if __name__ == '__main__':
    main()
