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


def load_npz(npz_path):
	data = np.load(npz_path, allow_pickle=True)
	out = {}
	# handle saved dicts which become 0-d object arrays
	for k in data.files:
		v = data[k]
		try:
			if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
				out[k] = v.item()
			else:
				out[k] = v
		except Exception:
			out[k] = v
	return out


def setup_smpl_model(device='cpu', model_folder='./LiDAR-HMR/smplx_models/'):
	if smplx is None:
		raise RuntimeError('smplx is not installed or failed to import')
	model = smplx.create(model_folder, model_type='smpl', gender='neutral', use_face_contour=False, ext='npz')
	if device == 'cuda':
		model = model.cuda()
	return model


def params_to_joints(smpl_model, smpl_dict):
	"""
	smpl_dict should contain: 'betas', 'pose_body', 'root_orient', 'trans', optionally 'global_dis'.
	Returns: joints (N_joints, 3) as numpy array in world coordinates.
	"""
	betas = np.asarray(smpl_dict.get('betas'))
	body_pose = np.asarray(smpl_dict.get('pose_body'))
	root_orient = np.asarray(smpl_dict.get('root_orient'))
	trans = np.asarray(smpl_dict.get('trans'))
	global_dis = np.asarray(smpl_dict.get('global_dis')) if 'global_dis' in smpl_dict else None

	if global_dis is not None:
		transl = trans + global_dis
	else:
		transl = trans

	# make tensors of correct shape (1, ...)
	device = next(smpl_model.parameters()).device if hasattr(smpl_model, 'parameters') else torch.device('cpu')
	betas_t = torch.from_numpy(betas).float().unsqueeze(0).to(device)
	body_pose_t = torch.from_numpy(body_pose).float().unsqueeze(0).to(device)
	root_t = torch.from_numpy(root_orient).float().unsqueeze(0).to(device)
	transl_t = torch.from_numpy(transl).float().unsqueeze(0).to(device)

	with torch.no_grad():
		try:
			out = smpl_model(betas=betas_t, body_pose=body_pose_t, global_orient=root_t, transl=transl_t)
		except TypeError:
			# fallback to older arg name
			out = smpl_model(betas=betas_t, body_pose=body_pose_t, global_orient=root_t, trans=transl_t)

	# try to get joints from output
	joints = None
	if hasattr(out, 'joints'):
		joints = out.joints.detach().cpu().numpy().squeeze(0)
	else:
		# compute from vertices via regressor
		verts = out.vertices.detach().cpu().numpy().squeeze(0)  # (V,3)
		J_reg = smpl_model.J_regressor
		try:
			if isinstance(J_reg, np.ndarray):
				joints = J_reg.dot(verts)
			else:
				# torch tensor or sparse
				J = J_reg.cpu().numpy()
				joints = J.dot(verts)
		except Exception:
			# last fallback: pick some vertices as approximate joints (pelvis only)
			joints = np.zeros((24, 3))
			joints[0] = transl  # pelvis

	return joints


def set_axes_equal(ax):
	"""Make 3D axes have equal scale."""
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


SMPL_BONE_LINK = np.array([
	[0, 1], [0, 2],
	[1, 4], [2, 5],
	[4, 7], [5, 8],
	[0, 3],
	[3, 6], [6, 9],
	[9, 12],
	[12, 15],
	[9, 13], [9, 14],
	[13, 16], [14, 17],
	[16, 18], [17, 19],
	[18, 20], [19, 21]
])


def plot_joints(joints, bone_links=SMPL_BONE_LINK, ax=None, color='blue', show=True, title=None, save_path=None):
	if ax is None:
		fig = plt.figure(figsize=(6, 8))
		ax = fig.add_subplot(111, projection='3d')
	else:
		fig = ax.get_figure()

	joints = np.asarray(joints)
	ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, s=20)

	for a, b in bone_links:
		if a < len(joints) and b < len(joints):
			xs = [joints[a, 0], joints[b, 0]]
			ys = [joints[a, 1], joints[b, 1]]
			zs = [joints[a, 2], joints[b, 2]]
			ax.plot(xs, ys, zs, c='k', linewidth=2)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	if title:
		ax.set_title(title)

	set_axes_equal(ax)

	if show:
		plt.show()
	# save figure if path provided
	if save_path:
		fig.savefig(save_path)
		plt.close(fig)


def plot_npz_file(npz_path, model_folder='./LiDAR-HMR/smplx_models/'):
	data = load_npz(npz_path)
	if 'SMPL_lidarhmr' not in data:
		raise ValueError('NPZ does not contain SMPL_lidarhmr key')

	smpl_dict = data['SMPL_lidarhmr']

	# create smpl model on cpu
	smpl_model = setup_smpl_model(device='cpu', model_folder=model_folder)
	joints = params_to_joints(smpl_model, smpl_dict)

	info = data.get('basic_info', {})
	title = None
	if isinstance(info, dict):
		title = f"{info.get('context_name','')}_{info.get('object_id','')}_{info.get('time_stamp','')}"

	plot_joints(joints, title=title, save_path='smpl_joints_plot.png')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--npz',default='/home/guangxun/test/lidar_hmr_results/training/12896629105712361308_4520_000_4540_000/KdiTz3bysrqJWpUnWIhbPw/1557857010597261.npz', help='path to the .npz file',required=False)
	parser.add_argument('--model_folder', default='./LiDAR-HMR/smplx_models/', help='SMPL model folder',required=False)
	args = parser.parse_args()
	plot_npz_file(args.npz, args.model_folder)


if __name__ == '__main__':
	main()

