"""Visualize synchronized pedestrian sequences with Matplotlib.

This version mirrors the aesthetics of `vis/plot_30person_animation.py` while
retaining the relative-time alignment and SMPL decoding pipeline built for the
Plotly renderer. It produces Matplotlib animations (MP4/GIF) with fixed camera,
ground grids, and root trajectories.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')  # ensure headless rendering works reliably
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import smplx
import torch

SMPL_BONES = [
	(0, 1), (0, 2),
	(1, 4), (2, 5),
	(4, 7), (5, 8),
	(0, 3), (3, 6), (6, 9),
	(9, 12), (12, 15),
	(9, 13), (9, 14),
	(13, 16), (14, 17),
	(16, 18), (17, 19),
	(18, 20), (19, 21),
]

ROOT_JOINT_IDX = 0


def _flatten_root_xy(joints: np.ndarray) -> np.ndarray:
	root_xy = joints[:, ROOT_JOINT_IDX, :2].reshape(-1, 2)
	mask = np.isfinite(root_xy).all(axis=1)
	return root_xy[mask]


def _flatten_z(joints: np.ndarray) -> np.ndarray:
	vals = joints[..., 2].reshape(-1)
	mask = np.isfinite(vals)
	return vals[mask]


def compute_xy_ranges(
	aligned_joints: Sequence[np.ndarray],
	padding: float,
	min_half_extent: float = 1.0,
	max_half_extent: Optional[float] = None,
	fallback: float = 10.0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
	ax = []
	ay = []
	for joints in aligned_joints:
		vals = _flatten_root_xy(joints)
		if vals.size == 0:
			continue
		ax.append(vals[:, 0])
		ay.append(vals[:, 1])
	if not ax:
		return (-fallback, fallback), (-fallback, fallback)
	all_x = np.concatenate(ax)
	all_y = np.concatenate(ay)
	min_x, max_x = float(np.min(all_x)), float(np.max(all_x))
	min_y, max_y = float(np.min(all_y)), float(np.max(all_y))
	center_x = 0.5 * (min_x + max_x)
	center_y = 0.5 * (min_y + max_y)
	span = max(max_x - min_x, max_y - min_y, min_half_extent * 2.0)
	half_extent = span / 2.0
	if max_half_extent is not None:
		half_extent = min(half_extent, max_half_extent)
	half_extent = max(half_extent, min_half_extent) + padding
	x_range = (center_x - half_extent, center_x + half_extent)
	y_range = (center_y - half_extent, center_y + half_extent)
	return x_range, y_range


def compute_z_limits(
	aligned_joints: Sequence[np.ndarray],
	fallback: Tuple[float, float] = (0.0, 3.0),
) -> Tuple[float, float]:
	vals = []
	for joints in aligned_joints:
		flat = _flatten_z(joints)
		if flat.size:
			vals.append(flat)
	if not vals:
		return fallback
	all_z = np.concatenate(vals)
	return float(np.min(all_z)), float(np.max(all_z))


def build_smpl(model_folder: str, device: torch.device) -> smplx.SMPL:
	model = smplx.create(model_folder, model_type='smpl', gender='neutral')
	return model.to(device)


def params_to_joints(
	model: smplx.SMPL,
	betas: np.ndarray,
	pose_body: np.ndarray,
	root_orient: np.ndarray,
	trans: np.ndarray,
	device: torch.device,
) -> np.ndarray:
	num_frames = pose_body.shape[0]
	if betas.ndim == 1:
		betas = np.repeat(betas[None, :], num_frames, axis=0)
	elif betas.shape[0] != num_frames:
		betas = np.repeat(betas[:1, :], num_frames, axis=0)

	params = {
		'betas': torch.tensor(betas, dtype=torch.float32, device=device),
		'body_pose': torch.tensor(pose_body, dtype=torch.float32, device=device),
		'global_orient': torch.tensor(root_orient, dtype=torch.float32, device=device),
		'transl': torch.tensor(trans, dtype=torch.float32, device=device),
	}
	with torch.no_grad():
		joints = model(**params).joints[:, :22, :]
	return joints.cpu().numpy()


def align_relative_times(sequences: Sequence[Dict]) -> Tuple[np.ndarray, List[np.ndarray]]:
	if not sequences:
		raise ValueError('No sequences to align.')
	all_timestamps = [np.asarray(seq['timestamps']).astype(np.int64) for seq in sequences]
	timeline = np.unique(np.concatenate(all_timestamps))
	if timeline.size == 0:
		raise ValueError('Unable to build timeline from empty timestamps.')
	aligned: List[np.ndarray] = []
	for seq, ts_array in zip(sequences, all_timestamps):
		lookup = {int(ts): idx for idx, ts in enumerate(ts_array)}
		frame_template = np.full(
			(timeline.size, seq['joints'].shape[1], seq['joints'].shape[2]),
			np.nan,
			dtype=seq['joints'].dtype,
		)
		for frame_idx, timestamp in enumerate(timeline):
			match = lookup.get(int(timestamp))
			if match is not None:
				frame_template[frame_idx] = seq['joints'][match]
		aligned.append(frame_template)
	return timeline, aligned


def load_context_sequences(
	data_root: str,
	split: str,
	context_name: str,
	model: smplx.SMPL,
	device: torch.device,
) -> List[Dict]:
	context_dir = os.path.join(data_root, split, context_name)
	if not os.path.isdir(context_dir):
		raise FileNotFoundError(f'Context directory not found: {context_dir}')
	sequences: List[Dict] = []
	for file_name in sorted(os.listdir(context_dir)):
		if not file_name.endswith('.npz'):
			continue
		path = os.path.join(context_dir, file_name)
		data = np.load(path, allow_pickle=True)
		pose_body = data['pose_body']
		root_orient = data['root_orient']
		trans = data['trans']
		betas = data['betas'] if 'betas' in data else np.zeros((1, 10), dtype=np.float32)
		timestamps = data['waymo_timestamps']
		joints = params_to_joints(model, betas, pose_body, root_orient, trans, device)
		# joints = joints[..., [0, 2, 1]]  # align axes with Matplotlib reference
		sequences.append(
			{
				'object_id': os.path.splitext(file_name)[0],
				'joints': joints,
				'timestamps': timestamps,
			}
		)
	if not sequences:
		raise RuntimeError('No pedestrian npz files found for the specified context.')
	return sequences


def plot_ground_grid(
	ax,
	x_range: Tuple[float, float],
	y_range: Tuple[float, float],
	ground_z: float,
	grid_count: int,
	scale: float,
) -> None:
	grid_count = max(2, grid_count)
	scale = max(scale, 0.1)
	x_center = 0.5 * (x_range[0] + x_range[1])
	y_center = 0.5 * (y_range[0] + y_range[1])
	x_half = 0.5 * (x_range[1] - x_range[0]) * scale
	y_half = 0.5 * (y_range[1] - y_range[0]) * scale
	if x_half <= 0:
		x_half = scale
	if y_half <= 0:
		y_half = scale
	x_min, x_max = x_center - x_half, x_center + x_half
	y_min, y_max = y_center - y_half, y_center + y_half
	xs = np.linspace(x_min, x_max, grid_count)
	ys = np.linspace(y_min, y_max, grid_count)
	X, Y = np.meshgrid(xs, ys)
	Z = np.full_like(X, ground_z)
	ax.plot_surface(X, Y, Z, color='whitesmoke', alpha=0.5, linewidth=0, antialiased=True)
	for x in xs:
		ax.plot([x, x], [y_min, y_max], [ground_z, ground_z], color='gray', linewidth=0.5)
	for y in ys:
		ax.plot([x_min, x_max], [y, y], [ground_z, ground_z], color='gray', linewidth=0.5)


def _empty_offsets() -> Tuple[List[float], List[float], List[float]]:
	empty: List[float] = []
	return empty, empty, empty


def _clear_line(line) -> None:
	line.set_data([], [])
	line.set_3d_properties([])


def _trajectory_segment(joints: np.ndarray, frame_idx: int, history_len: int) -> np.ndarray:
	history_len = max(1, history_len)
	start = max(0, frame_idx - history_len + 1)
	segment = joints[start:frame_idx + 1, ROOT_JOINT_IDX, :]
	mask = np.isfinite(segment).all(axis=1)
	return segment[mask]


def build_animation(
	aligned_joints: Sequence[np.ndarray],
	x_range: Tuple[float, float],
	y_range: Tuple[float, float],
	z_range: Tuple[float, float],
	ground_z: float,
	grid_count: int,
	grid_scale: float,
	history_len: int,
	fps: int,
	fig_width: float,
	fig_height: float,
	dpi: int,
	elev: float,
	azim: float,
	camera_dist: float,
) -> Tuple[plt.Figure, animation.FuncAnimation]:
	frame_count = len(aligned_joints[0]) if aligned_joints else 0
	if frame_count == 0:
		raise RuntimeError('No frames available after alignment.')
	fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
	ax = fig.add_subplot(111, projection='3d')
	ax.set_axis_off()
	ax.set_xlim(list(x_range))
	ax.set_ylim(list(y_range))
	ax.set_zlim(list(z_range))
	x_span = max(x_range[1] - x_range[0], 1e-3)
	y_span = max(y_range[1] - y_range[0], 1e-3)
	z_span = max(z_range[1] - z_range[0], 1e-3)
	if hasattr(ax, 'set_box_aspect'):
		ax.set_box_aspect((x_span, y_span, z_span))
	ax.view_init(elev=elev, azim=azim)
	if hasattr(ax, 'dist'):
		ax.dist = max(1.0, camera_dist)
	plot_ground_grid(ax, x_range, y_range, ground_z, grid_count, grid_scale)
	colors = plt.cm.tab20(np.linspace(0, 1, max(len(aligned_joints), 20)))
	scatters = []
	bone_lines = []
	traj_lines = []
	for idx in range(len(aligned_joints)):
		color = colors[idx % colors.shape[0]]
		scatters.append(ax.scatter([], [], [], s=20, color=color, depthshade=False))
		lines = []
		for _ in SMPL_BONES:
			line, = ax.plot([], [], [], color=color, linewidth=3)
			lines.append(line)
		bone_lines.append(lines)
		traj_line, = ax.plot([], [], [], color=color, linestyle='dashed', linewidth=3)
		traj_lines.append(traj_line)

	def update(frame_idx: int):
		artists = []
		for obj_idx, joints in enumerate(aligned_joints):
			frame = joints[frame_idx]
			valid = np.isfinite(frame).all()
			if not valid:
				scatters[obj_idx]._offsets3d = _empty_offsets()
				for line in bone_lines[obj_idx]:
					_clear_line(line)
				_clear_line(traj_lines[obj_idx])
				continue
			scatters[obj_idx]._offsets3d = (
				frame[:, 0],
				frame[:, 1],
				frame[:, 2],
			)
			artists.append(scatters[obj_idx])
			for line, (start, end) in zip(bone_lines[obj_idx], SMPL_BONES):
				segment = frame[[start, end], :]
				line.set_data(segment[:, 0], segment[:, 1])
				line.set_3d_properties(segment[:, 2])
				artists.append(line)
			traj = _trajectory_segment(joints, frame_idx, history_len)
			if traj.size == 0:
				_clear_line(traj_lines[obj_idx])
			else:
				traj_lines[obj_idx].set_data(traj[:, 0], traj[:, 1])
				traj_lines[obj_idx].set_3d_properties(np.full(traj.shape[0], ground_z))
			artists.append(traj_lines[obj_idx])
		return artists

	interval_ms = int(round(1000.0 / fps))
	ani = animation.FuncAnimation(
		fig,
		update,
		frames=frame_count,
		interval=interval_ms,
		blit=False,
	)
	return fig, ani


def save_animation(
	fig: plt.Figure,
	ani: animation.FuncAnimation,
	mp4_path: str | None,
	gif_path: str | None,
	fps: int,
	dpi: int,
	bitrate: int,
) -> None:
	if mp4_path:
		output_dir = os.path.dirname(mp4_path)
		if output_dir:
			os.makedirs(output_dir, exist_ok=True)
		try:
			ani.save(
				mp4_path,
				writer='ffmpeg',
				fps=fps,
				dpi=dpi,
				# savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0},
				# extra_args=['-pix_fmt', 'yuv420p', '-b:v', f'{bitrate}k'],
			)
		except FileNotFoundError as exc:
			raise RuntimeError('ffmpeg not found. Install ffmpeg or add it to PATH to enable MP4 export.') from exc
		print(f'[INFO] MP4 saved to {mp4_path}')
	if gif_path:
		output_dir = os.path.dirname(gif_path)
		if output_dir:
			os.makedirs(output_dir, exist_ok=True)
		ani.save(
			gif_path,
			writer='pillow',
			fps=fps,
			savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0},
		)
		print(f'[INFO] GIF saved to {gif_path}')
	plt.close(fig)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Matplotlib visualization of pedestrian contexts (relative-time aligned).')
	parser.add_argument('--data-root', default='/dataset/guangxun/test/generated_1/final_10fps', help='Root directory containing split/context/object npz files')
	parser.add_argument('--split', default='training', help='Dataset split name')
	parser.add_argument('--context', default='11252086830380107152_1540_000_1560_000', help='Context folder to visualize')
	parser.add_argument('--model-folder', default='./LiDAR-HMR/smplx_models', help='SMPL model directory')
	parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Torch device to run SMPL on')
	parser.add_argument('--xy-limit', type=float, default=None, help='Optional fixed half-extent (meters) for x/y axes (auto if omitted)')
	parser.add_argument('--xy-max-limit', type=float, default=20.0, help='Maximum half-extent (meters) when auto-computing x/y ranges')
	parser.add_argument('--xy-padding', type=float, default=1.5, help='Extra meters added around auto-computed x/y ranges')
	parser.add_argument('--grid-count', type=int, default=20, help='Grid density for the drawn ground plane')
	parser.add_argument('--grid-scale', type=float, default=1.7, help='Scale factor applied to the auto x/y ranges when drawing the ground plane')
	parser.add_argument('--ground-z', type=float, default=None, help='Optional explicit ground plane height (auto if omitted)')
	parser.add_argument('--ground-padding', type=float, default=0.0, help='Meters to push the auto ground plane below the lowest joint')
	parser.add_argument('--z-min', type=float, default=None, help='Optional lower z-axis bound (auto if omitted)')
	parser.add_argument('--z-max', type=float, default=None, help='Optional upper z-axis bound (auto if omitted)')
	parser.add_argument('--z-padding', type=float, default=0.5, help='Extra meters added above the tallest joint when auto z range is used')
	parser.add_argument('--z-max-span', type=float, default=None, help='Maximum vertical span (meters) allowed when auto-computing z range')
	parser.add_argument('--history-len', type=int, default=10, help='Number of past frames to display as root trajectories')
	parser.add_argument('--fps', type=int, default=10, help='Playback FPS for the saved animation')
	parser.add_argument('--fig-width', type=float, default=25.0, help='Matplotlib figure width in inches')
	parser.add_argument('--fig-height', type=float, default=15.0, help='Matplotlib figure height in inches')
	parser.add_argument('--dpi', type=int, default=150, help='Figure DPI used during saving')
	parser.add_argument('--elev', type=float, default=20.0, help='Elevation angle for the 3D camera')
	parser.add_argument('--azim', type=float, default=-60.0, help='Azimuth angle for the 3D camera')
	parser.add_argument('--camera-dist', type=float, default=6.0, help='Axes3D distance parameter (smaller values zoom in)')
	parser.add_argument('--mp4-output', default=None, help='Path to the MP4 file to save (defaults to vis/matplotlib_videos/<context>.mp4)')
	parser.add_argument('--skip-mp4', action='store_true', help='Skip saving the MP4 animation')
	parser.add_argument('--bitrate', type=int, default=8000, help='Target video bitrate (kbps) for ffmpeg writer')
	parser.add_argument('--gif-output', default=None, help='Optional GIF output path (skipped if omitted)')
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	device = torch.device(args.device)
	model = build_smpl(args.model_folder, device)
	sequences = load_context_sequences(args.data_root, args.split, args.context, model, device)
	timestamps, aligned_joints = align_relative_times(sequences)
	if args.xy_limit is not None:
		limit = max(abs(args.xy_limit), 1.0)
		x_range = (-limit, limit)
		y_range = (-limit, limit)
	else:
		max_half = None if args.xy_max_limit is None else max(args.xy_max_limit, 1.0)
		x_range, y_range = compute_xy_ranges(
			aligned_joints,
			padding=args.xy_padding,
			max_half_extent=max_half,
		)
	z_min_data, z_max_data = compute_z_limits(aligned_joints)
	auto_ground = z_min_data - args.ground_padding
	auto_top = z_max_data + args.z_padding
	if args.z_max_span is not None:
		max_span = max(args.z_max_span, 0.5)
		current_span = auto_top - auto_ground
		if current_span > max_span:
			auto_top = auto_ground + max_span
	ground_z = args.ground_z if args.ground_z is not None else auto_ground
	z_lower = args.z_min if args.z_min is not None else ground_z
	z_upper = args.z_max if args.z_max is not None else auto_top
	if ground_z < z_lower:
		z_lower = ground_z
	if ground_z > z_upper:
		z_upper = ground_z + 1.0
	if z_upper <= z_lower:
		z_upper = z_lower + 1.0
	fps = max(1, args.fps)
	fig, ani = build_animation(
		aligned_joints,
		x_range,
		y_range,
		(z_lower, z_upper),
		ground_z,
		max(2, args.grid_count),
		max(0.1, args.grid_scale),
		max(1, args.history_len),
		fps,
		args.fig_width,
		args.fig_height,
		args.dpi,
		args.elev,
		args.azim,
		args.camera_dist,
	)
	safe_context = args.context.replace(os.sep, '_')
	mp4_path = None if args.skip_mp4 else args.mp4_output or os.path.join('vis', 'matplotlib_videos', f'{safe_context}.mp4')
	gif_path = args.gif_output
	save_animation(
		fig,
		ani,
		mp4_path,
		gif_path,
		fps,
		args.dpi,
		args.bitrate,
	)
	print('[INFO] Visualization complete.')


if __name__ == '__main__':
	main()
