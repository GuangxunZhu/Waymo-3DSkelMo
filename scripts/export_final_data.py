import argparse
import os
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import pickle
from collections import defaultdict
import glob
from tqdm import tqdm
import pickle as pkl
from tqdm import tqdm

def read_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def read_np_file(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data


def collect_data_from_save(generated_results_path, data_metas_path, interpolation_results_path, collect_clips_save_dir):

    root_results_path= generated_results_path + '/*/motion_inbetweening/*/clip_0/*30fps.npz'


    root_results_path = glob.glob(root_results_path)


    # save_dir = '/home/guangxun/dataset/waymo/waymo_amassformat/frenet_30fps_improved/final_version/generated_pesudo5/clips'

    for results_path in tqdm(root_results_path):
        result = results_path.split('/')[-1]

        results_data = np.load(results_path)

        cluster = results_path.split('/')[3].split('_')
        cluster = '_'.join(cluster[:3])

        trans = results_data['trans']
        # trans[:,2] -=0.85
        # betas = results_data['betas']
        # betas = np.repeat(betas[np.newaxis], len(trans), axis=0)

        root_orient = results_data['poses'][:, :3]
        pose_body = results_data['poses'][:, 3:72]

        index = int(result.split('_')[1])

        data_metas_file = os.path.join(data_metas_path, cluster, 'data_metas.npz')
        data_metas = np.load(data_metas_file, allow_pickle=True)

        data_meta = data_metas['data_metas'][index]

        processed_data_path = data_meta['filename']
        clip_index = data_meta['clip_idx']
        valid_indices = data_meta['valid_indices']


        split = processed_data_path.split('/')[-3]
        context_name = processed_data_path.split('/')[-2]
        object_id = processed_data_path.split('/')[-1].rsplit('-', 1)[0]


        # interpolation_results_path = f'/home/guangxun/dataset/waymo/waymo_amassformat/frenet_30fps_improved/final_version/30fps/training/{context_name}/{object_id}.npz'
        interpolation_results_file = os.path.join(interpolation_results_path, 'training', context_name, f'{object_id}.npz')
        original_split = 'training'
        if not os.path.exists(interpolation_results_file):
            interpolation_results_file = os.path.join(interpolation_results_path, 'validation', context_name, f'{object_id}.npz')
            original_split = 'validation'




        clip_data = {'context_name': context_name,
                                     'split': original_split,
                                'object_id': object_id,
                                'clip_index': clip_index,
                                'trans':trans,
                                'root_orient': root_orient,
                                'pose_body': pose_body,
                                'valid_indices': valid_indices,
                                }
        
        save_path = os.path.join(collect_clips_save_dir, original_split, context_name)
        # save_path = os.path.join(save_dir,cluster,original_split, context_name)

        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f'{object_id}-{clip_index}.npz')

        np.savez(save_path, **clip_data)

def merge_clips(clips_path, humor_root, amass_root):

    # 根目录路径
    # clips_path = '/home/guangxun/dataset/waymo/waymo_amassformat/frenet_30fps_improved/final_version/generated_pesudo5/clips'
    split_list = ['training', 'validation']

    # humor_root = '/home/guangxun/dataset/waymo/waymo_amassformat/frenet_30fps_improved/final_version/30fps_humor'

    # amass_root = '/home/guangxun/dataset/waymo/waymo_amassformat/frenet_30fps_improved/final_version/30fps'
    


    # 创建一个字典，用于存储同一目标名称的文件
    target_files = defaultdict(list)

    # 遍历 training 和 validation 文件夹
    for split in split_list:
        data_path = os.path.join(clips_path, split)

        # 遍历所有子文件夹
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.npz'):
                    # 提取目标名称（文件名的前半部分）
                    target_name = file.rsplit('-', 1)[0]
                    # 将文件路径添加到对应目标名称的列表中
                    target_files[target_name].append(os.path.join(root, file))

    # 对每个目标名称的文件列表进行排序
    for target_name, file_list in target_files.items():
        target_files[target_name] = sorted(file_list, key=lambda x: int(x.rsplit('-', 1)[-1].split('.')[0]))
    

    for target_name, file_list in tqdm(target_files.items()) :

        combined_trans = []
        combined_root_orient = []
        combined_pose_body = []

        for i, file_path in enumerate(file_list):
            data = np.load(file_path, allow_pickle=True)

            # 提取数据
            valid_indices = data['valid_indices']
            root_orient = data['root_orient'][valid_indices]
            trans = data['trans'][valid_indices]
            pose_body = data['pose_body'][valid_indices]

            context_name = str(data['context_name'])
            split_name = str(data['split']) 
            object_name = str(data['object_id'])

            # 如果不是第一个文件，去掉前 2 帧（overlap 部分）
            if i > 0:
                root_orient = root_orient[2:]
                trans = trans[2:]
                pose_body = pose_body[2:]

            # 合并数据
            combined_trans.append(trans)
            combined_root_orient.append(root_orient)
            combined_pose_body.append(pose_body)

        # 拼接所有数据
        combined_trans = np.concatenate(combined_trans, axis=0)
        combined_root_orient = np.concatenate(combined_root_orient, axis=0)
        combined_pose_body = np.concatenate(combined_pose_body, axis=0)

        # 恢复humor处理的数据
        humor_path = os.path.join(humor_root, split_name, context_name, f'{object_name}.npz')
        humor_data = np.load(humor_path, allow_pickle=True)

        floor_height = humor_data['floor_height']
        betas = humor_data['betas']

        # first_frame_trans = humor_data['first_frame_trans']

        combined_trans[:, 2] += floor_height
        combined_trans[:, 2] -= 0.85


        # 恢复插值的数据
        interpolation_path = os.path.join(amass_root, split_name, context_name, f'{object_name}.npz')

        interpolation_data = np.load(interpolation_path, allow_pickle=True)

        first_frame_trans = interpolation_data['first_frame_trans']

        combined_trans += first_frame_trans


        waymo_time_stamps = interpolation_data['waymo_timestamps']

        fps30_timestamps = interpolation_data['fps30_timestamps']

        # interpolation_trans = interpolation_data['trans']

        # waymo_index_in_30fps = [fps30_timestamps.tolist().index(waymo_time) for waymo_time in waymo_time_stamps if waymo_time in fps30_timestamps]

        # original_trans = interpolation_trans[waymo_index_in_30fps]

        combined_index_in_30fps = [fps30_timestamps.tolist().index(waymo_time)-1 for waymo_time in waymo_time_stamps if waymo_time in fps30_timestamps]

        combined_trans_10fps = combined_trans[combined_index_in_30fps]
        combined_pose_body_10fps = combined_pose_body[combined_index_in_30fps]
        combined_root_orient_10fps = combined_root_orient[combined_index_in_30fps]


        output_30fps = {
            'trans': combined_trans,
            'root_orient': combined_root_orient,
            'pose_body': combined_pose_body,
            'betas': betas,
            '30fps_timestamps': fps30_timestamps[1:-1],
            'waymo_timestamps': waymo_time_stamps,
        }

        output_10fps = {
            'trans': combined_trans_10fps,
            'root_orient': combined_root_orient_10fps,
            'pose_body': combined_pose_body_10fps,
            'betas': betas,
            # '30fps_timestamps': fps30_timestamps[1:-1],
            'waymo_timestamps': waymo_time_stamps,
        }

        output_30fps_path = os.path.join(clips_path.rsplit('/',1)[0],'final_30fps', split_name, context_name)
        os.makedirs(output_30fps_path, exist_ok=True)
        output_30fps_path = os.path.join(output_30fps_path, f'{object_name}.npz')
        np.savez(output_30fps_path, **output_30fps)

        output_10fps_path = os.path.join(clips_path.rsplit('/',1)[0],'final_10fps', split_name, context_name)
        os.makedirs(output_10fps_path, exist_ok=True)
        output_10fps_path = os.path.join(output_10fps_path, f'{object_name}.npz')
        np.savez(output_10fps_path, **output_10fps)
        



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-results-path", default="NeMF/save/waymo")
    parser.add_argument("--data-metas-path", default="NeMF/data/waymo/generative")
    parser.add_argument("--interpolation-results-path", default="/dataset/guangxun/test/amass_format/30fps")
    parser.add_argument("--clips-save-dir", default="/dataset/guangxun/test/generated/clips")
    parser.add_argument("--humor-root", default="/dataset/guangxun/test/humor")

    
    args = parser.parse_args()
    generated_results_path = args.gen_results_path
    data_metas_path = args.data_metas_path
    interpolation_results_path = args.interpolation_results_path
    collect_clips_save_dir = args.clips_save_dir
    humor_root = args.humor_root
    collect_data_from_save(generated_results_path, data_metas_path, interpolation_results_path, collect_clips_save_dir)

    merge_clips(collect_clips_save_dir, humor_root, interpolation_results_path)

    # calculate_distances()
   
    