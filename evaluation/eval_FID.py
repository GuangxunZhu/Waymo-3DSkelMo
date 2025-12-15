import numpy as np
from scipy.linalg import sqrtm
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation.tools.evaluator_wrapper import EvaluatorMDMWrapper
from scipy import linalg
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm


from evaluation.tools.motion_process import recover_from_ric
from evaluation.tools.rotation2xyz import Rotation2xyz
from evaluation.tools.metrics import calculate_skating_ratio


class FootskatingDataset(data.Dataset):
    def __init__(self, dataset_root, mean_std_path, is_save=False):

        self.mean = np.load(os.path.join(mean_std_path, 'Mean.npy'))
        self.std = np.load(os.path.join(mean_std_path, 'Std.npy'))

        self.data_dict = {}
        self.name_list= []
        print('Loading dataset')

        for filename in tqdm(os.listdir(dataset_root)) :
            file_path = os.path.join(dataset_root, filename)
            motion = np.load(file_path)
            if (len(motion)) < 2:
                    continue
            # motion = motion[:196]
            self.data_dict[filename[:-4]] = {
                'motion': motion,
                'length': len(motion),
            }
            self.name_list.append(filename[:-4])

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, item):
        idx =  item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length = data['motion'], data['length']

        "Z Normalization"
        # eps = 1e-8
        # std, mean = self.std, self.mean
        # motion = (motion - mean) / (std+eps)

        if m_length < 196:
            motion = np.concatenate([motion, np.zeros((196 - m_length, motion.shape[1]))], axis=0)

        return motion
    
class FidDataset(data.Dataset):
    def __init__(self, dataset_root, mean_std_path, is_save=False):

        self.mean = np.load(os.path.join(mean_std_path, 'Mean.npy'))
        self.std = np.load(os.path.join(mean_std_path, 'Std.npy'))

        self.data_dict = {}
        self.name_list= []
        print('Loading dataset')

        for filename in tqdm(os.listdir(dataset_root)) :
            file_path = os.path.join(dataset_root, filename)
            motion = np.load(file_path)
            if (len(motion)) < 30 or (len(motion) >= 200):
                    continue
            motion = motion[:196]
            self.data_dict[filename[:-4]] = {
                'motion': motion,
                'length': len(motion),
            }
            self.name_list.append(filename[:-4])

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, item):
        idx =  item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length = data['motion'], data['length']

        "Z Normalization"
        eps = 1e-8
        std, mean = self.std, self.mean
        motion = (motion - mean) / (std+eps)

        if m_length < 196:
            motion = np.concatenate([motion, np.zeros((196 - m_length, motion.shape[1]))], axis=0)

        return motion, m_length

# ------------------- core -------------------
def calculate_activation_statistics(activations):
    """Calculation of the statistics used by the FID."""
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative dataset set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative dataset set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

# ------------------- feature extraction flow -------------------
def extract_features(evaluator, motion_data, m_lens):
    """use EvaluatorMDMWrapper to extract motion features"""


    motion_embeddings = evaluator.get_motion_embeddings(
                motions=motion_data, m_lens=m_lens)
    return motion_embeddings.cpu().numpy()



def fid(gen_dataset_path, amass_dataset_path):
    
    batch_size = 16
    num_workers = 1
    
    # initialize evaluator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = EvaluatorMDMWrapper(dataset_name='humanml', device=device)

    gt_dataset = FidDataset(dataset_root=f'{amass_dataset_path}/new_joint_vecs', mean_std_path=amass_dataset_path)

    gen_dataset = FidDataset(dataset_root=f'{gen_dataset_path}/new_joint_vecs', mean_std_path=gen_dataset_path)

    gt_loader = DataLoader(gt_dataset, batch_size=batch_size, num_workers=num_workers)

    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, num_workers=num_workers)

    gt_motion_embeddings = []

    print('Extracting gt motion embeddings...')
    for gt_motion, gt_length in tqdm(gt_loader): # gt_motion: (16, 196,263)
        motion_embeddings = evaluator.get_motion_embeddings(motions=gt_motion, m_lens=gt_length)
        gt_motion_embeddings.append(motion_embeddings.cpu().numpy())

    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    mu_real, sigma_real = calculate_activation_statistics(gt_motion_embeddings)

    gen_motion_embeddings = []
    print('Extracting gen motion embeddings...')

    for gen_motion, gen_length in tqdm(gen_loader):
        motion_embeddings = evaluator.get_motion_embeddings(motions=gen_motion, m_lens=gen_length)
        if np.isnan(motion_embeddings.cpu().numpy()).any():
            # print("nan")
            continue
        gen_motion_embeddings.append(motion_embeddings.cpu().numpy())

    gen_motion_embeddings = np.concatenate(gen_motion_embeddings, axis=0)
    mu_gen, sigma_gen = calculate_activation_statistics(gen_motion_embeddings)

    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    print(f"FID Score: {fid_score:.4f}")
    
def foot_skating_ratio(gen_dataset_path):

    batch_size = 1
    num_workers = 1
    
    # initialize evaluator
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen_dataset = FootskatingDataset(dataset_root=f'{gen_dataset_path}/new_joint_vecs', mean_std_path=gen_dataset_path)

    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, num_workers=num_workers)

    skate_ratio_list = []
    all_size = 0
    rot2xyz = Rotation2xyz(device=device)

    for gen_motion in tqdm(gen_loader):

        motion_sk = gen_motion[:,None]  # 1,1,196,263
        gt_skel_motions = recover_from_ric(motion_sk.float(), 22, abs_3d=False) # 1,1,196,22,3
        gt_skel_motions = gt_skel_motions.view(-1, *gt_skel_motions.shape[2:]).permute(0, 2, 3, 1) #1,22,3,196
        gt_skel_motions = rot2xyz(x=gt_skel_motions, mask=None, pose_rep='xyz', glob=True, translation=True,
                                            jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None, get_rotations_back=False)
        skate_ratio, skate_vel = calculate_skating_ratio(gt_skel_motions)  # [batch_size]

        skate_ratio_list.append(skate_ratio)
        all_size += skate_ratio.shape[0]

    skate_ratio_all = np.concatenate(skate_ratio_list, axis=0)
    skate_ratio_sum = skate_ratio_all.sum()
    skate_ratio = skate_ratio_sum / all_size
    print(f'Skating Ratio: {skate_ratio:.6f}')

if __name__ == "__main__":

    gen_dataset_path = 'FID_data/new_joints'
    amass_dataset_path = '/home/guangxun/code/HumanML3D/HumanML3D'

    fid(gen_dataset_path, amass_dataset_path)
    foot_skating_ratio(gen_dataset_path)

    