import os
import sys
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
#
import numpy as np
import pandas as pd
#
import torch
from torch.utils.data import Dataset
# from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
#
from utils.utils import from_numpy, from_tensor
from data_av2_refine.argo_preprocess import TemporalData
os.sys.path.insert(0, '/data4/xiaoliwen/motion_forecasting/SRefiner-GitHub/data_av2_refine/')
os.sys.path.insert(0, '/data4/xiaoliwen/motion_forecasting/SRefiner-GitHub/data_interaction_refine/')
import pickle
import glob

'''
data
            'y_hat_final': [6,n,60,2] agent coords
            'embeds_final': [6,n,134]
            'y_gt_agentcoords': [n,60,2]
            'x_padding_mask': [n,110]  1:invalid  0:valid
            'x_scored': [n]
            'x_heading': [n]
            'x_centers': [n,2]

            'lane_positions': [n_lane, 20, 2] av coords
            'lane_centers': [n_lane, 2] av coords
            'lane_angles': [n_lane] av coords
            'lane_attr': [n_lane, 3] [lane_type, lane_width, is_intersection]
            'is_intersections': [n_lane]
            'lane_padding_mask': [n_lane, 20]  1:invalid  0:valid

            'num_actors': [1]
            'num_lanes': [1]
            'scenario_id': scenario_id str()
            'origin': [2]
            'theta': [1]

'''

# fmae
class AV2_Refine_multiagent(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 mode: str,
                 obs_len: int = 50,
                 pred_len: int = 60,
                 verbose: bool = False):
        self.mode = mode
        self.verbose = verbose

        self.dataset_files = []
        
        

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len


        split = mode
        self._split = split
        

        if split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test'
        else:
            raise ValueError(split + ' is not valid')
        
        self.dataset_dir = dataset_dir
        self._data_paths = glob.glob(os.path.join(self.dataset_dir, self._directory, "*.pkl"))

        self.dataset_len = len(self._data_paths)

        if self.verbose:
            print('[Dataset] Dataset Info:')
            print('-- mode: ', self.mode)
            print('-- total frames: ', self.dataset_len)
            print('-- obs_len: ', self.obs_len)
            print('-- pred_len: ', self.pred_len)
            print('-- seq_len: ', self.seq_len)

        

        super(AV2_Refine_multiagent, self).__init__()


    def len(self) -> int:
        return len(self._data_paths)
    
    def __len__(self):
        return len(self._data_paths)

    def __getitem__(self, idx) -> Data:
        with open(self._data_paths[idx], 'rb') as handle:
            data = pickle.load(handle)
        for k,v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.type(torch.float32)

        data["num_actors"] = data["num_actors"].long() # [1]
        data["num_vehicles"] = data["num_actors"] # [1]
        data["num_nodes"] = data["num_actors"] # [1]
        lane_positions = data["lane_positions"] # [n_lane, 20, 2]
        lane_positions = lane_positions[:, ::2, :] # [n_lane, 10, 2]
        lane_vectors = lane_positions[:, 1:, :] - lane_positions[:, :-1, :] # [n_lane, 9, 2]
        lane_vectors = torch.cat((lane_vectors, lane_vectors[:, -1, :].unsqueeze(1)), dim=1) # [n_lane, 10, 2]
        lane_attr = data["lane_attr"] # [n_lane, 3]
        lane_attr = lane_attr.unsqueeze(1).expand(-1, 10, -1) # [n_lane, 10, 3]
        lane_padding_mask = data["lane_padding_mask"][:, ::2].bool() # [n_lane, 10]  1:invalid  0:valid
        lane_valid_mask = ~lane_padding_mask # [n_lane, 10] 1:valid  0:invalid

        lane_positions = lane_positions.reshape(-1, 2) # [n_lane*10, 2]
        lane_vectors = lane_vectors.reshape(-1, 2) # [n_lane*10, 2]
        lane_attr = lane_attr.reshape(-1, 3) # [n_lane*10, 3]
        lane_valid_mask = lane_valid_mask.reshape(-1) # [n_lane*10] 1:valid  0:invalid

        tar_lane_positions = lane_positions[lane_valid_mask] # [n_tar_lane, 2]
        tar_lane_vectors = lane_vectors[lane_valid_mask] # [n_tar_lane, 2]
        tar_lane_attr = lane_attr[lane_valid_mask] # [n_tar_lane, 3]

        num_tar_lane = tar_lane_positions.shape[0] # [1]

        data["y_hat_final"] = data["y_hat_final"].permute(1,0,2,3) # [n,6,60,2]
        data["embeds_final"] = data["embeds_final"].permute(1,0,2) # [n,6,134]

        data["tar_lane_positions"] = tar_lane_positions # [n_tar_lane, 2]
        data["tar_lane_vectors"] = tar_lane_vectors # [n_tar_lane, 2]
        data["tar_lane_attr"] = tar_lane_attr # [n_tar_lane, 3]
        data["tar_lane_points_num"] = num_tar_lane # [1]
        data['x_padding_mask'] = data['x_padding_mask'].bool() # [n, 110] 1为valid，0 为 invalid
        data['x_scored'] = data['x_scored'].bool() # [n] 1为valid，0 为 invalid

        data["y_hat_final_avheading"] = data["y_hat_final_avheading"].permute(1,0,2,3) # [n,6,60,2]

        data = Data.from_dict(data)

        
        return data

        
    def collate_fn(self, batch: List[Any]) -> Dict[str, Any]:
        
        return batch