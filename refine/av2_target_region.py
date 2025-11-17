# import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from refine import MLPDecoder, MLPDeltaDecoder, MLPDeltaDecoderPi, MLPDeltaDecoderScore
from refine.local_encoder import ALEncoderWithAo_av2, AAEncoderWithAo_CrossAttention
from itertools import permutations
from data_av2_refine import TemporalData
from data_av2_refine import DistanceDropEdge
from torch_geometric.utils import subgraph
from itertools import product
import numpy as np
from data_av2_refine.argo_preprocess import init_weights
from torch_geometric.utils import dense_to_sparse
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from typing import List, Optional
import math
from refine.embedding import SingleInputEmbedding


class MLP(nn.Module):
    def __init__(self, embed_dim, out_channels) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)
    



class TargetRegion_av2_multiagent(nn.Module):

    def __init__(self,
                 future_steps: int,
                 num_modes: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 refine_num: int,
                 seg_num: int,
                 local_radius: int,
                 **kwargs) -> None:
        super(TargetRegion_av2_multiagent, self).__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps        
        self.embed_dim = embed_dim
        self.radius = local_radius
        self.refine_num = refine_num
        self.seg_num = seg_num

        self.fc_encoder = MLP(embed_dim=134, out_channels=64)

        fusion_module = []
        for i in range(self.seg_num):
            fusion_module.append(ALEncoderWithAo_av2(node_dim=node_dim,
                                        edge_dim=edge_dim,
                                        embed_dim=embed_dim))
        self.target_al_encoder = nn.Sequential(*fusion_module)

        
        self.refine_decoder = MLPDeltaDecoder(local_channels = embed_dim,
                                        global_channels = embed_dim,
                                        future_steps = future_steps // self.seg_num,   # cut to chunk
                                        num_modes = num_modes,
                                        with_cumsum=0)

        dec_pi_module = []
        dec_pi_module.append(MLPDeltaDecoderPi(embed_dim=embed_dim,))
        self.refine_pi_decoder = nn.Sequential(*dec_pi_module)

        self.pos_embed = nn.Parameter(torch.zeros(self.refine_num+1, 1, embed_dim))

        score_module = []
        score_module.append(nn.GRU(input_size=embed_dim,hidden_size=embed_dim))
        score_module.append(MLPDeltaDecoderScore(embed_dim=embed_dim, with_last=False))
        self.refine_score_decoder = nn.Sequential(*score_module)

        self.drop_edge = DistanceDropEdge(local_radius)

        vehicle_module = []
        for i in range(self.seg_num):
            vehicle_module.append(AAEncoderWithAo_CrossAttention(node_dim=node_dim,
                                        edge_dim=edge_dim,
                                        embed_dim=embed_dim))
        self.vehicle_module = nn.Sequential(*vehicle_module)

        y_emb = []
        for i in range(self.seg_num):
            y_emb.append(SingleInputEmbedding(in_channel=12, out_channel=embed_dim))
        self.y_emb = nn.Sequential(*y_emb)

        self.lane_attr_embed = nn.Sequential(
            nn.Linear(3, 2*embed_dim),
            nn.ReLU(),
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.ReLU(),
            nn.Linear(2*embed_dim, embed_dim)
        )

        self.apply(init_weights)


    def forward(self, y_hat, ego_embed, valid_mask, data):
        '''
        y_hat: [n_in_batch,6,30,2]
        ego_embed: [n_in_batch,6,128]
        '''
        n_in_batch = y_hat.shape[0]

        y_hat = y_hat.reshape(y_hat.shape[0]*y_hat.shape[1], 60, 2) # [n_in_batch*6, 60, 2]
        y_hat_init = y_hat # [n_in_batch*6, 60, 2]
        
        rotate_local_modes = data.rotate_agent2av.unsqueeze(1).repeat(1,self.num_modes, 1, 1).reshape(n_in_batch*self.num_modes, 2, 2) # [n_in_batch*6,2,2] agent --> av
        data_local_origin_modes = data.x_centers.unsqueeze(1).repeat(1,self.num_modes, 1).reshape(n_in_batch*self.num_modes, 2) # [n_in_batch*6,2]  第50帧agent的坐标 av coords
        av_to_agent_mat = data.rotate_av2agent.unsqueeze(1).repeat(1,self.num_modes, 1, 1).reshape(n_in_batch*self.num_modes, 2, 2) # [n_in_batch*6,2,2] av --> agent
        
        new_agent_index = torch.arange(n_in_batch*self.num_modes).to(ego_embed.device) # n*f f1 f2 ... fn # [n_in_batch*6]: 0, 1, 2 ... n_in_batch*6-1

        
        tar_lane_positions = data.tar_lane_positions # [16992, 2]
        tar_lane_vectors = data.tar_lane_vectors # [16992, 2]
        tar_lane_attr = self.lane_attr_embed(data.tar_lane_attr) # [16992,64]
        
        trajs = []
        pis = []
        scores = []
        embeds = []

        ego_embed = self.fc_encoder(ego_embed) # [n_in_batch*6,64] <-- [n_in_batch*6,134] 降维

        ego_embed = ego_embed.reshape(n_in_batch*self.num_modes, -1) # [n_in_batch*6,64]
        score = self.refine_score_decoder[0]((ego_embed.unsqueeze(0)))[0][-1] # [n_in_batch*6,64]
        score = self.refine_score_decoder[1](score.reshape(self.num_modes, n_in_batch, -1)+self.pos_embed[:1]) # [n_in_batch,6]
        embeds.append(ego_embed.detach()) # append([n_in_batch*6,64])
        scores.append(score) # append([n_in_batch,6])

        vehicle_reg_mask = ~data['x_padding_mask'] # [n_batch, 110] 1为可见，0为不可见


        vehicle_mask_bool = valid_mask # [n_batch]  1为 valid，0为 invalid
        vehicle_mask = valid_mask # [n_batch]  1为 valid，0为 invalid



        # 车道的
        tar_index_1 = []
        tar_index_2 = []
        tar_index_3 = []
        tar_index_4 = []

        tar_lane_actor_vectors_1 = []
        tar_lane_actor_vectors_2 = []
        tar_lane_actor_vectors_3 = []
        tar_lane_actor_vectors_4 = []
        split_len = 0

        # 轨迹的
        node_split_len = 0
        node_tar_index = []
        hardmask = []
        mode_index = []

        for i, tar_lane_point_num in enumerate(data.tar_lane_points_num): # batch

            # 计算轨迹信息
            vehicle_num = data.num_vehicles[i] # 该场景中车辆数
            vehicle_index_lo, vehicle_index_hi = node_split_len, node_split_len+vehicle_num # 每个场景中 vehicle 索引的最小值和最大值

            vehicle_mask_bool_i = vehicle_mask_bool[vehicle_index_lo:vehicle_index_hi] # [n]

            # 通过 padding mask 得到 hard mask
            vehicle_mask_i = vehicle_mask[vehicle_index_lo:vehicle_index_hi] # [n]
            vehicle_mask_i = vehicle_mask_i.unsqueeze(-1).repeat(1, vehicle_num) # [nk, nq]
            vehicle_mask_i.fill_diagonal_(False) # 对角线赋 0，自己看不到自己
            vehicle_mask_i = vehicle_mask_i.unsqueeze(0).repeat(6, 1, 1) # [6, nk, nq]
            vehicle_mask_i = vehicle_mask_i[:, vehicle_mask_bool_i] # [6, nk_valid, nq]
            vehicle_mask_i = vehicle_mask_i[:, :, vehicle_mask_bool_i] # [6, nk_valid, nq_valid]
            vehicle_mask_i = vehicle_mask_i.reshape(-1) # [6*nk_valid*nq_valid]
            hardmask.append(vehicle_mask_i)

            ## 得到 q 的索引
            # inter-world agent interaction 
            index_this_q = torch.arange(vehicle_index_lo*6, vehicle_index_hi*6)  # 当前 batch q 的索引 [n*6]
            index_this_k = index_this_q.reshape(vehicle_num, self.num_modes) # [n,6]
            index_this_k = index_this_k[vehicle_mask_bool_i] # [n_valid, 6]
            index_this_k = index_this_k.permute(1,0) # [6, n_valid]
            index_i = []
            for mode in range(index_this_k.shape[0]):
                index_this_k_mode = index_this_k[mode] # [n_valid]
                index_this_q_mode = index_this_k_mode # [n_valid]
                index_i.append(torch.cartesian_prod(new_agent_index[index_this_q_mode].long(), new_agent_index[index_this_k_mode].long())) # [nk_valid*nq_valid, 2]
            index_i = torch.stack(index_i, dim=0) # [6, nk_valid*nq_valid, 2]
            index_i = index_i.reshape(-1, 2) # [6*nk_valid*nq_valid, 2]
            node_tar_index.append(index_i)


            node_split_len = vehicle_index_hi

        node_tar_index = torch.cat(node_tar_index).t().contiguous().to(ego_embed.device)
        hardmask = torch.cat(hardmask).contiguous().to(ego_embed.device)



        for refine_iter in range(self.refine_num):

            if refine_iter == 0:
                y_hat_agent_cord = y_hat_init.clone() # [n_in_batch*6, 60, 2]
                y_hat = torch.bmm(y_hat_init, rotate_local_modes)+data_local_origin_modes.unsqueeze(1) # [n_in_batch*6, 60, 2] global 坐标系
            else:
                y_hat_init = y_hat_init + y_hat_delta # [n_in_batch*6, 60, 2]
                y_hat_agent_cord = y_hat_init.clone()
                y_hat = torch.bmm(y_hat_init, rotate_local_modes)+data_local_origin_modes.unsqueeze(1) # [n_in_batch*6, 60, 2] global coord
            
            idx = [-46, -31, -16, -1]

            target_hats = [y_hat[:, id].reshape(n_in_batch, self.num_modes, -1) for id in idx] # List[4] target_hats[0]:[n_in_batch,6,2] target_hats[1]:[n_in_batch,6,2] 

            refine_cum_sum = []

            ego_embed = ego_embed.reshape(n_in_batch*self.num_modes, -1) # [n_in_batch*6,64]


            
            # 车道的
            tar_index_1 = []
            tar_index_2 = []
            tar_index_3 = []
            tar_index_4 = []
            tar_lane_actor_vectors_1 = []
            tar_lane_actor_vectors_2 = []
            tar_lane_actor_vectors_3 = []
            tar_lane_actor_vectors_4 = []
            split_len = 0
            split_len_vehicle = 0

            # # 轨迹的
            trajs_edge_attr_1 = []
            trajs_dist_1 = []
            trajs_edge_attr_2 = []
            trajs_dist_2 = []
            trajs_edge_attr_3 = []
            trajs_dist_3 = []
            trajs_edge_attr_4 = []
            trajs_dist_4 = []
            aa_mask_1 = []
            aa_mask_2 = []

            for i, tar_lane_point_num in enumerate(data.tar_lane_points_num): # batch

                
                num_point = tar_lane_point_num  # 2502 每一个sample的场景中车道向量的总数
                index_lo, index_hi = split_len, split_len + num_point # 每一个场景中车道向量的索引的最小值和最大值
                tar_lane_positions_i = tar_lane_positions[index_lo:index_hi] # [2502,2] 一个sample中车道向量
                num_vehicle = data.num_vehicles[i] # 一个场景中的车辆数
                index_lo_vehicles, index_hi_vehicles = split_len_vehicle, split_len_vehicle + num_vehicle # 每个场景中车辆索引的最大值和最小值
                vehicle_mask_bool_i = vehicle_mask_bool[index_lo_vehicles:index_hi_vehicles] # [n]
                num_vehicle_valid = vehicle_mask_bool_i.sum() # [1]

                y_hat_i = y_hat.detach()[index_lo_vehicles*6:index_hi_vehicles*6] # 一个sample中的车辆 global coord [n*6, 30, 2]
                target_hats_i = [target_hats[0][index_lo_vehicles:index_hi_vehicles][vehicle_mask_bool_i].reshape(num_vehicle_valid*self.num_modes, 2), 
                                 target_hats[1][index_lo_vehicles:index_hi_vehicles][vehicle_mask_bool_i].reshape(num_vehicle_valid*self.num_modes, 2),
                                 target_hats[2][index_lo_vehicles:index_hi_vehicles][vehicle_mask_bool_i].reshape(num_vehicle_valid*self.num_modes, 2),
                                 target_hats[3][index_lo_vehicles:index_hi_vehicles][vehicle_mask_bool_i].reshape(num_vehicle_valid*self.num_modes, 2)] # List[2] target_hats[0]:[n_valid*6,2] target_hats[1]:[n_valid*6,2] 

                ##  计算轨迹信息    
                # 第一个target
                y_hat_i_1 = y_hat_i[:, :15, :] # [n*6, 15, 2]
                y_hat_i_1 = y_hat_i_1.reshape(num_vehicle, 6, 15, 2)
                y_hat_i_1 = y_hat_i_1[vehicle_mask_bool_i] # [n_valid, 6, 15, 2]
                y_hat_i_1 = y_hat_i_1.permute(1,0,2,3) # [6, n_valid, 15, 2]
                offset_trajs_av_1 = y_hat_i_1.unsqueeze(2).expand(-1, -1, num_vehicle_valid, -1, -1) - y_hat_i_1.unsqueeze(1).expand(-1, num_vehicle_valid, -1, -1, -1) # [6, nk_valid, nq_valid, 15, 2]
                dist_trajs_av_1 = torch.norm(offset_trajs_av_1, p=2, dim=-1) # [6, nk_valid, nq_valid, 15]
                dist_trajs_av_1, dist_trajs_av_1_idx = dist_trajs_av_1.min(-1) # [6, nk_valid, nq_valid]   [6, nk_valid, nq_valid]
                offset_trajs_av_1 = torch.gather(offset_trajs_av_1, # [6, nk_valid, nq_valid, 15, 2]
                                               -2,
                                               dist_trajs_av_1_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                                               ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                
                # 第二个 target
                y_hat_i_2 = y_hat_i[:, 15:30, :] # [n*6, 15, 2]
                y_hat_i_2 = y_hat_i_2.reshape(num_vehicle, 6, 15, 2)
                y_hat_i_2 = y_hat_i_2[vehicle_mask_bool_i] # [n_valid, 6, 15, 2]
                y_hat_i_2 = y_hat_i_2.permute(1,0,2,3) # [6, n_valid, 15, 2]
                offset_trajs_av_2 = y_hat_i_2.unsqueeze(2).expand(-1, -1, num_vehicle_valid, -1, -1) - y_hat_i_2.unsqueeze(1).expand(-1, num_vehicle_valid, -1, -1, -1) # [6, nk_valid, nq_valid, 15, 2]
                dist_trajs_av_2 = torch.norm(offset_trajs_av_2, p=2, dim=-1) # [6, nk_valid, nq_valid, 15]
                dist_trajs_av_2, dist_trajs_av_2_idx = dist_trajs_av_2.min(-1) # [6, nk_valid, nq_valid]   [6, nk_valid, nq_valid]
                offset_trajs_av_2 = torch.gather(offset_trajs_av_2, # [6, nk_valid, nq_valid, 15, 2]
                                               -2,
                                               dist_trajs_av_2_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                                               ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                
                # 第三个 target
                y_hat_i_3 = y_hat_i[:, 30:45, :] # [n*6, 15, 2]
                y_hat_i_3 = y_hat_i_3.reshape(num_vehicle, 6, 15, 2)
                y_hat_i_3 = y_hat_i_3[vehicle_mask_bool_i] # [n_valid, 6, 15, 2]
                y_hat_i_3 = y_hat_i_3.permute(1,0,2,3) # [6, n_valid, 15, 2]
                offset_trajs_av_3 = y_hat_i_3.unsqueeze(2).expand(-1, -1, num_vehicle_valid, -1, -1) - y_hat_i_3.unsqueeze(1).expand(-1, num_vehicle_valid, -1, -1, -1) # [6, nk_valid, nq_valid, 15, 2]
                dist_trajs_av_3 = torch.norm(offset_trajs_av_3, p=2, dim=-1) # [6, nk_valid, nq_valid, 15]
                dist_trajs_av_3, dist_trajs_av_3_idx = dist_trajs_av_3.min(-1) # [6, nk_valid, nq_valid]   [6, nk_valid, nq_valid]
                offset_trajs_av_3 = torch.gather(offset_trajs_av_3, # [6, nk_valid, nq_valid, 15, 2]
                                               -2,
                                               dist_trajs_av_3_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                                               ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                
                # 第四个 target
                y_hat_i_4 = y_hat_i[:, 45:, :] # [n*6, 15, 2]
                y_hat_i_4 = y_hat_i_4.reshape(num_vehicle, 6, 15, 2)
                y_hat_i_4 = y_hat_i_4[vehicle_mask_bool_i] # [n_valid, 6, 15, 2]
                y_hat_i_4 = y_hat_i_4.permute(1,0,2,3) # [6, n_valid, 15, 2]
                offset_trajs_av_4 = y_hat_i_4.unsqueeze(2).expand(-1, -1, num_vehicle_valid, -1, -1) - y_hat_i_4.unsqueeze(1).expand(-1, num_vehicle_valid, -1, -1, -1) # [6, nk_valid, nq_valid, 15, 2]
                dist_trajs_av_4 = torch.norm(offset_trajs_av_4, p=2, dim=-1) # [6, nk_valid, nq_valid, 15]
                dist_trajs_av_4, dist_trajs_av_4_idx = dist_trajs_av_4.min(-1) # [6, nk_valid, nq_valid]   [6, nk_valid, nq_valid]
                offset_trajs_av_4 = torch.gather(offset_trajs_av_4, # [6, nk_valid, nq_valid, 15, 2]
                                               -2,
                                               dist_trajs_av_4_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                                               ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                

                av_to_agent_mat_i = av_to_agent_mat[index_lo_vehicles*6:index_hi_vehicles*6] # [nq*6, 2, 2]
                av_to_agent_mat_i = av_to_agent_mat_i.reshape(num_vehicle, 6, 2, 2) # [nq, 6, 2, 2]
                av_to_agent_mat_i = av_to_agent_mat_i[vehicle_mask_bool_i] # [nq_valid, 6, 2, 2]
                av_to_agent_mat_i = av_to_agent_mat_i.permute(1,0,2,3) # [6, nq_valid, 2, 2]
                av_to_agent_mat_i = av_to_agent_mat_i.reshape(6*num_vehicle_valid, 2, 2) # [6*nq_valid, 2, 2]

                offset_trajs_vehicle_1 = torch.bmm(offset_trajs_av_1.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                offset_trajs_vehicle_2 = torch.bmm(offset_trajs_av_2.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                offset_trajs_vehicle_3 = torch.bmm(offset_trajs_av_3.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                offset_trajs_vehicle_4 = torch.bmm(offset_trajs_av_4.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_agent_cord_i = y_hat_agent_cord[index_lo_vehicles*6:index_hi_vehicles*6] # [nk*6, 60, 2]
                y_hat_agent_cord_i_1 = y_hat_agent_cord_i[:, :15, :] # [nk*6, 15, 2]
                y_hat_agent_cord_i_2 = y_hat_agent_cord_i[:, 15:30, :] # [nk*6, 15, 2]
                y_hat_agent_cord_i_3 = y_hat_agent_cord_i[:, 30:45, :] # [nk*6, 15, 2]
                y_hat_agent_cord_i_4 = y_hat_agent_cord_i[:, 45:, :] # [nk*6, 15, 2]
                y_hat_agent_cord_i_1 = y_hat_agent_cord_i_1.reshape(num_vehicle, self.num_modes, 15, 2) # [nk, 6, 15, 2]
                y_hat_agent_cord_i_1 = y_hat_agent_cord_i_1[vehicle_mask_bool_i] # [nk_valid, 6, 15, 2]
                y_hat_agent_cord_i_1 = y_hat_agent_cord_i_1.permute(1,0,2,3) # [6, nk_valid, 15, 2]
                y_hat_agent_cord_i_2 = y_hat_agent_cord_i_2.reshape(num_vehicle, self.num_modes, 15, 2) # [nk, 6, 15, 2]
                y_hat_agent_cord_i_2 = y_hat_agent_cord_i_2[vehicle_mask_bool_i] # [nk_valid, 6, 15, 2]
                y_hat_agent_cord_i_2 = y_hat_agent_cord_i_2.permute(1,0,2,3) # [6, nk_valid, 15, 2]
                y_hat_agent_cord_i_3 = y_hat_agent_cord_i_3.reshape(num_vehicle, self.num_modes, 15, 2) # [nk, 6, 15, 2]
                y_hat_agent_cord_i_3 = y_hat_agent_cord_i_3[vehicle_mask_bool_i] # [nk_valid, 6, 15, 2]
                y_hat_agent_cord_i_3 = y_hat_agent_cord_i_3.permute(1,0,2,3) # [6, nk_valid, 15, 2]
                y_hat_agent_cord_i_4 = y_hat_agent_cord_i_4.reshape(num_vehicle, self.num_modes, 15, 2) # [nk, 6, 15, 2]
                y_hat_agent_cord_i_4 = y_hat_agent_cord_i_4[vehicle_mask_bool_i] # [nk_valid, 6, 15, 2]
                y_hat_agent_cord_i_4 = y_hat_agent_cord_i_4.permute(1,0,2,3) # [6, nk_valid, 15, 2]
                y_hat_agent_cord_target_1 = torch.gather(
                    y_hat_agent_cord_i_1.unsqueeze(2).expand(-1, -1, num_vehicle_valid, -1, -1), # [6, nk_valid, nq_valid, 30, 2]
                    -2,
                    dist_trajs_av_1_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2).reshape(-1, 2) # [6*nk_valid*nq_valid, 2]
                y_hat_agent_cord_target_2 = torch.gather(
                    y_hat_agent_cord_i_2.unsqueeze(2).expand(-1, -1, num_vehicle_valid, -1, -1), # [6, nk_valid, nq_valid, 30, 2]
                    -2,
                    dist_trajs_av_2_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2).reshape(-1, 2) # [6*nk_valid*nq_valid, 2]
                y_hat_agent_cord_target_3 = torch.gather(
                    y_hat_agent_cord_i_3.unsqueeze(2).expand(-1, -1, num_vehicle_valid, -1, -1), # [6, nk_valid, nq_valid, 30, 2]
                    -2,
                    dist_trajs_av_3_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2).reshape(-1, 2) # [6*nk_valid*nq_valid, 2]
                y_hat_agent_cord_target_4 = torch.gather(
                    y_hat_agent_cord_i_4.unsqueeze(2).expand(-1, -1, num_vehicle_valid, -1, -1), # [6, nk_valid, nq_valid, 30, 2]
                    -2,
                    dist_trajs_av_4_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2).reshape(-1, 2) # [6*nk_valid*nq_valid, 2]


                # target 1 处速度 vector
                y_hat_velocity_i_1 = y_hat_i_1[:, :, 1:, :] - y_hat_i_1[:, :, :-1, :] # [6, n_valid, 14, 2]  global_coords
                y_hat_velocity_i_1 = torch.cat((y_hat_velocity_i_1, y_hat_velocity_i_1[:, :, -1, :].unsqueeze(-2)), dim=-2) # [6, n_valid, 15, 2]   global_coords
                y_hat_velocity_i_1_vectorq = y_hat_velocity_i_1.unsqueeze(1).expand(-1, num_vehicle_valid, -1, -1, -1) # [6, nk_valid, nq_valid, 15, 2] global coords
                y_hat_velocity_i_1_vectork = y_hat_velocity_i_1.unsqueeze(2).expand(-1, -1, num_vehicle_valid, -1, -1) # [6, nk_valid, nq_valid, 15, 2] global coords
                y_hat_velocity_i_1_vectorq_target = torch.gather(
                    y_hat_velocity_i_1_vectorq, # [6, nk_valid, nq_valid, 15, 2] global coords
                    -2, 
                    dist_trajs_av_1_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                y_hat_velocity_i_1_vectork_target = torch.gather(
                    y_hat_velocity_i_1_vectork, # [6, nk_valid, nq_valid, 15, 2] global coords
                    -2, 
                    dist_trajs_av_1_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                y_hat_velocity_i_1_vectorq_target = torch.bmm(y_hat_velocity_i_1_vectorq_target.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_velocity_i_1_vectork_target = torch.bmm(y_hat_velocity_i_1_vectork_target.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_velocity_i_1_vector_target = torch.cat((y_hat_velocity_i_1_vectorq_target, y_hat_velocity_i_1_vectork_target), dim=-1)  # [6*nk_valid*nq_valid,4] agent coord

                # target 1 处 acceleration vector
                y_hat_acceleration_i_1 = y_hat_velocity_i_1[:, :, 1:, :] - y_hat_velocity_i_1[:, :, :-1, :] # [6, n_valid, 14, 2]  global_coords
                y_hat_acceleration_i_1 = torch.cat((y_hat_acceleration_i_1, y_hat_acceleration_i_1[:, :, -1, :].unsqueeze(-2)), dim=-2) # [6, n_valid, 15, 2]   global_coords
                y_hat_acceleration_i_1_vectorq = y_hat_acceleration_i_1.unsqueeze(1).expand(-1, num_vehicle_valid, -1, -1, -1) # [6, nk_valid, nq_valid, 15, 2] global coords
                y_hat_acceleration_i_1_vectork = y_hat_acceleration_i_1.unsqueeze(2).expand(-1, -1, num_vehicle_valid, -1, -1) # [6, nk_valid, nq_valid, 15, 2] global coords
                y_hat_acceleration_i_1_vectorq_target = torch.gather(
                    y_hat_acceleration_i_1_vectorq, # [6, nk_valid, nq_valid, 15, 2] global coords
                    -2, 
                    dist_trajs_av_1_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                y_hat_acceleration_i_1_vectork_target = torch.gather(
                    y_hat_acceleration_i_1_vectork, # [6, nk_valid, nq_valid, 15, 2] global coords
                    -2, 
                    dist_trajs_av_1_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                y_hat_acceleration_i_1_vectorq_target = torch.bmm(y_hat_acceleration_i_1_vectorq_target.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_acceleration_i_1_vectork_target = torch.bmm(y_hat_acceleration_i_1_vectork_target.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_acceleration_i_1_vector_target = torch.cat((y_hat_acceleration_i_1_vectorq_target, y_hat_acceleration_i_1_vectork_target), dim=-1)  # [6*nk_valid*nq_valid,4] agent coord

                # target 2 处速度 vector
                y_hat_velocity_i_2 = y_hat_i_2[:, :, 1:, :] - y_hat_i_2[:, :, :-1, :] # [6, n_valid, 14, 2]  global_coords
                y_hat_velocity_i_2 = torch.cat((y_hat_velocity_i_2, y_hat_velocity_i_2[:, :, -1, :].unsqueeze(-2)), dim=-2) # [6, n_valid, 15, 2]   global_coords
                y_hat_velocity_i_2_vectorq = y_hat_velocity_i_2.unsqueeze(1).expand(-1, num_vehicle_valid, -1, -1, -1) # [6, nk_valid, nq_valid, 15, 2]  global coords
                y_hat_velocity_i_2_vectork = y_hat_velocity_i_2.unsqueeze(2).expand(-1, -1, num_vehicle_valid, -1, -1) # [6, nk_valid, nq_valid, 15, 2]  global coords
                y_hat_velocity_i_2_vectorq_target = torch.gather(
                    y_hat_velocity_i_2_vectorq, # [6, nk_valid, nq_valid, 15, 2] global coords
                    -2, 
                    dist_trajs_av_2_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                y_hat_velocity_i_2_vectork_target = torch.gather(
                    y_hat_velocity_i_2_vectork, # [6, nk_valid, nq_valid, 15, 2] global coords
                    -2, 
                    dist_trajs_av_2_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                y_hat_velocity_i_2_vectorq_target = torch.bmm(y_hat_velocity_i_2_vectorq_target.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_velocity_i_2_vectork_target = torch.bmm(y_hat_velocity_i_2_vectork_target.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_velocity_i_2_vector_target = torch.cat((y_hat_velocity_i_2_vectorq_target, y_hat_velocity_i_2_vectork_target), dim=-1)  # [6*nk_valid*nq_valid,4] agent coord

                # target 2 处 acceleration vector
                y_hat_acceleration_i_2 = y_hat_velocity_i_2[:, :, 1:, :] - y_hat_velocity_i_2[:, :, :-1, :] # [6, n_valid, 14, 2]  global_coords
                y_hat_acceleration_i_2 = torch.cat((y_hat_acceleration_i_2, y_hat_acceleration_i_2[:, :, -1, :].unsqueeze(-2)), dim=-2) # [6, n_valid, 15, 2]   global_coords
                y_hat_acceleration_i_2_vectorq = y_hat_acceleration_i_2.unsqueeze(1).expand(-1, num_vehicle_valid, -1, -1, -1) # [6, nk_valid, nq_valid, 15, 2] global coords
                y_hat_acceleration_i_2_vectork = y_hat_acceleration_i_2.unsqueeze(2).expand(-1, -1, num_vehicle_valid, -1, -1) # [6, nk_valid, nq_valid, 15, 2] global coords
                y_hat_acceleration_i_2_vectorq_target = torch.gather(
                    y_hat_acceleration_i_2_vectorq, # [6, nk_valid, nq_valid, 15, 2] global coords
                    -2, 
                    dist_trajs_av_2_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                y_hat_acceleration_i_2_vectork_target = torch.gather(
                    y_hat_acceleration_i_2_vectork, # [6, nk_valid, nq_valid, 15, 2] global coords
                    -2, 
                    dist_trajs_av_2_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                y_hat_acceleration_i_2_vectorq_target = torch.bmm(y_hat_acceleration_i_2_vectorq_target.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_acceleration_i_2_vectork_target = torch.bmm(y_hat_acceleration_i_2_vectork_target.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_acceleration_i_2_vector_target = torch.cat((y_hat_acceleration_i_2_vectorq_target, y_hat_acceleration_i_2_vectork_target), dim=-1)  # [6*nk_valid*nq_valid,4] agent coord


                # target 3 处速度 vector
                y_hat_velocity_i_3 = y_hat_i_3[:, :, 1:, :] - y_hat_i_3[:, :, :-1, :] # [6, n_valid, 14, 2]  global_coords
                y_hat_velocity_i_3 = torch.cat((y_hat_velocity_i_3, y_hat_velocity_i_3[:, :, -1, :].unsqueeze(-2)), dim=-2) # [6, n_valid, 15, 2]   global_coords
                y_hat_velocity_i_3_vectorq = y_hat_velocity_i_3.unsqueeze(1).expand(-1, num_vehicle_valid, -1, -1, -1) # [6, nk_valid, nq_valid, 15, 2]  global coords
                y_hat_velocity_i_3_vectork = y_hat_velocity_i_3.unsqueeze(2).expand(-1, -1, num_vehicle_valid, -1, -1) # [6, nk_valid, nq_valid, 15, 2]  global coords
                y_hat_velocity_i_3_vectorq_target = torch.gather(
                    y_hat_velocity_i_3_vectorq, # [6, nk_valid, nq_valid, 15, 2] global coords
                    -2, 
                    dist_trajs_av_3_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                y_hat_velocity_i_3_vectork_target = torch.gather(
                    y_hat_velocity_i_3_vectork, # [6, nk_valid, nq_valid, 15, 2] global coords
                    -2, 
                    dist_trajs_av_3_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                y_hat_velocity_i_3_vectorq_target = torch.bmm(y_hat_velocity_i_3_vectorq_target.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_velocity_i_3_vectork_target = torch.bmm(y_hat_velocity_i_3_vectork_target.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_velocity_i_3_vector_target = torch.cat((y_hat_velocity_i_3_vectorq_target, y_hat_velocity_i_3_vectork_target), dim=-1)  # [6*nk_valid*nq_valid,4] agent coord

                # target 3 处 acceleration vector
                y_hat_acceleration_i_3 = y_hat_velocity_i_3[:, :, 1:, :] - y_hat_velocity_i_3[:, :, :-1, :] # [6, n_valid, 14, 2]  global_coords
                y_hat_acceleration_i_3 = torch.cat((y_hat_acceleration_i_3, y_hat_acceleration_i_3[:, :, -1, :].unsqueeze(-2)), dim=-2) # [6, n_valid, 15, 2]   global_coords
                y_hat_acceleration_i_3_vectorq = y_hat_acceleration_i_3.unsqueeze(1).expand(-1, num_vehicle_valid, -1, -1, -1) # [6, nk_valid, nq_valid, 15, 2] global coords
                y_hat_acceleration_i_3_vectork = y_hat_acceleration_i_3.unsqueeze(2).expand(-1, -1, num_vehicle_valid, -1, -1) # [6, nk_valid, nq_valid, 15, 2] global coords
                y_hat_acceleration_i_3_vectorq_target = torch.gather(
                    y_hat_acceleration_i_3_vectorq, # [6, nk_valid, nq_valid, 15, 2] global coords
                    -2, 
                    dist_trajs_av_3_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                y_hat_acceleration_i_3_vectork_target = torch.gather(
                    y_hat_acceleration_i_3_vectork, # [6, nk_valid, nq_valid, 15, 2] global coords
                    -2, 
                    dist_trajs_av_3_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                y_hat_acceleration_i_3_vectorq_target = torch.bmm(y_hat_acceleration_i_3_vectorq_target.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_acceleration_i_3_vectork_target = torch.bmm(y_hat_acceleration_i_3_vectork_target.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_acceleration_i_3_vector_target = torch.cat((y_hat_acceleration_i_3_vectorq_target, y_hat_acceleration_i_3_vectork_target), dim=-1)  # [6*nk_valid*nq_valid,4] agent coord



                # target 4 处速度 vector
                y_hat_velocity_i_4 = y_hat_i_4[:, :, 1:, :] - y_hat_i_4[:, :, :-1, :] # [6, n_valid, 14, 2]  global_coords
                y_hat_velocity_i_4 = torch.cat((y_hat_velocity_i_4, y_hat_velocity_i_4[:, :, -1, :].unsqueeze(-2)), dim=-2) # [6, n_valid, 15, 2]   global_coords
                y_hat_velocity_i_4_vectorq = y_hat_velocity_i_4.unsqueeze(1).expand(-1, num_vehicle_valid, -1, -1, -1) # [6, nk_valid, nq_valid, 15, 2]  global coords
                y_hat_velocity_i_4_vectork = y_hat_velocity_i_4.unsqueeze(2).expand(-1, -1, num_vehicle_valid, -1, -1) # [6, nk_valid, nq_valid, 15, 2]  global coords
                y_hat_velocity_i_4_vectorq_target = torch.gather(
                    y_hat_velocity_i_4_vectorq, # [6, nk_valid, nq_valid, 15, 2] global coords
                    -2, 
                    dist_trajs_av_4_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                y_hat_velocity_i_4_vectork_target = torch.gather(
                    y_hat_velocity_i_4_vectork, # [6, nk_valid, nq_valid, 15, 2] global coords
                    -2, 
                    dist_trajs_av_4_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                y_hat_velocity_i_4_vectorq_target = torch.bmm(y_hat_velocity_i_4_vectorq_target.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_velocity_i_4_vectork_target = torch.bmm(y_hat_velocity_i_4_vectork_target.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_velocity_i_4_vector_target = torch.cat((y_hat_velocity_i_4_vectorq_target, y_hat_velocity_i_4_vectork_target), dim=-1)  # [6*nk_valid*nq_valid,4] agent coord

                # target 4 处 acceleration vector
                y_hat_acceleration_i_4 = y_hat_velocity_i_4[:, :, 1:, :] - y_hat_velocity_i_4[:, :, :-1, :] # [6, n_valid, 14, 2]  global_coords
                y_hat_acceleration_i_4 = torch.cat((y_hat_acceleration_i_4, y_hat_acceleration_i_4[:, :, -1, :].unsqueeze(-2)), dim=-2) # [6, n_valid, 15, 2]   global_coords
                y_hat_acceleration_i_4_vectorq = y_hat_acceleration_i_4.unsqueeze(1).expand(-1, num_vehicle_valid, -1, -1, -1) # [6, nk_valid, nq_valid, 15, 2] global coords
                y_hat_acceleration_i_4_vectork = y_hat_acceleration_i_4.unsqueeze(2).expand(-1, -1, num_vehicle_valid, -1, -1) # [6, nk_valid, nq_valid, 15, 2] global coords
                y_hat_acceleration_i_4_vectorq_target = torch.gather(
                    y_hat_acceleration_i_4_vectorq, # [6, nk_valid, nq_valid, 15, 2] global coords
                    -2, 
                    dist_trajs_av_4_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                y_hat_acceleration_i_4_vectork_target = torch.gather(
                    y_hat_acceleration_i_4_vectork, # [6, nk_valid, nq_valid, 15, 2] global coords
                    -2, 
                    dist_trajs_av_4_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2) # [6, nk_valid, nq_valid, 1, 2]
                ).squeeze(-2) # [6, nk_valid, nq_valid, 2]
                y_hat_acceleration_i_4_vectorq_target = torch.bmm(y_hat_acceleration_i_4_vectorq_target.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_acceleration_i_4_vectork_target = torch.bmm(y_hat_acceleration_i_4_vectork_target.permute(0,2,1,3).reshape(6*num_vehicle_valid, num_vehicle_valid, 2), # [6*nq_valid,nk_valid,2]
                                                 av_to_agent_mat_i # [6*nq_valid, 2, 2]
                                                 ).reshape(6, num_vehicle_valid, num_vehicle_valid, 2).permute(0,2,1,3).reshape(-1, 2) # [6*nk_valid*nq_valid,2] agent coord
                y_hat_acceleration_i_4_vector_target = torch.cat((y_hat_acceleration_i_4_vectorq_target, y_hat_acceleration_i_4_vectork_target), dim=-1)  # [6*nk_valid*nq_valid,4] agent coord





                trajs_edge_attr_i_1 = torch.cat((offset_trajs_vehicle_1, y_hat_agent_cord_target_1, y_hat_velocity_i_1_vector_target, y_hat_acceleration_i_1_vector_target), dim=-1) # [6*nk_valid*nq_valid, 12]
                trajs_edge_attr_i_2 = torch.cat((offset_trajs_vehicle_2, y_hat_agent_cord_target_2, y_hat_velocity_i_2_vector_target, y_hat_acceleration_i_2_vector_target), dim=-1) # [6*nk_valid*nq_valid, 12]
                trajs_edge_attr_i_3 = torch.cat((offset_trajs_vehicle_3, y_hat_agent_cord_target_3, y_hat_velocity_i_3_vector_target, y_hat_acceleration_i_3_vector_target), dim=-1) # [6*nk_valid*nq_valid, 12]
                trajs_edge_attr_i_4 = torch.cat((offset_trajs_vehicle_4, y_hat_agent_cord_target_4, y_hat_velocity_i_4_vector_target, y_hat_acceleration_i_4_vector_target), dim=-1) # [6*nk_valid*nq_valid, 12]

                trajs_edge_attr_1.append(trajs_edge_attr_i_1.reshape(-1, 12)) # [6*nk_valid*nq_valid, 12]
                trajs_dist_1.append(dist_trajs_av_1.reshape(-1)) # [6*nk_valid*nq_valid]
                trajs_edge_attr_2.append(trajs_edge_attr_i_2.reshape(-1, 12)) # [6*nk_valid*nq_valid, 12]
                trajs_dist_2.append(dist_trajs_av_2.reshape(-1)) # [6*nk_valid*nq_valid]
                trajs_edge_attr_3.append(trajs_edge_attr_i_3.reshape(-1, 12)) # [6*nk_valid*nq_valid, 12]
                trajs_dist_3.append(dist_trajs_av_3.reshape(-1)) # [6*nk_valid*nq_valid]
                trajs_edge_attr_4.append(trajs_edge_attr_i_4.reshape(-1, 12)) # [6*nk_valid*nq_valid, 12]
                trajs_dist_4.append(dist_trajs_av_4.reshape(-1)) # [6*nk_valid*nq_valid]


                # 计算地图信息

                # 第一个 target 处计算
                tar_lane_actor_vectors_i = \
                    tar_lane_positions_i.repeat_interleave(num_vehicle_valid*self.num_modes, dim=0) - target_hats_i[0].repeat(tar_lane_positions_i.size(0), 1) # [n_valid*6*2502, 2] 车道向量与目标位置两两之间的offset。 同在 global 坐标系下计算的

                index_this = torch.arange(index_lo_vehicles*6, index_hi_vehicles*6)  # 当前 batch q 的索引
                index_this = index_this.reshape(num_vehicle, self.num_modes) # [n,6]
                index_this = index_this[vehicle_mask_bool_i] # [n_valid,6]
                index_this = index_this.reshape(-1) # [n_valid*6]
                index_i = torch.cartesian_prod(torch.arange(index_lo, index_hi).long().to(ego_embed.device), new_agent_index[index_this].long()) # [n_valid*6*2502, 2] 索引 [[0,0,0,0,0,0,1,1,1,1,1,1,...,2051,2051,2051,2051,2051,2051], [i,i,i,...,i]
                tar_index_1.append(index_i) # append([6*2052, 2])
                tar_lane_actor_vectors_1.append(tar_lane_actor_vectors_i) # p*f # append([6*2052, 2])

                # 第二个 target 处计算
                tar_lane_actor_vectors_i = \
                    tar_lane_positions_i.repeat_interleave(num_vehicle_valid*self.num_modes, dim=0) - target_hats_i[1].repeat(tar_lane_positions_i.size(0), 1) # [n_valid*6*2502, 2] 车道向量与目标位置两两之间的offset。 同在 global 坐标系下计算的
                tar_index_2.append(index_i) # append([6*2052, 2])
                tar_lane_actor_vectors_2.append(tar_lane_actor_vectors_i) # p*f # append([6*2052, 2])

                # 第三个 target 处计算
                tar_lane_actor_vectors_i = \
                    tar_lane_positions_i.repeat_interleave(num_vehicle_valid*self.num_modes, dim=0) - target_hats_i[2].repeat(tar_lane_positions_i.size(0), 1) # [n_valid*6*2502, 2] 车道向量与目标位置两两之间的offset。 同在 global 坐标系下计算的
                tar_index_3.append(index_i) # append([6*2052, 2])
                tar_lane_actor_vectors_3.append(tar_lane_actor_vectors_i) # p*f # append([6*2052, 2])

                # 第四个 target 处计算
                tar_lane_actor_vectors_i = \
                    tar_lane_positions_i.repeat_interleave(num_vehicle_valid*self.num_modes, dim=0) - target_hats_i[3].repeat(tar_lane_positions_i.size(0), 1) # [n_valid*6*2502, 2] 车道向量与目标位置两两之间的offset。 同在 global 坐标系下计算的
                tar_index_4.append(index_i) # append([6*2052, 2])
                tar_lane_actor_vectors_4.append(tar_lane_actor_vectors_i) # p*f # append([6*2052, 2])

                split_len = index_hi
                split_len_vehicle = index_hi_vehicles

                


            # 与轨迹的交互
            trajs_edge_attr_1 = torch.cat(trajs_edge_attr_1).to(ego_embed.device) # [28860, 12]
            trajs_dist_1 = torch.cat(trajs_dist_1, dim=0) # [28860]

            trajs_edge_attr_2 = torch.cat(trajs_edge_attr_2).to(ego_embed.device) # [28860, 12]
            trajs_dist_2 = torch.cat(trajs_dist_2, dim=0) # [28860]

            trajs_edge_attr_3 = torch.cat(trajs_edge_attr_3).to(ego_embed.device) # [28860, 12]
            trajs_dist_3 = torch.cat(trajs_dist_3, dim=0) # [28860]

            trajs_edge_attr_4 = torch.cat(trajs_edge_attr_4).to(ego_embed.device) # [28860, 12]
            trajs_dist_4 = torch.cat(trajs_dist_4, dim=0) # [28860]


                
            # 与车道的交互
            tar_lane_actor_index_1 = torch.cat(tar_index_1).t().contiguous().to(ego_embed.device) # [2,101952]
            tar_lane_actor_index_2 = torch.cat(tar_index_2).t().contiguous().to(ego_embed.device) # [2,101952]
            tar_lane_actor_index_3 = torch.cat(tar_index_3).t().contiguous().to(ego_embed.device) # [2,101952]
            tar_lane_actor_index_4 = torch.cat(tar_index_4).t().contiguous().to(ego_embed.device) # [2,101952]

            tar_lane_actor_vectors_1 = torch.cat(tar_lane_actor_vectors_1).to(ego_embed.device) # [101952,2] 车道向量与目标位置两两之间的offset。缺失道路本身的vector（方向）信息
            tar_lane_actor_vectors_2 = torch.cat(tar_lane_actor_vectors_2).to(ego_embed.device) # [101952,2] 车道向量与目标位置两两之间的offset。缺失道路本身的vector（方向）信息
            tar_lane_actor_vectors_3 = torch.cat(tar_lane_actor_vectors_3).to(ego_embed.device) # [101952,2] 车道向量与目标位置两两之间的offset。缺失道路本身的vector（方向）信息
            tar_lane_actor_vectors_4 = torch.cat(tar_lane_actor_vectors_4).to(ego_embed.device) # [101952,2] 车道向量与目标位置两两之间的offset。缺失道路本身的vector（方向）信息


            mask_1 = torch.norm(tar_lane_actor_vectors_1, p=2, dim=-1) < self.radius
            mask_2 = torch.norm(tar_lane_actor_vectors_2, p=2, dim=-1) < self.radius
            mask_3 = torch.norm(tar_lane_actor_vectors_3, p=2, dim=-1) < self.radius
            mask_4 = torch.norm(tar_lane_actor_vectors_4, p=2, dim=-1) < self.radius


            tar_lane_actor_index_1 = tar_lane_actor_index_1[:, mask_1] # [2,1521] <-- [2,101952]
            tar_lane_actor_vectors_1 = tar_lane_actor_vectors_1[mask_1] # [1521,2] <-- [101952,2]
            tar_lane_actor_index_2 = tar_lane_actor_index_2[:, mask_2] # [2,1521] <-- [2,101952]
            tar_lane_actor_vectors_2 = tar_lane_actor_vectors_2[mask_2] # [1521,2] <-- [101952,2]
            tar_lane_actor_index_3 = tar_lane_actor_index_3[:, mask_3] # [2,1521] <-- [2,101952]
            tar_lane_actor_vectors_3 = tar_lane_actor_vectors_3[mask_3] # [1521,2] <-- [101952,2]
            tar_lane_actor_index_4 = tar_lane_actor_index_4[:, mask_4] # [2,1521] <-- [2,101952]
            tar_lane_actor_vectors_4 = tar_lane_actor_vectors_4[mask_4] # [1521,2] <-- [101952,2]

            vec_ao_1 = data_local_origin_modes - target_hats[0].reshape(n_in_batch*self.num_modes, -1) # [n_in_batch*6,2] 初始点与目标点的offset
            vec_ao_2 = data_local_origin_modes - target_hats[1].reshape(n_in_batch*self.num_modes, -1) # [n_in_batch*6,2] 初始点与目标点的offset
            vec_ao_3 = data_local_origin_modes - target_hats[2].reshape(n_in_batch*self.num_modes, -1) # [n_in_batch*6,2] 初始点与目标点的offset
            vec_ao_4 = data_local_origin_modes - target_hats[3].reshape(n_in_batch*self.num_modes, -1) # [n_in_batch*6,2] 初始点与目标点的offset

            rotate_mat_ego = data.rotate_av2agent # [n_in_batch,2,2]
            rotate_mat_ego = rotate_mat_ego.unsqueeze(1).repeat(1, self.num_modes, 1, 1).reshape(n_in_batch*self.num_modes, 2, 2) # [n_in_batch*6,2,2]

            theta_now_1 = torch.atan2(target_hats[0].reshape(n_in_batch*self.num_modes, -1)[..., 1:2] - y_hat[:,idx[0]-1,1:2],
                                    target_hats[0].reshape(n_in_batch*self.num_modes, -1)[..., 0:1] - y_hat[:,idx[0]-1,:1]) # [n_in_batch*6,1] 在global坐标系下 目标位置方向相对于原点的方向
            theta_now_2 = torch.atan2(target_hats[1].reshape(n_in_batch*self.num_modes, -1)[..., 1:2] - y_hat[:,idx[1]-1,1:2],
                                    target_hats[1].reshape(n_in_batch*self.num_modes, -1)[..., 0:1] - y_hat[:,idx[1]-1,:1]) # [n_in_batch*6,1] 在global坐标系下 目标位置方向相对于原点的方向
            theta_now_3 = torch.atan2(target_hats[2].reshape(n_in_batch*self.num_modes, -1)[..., 1:2] - y_hat[:,idx[2]-1,1:2],
                                    target_hats[2].reshape(n_in_batch*self.num_modes, -1)[..., 0:1] - y_hat[:,idx[2]-1,:1]) # [n_in_batch*6,1] 在global坐标系下 目标位置方向相对于原点的方向
            theta_now_4 = torch.atan2(target_hats[3].reshape(n_in_batch*self.num_modes, -1)[..., 1:2] - y_hat[:,idx[3]-1,1:2],
                                    target_hats[3].reshape(n_in_batch*self.num_modes, -1)[..., 0:1] - y_hat[:,idx[3]-1,:1]) # [n_in_batch*6,1] 在global坐标系下 目标位置方向相对于原点的方向
            
            rotate_mat_tar_1 = torch.cat(
                (
                    torch.cat((torch.cos(theta_now_1), -torch.sin(theta_now_1)), -1).unsqueeze(-2),
                    torch.cat((torch.sin(theta_now_1), torch.cos(theta_now_1)), -1).unsqueeze(-2)
                ),
                -2
            ) # [n_in_batch*6,2,2]
            rotate_mat_tar_2 = torch.cat(
                (
                    torch.cat((torch.cos(theta_now_2), -torch.sin(theta_now_2)), -1).unsqueeze(-2),
                    torch.cat((torch.sin(theta_now_2), torch.cos(theta_now_2)), -1).unsqueeze(-2)
                ),
                -2
            ) # [n_in_batch*6,2,2]
            rotate_mat_tar_3 = torch.cat(
                (
                    torch.cat((torch.cos(theta_now_3), -torch.sin(theta_now_3)), -1).unsqueeze(-2),
                    torch.cat((torch.sin(theta_now_3), torch.cos(theta_now_3)), -1).unsqueeze(-2)
                ),
                -2
            ) # [n_in_batch*6,2,2]
            rotate_mat_tar_4 = torch.cat(
                (
                    torch.cat((torch.cos(theta_now_4), -torch.sin(theta_now_4)), -1).unsqueeze(-2),
                    torch.cat((torch.sin(theta_now_4), torch.cos(theta_now_4)), -1).unsqueeze(-2)
                ),
                -2
            ) # [n_in_batch*6,2,2]

            rotate_mat_ego_1 = rotate_mat_tar_1.reshape(n_in_batch*self.num_modes, 2, 2) # [n_in_batch*6,2,2] # global -> agent
            rotate_mat_ego_2 = rotate_mat_tar_2.reshape(n_in_batch*self.num_modes, 2, 2) # [n_in_batch*6,2,2] # global -> agent
            rotate_mat_ego_3 = rotate_mat_tar_3.reshape(n_in_batch*self.num_modes, 2, 2) # [n_in_batch*6,2,2] # global -> agent
            rotate_mat_ego_4 = rotate_mat_tar_4.reshape(n_in_batch*self.num_modes, 2, 2) # [n_in_batch*6,2,2] # global -> agent



            # global aa interaction
            trajs_edge_attr_1 = self.y_emb[0](trajs_edge_attr_1) # [2000, 64]
            hardmask = hardmask.bool()
            if node_tar_index.shape[1] == hardmask.shape[0]:
                node_tar_index = node_tar_index[:, hardmask]
            trajs_edge_attr_1 = trajs_edge_attr_1[hardmask]
            ego_embed = self.vehicle_module[0](
                x=(ego_embed, ego_embed),
                edge_index = node_tar_index,
                edge_attr = trajs_edge_attr_1,
                softmask = None,
                hardmask = None
            )
            
            # 与 lane 交互
            ego_embed = self.target_al_encoder[0](x=(tar_lane_vectors, ego_embed),
                                            edge_index=tar_lane_actor_index_1,
                                            edge_attr=tar_lane_actor_vectors_1,
                                            lane_attr=tar_lane_attr,
                                            vec_ao=vec_ao_1,
                                            rotate_mat=rotate_mat_ego_1) # [n_in_batch*6,64]

            refine_y_hat_delta = self.refine_decoder(ego_embed + self.pos_embed[refine_iter+1]) # [6,n_in_batch,15,4]
            refine_cum_sum.append(refine_y_hat_delta)



 

        
            # global aa interaction
            trajs_edge_attr_2 = self.y_emb[1](trajs_edge_attr_2) # [2000, 64]
            trajs_edge_attr_2 = trajs_edge_attr_2[hardmask]
            ego_embed = self.vehicle_module[1](
                x=(ego_embed, ego_embed),
                edge_index = node_tar_index,
                edge_attr = trajs_edge_attr_2,
                softmask = None,
                hardmask = None
            )
            
            # 与 lane 交互
            ego_embed = self.target_al_encoder[1](x=(tar_lane_vectors, ego_embed),
                                            edge_index=tar_lane_actor_index_2,
                                            edge_attr=tar_lane_actor_vectors_2,
                                            lane_attr=tar_lane_attr,
                                            vec_ao=vec_ao_2,
                                            rotate_mat=rotate_mat_ego_2) # [n_in_batch*6,64]

            refine_y_hat_delta = self.refine_decoder(ego_embed + self.pos_embed[refine_iter+1]) # [6,n_in_batch,15,4]
            refine_cum_sum.append(refine_y_hat_delta)


            

            ## 第三段
            # global aa interaction
            trajs_edge_attr_3 = self.y_emb[2](trajs_edge_attr_3) # [2000, 64]
            trajs_edge_attr_3 = trajs_edge_attr_3[hardmask]
            ego_embed = self.vehicle_module[2](
                x=(ego_embed, ego_embed),
                edge_index = node_tar_index,
                edge_attr = trajs_edge_attr_3,
                softmask = None,
                hardmask = None
            )
            
            # 与 lane 交互
            ego_embed = self.target_al_encoder[2](x=(tar_lane_vectors, ego_embed),
                                            edge_index=tar_lane_actor_index_3,
                                            edge_attr=tar_lane_actor_vectors_3,
                                            lane_attr=tar_lane_attr,
                                            vec_ao=vec_ao_3,
                                            rotate_mat=rotate_mat_ego_3) # [n_in_batch*6,64]

            refine_y_hat_delta = self.refine_decoder(ego_embed + self.pos_embed[refine_iter+1]) # [6,n_in_batch,15,4]
            refine_cum_sum.append(refine_y_hat_delta)


            ## 第四段
            # global aa interaction
            trajs_edge_attr_4 = self.y_emb[3](trajs_edge_attr_4) # [2000, 64]
            trajs_edge_attr_4 = trajs_edge_attr_4[hardmask]
            ego_embed = self.vehicle_module[3](
                x=(ego_embed, ego_embed),
                edge_index = node_tar_index,
                edge_attr = trajs_edge_attr_4,
                softmask = None,
                hardmask = None
            )
            
            # 与 lane 交互
            ego_embed = self.target_al_encoder[3](x=(tar_lane_vectors, ego_embed),
                                            edge_index=tar_lane_actor_index_4,
                                            edge_attr=tar_lane_actor_vectors_4,
                                            lane_attr=tar_lane_attr,
                                            vec_ao=vec_ao_4,
                                            rotate_mat=rotate_mat_ego_4) # [n_in_batch*6,64]

            refine_y_hat_delta = self.refine_decoder(ego_embed + self.pos_embed[refine_iter+1]) # [6,n_in_batch,15,4]
            refine_cum_sum.append(refine_y_hat_delta)







            ego_embed = ego_embed.reshape(n_in_batch, self.num_modes, -1) # [n_in_batch,6,64]

            refine_y_hat_delta = torch.cat(refine_cum_sum, dim=-2).view(n_in_batch, self.num_modes, 60, 4) # [n_in_batch,6,60,4]


            refine_pi = self.refine_pi_decoder[0](ego_embed + self.pos_embed[refine_iter+1:refine_iter+2]) # [n_in_batch,6]
            pis.append(refine_pi.permute(1,0)) # append([n_in_batch,6])

            ego_embed = ego_embed.reshape(n_in_batch*self.num_modes, -1) # [n_in_batch*6,64]
            embeds_before = torch.stack(embeds, 0) # [1,n_in_batch*6,64]
            score_input = torch.cat((embeds_before, ego_embed.unsqueeze(0)), 0) # [2,n_in_batch*6,64]
            score = self.refine_score_decoder[0](score_input)[0][-1] # [n_in_batch*6,64]
            score = self.refine_score_decoder[1](score.reshape(self.num_modes, n_in_batch, -1)+self.pos_embed[refine_iter+1:refine_iter+2]) # [n_in_batch,6]
            embeds.append(ego_embed.detach()) # append([n_in_batch*6,64])
            ego_embed = ego_embed.reshape(n_in_batch, self.num_modes, -1) # [n_in_batch,6,64]
            scores.append(score) # append([n_in_batch,6])

            ego_embed = ego_embed.detach()
            y_hat_delta = refine_y_hat_delta.reshape(n_in_batch*self.num_modes, -1, 4)[...,:2].detach() # [n_in_batch*6,60,2]

            trajs.append(refine_y_hat_delta.permute(1,0,2,3)) # append([6,n_in_batch,60,4])

        ret_pis = pis, scores
        '''
        pis: List[5] pis[0]:[n_in_batch,6]
        scores: List[6] scores[0]:[n_in_batch,6]
        '''
        return trajs, ret_pis
        '''
        trajs: List[5] trajs[0]:[6,n_in_batch,30,4]
        '''


