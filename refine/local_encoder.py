from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from refine import MultipleInputEmbedding
from refine import SingleInputEmbedding
from data_av2_refine import DistanceDropEdge
from data_av2_refine import TemporalData
from data_av2_refine.argo_preprocess import init_weights


class ALEncoderWithAo_av2(MessagePassing):

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(ALEncoderWithAo_av2, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lane_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim, node_dim], out_channel=embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.side_embed = nn.Parameter(torch.Tensor(3, embed_dim))
        nn.init.normal_(self.side_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: Tuple[torch.Tensor, torch.Tensor],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                lane_attr: torch.Tensor,
                vec_ao: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:
        x_lane, x_actor = x
        x_actor = x_actor + self._mha_block(self.norm1(x_actor), x_lane, edge_index, edge_attr, lane_attr, vec_ao, rotate_mat, size)
        x_actor = x_actor + self._ff_block(self.norm2(x_actor))
        return x_actor

    def message(self,
                edge_index: Adj,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                lane_attr_j,
                vec_ao,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        if rotate_mat is None:
            x_j = self.lane_embed([x_j, edge_attr],
                                  [lane_attr_j])
        else:
            if lane_attr_j != None:
                # import pdb
                # pdb.set_trace()
                rotate_mat = rotate_mat[edge_index[1]]

                vec_ao = vec_ao[edge_index[1]]
                # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                # starter.record()
                # test = self.is_intersection_embed[is_intersections_j]
                # # ender.record()
                # # # WAIT FOR GPU SYNC
                # # torch.cuda.synchronize()
                # # curr_time = starter.elapsed_time(ender)
                # # print(f'para index: {curr_time}')
                x_j = self.lane_embed([torch.bmm(x_j.unsqueeze(-2), rotate_mat).squeeze(-2),
                                    torch.bmm(edge_attr.unsqueeze(-2), rotate_mat).squeeze(-2),
                                    torch.bmm(vec_ao.unsqueeze(-2), rotate_mat).squeeze(-2)],
                                    [lane_attr_j])
            else:
                rotate_mat = rotate_mat[edge_index[1]]

                vec_ao = vec_ao[edge_index[1]]
                x_j = self.lane_embed([torch.bmm(x_j.unsqueeze(-2), rotate_mat).squeeze(-2),
                                    torch.bmm(edge_attr.unsqueeze(-2), rotate_mat).squeeze(-2),
                                    torch.bmm(vec_ao.unsqueeze(-2), rotate_mat).squeeze(-2)])
                
        query = self.lin_q(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        x_actor = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_actor))
        return inputs + gate * (self.lin_self(x_actor) - inputs)

    def _mha_block(self,
                   x_actor: torch.Tensor,
                   x_lane: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   lane_attr: torch.Tensor,
                   vec_ao:torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size) -> torch.Tensor:
        # import pdb
        # pdb.set_trace()
        x_actor = self.out_proj(self.propagate(edge_index=edge_index, x=(x_lane, x_actor), edge_attr=edge_attr,
                                               lane_attr=lane_attr, vec_ao = vec_ao,
                                               rotate_mat=rotate_mat, size=size))
        return self.proj_drop(x_actor)

    def _ff_block(self, x_actor: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_actor)


class AAEncoderWithAo_CrossAttention(MessagePassing):

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(AAEncoderWithAo_CrossAttention, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.aggre_embed = nn.Linear(embed_dim*2, embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        # self.is_intersection_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        # self.turn_direction_embed = nn.Parameter(torch.Tensor(3, embed_dim))
        # self.traffic_control_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        # nn.init.normal_(self.is_intersection_embed, mean=0., std=.02)
        # nn.init.normal_(self.turn_direction_embed, mean=0., std=.02)
        # nn.init.normal_(self.traffic_control_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: Tuple[torch.Tensor, torch.Tensor],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                softmask = None,
                hardmask = None, 
                vec_ao=None,
                rotate_mat = None,
                size: Size = None) -> torch.Tensor:
        x_vehicle, x_actor = x
        # is_intersections = is_intersections.long()
        # turn_directions = turn_directions.long()
        # traffic_controls = traffic_controls.long()
        x_actor = x_actor + self._mha_block(self.norm1(x_actor), x_vehicle, edge_index, edge_attr, softmask, hardmask, vec_ao, rotate_mat, size)
        x_actor = x_actor + self._ff_block(self.norm2(x_actor))
        return x_actor

    def message(self,
                edge_index: Adj,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                softmask,
                hardmask, 
                vec_ao,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:

        if edge_attr != None:
            x_j = self.aggre_embed(torch.cat((x_j, edge_attr), dim=-1))
        
        query = self.lin_q(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale

        # softmask
        if softmask != None:
            alpha = alpha * softmask.unsqueeze(-1)
        # hardmask
        if hardmask != None:
            # hardmask[hardmask == 0.0] = 0.0
            # alpha = alpha * hardmask.unsqueeze(-1)
            hardmask = hardmask.unsqueeze(-1).expand(-1, self.num_heads)
            alpha = alpha.masked_fill(hardmask==0, float('-inf'))

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha) # [n_edge, 8]
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        x_actor = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_actor))
        return inputs + gate * (self.lin_self(x_actor) - inputs)

    def _mha_block(self,
                   x_actor: torch.Tensor,
                   x_lane: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   softmask, 
                   hardmask, 
                   vec_ao:torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size) -> torch.Tensor:
        # import pdb
        # pdb.set_trace()
        x_actor = self.out_proj(self.propagate(edge_index=edge_index, x=(x_lane, x_actor), edge_attr=edge_attr, softmask=softmask, hardmask=hardmask, vec_ao = vec_ao,
                                               rotate_mat=rotate_mat, size=size))
        return self.proj_drop(x_actor)

    def _ff_block(self, x_actor: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_actor)

