# import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import LaplaceNLLLoss, LaplaceNLLLoss_woscale
from losses import SoftTargetCrossEntropyLoss
from losses import ScoreRegL1Loss
from metrics import ADE
from metrics import FDE
from metrics import MR
from metrics import minJointADE
from metrics import minJointFDE
from metrics import ActorMR
from collections import OrderedDict

from data_av2_refine import TemporalData
from torch_geometric.data import Data
from torch_geometric.utils import unbatch

from refine.av2_target_region import TargetRegion_av2_multiagent

class Refine_multiagent_AV2(nn.Module):

    def __init__(self,
                 cls_temperture: int,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 rotate: bool,

                 future_steps: int,
                 num_modes: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 seg_num: int,
                 refine_num: int,
                 refine_radius: int,
                 **kwargs) -> None:
        super(Refine_multiagent_AV2, self).__init__()

        self.cls_temperture = cls_temperture

        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max

        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate

        self.refine_num = refine_num

        self.target_encoder = TargetRegion_av2_multiagent(
                                          future_steps=future_steps,
                                          num_modes=num_modes,
                                          node_dim=node_dim,
                                          edge_dim=edge_dim,
                                          embed_dim=embed_dim,
                                          refine_num=refine_num,
                                          seg_num=seg_num,
                                          refine_radius=refine_radius,
                                          **kwargs)
    

        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')
        self.score_loss = ScoreRegL1Loss()

        self.minADE = ADE()
        self.minFDE = FDE()
        self.minMR = MR()

        self.minJointADE = minJointADE()
        self.minJointFDE = minJointFDE()
        self.actormr = ActorMR()

    def to_global_coord(self, data):

        rotate_local = torch.empty(data['y_hat_final'].shape[0], 2, 2, device=data['y_hat_final'].device)
        sin_vals_angle = torch.sin(data["x_heading"])
        cos_vals_angle = torch.cos(data["x_heading"])
        rotate_local[:, 0, 0] = cos_vals_angle
        rotate_local[:, 0, 1] = -sin_vals_angle
        rotate_local[:, 1, 0] = sin_vals_angle
        rotate_local[:, 1, 1] = cos_vals_angle
        # av to agent
        data.rotate_av2agent = rotate_local # [n_batch,2,2]

        rotate_local = torch.empty(data['y_hat_final'].shape[0], 2, 2, device=data['y_hat_final'].device)
        sin_vals_angle = torch.sin(-data["x_heading"])
        cos_vals_angle = torch.cos(-data["x_heading"])
        rotate_local[:, 0, 0] = cos_vals_angle
        rotate_local[:, 0, 1] = -sin_vals_angle
        rotate_local[:, 1, 0] = sin_vals_angle
        rotate_local[:, 1, 1] = cos_vals_angle
        # agent to av
        data.rotate_agent2av = rotate_local # [n_batch,2,2]

        
    def forward(self, y_hat, embeds, valid_mask, data):
                
        ys_hat_ego = y_hat # [n_in_batch,6,60,2]
        
        ys_refine, pis_refine  = self.target_encoder(y_hat, embeds, valid_mask, data)

        # concat for later laplace sigma computation.
        return torch.cat((ys_hat_ego, ys_hat_ego), -1), None, ys_refine, pis_refine

    def training_step(self, data):
        data = data.to(self.device)
        self.to_global_coord(data)

        bs = len(data['scenario_id'])

        y_hat = data["y_hat_final"] # [n_batch,6,60,2]
        embeds = data["embeds_final"] # [n_batch,6,134]
        x_scored = data["x_scored"] # [n_batch]
        
        valid_step = ~data['x_padding_mask'] # [n_batch, 110]
        valid_mask = valid_step.sum(-1) > 109 # [n_batch]  
        valid_mask = valid_mask | x_scored # [n_batch]
        reg_mask = valid_step[:, 50:] # [n_batch, 60]
        y_gt = data["y_gt_agentcoords"] # [n_batch,60,2]

        ys_hat_ego, _, refine_y_hat_deltas, refine_pis = self.forward(y_hat, embeds, valid_mask, data)
        ys_hat_ego = ys_hat_ego.permute(1,0,2,3)
        '''
        ys_hat_ego: [6,n_in_batch,60,4]
        refine_y_hat_deltas: List[5]: refine_y_hat_deltas[0]:[6,n_in_batch,60,4]
        refine_pis:
            pis: List[5] pis[0]:[n_in_batch,6]
            scores: List[6] scores[0]:[n_in_batch,6]
        '''
        reg_loss_refines = 0
        

        refine_y_hat = ys_hat_ego # [6,n_in_batch,60,4]

        for i in range(self.refine_num):
            refine_y_hat_i = refine_y_hat_deltas[i] # [6,n_in_batch,30,4]
            refine_y_hat[...,:2] = refine_y_hat[...,:2] + refine_y_hat_i[...,:2]
            refine_y_hat[...,2:] = refine_y_hat_i[...,2:]

            # joint ADE match  scored
            reg_mask_scored = reg_mask # [n_batch, 60]
            reg_mask_scored[~x_scored] = False # [n_batch, 60]
            reg_mask_scored = reg_mask_scored.unsqueeze(0).float().expand(6,-1,-1) # [6,n_batch,60]
            ade = (
                torch.norm(refine_y_hat[..., :2] - y_gt.unsqueeze(0).expand(6,-1,-1,-1), dim=-1) * reg_mask_scored
            ).sum(dim=(-1)) / (reg_mask_scored.sum(dim=(-1))+1e-3) # [6,n_batch]
            ade = ade.permute(1,0) # [n_batch,6]
            ade = ade[x_scored] # [n_batch_scored]
            scored_batch = data['batch'][x_scored] # [n_batch_scored]
            joint_errors = [error.sum(dim=0, keepdim=True) for error in unbatch(ade, scored_batch)]
            joint_errors = torch.cat(joint_errors, dim=0)    #[b,6]
            best_mode_index = joint_errors.argmin(dim=-1)     #[b]

            valid_batch = data['batch'][valid_mask] # [n_batch_valid]
            num_agent_pre_batch = torch.bincount(valid_batch) # [b]
            best_mode_index = best_mode_index.repeat_interleave(num_agent_pre_batch, 0)     #[n_batch_valid]
            refine_y_hat_best_valid = torch.gather(
                refine_y_hat[:, valid_mask], # [6, n_batch_valid, 60, 4] agent coords
                0,
                best_mode_index.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 60, 4) # [1, n_batch_valid, 60, 4]
            ).squeeze(0) # [n_batch_valid, 60, 4]
            reg_mask_valid = reg_mask[valid_mask] # [n_batch_valid,60]
            y_gt_valid = y_gt[valid_mask] # [n_batch_valid,60,2]
            reg_loss_refine = self.reg_loss(refine_y_hat_best_valid[reg_mask_valid], y_gt_valid[reg_mask_valid]) # [1]
            reg_loss_refines += reg_loss_refine
            

        
        loss = reg_loss_refines/self.refine_num

        return loss
  
    def validation_step(self, data, vis=False):
        data = data.to(self.device)
        self.to_global_coord(data)
        
        bs = len(data['scenario_id'])

        y_hat = data["y_hat_final"] # [n_batch,6,60,2]
        embeds = data["embeds_final"] # [n_batch,6,134]
        x_scored = data["x_scored"] # [n_batch]
                
        valid_step = ~data['x_padding_mask'] # [n_batch, 110]
        valid_mask = valid_step[:, :50].sum(-1) > 49 # [n_batch]
        valid_mask = valid_mask | x_scored # [n_batch]
        reg_mask_all = torch.ones_like(valid_step[:, 50:])


        y_gt = data["y_gt_agentcoords"] # [n_batch,60,2]
        
        y_hat_init_ego, _, refine_y_hat_deltas, refine_pis = self(y_hat, embeds, valid_mask, data)
        y_hat_init_ego = y_hat_init_ego.permute(1,0,2,3)
        '''
        ys_hat_init_ego: [6,n_in_batch,60,4]
        refine_y_hat_deltas: List[5]: refine_y_hat_deltas[0]:[6,n_in_batch,60,4]
        refine_pis:
            pis: List[5] pis[0]:[n_in_batch,6]
            scores: List[6] scores[0]:[n_in_batch,6]
        '''

        refine_y_hat = y_hat_init_ego.clone()
 
        for i in range(self.refine_num):
            refine_y_hat[...,:2] += refine_y_hat_deltas[i][...,:2]
            refine_y_hat[...,2:] = refine_y_hat_deltas[i][...,2:]

        
        y_hat_all = refine_y_hat[:, :, :, :2] # [6, n_batch, 60, 2]

        agent_batch = data['batch'][x_scored]
        refine_y_hat_batch = unbatch(y_hat_all.permute(1,0,2,3)[x_scored], agent_batch)                   #[(n1,K,F,2),...,(nb,K,F,2)]
        gt_y_batch = unbatch(y_gt[x_scored], agent_batch)                                   #[(n1,F,2),...,(nb,F,2)]
        reg_mask_valid_batch = unbatch(reg_mask_all[x_scored], agent_batch)               #[(n1,F),...,(nb,F)]
        self.minJointFDE(refine_y_hat_batch, gt_y_batch, reg_mask_valid_batch)
        self.minJointADE(refine_y_hat_batch, gt_y_batch, reg_mask_valid_batch)
        self.actormr(refine_y_hat_batch, gt_y_batch, reg_mask_valid_batch)

        
        if vis == True:
            fde = fde_agent.min(dim=0)[0]
            return p1_data['traj'].squeeze(0), y_hat_agent, fde


    
    def metrics_reset(self):
        self.minJointFDE.reset()
        self.minJointADE.reset()
        self.actormr.reset()
  
    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay)) if 'encoder_phase1' not in param_name],
             "lr": self.lr,
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay)) if 'encoder_phase1' not in param_name],
             "lr": self.lr,
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Refine')
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--local_radius', type=int, default=-1)
        parser.add_argument('--cls_temperture', type=int, default=1)
        

        parser.add_argument('--future_steps', type=int, default=60)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2)
        parser.add_argument('--edge_dim', type=int, default=2)
        parser.add_argument('--embed_dim', type=int, required=True)
        parser.add_argument('--seg_num', type=int, default=2)
        parser.add_argument('--refine_num', type=int, required=True)
        parser.add_argument('--refine_radius', type=int, default=-1)
        return parent_parser





