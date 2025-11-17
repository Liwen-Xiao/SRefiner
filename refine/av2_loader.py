import os
import random
from typing import Any, Dict, List, Tuple, Union
import argparse
from importlib import import_module
#
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
#
from utils.optimizer import Optimizer
from utils.evaluator import TrajPredictionEvaluator
from refine.av2_refine import *
from refine.av2_refine_dataset import *



class Loader_av2_multiagent:
    '''
        Get and return dataset, network, loss_fn, optimizer, evaluator
    '''

    def __init__(self, args, device, is_ddp=False, world_size=1, local_rank=0, verbose=True):
        self.args = args
        self.device = device
        self.is_ddp = is_ddp
        self.world_size = world_size
        self.local_rank = local_rank
        self.resume = False
        self.verbose = verbose
        self.resume_epoch = 0

    def print(self, info):
        if self.verbose:
            print(info)

    def set_resmue(self, model_path):
        self.resume = True
        if not model_path.endswith(".tar"):
            assert False, "Model path error - '{}'".format(model_path)
        self.ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.resume_epoch = self.ckpt['epoch']

    def load(self):
        # dataset
        dataset = self.get_dataset()
        # network
        model = self.get_model()
        # optimizer
        optimizer, scheduler = self.get_optimizer(model)

        if self.resume == False:
            return dataset, model, optimizer, scheduler, 0
        else:
            return dataset, model, optimizer, scheduler, self.resume_epoch

    def get_dataset(self):

        dataset_dir = self.args.features_dir
        

        if self.args.mode == 'train' or self.args.mode == 'val':
            train_set = AV2_Refine_multiagent(dataset_dir,
                                                                 mode='train',
                                                                 obs_len=50,
                                                                 pred_len=60,
                                                                 verbose=self.verbose,)
            val_set = AV2_Refine_multiagent(dataset_dir,
                                                               mode='val',
                                                               obs_len=50,
                                                               pred_len=60,
                                                               verbose=self.verbose,
                                                               )
            
            return train_set, val_set
        elif self.args.mode == 'test':
            test_set = AV2_Refine_multiagent(dataset_dir,
                                                                mode='test',
                                                                obs_len=50,
                                                                pred_len=60,
                                                                verbose=self.verbose)
            return test_set
        else:
            assert False, "Unknown mode"

    def get_model(self):
        model = Refine_multiagent_AV2(**vars(self.args))

        # print network params
        total_num = sum(p.numel() for p in model.parameters())
        self.print('[Loader] network params:')
        self.print('-- total: {}'.format(total_num))
        subnets = list()
        for name, param in model.named_parameters():
            subnets.append(name.split('.')[0])
        subnets = list(set(subnets))
        for subnet in subnets:
            numelem = 0
            for name, param in model.named_parameters():
                if name.startswith(subnet):
                    numelem += param.numel()
            self.print('-- {} {}'.format(subnet, numelem))

        if self.resume:
            model.load_state_dict(self.ckpt["state_dict"], strict=False)

        if self.is_ddp:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)  # SyncBN
            model = model.to(self.device)
            model.device = self.device
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        else:
            model = model.to(self.device)
            model.device = self.device

        return model

    def get_optimizer(self, model):
        [optimizer], [scheduler] = model.module.configure_optimizers()

        return optimizer, scheduler

    def network_name(self):
        net_name = 'Refine'
        return net_name
