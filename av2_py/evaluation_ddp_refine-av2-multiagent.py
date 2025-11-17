import os
import sys

os.sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
print(os.getcwd())
print(sys.path)

import time
import copy
import subprocess
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import faulthandler
from tqdm import tqdm
import numpy as np
#
import torch
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
#
from utils.logger import Logger
from utils.utils import AverageMeterForDict
from utils.utils import save_ckpt, set_seed, str2bool, distributed_mean
from torch_geometric.data import Data, Batch

from refine.av2_loader import *
from refine.av2_refine_dataset import *
from refine.av2_refine import *



def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", type=str, help="Mode, train/val/test")
    parser.add_argument("--features_dir", required=True, default="", type=str, help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Val batch size")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--logger_writer", action="store_true", help="Enable tensorboard")
    parser.add_argument("--rank_metric", required=False, type=str, default="brier_fde_k", help="Ranking metric")
    parser.add_argument("--no_pbar", action="store_true", help="Hide progress bar")
    parser.add_argument("--model_path", required=False, type=str, help="path to the saved model")
    parser.add_argument("--file_name", type=str, default=None)
    parser = Refine_multiagent_AV2.add_model_specific_args(parser)
    return parser.parse_args()


def main():
    args = parse_arguments()
    faulthandler.enable()
    start_time = time.time()

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()

    is_main = True if local_rank == 0 else False

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.file_name == None:
        log_dir = "log/" + date_str
    else:
        log_dir = "log/" + date_str + "_" + args.file_name[:-3]
    logger = Logger(date_str=log_dir[4:], enable=is_main, log_dir=log_dir,
                    enable_flags={'writer': False, 'mailbot': False})
    
    logger.print(args)
    # log basic info
    logger.log_basics(args=args, datetime=date_str)

    loader = Loader_av2_multiagent(args, device, is_ddp=True, world_size=world_size, local_rank=local_rank, verbose=is_main)
    logger.print('[Resume] Loading state_dict from {}'.format(args.model_path))
    loader.set_resmue(args.model_path)
    (train_set, val_set), net, optimizer, scheduler, resume_epoch = loader.load()
    logger.print(net)
    resume_epoch = 0

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    dl_val = DataLoader(val_set,
                        batch_size=args.val_batch_size,
                        num_workers=48,
                        collate_fn=val_set.collate_fn,
                        drop_last=True,
                        sampler=val_sampler,
                        pin_memory=True)

    

    with torch.no_grad():
        val_start = time.time()
        dl_val.sampler.set_epoch(0)
        net.eval()
        net.module.metrics_reset() # 指标清零
        for i, data in enumerate(tqdm(dl_val, disable=(not is_main) or args.no_pbar, ncols=80)):
            data = Batch.from_data_list(data)    
            net.module.validation_step(data)

        print('[Validation], time cost: {:.3} mins'.format((time.time() - val_start) / 60.0))
        print('--minJointADE: {:.4}, minJointFDE: {:.4}, ActorMR: {:.4}'.format(net.module.minJointADE.compute(), net.module.minJointFDE.compute(), net.module.actormr.compute()))
        print('sample num: ', net.module.minJointFDE.count)

    dist.destroy_process_group()
    logger.print('\nExit...\n')


if __name__ == "__main__":
    main()
