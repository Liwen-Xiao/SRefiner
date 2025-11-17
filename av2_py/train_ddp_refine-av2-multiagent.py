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
# from torch_geometric.data import DataLoader
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
    parser.add_argument("--train_epoches", type=int, default=10, help="Number of epoches for training")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation intervals")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--logger_writer", action="store_true", help="Enable tensorboard")
    parser.add_argument("--rank_metric", required=False, type=str, default="brier_fde_k", help="Ranking metric")
    parser.add_argument("--resume", action="store_true", help="Resume training")
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
    set_seed(args.seed + local_rank)

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
                    enable_flags={'writer': args.logger_writer, 'mailbot': False})
    logger.print(args)
    # log basic info
    logger.log_basics(args=args, datetime=date_str)

    loader = Loader_av2_multiagent(args, device, is_ddp=True, world_size=world_size, local_rank=local_rank, verbose=is_main)
    if args.resume:
        logger.print('[Resume] Loading state_dict from {}'.format(args.model_path))
        loader.set_resmue(args.model_path)
    (train_set, val_set), net, optimizer, scheduler, resume_epoch = loader.load()
    logger.print(net)
    resume_epoch = 0
    # print(resume_epoch)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    dl_train = DataLoader(train_set,
                          batch_size=args.train_batch_size,
                          num_workers=48,
                          collate_fn=train_set.collate_fn,
                          drop_last=True,
                          sampler=train_sampler,
                          pin_memory=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    dl_val = DataLoader(val_set,
                        batch_size=args.val_batch_size,
                        num_workers=48,
                        collate_fn=val_set.collate_fn,
                        drop_last=True,
                        sampler=val_sampler,
                        pin_memory=True)

    niter = 0
    best_metric = 1e6
    rank_metric = args.rank_metric
    net_name = loader.network_name()

    for epoch in range(args.train_epoches - resume_epoch):
        epoch += resume_epoch
        dist.barrier()  # sync
        logger.print('\nEpoch {}'.format(epoch))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # * Train
        dl_train.sampler.set_epoch(epoch)
        epoch_start = time.time()
        train_loss_meter = AverageMeterForDict()
        net.train()
        for i, data in enumerate(tqdm(dl_train, disable=(not is_main) or args.no_pbar, ncols=80)):    
            data = Batch.from_data_list(data)
            loss = net.module.training_step(data)     
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_out={}
            loss_out['loss'] = loss
            train_loss_meter.update(loss_out)
            niter += world_size * args.train_batch_size
            logger.add_dict(loss_out, niter, prefix='train/')

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        max_memory = torch.cuda.max_memory_allocated(device=device) // 2 ** 20

        loss_avg = train_loss_meter.metrics['loss'].avg
        logger.print('[Training] Avg. loss: {:.6}, time cost: {:.3} mins, lr: {:.3}, peak mem: {} MB'.
                     format(loss_avg, (time.time() - epoch_start) / 60.0, lr, max_memory))

        logger.add_scalar('train/lr', lr, it=epoch)
        logger.add_scalar('train/max_mem', max_memory, it=epoch)

        dist.barrier()  # sync
        if ((epoch + 1) % args.val_interval == 0) or epoch > int(args.train_epoches / 2):
            # * Validation
            with torch.no_grad():
                val_start = time.time()
                dl_val.sampler.set_epoch(epoch)
                net.eval()
                net.module.metrics_reset() # 指标清零
                for i, data in enumerate(tqdm(dl_val, disable=(not is_main) or args.no_pbar, ncols=80)):
                    data = Batch.from_data_list(data)
                    net.module.validation_step(data)

                logger.print('[Validation], time cost: {:.3} mins'.format((time.time() - val_start) / 60.0))
                logger.print('--minJointADE: {:.4}, minJointFDE: {:.4}'.format(net.module.minJointADE.compute(), net.module.minJointFDE.compute()))
                
                if is_main:
                    if net.module.minJointFDE.compute() < best_metric:
                        model_name = date_str + '_{}_ddp_best.tar'.format(net_name)
                        save_ckpt(net.module, optimizer, epoch, 'saved_models/', model_name)
                        best_metric = net.module.minJointFDE.compute()
                        logger.print('Save the model: {}, {}: {:.4}, epoch: {}'.format(
                            model_name, rank_metric, best_metric, epoch))

        if is_main:
            model_name = date_str + '_{}_ddp_ckpt_epoch{}.tar'.format(net_name, epoch)
            save_ckpt(net.module, optimizer, epoch, 'saved_models/', model_name)
            logger.print('Save the model to {}'.format('saved_models/' + model_name))

    logger.print("\nTraining completed in {:.2f} mins".format((time.time() - start_time) / 60.0))

    if is_main:
        # save trained model
        model_name = date_str + '_{}_ddp_epoch{}.tar'.format(net_name, args.train_epoches)
        save_ckpt(net.module, optimizer, epoch, 'saved_models/', model_name)
        logger.print('Save the model to {}'.format('saved_models/' + model_name))

    dist.destroy_process_group()
    logger.print('\nExit...\n')


if __name__ == "__main__":
    main()
