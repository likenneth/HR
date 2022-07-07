# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import copy
import itertools

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models

DEBUG = False
OC = True

class CatDataLoaders(object):
    def __init__(self, datasets, batch_sizes, cfg):
        self.dataloaders = []
        self.batch_sizes = batch_sizes
        for i, (dataset, batch_size) in enumerate(zip(datasets, batch_sizes)):
            self.dataloaders.append(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size*len(cfg.GPUS),
                    shuffle=cfg.TRAIN.SHUFFLE,
                    num_workers=cfg.WORKERS // len(datasets),
                    pin_memory=cfg.PIN_MEMORY# if i == 0 else False, 
                )
            )
            print(f"=> {dataset.name} has {len(dataset)} perons, per-GPU batch size of {batch_size}, and {len(self.dataloaders[-1])} iters per epoch")
        print(f"=> Concatenate Dataset Created with batch size {batch_sizes} and length {len(self)}")

    def __len__(self, ):
        return len(self.dataloaders[0])

    def __iter__(self):
        # any object with a __iter__() is an iterable
        # any object with a __next__() is an iterator
        # this object is both
        # shuffle happens in iter(data_loader)
        self.loader_iter = []
        for i, data_loader in enumerate(self.dataloaders):
            if i == 0:
                self.loader_iter.append(iter(data_loader))  # will raise StopIteration once COCO is drain, as required by __next__
            else:
                self.loader_iter.append(iter(data_loader))  # TODO: now assuming additional datasets have more iterations than COCO
                # done: fix memory linear increasing bug, due to the implememntation of itertools.cycle
        return self

    def __next__(self):
        # done: additional dataset data is always at the end of the list, but for DP it should be OK
        input_ct, target_ct, target_weight_ct, meta_ct = [], [], [], []
        for data_iter in self.loader_iter:
            input, target, target_weight, meta = next(data_iter)
            input_ct.append(input)  # B, 3, W, H
            target_ct.append(target)  # B, 17, W // 4, H // 4
            target_weight_ct.append(target_weight)  # B, 17, 1
            meta_ct.append(meta)  # a dict of list or tensors
        input = torch.cat(input_ct, dim=0)
        target = torch.cat(target_ct, dim=0)
        target_weight = torch.cat(target_weight_ct, dim=0)
        meta = {}
        for k in meta_ct[0].keys():
            if type(meta_ct[0][k]) == torch.Tensor:
                meta[k] = torch.cat([_[k] for _ in meta_ct], dim=0)
            elif type(meta_ct[0][k]) == list:
                meta[k] = [__ for _ in meta_ct for __ in _[k]]
            else:
                assert 0
        return input, target, target_weight, meta

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--exp',
                        help='name for this experirment',
                        type=str,
                        default='')                

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train', enforced_name=args.exp)

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=True)

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))
    # writer_dict['writer'].add_graph(model, (dump_input, ))

    logger.info(get_model_summary(model, dump_input))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # add FineGym etc. into training
    additional_pcts = []  # a list of int, batch size per GPU for additional datasets
    additional_trainset = []
    if cfg.FINEGYM.PCT > 0:
        additional_pcts.append(cfg.FINEGYM.PCT)
        additional_trainset.append(      
            eval('dataset.'+cfg.FINEGYM.DATASET)(
                cfg, cfg.FINEGYM.ROOT, cfg.FINEGYM.TRAIN_SET, True,
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
            )
        )
    # duplicate the snippet above and replace FINEGYM with PT21, etc.

    if sum(additional_pcts) < cfg.TRAIN.BATCH_SIZE_PER_GPU:
        additional_pcts.insert(0, cfg.TRAIN.BATCH_SIZE_PER_GPU - sum(additional_pcts))
        additional_trainset.insert(0, 
            eval('dataset.'+cfg.DATASET.DATASET)(
                cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
            )
        )
    elif sum(additional_pcts) == cfg.TRAIN.BATCH_SIZE_PER_GPU:
        pass  # not loading COCO for training
    else:
        assert 0

    valid_datasets = []
    coco_val = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_datasets.append(coco_val)

    if OC and hasattr(cfg, "OCHUMAN"):
        oc_val = eval('dataset.'+cfg.OCHUMAN.DATASET)(
            cfg, cfg.OCHUMAN.ROOT, cfg.OCHUMAN.VAL_SET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
        valid_datasets.append(oc_val)
        oc_test = eval('dataset.'+cfg.OCHUMAN.DATASET)(
            cfg, cfg.OCHUMAN.ROOT, cfg.OCHUMAN.TEST_SET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
        valid_datasets.append(oc_test)

    if DEBUG:
        train_dataset = additional_trainset[1]
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
            shuffle=cfg.TRAIN.SHUFFLE,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )
    else:
        train_loader = CatDataLoaders(additional_trainset, additional_pcts, cfg)

    valid_loaders = []
    for valid_dataset in valid_datasets:
        valid_loaders.append(
            torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
                shuffle=False,
                num_workers=cfg.WORKERS,
                pin_memory=cfg.PIN_MEMORY
            )
        )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        # evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loaders[0], valid_datasets[0], model, criterion,
            final_output_dir, tb_log_dir, writer_dict
        )  # valid_loaders[0] is always COCO val2017
        writer_dict['valid_global_steps'] += 1

        if OC:
            for valid_loader, valid_dataset in zip(valid_loaders[1:], valid_datasets[1:]):
                _ = validate(
                    cfg, valid_loader, valid_dataset, model, criterion,
                    final_output_dir, tb_log_dir, writer_dict
                )

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
