import os
import sys
# sys.path.insert(0, "../tools")
os.chdir("..")

import pprint
import argparse
import time
import json
import shutil

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from utils.utils import create_logger
from core.function import AverageMeter
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images

import dataset
import models

DEBUG = False

def parse_args():
    parser = argparse.ArgumentParser(description='Extract Keypoints with HRNet trained with HRNet repo for ONE video')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=None,
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

    # ken
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world', type=int, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--pt', action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.pt:
        args.cfg = "experiments/coco/hrnet/coco_ptswcthr.yaml"
        dscfg = cfg.POSETRACK
    else:
        args.cfg = "experiments/coco/hrnet/coco_fgswcthr.yaml"
        dscfg = cfg.FINEGYM
    update_config(cfg, args)
    json_paths = [_ for _ in os.listdir(dscfg.PSEUDO_LABEL) if _.endswith(".json")][args.rank::args.world]
    args.save_path = os.path.join("smoother", args.save_path)

    if os.path.exists(args.save_path) and os.path.isdir(args.save_path):
        shutil.rmtree(args.save_path)
        print(f"Removing existing {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'wo_smooth', enforced_name=args.exp)

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    for json_path in json_paths:
        if os.path.exists(os.path.join(args.save_path, json_path)):
            continue  # avoid redoing

        val_dataset = eval('dataset.'+dscfg.DATASET)(
            cfg, dscfg.ROOT, dscfg.TRAIN_SET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), 
            json_index=json_path, 
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=False, 
        )

        # perf_indicator = validate(
        #     cfg, valid_loaders[0], valid_datasets[0], model, criterion,
        #     final_output_dir, tb_log_dir
        # )

        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        data_prefix = "_".join([val_dataset.name, val_dataset.image_set, ""])

        # switch to evaluate mode
        model.eval()

        num_samples = len(val_dataset)
        all_preds = np.zeros(
            (num_samples, cfg.MODEL.NUM_JOINTS, 3),
            dtype=np.float32
        )
        all_boxes = np.zeros((num_samples, 6))
        track_ids = np.zeros((num_samples, ))
        image_path = []
        idx = 0
        with torch.no_grad():
            end = time.time()
            for i, (input, target, target_weight, meta) in enumerate(val_loader):
                # compute output
                outputs = model(input)
                if isinstance(outputs, list):
                    output = outputs[-1]
                else:
                    output = outputs

                if cfg.TEST.FLIP_TEST:
                    input_flipped = input.flip(3)
                    outputs_flipped = model(input_flipped)

                    if isinstance(outputs_flipped, list):
                        output_flipped = outputs_flipped[-1]
                    else:
                        output_flipped = outputs_flipped

                    output_flipped = flip_back(output_flipped.cpu().numpy(),
                                            val_dataset.flip_pairs)
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    if cfg.TEST.SHIFT_HEATMAP:
                        output_flipped[:, :, :, 1:] = \
                            output_flipped.clone()[:, :, :, 0:-1]

                    output = (output + output_flipped) * 0.5

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                loss = criterion(output, target, target_weight)

                num_images = input.size(0)
                # measure accuracy and record loss
                losses.update(loss.item(), num_images)
                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                                target.cpu().numpy())

                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                score = meta['score'].numpy()
                track_id = meta['track_id'].numpy()

                preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), c, s)

                if DEBUG and i == 0:
                    prefix = os.path.join(args.save_path, f"{json_path[:-5]}_avg_{i}")
                    save_debug_images(cfg, input, meta, target, pred*4, output, prefix)

                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                track_ids[idx:idx + num_images] = track_id
                image_path.extend(meta['image'])

                idx += num_images
            
        assert idx == num_samples == len(image_path), str(idx) + " " + str(num_samples) + " " + str(len(image_path))

        val_dataset.dump_with_updated(all_preds, os.path.join(args.save_path, json_path))

if __name__ == '__main__':
    main()
