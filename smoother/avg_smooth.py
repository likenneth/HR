import os
import sys
# sys.path.insert(0, "../tools")
os.chdir("..")

import pprint
import argparse
import time
import json
from itertools import groupby
from scipy.interpolate import LinearNDInterpolator

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
from core.inference import get_final_preds, get_ori_coords
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
                        default='experiments/coco/hrnet/coco_fgswcthr.yaml',
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
    parser.add_argument('--window', type=int, required=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    json_paths = list(os.listdir(cfg.FINEGYM.PSEUDO_LABEL))[args.rank::args.world]
    args.save_path = os.path.join("smoother", args.save_path)
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

        val_dataset = eval('dataset.'+cfg.FINEGYM.DATASET)(
            cfg, cfg.FINEGYM.ROOT, cfg.FINEGYM.TRAIN_SET, False,
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

        idx = 0
        # containers
        num_samples = len(val_dataset)
        all_boxes = np.zeros((num_samples, 6))
        track_ids = np.zeros((num_samples, ))  # a list of track ids starting 0 with different lengths
        image_path = []
        dense_grid_container = np.zeros((num_samples, cfg.MODEL.HEATMAP_SIZE[1] * cfg.MODEL.HEATMAP_SIZE[0], 2), dtype=np.float32)  # [B, HW, 2]
        p_container = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS, cfg.MODEL.HEATMAP_SIZE[1] * cfg.MODEL.HEATMAP_SIZE[0]), dtype=np.float32)  # [B, #J, HW]

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
                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(), target.cpu().numpy())

                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                score = meta['score'].numpy()
                track_id = meta['track_id'].numpy()

                # preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), c, s)  # output, [B, J, H, W]
                dense_grids, ps =  get_ori_coords(cfg, output.clone().cpu().numpy(), c, s)  # [B, HW, 2], [B, J, HW], global, (to left, from top) = (x, y), when indexing, should use [y, x]

                if DEBUG and i == 0:
                    prefix = os.path.join(args.save_path, f"{json_path[:-5]}_avg_{i}")
                    # note that the input to debug() is pred not preds, pred is in relative coordinates, preds is in global coordinates
                    save_debug_images(cfg, input, meta, target, pred*4, output, prefix)  # output, [B, J, W=96, H=72], tensor
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)  # area
                all_boxes[idx:idx + num_images, 5] = score
                track_ids[idx:idx + num_images] = track_id
                image_path.extend(meta['image'])
                dense_grid_container[idx:idx + num_images, :] = dense_grids
                p_container[idx:idx + num_images, :] = ps

                idx += num_images
            
        assert idx == num_samples == len(image_path), str(idx) + " " + str(num_samples) + " " + str(len(image_path))

        hw = int(args.window // 2)  # half window size
        all_preds = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS, 3), dtype=np.float32)  # [B, #J, 3]
        # smoothing: generate a new all_preds from dense_grid_container ([[B, HW, 2]]) and p_container ([B, #J, HW])
        idx = 0
        for track_id, g in groupby(track_ids.tolist()):
            
            num_images = len(list(g))
            # TODO: extend to more interpolation methods from here: https://docs.scipy.org/doc/scipy/reference/interpolate.html#multivariate-interpolation
            for t in range(num_images):
                start_frame = max(0, t - hw)  # index of the start frame within this track
                end_frame = min(num_images - 1, t + hw)
                actual_ws = end_frame - start_frame + 1  # i.e., N
                # TODO: add ways to propagate coordinates between frames
                dense_grids = dense_grid_container[int(idx + start_frame): int(idx + end_frame + 1)]  # [N=window size per, HW, 2]
                X_ = dense_grids[:, :cfg.MODEL.HEATMAP_SIZE[0], 0].flatten()  # [NW]
                Y_ = dense_grids[:, 0::cfg.MODEL.HEATMAP_SIZE[0], 1].flatten()  # [NH]
                X, Y = np.meshgrid(X_, Y_)  # both [NH, NW]
                denser_ps = np.zeros((cfg.MODEL.NUM_JOINTS, cfg.MODEL.HEATMAP_SIZE[1] * actual_ws, cfg.MODEL.HEATMAP_SIZE[0] * actual_ws), dtype=np.float32)  # [#J, NH, NW]
                for j in range(cfg.MODEL.NUM_JOINTS):
                    # TODO: hack the following line to do smoothing with different weights to neighbouring frames
                    ps = p_container[idx + start_frame: idx + end_frame + 1, j].flatten()  # [NHW]
                    # interpolator takes in N*H*W points in 2D original-resolution plane, and gives out N*NH*NW points
                    interp = LinearNDInterpolator(dense_grids.reshape(ps.shape[0], 2), ps)
                    denser_ps[j] = interp(X, Y)  # [NH, NW]

                width = denser_ps.shape[-1]  # i.e., NW
                heatmaps_reshaped = denser_ps.reshape((cfg.MODEL.NUM_JOINTS, -1))  # [#J, NHNW]
                indices = np.argmax(heatmaps_reshaped, 1)  # [#J], index within NHNW to index from X_, Y_, which are the real float coordinates
                maxvals = np.amax(heatmaps_reshaped, 1)  # [#J]
                all_preds[idx + t, :, 0] = X_[indices % width]
                all_preds[idx + t, :, 1] = Y_[np.floor(indices / width).astype(np.int64)]
                all_preds[idx + t, :, 2] = maxvals
            idx += num_images

        assert idx == num_samples == len(image_path), str(idx) + " " + str(num_samples) + " " + str(len(image_path))

        # dump results for this event
        with open(os.path.join(cfg.FINEGYM.PSEUDO_LABEL, json_path), "rb") as f:
            video_annos = json.load(f)  # 'images', 'annotations', 'categories'

        assert num_samples == len(video_annos["annotations"]), str(num_samples) + " " + str(len(video_annos["annotations"]))
        for idx in range(num_samples):  # because the dataloader does not shuffle and there is no re-used track id in one video
            video_annos["annotations"][idx]["keypoints"] = all_preds[idx].flatten().tolist()
            video_annos["annotations"][idx]["scores"] = all_preds[idx, -1].flatten().tolist()

        with open(os.path.join(args.save_path, json_path), "w") as f:
            json.dump(video_annos, f)

if __name__ == '__main__':
    main()
