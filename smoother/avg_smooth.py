import os
import sys
from xmlrpc.client import boolean

from matplotlib.colors import to_rgba
# sys.path.insert(0, "../tools")
os.chdir("..")

import pprint
import argparse
import time
import shutil
import json
from itertools import groupby, repeat
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm
import multiprocessing
from numba import njit

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
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--pt', action='store_true')

    args = parser.parse_args()
    return args

def weight_func1(distance):
    # distance, int, between this frame and t
    return 1 / (abs(distance) + 1)

# @njit(parallel=True)
def worker(cfg, boxes, outputs, weights):
    # boxes, [N=window size per, 6], np.array
    # outputs, an N-long list of [#J, W=96, H=72], np.array, each element from 0 to 1, not a probability distribution spatially
    # weights, a list of N long, sums not necessarily to 1, only has one == 1 inside
    # returned, [#J, W, H], heatmap that sums to 1
    # TODO: extend to more interpolation methods from here: https://docs.scipy.org/doc/scipy/reference/interpolate.html#multivariate-interpolation
    t = weights.index(1)  # the index of the main frame to process, out of 0 to N - 1
    dense_grids, ps =  get_ori_coords(cfg, outputs[t], boxes[t, 0:2], boxes[t, 2:4])  # [HW, 2], [J, HW], global, (to left, from top) = (x, y), when indexing, should use [y, x]

    X_ = dense_grids[:cfg.MODEL.HEATMAP_SIZE[0], 0].flatten()  # [W]
    Y_ = dense_grids[0::cfg.MODEL.HEATMAP_SIZE[0], 1].flatten()  # [H]
    X, Y = np.meshgrid(X_, Y_)  # both [H, W]

    tbr = np.zeros(outputs[0].shape)
    for i, weight in enumerate(weights):
        if i == t:  # is main
            tbr += weight * outputs[i]
        else:
            # map non-main frmae boxes to the global coordinates --> get an interpolator --> query with main frame grid
            dense_grids_per, ps_per =  get_ori_coords(cfg, outputs[i], boxes[i, 0:2], boxes[i, 2:4])  # [HW, 2], [J, HW], global, (to left, from top) = (x, y), when indexing, should use [y, x]
            tba = np.zeros(outputs[0].shape, dtype=np.float32)  # [#J, H, W]
            for j in range(cfg.MODEL.NUM_JOINTS):
                interp = LinearNDInterpolator(dense_grids_per, ps_per[j])
                tba[j] = interp(X, Y)
            tbr += weight * tba
    return tbr#  / tbr.sum(axis=-1).sum(axis=-1)[:, None, None]

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

    # if os.path.exists(args.save_path) and os.path.isdir(args.save_path):
    #     shutil.rmtree(args.save_path)
    #     print(f"Removing existing {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'avg_smooth', enforced_name=args.exp)

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

        idx = 0
        # containers
        num_samples = len(val_dataset)
        all_boxes = np.zeros((num_samples, 6))
        track_ids = np.zeros((num_samples, ))  # a list of track ids starting 0 with different lengths
        input_container = []  # a list of CPU torch tensors
        output_container = []  # a list of numpy arrays

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

                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)  # area
                all_boxes[idx:idx + num_images, 5] = score
                track_ids[idx:idx + num_images] = track_id
                input_container.extend([_[0].numpy() for _  in torch.split(input.detach().cpu(), split_size_or_sections=1, dim=0)])
                output_container.extend([_[0].numpy() for _ in torch.split(output.detach().cpu(), split_size_or_sections=1, dim=0)])
                idx += num_images
            
        assert idx == num_samples == len(input_container) == len(output_container), str(idx) + " " + str(num_samples) + " " + str(len(input_container)) + " " + str(len(output_container))
        print("=> loop 1 finished")
        hw = int(args.window // 2)  # half window size
        import pickle
        with open("output.pkl", "wb") as f:
            pickle.dump(output_container, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Create the input list for parallizing
        idx = 0
        x1_container = []
        x2_container = []
        x3_container = []
        for track_id, g in groupby(track_ids.tolist()):            
            num_images = len(list(g))
            for t in range(num_images):
                start_frame = max(0, t - hw)  # index of the start frame within this track
                end_frame = min(num_images - 1, t + hw)
                actual_ws = end_frame - start_frame + 1  # i.e., N
                # TODO: add ways to propagate coordinates between frames
                boxes = all_boxes[idx + start_frame: idx + end_frame + 1]  # [N=window size per, 6]
                outputs = output_container[idx + start_frame: idx + end_frame + 1]  # [N, #J, W=96, H=72]
                x1_container.append(boxes)
                x2_container.append(outputs)
                x3_container.append([weight_func1(t - ttt) for ttt in range(start_frame, end_frame + 1)])
            idx += num_images
        assert idx == num_samples == len(x1_container) == len(x2_container) == len(x3_container), str(idx) + " " + str(num_samples) + " " + str(len(x1_container)) + " " + str(len(x2_container)) + " " + str(len(x3_container))
        print("=> loop 2 finished")

        # Run multi-processing
        num_proc = max(1, multiprocessing.cpu_count() - 1)  # use all processors
        p = multiprocessing.Pool(num_proc)
        refined_output_container = []  # a list of numpy arrays, [#J, W, H]
        for refined_output in p.starmap(worker, tqdm(zip(repeat(cfg), x1_container, x2_container, x3_container), disable=False, total=num_samples, desc=f"=> using {num_proc} processors for {json_path}")):
            refined_output_container.append(refined_output)
        p.close()
        p.join()
        print("=> loop 3 finished")

        all_preds = []  # B-long list of [#J, 3]
        for idx, (refined_output, boxes) in enumerate(zip(refined_output_container, all_boxes)):
            c = boxes[0:2]
            s = boxes[2:4]
            pred, maxvals = get_final_preds(cfg, refined_output, c, s)  # [#J, 2], [#J, 1]
            all_preds.append(np.concatenate([pred, maxvals], axis=-1))
            if args.vis and idx % args.vis == hw + 1:
                prefix = os.path.join(args.save_path, f"{json_path[:-5]}_avg2_{idx // args.vis}")
                # note that the input to debug() is pred not preds, pred is in relative coordinates, preds is in global coordinates
                save_debug_images(
                    cfg, 
                    input=input_container[idx - hw: idx + hw + 1] + [input_container[idx], ],  # window + 1 rows, with the bottom row being the main image
                    meta=None, target=None, joints_pred=None, 
                    output=output_container[idx - hw: idx + hw + 1] + [refined_output, ], 
                    prefix=prefix
                )  # refined_output, [B, #J, W=96, H=72], tensor
        all_preds = np.stack(all_preds, axis=0)
        print("=> loop 4 finished")

        val_dataset.dump_with_updated(all_preds, os.path.join(args.save_path, json_path))

if __name__ == '__main__':
    main()
