# CUDA_VISIBLE_DEVICES=1 python infer_finegym_swin.py

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import json
import cv2

# import decord, cause segmentation fault if imported too early
import mmcv
import numpy as np
import ipdb
from tqdm import tqdm

from pyskl.smp import mrlines

DEBUG = False

video_path = "/private/home/keli22/datasets/FineGym/processed_frames"

try:
    import mmdet
    from mmdet.apis import init_detector, inference_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    import mmpose
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

default_mmdet_root = osp.dirname(mmdet.__path__[0])
default_mmpose_root = osp.dirname(mmpose.__path__[0])
default_det_config = f'{default_mmdet_root}/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
default_det_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'
default_pose_config = f'{default_mmpose_root}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288.py'
default_pose_ckpt = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth'


def extract_image(path_in_anno):
    im = cv2.imread(os.path.join(path_in_anno), 1)
    return im

def detection_inference(model, frames):
    results = []
    for frame in frames:
        result = inference_detector(model, frame)
        results.append(result)
    return results

def pose_inference(model, frames, det_results):
    assert len(frames) == len(det_results)
    total_frames = len(frames)
    num_person = max([len(x) for x in det_results])
    bb = np.zeros((num_person, total_frames, 5), dtype=np.float32)
    kp = np.zeros((num_person, total_frames, 17, 3), dtype=np.float32)

    for i, (f, d) in enumerate(zip(frames, det_results)):
        # Align input format
        dd = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, dd, format='xyxy')[0]
        for j, (x, item) in enumerate(zip(list(d), pose)):
            bb[j, i] = x
            kp[j, i] = item['keypoints']
    return bb, kp

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    # * Both mmdet and mmpose should be installed from source
    parser.add_argument('--mmdet-root', type=str, default=default_mmdet_root)
    parser.add_argument('--mmpose-root', type=str, default=default_mmpose_root)
    parser.add_argument('--det-config', type=str, default=default_det_config)
    parser.add_argument('--det-ckpt', type=str, default=default_det_ckpt)
    parser.add_argument('--pose-config', type=str, default=default_pose_config)
    parser.add_argument('--pose-ckpt', type=str, default=default_pose_ckpt)
    # * Only det boxes with score larger than det_score_thr will be kept
    parser.add_argument('--det-score-thr', type=float, default=0.5)
    # * Only det boxes with large enough sizes will be kept,
    parser.add_argument('--det-area-thr', type=float, default=100)
    # * Accepted formats for each line in video_list are:
    # * 1. "xxx.mp4" ('label' is missing, the dataset can be used for inference, but not training)
    # * 2. "xxx.mp4 label" ('label' is an integer (category index),
    # * the result can be used for both training & testing)
    # * All lines should take the same format.
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    tbd = []
    rank, world_size = args.rank, args.world_size
    if rank == -1:
        for i in tqdm(range(world_size)):
            with open(osp.join("/private/home/keli22/HR/corrtrack/baselines/data/detections", f"FineGym_swin_bb_thres_{args.det_score_thr}_{i}from{world_size}.json"), "rb") as f:
                ldd = json.load(f)
            tbd.extend(ldd)
        mmcv.dump(tbd, osp.join("/private/home/keli22/HR/corrtrack/baselines/data/detections", f"FineGym_swin_bb_thres_{args.det_score_thr}.json"))
        return
    my_part = list(os.listdir(video_path))[rank::world_size]
    
    det_model = init_detector(args.det_config, args.det_ckpt, device='cuda')
    assert det_model.CLASSES[0] == 'person', 'A detector trained on COCO is required'
    # pose_model = init_pose_model(args.pose_config, args.pose_ckpt, device='cuda')

    for vidx, video_name in enumerate(tqdm(my_part)):

        frames = list(os.listdir(osp.join(video_path, video_name)))
        if DEBUG:
            frames = frames[:2]
        # vid_id = video_name.split("_")[0]  # str
        video_level_info = {
            "keypoints": [], 
            "file_id": video_name + ".json",
            "seq_name": video_name, 
            "tot_frames": len(frames), 
            "track_id": -1, 
            "vid_id": video_name, 
        }

        for frame_file_name in frames:
            try:
                frame_id = int(frame_file_name.split(".")[0])
            except:
                print(osp.join(video_path, video_name), frame_file_name)
            frame_level_info = {
                "frame_idx": frame_id, 
                "image_location": f"processed_frames/{video_name}/{frame_file_name}", 
                "img": f"{video_name}_{frame_id:03}"
            }
            img = extract_image(osp.join(video_path, video_name, frame_file_name))
            res = inference_detector(det_model, img)[0][0]  # [#person_t, 5] with varying #person_t for each time step
            res = res[res[:, 4] >= args.det_score_thr]
            box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])  # in xyxy format
            assert np.all(box_areas >= 0)
            res = res[box_areas >= args.det_area_thr]
            # xyxy to xywh
            w = res[:, 2] - res[:, 0]
            h = res[:, 3] - res[:, 1]
            res[:, 2] = w
            res[:, 3] = h
            
            for pidx in range(res.shape[0]):
                person_level_info = {
                    "bbox": list(res[pidx]), 
                    'bbox_score': res[pidx, -1].item(), 
                }
                person_level_info.update(frame_level_info)
                person_level_info.update(video_level_info)
                tbd.append(person_level_info)
        if DEBUG:
            break

    mmcv.dump(tbd, osp.join("/private/home/keli22/HR/corrtrack/baselines/data/detections", f"FineGym_swin_bb_thres_{args.det_score_thr}_{rank}from{world_size}.json"))


if __name__ == '__main__':
    main()
