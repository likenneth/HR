# Make it single GPU

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import json

# import decord, cause segmentation fault if imported too early
import mmcv
import cv2
import numpy as np
# import torch.distributed as dist
# from mmcv.runner import get_dist_info, init_dist
from tqdm import tqdm

from pyskl.smp import mrlines

DEBUG = False

dataset_root = "/private/home/keli22/datasets/FineGym"
video_path = "/private/home/keli22/HR/data/finegym/processed_frames"

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
    im = cv2.imread(os.path.join(dataset_root, path_in_anno), 1)
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
    parser.add_argument('--det-score-thr', type=float, default=0.7)
    # * Only det boxes with large enough sizes will be kept,
    parser.add_argument('--det-area-thr', type=float, default=1600)
    # * Accepted formats for each line in video_list are:
    # * 1. "xxx.mp4" ('label' is missing, the dataset can be used for inference, but not training)
    # * 2. "xxx.mp4 label" ('label' is an integer (category index),
    # * the result can be used for both training & testing)
    # * All lines should take the same format.
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world', type=int, required=True)
    args = parser.parse_args()
    return args

# def get_annos():
#     with open("/private/home/keli22/datasets/FineGym/finegym_annotation_info_v1.1.json", "r") as f:
#         gym_annot = json.load(f)
#     action_list = []
#     for video, events in gym_annot.items():
#         for event, event_details in events.items():
#             if event_details["segments"]:
#                 for action, action_details in event_details["segments"].items():
#                     relative_time_stamp = [action_details["timestamps"][0][0], action_details["timestamps"][-1][-1]]  # relative to events
#                     fps = 25
#                     s = round(relative_time_stamp[0] * fps)
#                     e = round(relative_time_stamp[1] * fps)
#                     if e <= s:
#                         continue
#                     action_list.append(
#                         {
#                             "video_path": os.path.join(video_path, f"{video}_{event}.mp4"), 
#                             "time_stamp": relative_time_stamp, 
#                             "action_name": f"{video}_{event}_{action}", 
#                         }
#                     )

#     return action_list

def main():
    args = parse_args()
    os.makedirs("FullFineGym_Swin_HRNet", exist_ok=True) 
    rank, world= args.rank, args.world
    my_part = sorted(list(os.listdir(video_path)))[rank::world]

    det_model = init_detector(args.det_config, args.det_ckpt, device='cuda')
    assert det_model.CLASSES[0] == 'person', 'A detector trained on COCO is required'
    pose_model = init_pose_model(args.pose_config, args.pose_ckpt, device='cuda')

    for video_name in tqdm(my_part):
        if os.path.exists(os.path.join("FullFineGym_Swin_HRNet", f'{video_name}.json')):
            continue  # avoid redoing
        anno = {}
        # fps = 25
        # s = round(anno['time_stamp'][0] * fps)
        # e = round(anno['time_stamp'][1] * fps)
        frames = [extract_image(os.path.join(video_path, video_name, _)) for _ in sorted(list(os.listdir(os.path.join(video_path, video_name))))]#[s:e+1]
        det_results_ = detection_inference(det_model, frames)  # det_results_: a T-long list
        # * Get detection results for human
        det_results = [x[0][0] for x in det_results_]  # x: a tuple for (detection, segmentation) result; x[0], a 80-long list of np.array of shape 10*5; for each image; 80 = #cls
        for i, res in enumerate(det_results):
            res = res[res[:, 4] >= args.det_score_thr]
            box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
            assert np.all(box_areas >= 0)
            res = res[box_areas >= args.det_area_thr]
            det_results[i] = res

        # det_results: a T-long list of [#person_t, 5] with varying #person_t for each time step
        zero_filled_bb, pose_results = pose_inference(pose_model, frames, det_results)  # [max #person, #frame, 5], [max #person, #frame, 17, 3], use [0, 0, 0] to fill persons not appearing
        # xyxy to xywh
        w = zero_filled_bb[..., 2] - zero_filled_bb[..., 0]
        h = zero_filled_bb[..., 3] - zero_filled_bb[..., 1]
        zero_filled_bb[..., 2] = w
        zero_filled_bb[..., 3] = h
        
        shape = frames[0].shape[:2]
        anno["bb"] = zero_filled_bb[..., :4]
        anno["bb_score"] = zero_filled_bb[..., 4]
        anno['img_shape'] = anno['original_shape'] = shape
        anno['total_frames'] = len(frames)
        anno['num_person_raw'] = pose_results.shape[0]
        anno['keypoint'] = pose_results[..., :2]#.astype(np.float16)
        anno['keypoint_score'] = pose_results[..., 2]#.astype(np.float16)

        mmcv.dump(anno, osp.join("FullFineGym_Swin_HRNet", f'{video_name}.json'))

if __name__ == '__main__':
    main()
