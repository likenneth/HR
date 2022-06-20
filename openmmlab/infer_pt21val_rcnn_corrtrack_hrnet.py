# cd openmmlab
# python infer_pt21val_rcnn_corrtrack_hrnet.py --json_file /private/home/keli22/CorrTrack/baselines/outputs/tracking_baselines/corrtrack_baseline/pose_3_stage_corr_tracking/jt_thres_0.1_duplicate_ratio_0.6_oks_0.2_corr_threshold_0.3_win_len_2_min_keypoints_2_min_track_len_3_break_tracks_True_pp_joint_threshold_0.3/sequences/000342_mpii_test.json

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from genericpath import exists
import os
import os.path as osp
import json
import cv2

import mmcv
import numpy as np
from tqdm import tqdm
import ipdb

from pyskl.smp import mrlines

DEBUG = False

# result_dir = "/private/home/keli22/CorrTrack/baselines/outputs/tracking_baselines/corrtrack_baseline/pose_3_stage_corr_tracking/jt_thres_0.1_duplicate_ratio_0.6_oks_0.2_corr_threshold_0.3_win_len_2_min_keypoints_2_min_track_len_3_break_tracks_True_pp_joint_threshold_0.3/sequences"
# gt_dir = "/private/home/keli22/datasets/PoseTrack21/posetrack_data/val"

dataset_root = "/private/home/keli22/datasets/PoseTrack21"

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

def pose_inference(model, frames, det_results):
    assert len(frames) == len(det_results)
    total_frames = len(frames)
    num_person = max([len(x) for x in det_results])
    bb = np.zeros((num_person, total_frames, 5), dtype=np.float32)
    kp = np.zeros((num_person, total_frames, 17, 3), dtype=np.float32)

    for i, (f, d) in enumerate(zip(frames, det_results)):
        # Align input format
        dd = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, dd, format='xywh')[0]
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
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--opt_root', type=str, required=True)
    args = parser.parse_args()
    return args

def get_annos(json_file):
    with open(json_file, "rb") as f:
        video_annos = json.load(f)  # 'images', 'annotations', 'categories'
    num_tracks = len(list(set([_["track_id"] for _ in video_annos["annotations"]])))  # id starts from 0

    new_annotations = []
    for track_id in range(num_tracks):
        track_per = []
        bboxes = [_ for _ in video_annos["annotations"] if _["track_id"]==track_id]
        # element of annotations are dict with keys: ['image_id', 'keypoints', 'scores', 'track_id', 'bbox']
        # bboxes = [_ for _ in bboxes if "bbox" in _]  # some of the bbox has "new_anno"=True, but does not have "bbox", are from propagate
        # print(bboxes[0]["image_id"], bboxes[-1]["image_id"], len(bboxes))  # we actaully need to sort it by image id, but it has been sorted
        # for those without bbox, the [:, -1] of "keypoints" are a duplicate of "scores"
        # for those with bbox, the [:, -1] of "keypoints" are binerized of "scores" by f x: x > 0

        for bbox in bboxes:
            if "bbox" not in bbox:
                kpts = np.array(bbox["keypoints"]).reshape(17, 3)
                visible_kpts = kpts[kpts[:, -1] > 0]  # [<=17, 3], invisible joints have coordinates 0
                x_min = visible_kpts[..., 0].min(axis=-1)  # []
                x_max = visible_kpts[..., 0].max(axis=-1)  # []
                y_min = visible_kpts[..., 1].min(axis=-1)  # []
                y_max = visible_kpts[..., 1].max(axis=-1)  # []
                x1, y1, w, h = x_min, y_min, x_max - x_min, y_max - y_min
                bbox["bbox"] = [x1.item(), y1.item(), w.item(), h.item()]

            # bbox.pop("keypoints")
            # bbox.pop("scores")  # this score is also used for mAP calculation (why?)
            bbox["category_id"] = 1
            track_per.append(bbox)
        new_annotations.append(track_per)
    video_annos["annotations"] = new_annotations
    return video_annos

def main():
    args = parse_args()
    annos = get_annos(args.json_file)  # coco format but annnotations segmented by tracks
    pose_model = init_pose_model(args.pose_config, args.pose_ckpt, device='cuda')

    # build image index from
    image_id2relative_path = {}
    for image in annos["images"]:
        image_id2relative_path[image["image_id"]] = image["file_name"]

    new_annotations = []
    for my_part in annos["annotations"]:  # a list of person bboxes from the same track
        # ipdb.set_trace()
        # prepare for mmpose API
        frames = []  # a T-long list of cv2 loaded images, for the track
        det_results = []  # det_results: a T-long list of [#person_t=1, 5] with varying #person_t for each time step
        for anno in my_part:
            frames.append(extract_image(image_id2relative_path[anno["image_id"]]))
            bbox = anno["bbox"]  # bbox in format xywh, from CorrTrack
            bbox.append(1.)  # a confidence value for the API of pose_inference
            det_results.append(np.array(bbox)[None, :])
        zero_filled_bb, pose_results = pose_inference(pose_model, frames, det_results)
        # pose_results, [max #person=1, #frame, 17, 3], use [0, 0, 0] to fill persons not appearing

        # TODO: implement temporal smoothing

        # process returned from mmpose
        for t, anno in enumerate(my_part):
            if 1 or "bbox" in anno:  # do not update if the pose is propagated
                anno['keypoints'] = pose_results[0, t].flatten().tolist()  # a list of 51
                anno["scores"] = pose_results[0, t, :, -1].flatten().tolist()  # a list of 17
        
        new_annotations.extend(my_part)

    annos["annotations"] = new_annotations
    os.makedirs(args.opt_root, exist_ok=True)
    video_name = osp.basename(args.json_file)
    mmcv.dump(annos, osp.join(args.opt_root, video_name))


if __name__ == '__main__':
    main()
