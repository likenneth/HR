import sys
sys.path.insert(0, '/private/home/keli22/HR/corrtrack/baselines')

import torch
import random
import torch.backends.cudnn as cudnn
import argparse
import json
import numpy as np
import cv2
import os
import os.path as osp
import time

from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import functional as F

from models.pose_estimation.bn_inception2 import bninception
from common.utils.inference_utils import get_preds_for_pose, get_transform
from common.utils.data_utils import get_posetrack_eval_dummy

import mmcv
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


flipRef = [i - 1 for i in [1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16]]

DEBUG = False

def extract_image(path_in_anno):
    im = cv2.imread(os.path.join(path_in_anno), 1)
    return im

def apply_augmentation_test(example, img_dir, output_size=[256, 256]):
    im = cv2.imread(os.path.join(img_dir, example['image_location']), 1)

    x1, x2 = example['bbox'][0], example['bbox'][0] + example['bbox'][2]
    y1, y2 = example['bbox'][1], example['bbox'][1] + example['bbox'][3]

    crop_pos = [(x1 + x2) / 2, (y1 + y2) / 2]
    max_d = np.maximum(example['bbox'][2], example['bbox'][3])

    scales = [output_size[0] / float(max_d), output_size[1] / float(max_d)]

    param = {'rot': 0,
            'scale_x': 1,
            'scale_y': 1,
            'flip': 0,
            'tx': 0,
            'ty': 0}

    t_form = get_transform(param, crop_pos, output_size, scales)
    im_cv = cv2.warpAffine(im, t_form[0:2, :], (output_size[0], output_size[1]))
    img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
    imf = cv2.flip(img, 1)

    img = torch.from_numpy(img).float()
    img = torch.transpose(img, 1, 2)
    img = torch.transpose(img, 0, 1)
    img /= 255

    imf = torch.from_numpy(imf).float()
    imf = torch.transpose(imf, 1, 2)
    imf = torch.transpose(imf, 0, 1)
    imf /= 255

    warp = torch.from_numpy(np.linalg.inv(t_form))

    return img, imf, warp


class PoseTrack:
    def __init__(self, args, json_path, img_dir=None):
        json_name = osp.basename(json_path)
        video_name = json_name.split(".")[0]
        self.video_name = video_name
        self.output_size = [args.output_size_x, args.output_size_y]
        self.keep_track_id = args.keep_track_id
        self.img_dir = img_dir
        self.anno = []  # a list of person in one of the videos
        
        sys.stdout = open(os.devnull, 'w')
        det_model = init_detector(args.det_config, args.det_ckpt, device='cuda')
        sys.stdout = sys.__stdout__

        assert det_model.CLASSES[0] == 'person', 'A detector trained on COCO is required'

        with open(json_path, "rb") as f:
            image_metas = json.load(f)
        video_level_info = {
            "keypoints": [], 
            "file_id": video_name + ".json",
            "seq_name": video_name, 
            "tot_frames": len(image_metas["images"]), 
            "track_id": -1, 
            "vid_id": video_name, 
        }
        for image_meta in image_metas["images"]:
            frame_file_name = image_meta["file_name"]  # path from dataset root
            frame_id = int(osp.basename(frame_file_name).split(".")[0])
            frame_level_info = {
                "frame_idx": frame_id,  # the frame index if indexed from 1 as in the naming of the frames
                "image_location": frame_file_name, 
                "img": f"{video_name}_{frame_id:03}"  # this video name is action name, but frame id is by event
            }
            img = extract_image(osp.join(self.img_dir, frame_file_name))
            assert img is not None, print(f"Cannot find {frame_file_name}")
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
                self.anno.append(person_level_info)

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, ind):
        images, imagesf, warps = apply_augmentation_test(self.anno[ind], 
                                                         self.img_dir,
                                                         output_size=self.output_size)

        kpts = torch.from_numpy(np.array(self.anno[ind]['keypoints'])).float()
        bbx = torch.from_numpy(np.array(self.anno[ind]['bbox']))
        area = bbx[2] * bbx[3]

        meta = {'imgID': self.anno[ind]['img'], 
                'file_name': self.anno[ind]['image_location'],
                'warps': warps, 
                'bbox': bbx, 
                'seq_name': self.anno[ind]['seq_name'],
                'area': area}

        if self.keep_track_id:
            meta['track_id'] = self.anno[ind]['track_id']

        return {'images': images, 'imagesf': imagesf, 'meta': meta}

def estimate_poses(args, 
                   val_loader, 
                   pt_prefix, 
                   output_size, 
                   checkpoint_path):

    poseNet = bninception(num_stages=args.num_stages, 
                          out_ch=51, 
                          pretrained=False)

    poseNet = poseNet.cuda()
    checkpoint = torch.load(checkpoint_path)
    pretrained_dict = checkpoint['state_dict']
    dict_keys = list(pretrained_dict.keys())

    if 'module.' in dict_keys[0]:
        model = DataParallel(poseNet)
    else:
        model = poseNet
        
    model.load_state_dict(pretrained_dict)
    model.eval()

    sequences = {}
    seq_imgs = {}  # one ele for one video, with keys ['is_labeled', 'nframes', 'image_id', 'id', 'vid_id', 'file_name', 'has_labeled_person', 'ignore_regions_y', 'ignore_regions_x']

    sequence_list = [val_loader.dataset.video_name+".json", ]

    for seq_name in sequence_list:
        with open(os.path.join(args.dataset_path, 
                               'posetrack_data/', 
                               pt_prefix, 
                               seq_name), 'r') as f:
            anno = json.load(f)

        seq = seq_name.split('.')[0]
        seq_imgs[seq] = anno['images']

    seq_track_ctr = {}
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(val_loader):

            images = sampled_batch['images']
            imagesf = sampled_batch['imagesf']
            inputs = images.cuda()
            inputsf = imagesf.cuda()

            output = model(inputs)
            output_det = torch.sigmoid(output[1][:, 0:17, :, :])
            output_det = output_det.data.cpu()

            outputf = model(inputsf)
            output_detf = torch.sigmoid(outputf[1][:, 0:17, :, :])
            output_detf = output_detf.data.cpu()

            sr = output[1][:, 17:51, :, :].data.cpu()
            N = output_det.shape[0]

            for n in range(N):
                size_y = args.output_size_y // 4
                size_x = args.output_size_x // 4

                prs = torch.zeros(17, size_y, size_x)
                output_detf[n] = output_detf[n][flipRef]
                for j in range(17):
                    prs[j] = output_det[n][j] + torch.from_numpy(cv2.flip(output_detf[n][j].numpy(), 1))

                keypoints, scores = get_preds_for_pose(prs, 
                                                       sampled_batch['meta']['warps'][n], sr[n],
                                                       output_size=output_size,
                                                       joint_scores=True)

                meta_data = sampled_batch['meta']
                sequence_name = meta_data['seq_name'][n]
                if sequence_name not in seq_track_ctr:
                    seq_track_ctr[sequence_name] = -1

                seq_track_ctr[sequence_name] += 1
                if sequence_name not in sequences:
                    sequences[sequence_name] = get_posetrack_eval_dummy()

                seq_data = sequences[meta_data['seq_name'][n]]

                im_id = meta_data['imgID'][n]#.item()

                track_id = seq_track_ctr[sequence_name] 
                if args.keep_track_id:
                    track_id = int(meta_data['track_id'][n])
                anno = {'image_id': im_id,  # f"{video_name}_{frame_id:03}"  # this video name is ac action name, but frame id is by event
                        'keypoints': keypoints, 
                        'scores': scores, 
                        'track_id': track_id,
                        'bbox': meta_data['bbox'][n][:-1].tolist()}

                seq_data['annotations'].append(anno)

    os.makedirs(args.save_path, exist_ok=True)

    for seq_name in seq_imgs.keys():
        if seq_name not in sequences.keys():
            sequences[seq_name] = get_posetrack_eval_dummy()

        images = seq_imgs[seq_name]
        sequence_anno = sequences[seq_name]

        sequence_anno['images'] = images

        with open(os.path.join(args.save_path, '{}.json'.format(seq_name)), 'w') as f:
            json.dump(sequences[seq_name], f)


def generate_parameters():
    parser = argparse.ArgumentParser()

    # from mmlab
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    # parser.add_argument('--annotation_file_path', type=str, required=True)  # anno generation is merged in to this file now
    parser.add_argument('--checkpoint_path', type=str, required=True)
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

    # from Corrtrack
    parser.add_argument('--prefix', type=str, default='val')
    parser.add_argument('--joint_threshold', type=float, default=0.0)
    parser.add_argument('--output_size_x', type=int, default=288)
    parser.add_argument('--output_size_y', type=int, default=384)
    parser.add_argument('--num_stages', type=int, default=2)
    parser.add_argument('--keep_track_id', action='store_true')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world', type=int, required=True)
    args = parser.parse_args()

    return args


def main():
    args = generate_parameters()
    cudnn.benchmark = True

    # video is the path to a json file
    json_paths = sorted(list(os.listdir(args.result_path)))[args.rank::args.world]

    for json_path in tqdm(json_paths):
        if os.path.exists(os.path.join(args.save_path, json_path)):
            continue  # avoid redoing

        json_path = os.path.join(args.result_path, json_path)
        dataset = PoseTrack(args, json_path, img_dir=args.dataset_path)
        
        val_loader = DataLoader(dataset, 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=args.num_workers)

        estimate_poses(args, 
                    val_loader, 
                    pt_prefix=args.prefix, 
                    output_size=[args.output_size_x, args.output_size_y],
                    checkpoint_path=args.checkpoint_path)

if __name__ == '__main__':
    main()
