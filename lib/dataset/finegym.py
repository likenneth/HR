# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import pickle
from tqdm import tqdm
from itertools import repeat

import json_tricks as json
import numpy as np
import multiprocessing

from dataset.JointsDataset import JointsDataset
from nms.nms import oks_nms
from nms.nms import soft_oks_nms


logger = logging.getLogger(__name__)
DEBUG=False

def worker(fgds, action):
    return fgds.load_annotation_for_action(action)

class FineGymDataset(JointsDataset):
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.name = "FineGym"
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.fps = 25

        self.num_joints = 17
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.parent_ids = None
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self.joints_weight = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))

        self.kpt_conf_thres = cfg.FINEGYM.KPT_CONF_THRES  # only use label with higher confidence
        self.bb_conf_thres = cfg.FINEGYM.BB_CONF_THRES  # only use label with higher confidence
        action_annotations = []
        for part_file in os.listdir(cfg.FINEGYM.PSEUDO_LABEL):
            with open(os.path.join(cfg.FINEGYM.PSEUDO_LABEL, part_file), "rb") as input_file:
                read = pickle.load(input_file)
                action_annotations.extend(read)
        # 'time_stamp', 'action_name', 'bb', 'bb_score', 'img_shape', 'original_shape', 'total_frames', 'num_person_raw', 'keypoint', 'keypoint_score'
        # 'keypoint': [max #person, #frame, 17, 2], use [0, 0] to fill persons not appearing

        num_proc = max(1, multiprocessing.cpu_count() - 1)  # use all processors
        db = []
        
        p = multiprocessing.Pool(num_proc)
        bar = tqdm(total=len(action_annotations) if not DEBUG else 100)
        for res in p.starmap(worker, zip(repeat(self), action_annotations[:len(action_annotations) if not DEBUG else 100])):
            db.extend(res)
            bar.update(1)
        p.close()
        p.join()
        self.db = db
        # if is_train and cfg.FINEGYM.SELECT_DATA:
        #     self.db = self.select_data(self.db)

        logger.info('=> load {} samples for {}'.format(len(self.db), self.name))

    def load_annotation_for_action(self, action):
        tbr = []
        s = round(action['time_stamp'][0] * self.fps)  # starting 0
        e = round(action['time_stamp'][1] * self.fps)  # starting 0
        youtube_name, _, event_s, event_e, _, action_s, action_e = action["action_name"].split("_")
        event_folder = os.path.join(self.root, "processed_frames", "_".join([youtube_name, "E", event_s, event_e]))
        num_person, num_frame = action["bb"].shape[:2]
        total_frames = len(list(os.listdir(event_folder)))
        e = min(total_frames-1, e)  # starting 0
        assert num_frame == e - s + 1, f"{num_frame} vs {e - s + 1} in {event_folder}"
        for frame_index, glob_fidx in enumerate(range(s, e+1)):
            for person_index in range(num_person):
                if action['bb_score'][person_index, frame_index] > self.bb_conf_thres and np.any(action["keypoint"][person_index, frame_index, -1] > self.kpt_conf_thres):
                    kpt = action["keypoint"][person_index, frame_index]  # [17, 2]
                    kpt_score = action["keypoint_score"][person_index, frame_index]  # [17]
                    joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                    joints_3d[:, :2] = kpt[:, :2]
                    joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
                    joints_3d_vis[:, :2] = (kpt_score[:, None] > self.kpt_conf_thres).astype(np.float)
                    center, scale = self._box2cs(action['bb'][person_index, frame_index, :])
                    image_path = os.path.join(event_folder, f"{glob_fidx+1:03}.jpg")
                    assert os.path.exists(image_path), f"Cannot fine {image_path}"
                    tbr.append({
                        'image': image_path,
                        'center': center,
                        'scale': scale,
                        'joints_3d': joints_3d,
                        'joints_3d_vis': joints_3d_vis,
                        'filename': '',
                        'imgnum': 0,
                    })
        return tbr

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale
