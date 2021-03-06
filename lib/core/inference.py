# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from utils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    singular = batch_heatmaps.ndim == 3  # disable batch processing
    if singular:
        batch_heatmaps = batch_heatmaps[None, ]
        center = center[None, ]
        scale = scale[None, ]
    # batch_heatmaps, [B, J, H=96, W=72], np.array
    # center, [B, 2], np.array, in original image scalec, returned by the _xywh2cs() in finegym.py
    # scale, [B, 2], np.array, not necessary 0~1, returned by the _xywh2cs() in finegym.py
    coords, maxvals = get_max_preds(batch_heatmaps)  
    # coords, [B, #J, 2], int coordinates in the heatmap resolution, [..., 0] range from 0 to W - 1, [..., 1] ranges from 0 to H - 1
    # maxvals, [B, #J, 1], the value < 1

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:  # move the integer coordinates produced by max() by 0.25 relying neighboring heatmap values
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px] - hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    # preds, [B, #J, 2], float coordinates at original resolution
    if singular:
        preds = preds[0]
        maxvals = maxvals[0]
    return preds, maxvals

def get_ori_coords(config, batch_heatmaps, center, scale):
    # batch_heatmaps, [B, J, H=96, W=72], np.array
    # center, [B, 2], np.array, in original image scalec, returned by the _xywh2cs() in finegym.py
    # scale, [B, 2], np.array, not necessary 0~1, returned by the _xywh2cs() in finegym.py
    singular = batch_heatmaps.ndim == 3  # disable batch processing

    if singular:
        batch_heatmaps = batch_heatmaps[None, ]
        center = center[None, ]
        scale = scale[None, ]
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    nx, ny = np.meshgrid(np.arange(heatmap_height), np.arange(heatmap_width), indexing="ij")  # both [H, W], first < H, second < W
    coords = np.stack([ny.flatten(), nx.flatten()], axis=-1)  # [HW, 2]
    maxvals = batch_heatmaps.reshape(batch_heatmaps.shape[0], batch_heatmaps.shape[1], heatmap_height*heatmap_width) # [B, J, HW]

    preds = []  # a list of [HW, 2]
    # Transform back
    for i in range(batch_heatmaps.shape[0]):
        preds.append(transform_preds(coords, center[i], scale[i], [heatmap_width, heatmap_height]))

    tbr = np.stack(preds, axis=0)
    # tbr, [B, HW, 2], float coordinates at original resolution
    # maxvals, [B, J, HW]
    if singular:
        tbr = tbr[0]
        maxvals = maxvals[0]

    return tbr, maxvals
