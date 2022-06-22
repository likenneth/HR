This folder is forked from https://github.com/anDoer/PoseTrack21. Use `mm` environment for this sub-folder. See `openmmlab` for installation method. 

```
cd baselines
bash scripts/posetracking/corrtrack_wo_docker.sh 0
cd ..
```

For evaluation on PoseTrack
```
python eval/posetrack21/scripts/run_pose_estimation.py --GT_FOLDER ~/datasets/PoseTrack21/posetrack_data/val --TRACKERS_FOLDER baselines/outputs/tracking_baselines/corrtrack_baseline/pose_3_stage_corr_tracking/jt_thres_0.1_duplicate_ratio_0.6_oks_0.2_corr_threshold_0.3_win_len_2_min_keypoints_2_min_track_len_3_break_tracks_True_pp_joint_threshold_0.3/sequences
```
<!-- python eval/posetrack21/scripts/run_mot.py --GT_FOLDER ~/datasets/PoseTrack21/posetrack_mot/mot/val --TRACKERS_FOLDER baselines/outputs/tracking_baselines/corrtrack_baseline/pose_3_stage_corr_tracking/jt_thres_0.1_duplicate_ratio_0.6_oks_0.2_corr_threshold_0.3_win_len_2_min_keypoints_2_min_track_len_3_break_tracks_True_pp_joint_threshold_0.3/sequences -->
# PoseTrack21
Current research evaluates person search, multi-object tracking and multi-person pose estimation as separate tasks and on different datasets although these tasks are very akin to each other and comprise similar sub-tasks, e.g. person detection or appearance-based association of detected persons. Consequently, approaches on these respective tasks are eligible to complement each other. Therefore, we introduce PoseTrack21, a large-scale dataset for person search, multi-object tracking and multi-person pose tracking in real-world scenarios with a high diversity of poses. The dataset provides rich annotations like human pose annotations including annotations of joint occlusions, bounding box annotations even for small persons, and person-ids within and across video sequences. The dataset allows to evaluate multi-object tracking and multi-person pose tracking jointly with person re-identification or exploit structural knowledge of human poses to improve person search and tracking, particularly in the context of severe occlusions. With PoseTrack21, we want to encourage researchers to work on joint approaches that perform reasonably well on all three tasks.        

## How to get the dataset?
In order to obtain the entire dataset, please fill out [this document](https://docs.google.com/document/d/1unxTYm2nVH1Qr7iYtgFzkzPbu042c1MLyZUP8Nb7-Fs/edit?usp=sharing) and send it to **posetrack21[at]googlegroups[dot]com**.

**NOTE: Due to technical issues, we might not have received your request. In case you did not get your access token by now, please contact us again.**

Afterwards, please run the following command with you access token:
```
python3 download_dataset.py --save_path /target/root/path/of/the/dataset --token [your token]
```

## Structure of the dataset 
The dataset is organized as follows: 

    .
    ├── images                              # contains all images  
        ├── train
        ├── val
    ├── posetrack_data                      # contains annotations for pose reid tracking
        ├── train
            ├── 000001_bonn_train.json
            ├── ...
        ├── val
            ├── ...
    ├── posetrack_mot                       # contains annotations for multi-object tracking 
        ├── mot
            ├── train
                ├── 000001_bonn_train
                    ├── image_info.json
                    ├── gt
                        ├── gt.txt          # ground truth annotations in mot format
                        ├── gt_kpts.txt     # ground truth poses for each frame
                ├── ...
            ├── val
    ├── posetrack_person_search             # person search annotations
        ├── query.json
        ├── train.json
        ├── val.json

A detailed description of the respective dataset formats can be found [here](doc/dataset_structure.md).

## Usage 
Instructions on the evaluation of the respective tacks are provided [here](eval/README.md).

## Citation 
```
@inproceedings{doering22,
  title={PoseTrack21: A Dataset for Person Search, Multi-Object Tracking and Multi-Person Pose Tracking},
  author={Andreas Doering and Di Chen and Shanshan Zhang and Bernt Schiele and Juergen Gall},
  booktitle={CVPR},
  year={2022}
}
```
