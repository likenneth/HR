# cd ~/HR/corrtrack
# conda activate andoer
# python remove_head_tb.py

import os
import json
import shutil
from tqdm import tqdm

exp_name = "corrtrack_swin"  # OR corrrtack_baseline
input_dir = f"/private/home/keli22/HR/corrtrack/baselines/outputs/tracking_baselines/{exp_name}/pose_3_stage_corr_tracking/jt_thres_0.1_duplicate_ratio_0.6_oks_0.2_corr_threshold_0.3_win_len_2_min_keypoints_2_min_track_len_3_break_tracks_True_pp_joint_threshold_0.3/sequences/val"
output_dir = f"/private/home/keli22/HR/corrtrack/baselines/outputs/tracking_baselines/{exp_name}/head_tb_removed"

if os.path.exists(output_dir) and os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
    print(f"Removing existing {output_dir}")
os.makedirs(output_dir, exist_ok=True)

for json_name in tqdm(os.listdir(input_dir)):
    if not json_name.endswith(".json"):
        continue
    try:
        with open(os.path.join(input_dir, json_name), "rb") as f:
            read = json.load(f)
    except:
        print(json_name)
    for anno in read["annotations"]:
        for i in range(3, 15):
            anno["keypoints"][i] = 0.
    with open(os.path.join(output_dir, json_name), "w") as f:
        json.dump(read, f)

tbt = f"python eval/posetrack21/scripts/run_pose_estimation.py --GT_FOLDER ~/datasets/PoseTrack21/posetrack_data/val --TRACKERS_FOLDER {output_dir}"
os.system(tbt)