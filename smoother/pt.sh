python wo_smooth.py --exp coco --save_path PT21_wo_smooth \
--rank 0 --world 1 --pt \
POSETRACK.PSEUDO_LABEL '/private/home/keli22/HR/corrtrack/baselines/outputs/tracking_baselines/corrtrack_swin/pose_3_stage_corr_tracking/jt_thres_0.1_duplicate_ratio_0.6_oks_0.2_corr_threshold_0.3_win_len_2_min_keypoints_2_min_track_len_3_break_tracks_True_pp_joint_threshold_0.3/sequences/val' \
POSETRACK.KPT_CONF_THRES 0.0 GPUS '(0,1)'

python avg_smooth.py --exp coco --save_path dPT21_avg_smoothebug_w5 \
--rank 0 --world 1 --window 5 --pt \
POSTRACK.PSEUDO_LABEL '/private/home/keli22/HR/corrtrack/baselines/outputs/tracking_baselines/corrtrack_swin/pose_3_stage_corr_tracking/jt_thres_0.1_duplicate_ratio_0.6_oks_0.2_corr_threshold_0.3_win_len_2_min_keypoints_2_min_track_len_3_break_tracks_True_pp_joint_threshold_0.3/sequences/val' \
POSETRACK.KPT_CONF_THRES 0.0 GPUS '(0,1)'
