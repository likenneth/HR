# python wo_smooth.py --exp posetrack --save_path PT21_wo_smooth \
# --rank 0 --world 1 --pt \
# POSETRACK.PSEUDO_LABEL '/private/home/keli22/HR/corrtrack/baselines/outputs/tracking_baselines/corrtrack_swin/pose_3_stage_corr_tracking/jt_thres_0.1_duplicate_ratio_0.6_oks_0.2_corr_threshold_0.3_win_len_2_min_keypoints_2_min_track_len_3_break_tracks_True_pp_joint_threshold_0.3/sequences/' \
# POSETRACK.KPT_CONF_THRES 0.0 #GPUS '(0,1)'

# Step 1: take CorrTrack output and infer without smoothing, btw take care of some bb issue
for X in {0..127}
do
printf -v pdd "%03d" $X
if squeue -u keli22 | grep onfer${pdd}
then
echo onfer${pdd} has not been killed
else
srun --job-name=onfer${pdd} --cpus-per-task=10 --gpus-per-node=1 --nodes=1 --ntasks=1 --time=3-00:00:00 --partition=learnfair \
python avg_smooth.py --exp posetrack --save_path PT21_avg_smooth_w3 \
--rank $X \
--world 128 \
--window 3 \
--pt \
POSETRACK.PSEUDO_LABEL '/private/home/keli22/HR/corrtrack/baselines/outputs/tracking_baselines/corrtrack_swin/pose_3_stage_corr_tracking/jt_thres_0.1_duplicate_ratio_0.6_oks_0.2_corr_threshold_0.3_win_len_2_min_keypoints_2_min_track_len_3_break_tracks_True_pp_joint_threshold_0.3/sequences/' \
POSETRACK.KPT_CONF_THRES 0.0 GPUS '(0,)' &
fi
# break
done

# rm -rf debug
# python avg_smooth.py --exp posetrack --save_path debug \
# --rank 2 --world 170 --window 5 --pt --vis 10 \
# POSETRACK.PSEUDO_LABEL '/private/home/keli22/HR/corrtrack/baselines/outputs/tracking_baselines/corrtrack_swin/pose_3_stage_corr_tracking/jt_thres_0.1_duplicate_ratio_0.6_oks_0.2_corr_threshold_0.3_win_len_2_min_keypoints_2_min_track_len_3_break_tracks_True_pp_joint_threshold_0.3/sequences/' \
# POSETRACK.KPT_CONF_THRES 0.0 GPUS '(0,1)'