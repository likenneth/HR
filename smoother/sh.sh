# # for X in {0..9}; do bash sh.sh ; sleep 4h; done


# Step 1: take CorrTrack output and infer without smoothing, btw take care of some bb issue
for X in {0..127}
do
printf -v pdd "%03d" $X
if squeue -u keli22 | grep infer${pdd}
then
echo infer${pdd} has not been killed
else
srun --job-name=infer${pdd} --cpus-per-task=10 --gpus-per-node=1 --nodes=1 --ntasks=1 --time=3-00:00:00 --partition=learnfair \
python wo_smooth.py --exp coco20 --save_path COCO20_wo_smooth \
--rank $X \
--world 128 \
FINEGYM.PSEUDO_LABEL '/private/home/keli22/HR/corrtrack/baselines/outputs/tracking_baselines/corrtrack_finegym/pose_3_stage_corr_tracking/jt_thres_0.1_duplicate_ratio_0.6_oks_0.2_corr_threshold_0.3_win_len_2_min_keypoints_2_min_track_len_3_break_tracks_True_pp_joint_threshold_0.3' \
FINEGYM.KPT_CONF_THRES 0.0 GPUS '(0,)' &
fi
# break
done

# python avg_smooth2.py --exp coco20 --save_path debug \
# --rank 0 --world 128 --window 5 \
# FINEGYM.KPT_CONF_THRES 0.0 GPUS '(0,)'