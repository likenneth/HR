# IPT_DIR="/private/home/keli22/HR/corrtrack/baselines/outputs/tracking_baselines/corrtrack_baseline/pose_3_stage_corr_tracking/jt_thres_0.1_duplicate_ratio_0.6_oks_0.2_corr_threshold_0.3_win_len_2_min_keypoints_2_min_track_len_3_break_tracks_True_pp_joint_threshold_0.3/sequences/"
# OPT_DIR="/private/home/keli22/HR/openmmlab/PT_RCNN_CorrTrack_HRNet/val_allbb/"

# for X in $IPT_DIR/*.json
# do
# echo $X
# python infer_pt21val_rcnn_corrtrack_hrnet.py --json_file $X --opt_root $OPT_DIR
# break
# done

# for X in {0..127}
# do
# srun --job-name=infer_finegym_swin_hrnet --cpus-per-task=8 --gpus-per-node=1 --nodes=1 --ntasks=1 --time=2-00:00:00 --partition=learnfair python infer_finegym_swin_hrnet.py --world-size 128 --rank $X &
# done

# CUDA_VISIBLE_DEVICES=1 python infer_pt21x_swin.py --trainval val & 
# CUDA_VISIBLE_DEVICES=0 python infer_pt21x_swin.py --trainval train &
# CUDA_VISIBLE_DEVICES=1 python infer_pt21x_swin.py --trainval test &

# IPT_DIR="/private/home/keli22/datasets/FineGym/processed_frames"

# for X in $IPT_DIR/*/
# do
# srun --job-name=infer --cpus-per-task=10 --gpus-per-node=1 --nodes=1 --ntasks=1 --time=1-00:00:00 --partition=learnfair python infer_finegym_swin.py --video $X &
# done

# # person detection and pose estimation with HRNet trained on 20% of COCO, full FineGym events
# for X in {0..255}
# do
# printf -v pdd "%03d" $X
# if squeue -u keli22 | grep infer${pdd}
# then
# echo infer${pdd} has not been killed
# else
# srun --job-name=infer${pdd} --cpus-per-task=10 --gpus-per-node=1 --constraint volta32gb --nodes=1 --ntasks=1 --time=3-00:00:00 --partition=learnfair \
# python infer_finegym_swin_hrnet.py \
# --rank $X \
# --world-size 256 & 
# fi
# # break
# done

IPT_DIR="/private/home/keli22/HR/corrtrack/baselines/outputs/tracking_baselines/corrtrack_finegym/pose_3_stage_corr_tracking/jt_thres_0.1_duplicate_ratio_0.6_oks_0.2_corr_threshold_0.3_win_len_2_min_keypoints_2_min_track_len_3_break_tracks_True_pp_joint_threshold_0.3"
OPT_DIR="/private/home/keli22/HR/openmmlab/FullFineGym_Swin_CorrTrack_HRNet/"

for X in {0..127}
do
printf -v pdd "%03d" $X
if squeue -u keli22 | grep infer${pdd}
then
echo infer${pdd} has not been killed
else
srun --job-name=infer${pdd} --cpus-per-task=10 --gpus-per-node=1 --nodes=1 --ntasks=1 --time=1-00:00:00 --partition=learnfair \
python infer_finegym_swin_corrtrack_hrnet.py --result_path $IPT_DIR --save_path $OPT_DIR \
--rank $X \
--world 128 &
fi
# break
done