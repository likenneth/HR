#!/bin/bash

# $PWD should be ~/PoseTrack21Release/baselines
# bash scripts/posetracking/corrtrack_finegym.sh 0,1,2,3,4,5,6,7
# srun --job-name=corr_test --cpus-per-task=32 --gpus-per-node=1 --nodes=1 --ntasks=1 --time=2-00:00:00 --partition=learnfair bash scripts/posetracking/corrtrack_finegym.sh 0 &

USERNAME=keli22

# host paths
SRC_DIR="$PWD"

EXPERIMENT_FOLDER_NAME=corrtrack_diving
EXP=

# container paths
MODEL_DIR="$PWD/data/models/"
DATA_DIR="$PWD/data/detections/"

POSE_ESTIMATION_MODEL_PATH="$MODEL_DIR/pose_estimation_model_3_stage_lr_1e-5_wo_vis.pth"
DATASET_PATH="/private/home/$USERNAME/datasets/Diving48/"
CORR_MODEL_PATH="$MODEL_DIR/corrtrack_model.pth"

NUM_POSE_STAGES=3

INFERENCE_FOLDER_PATH="$SRC_DIR/outputs/tracking_baselines/"

POSE_FOLDER=pose_3_stage
POSE_NMS_FOLDER=pose_3_stage_nms
WARPED_POSE_FOLDER=pose_3_stage_warped
REFINED_POSES_FOLDER=pose_3_stage_warped_and_refined
REFINED_POSE_NMS_FOLDER=pose_3_stage_refined_nms
CORR_TRACKING_FOLDER=pose_3_stage_corr_tracking

# 1) pose estimation
# run single-person pose estimation on a given person bbox file (does the detection step contain NMS with IOU?)
echo "RUN POSE ESTIMATION"
for X in {0..127}
do
printf -v pdd "%03d" $X
srun --job-name=infer${pdd} --cpus-per-task=10 --gpus-per-node=1 --constraint volta32gb --nodes=1 --ntasks=1 --time=3-00:00:00 --partition=learnfair \
python corrtrack/inference/pose_estimation.py \
--result_path=${DATASET_PATH}/posetrack_data/ \
--save_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${POSE_FOLDER}/ \
--dataset_path=${DATASET_PATH} \
--prefix=$EXP \
--checkpoint_path=${POSE_ESTIMATION_MODEL_PATH} \
--joint_threshold=0.0 \
--output_size_x=288 \
--output_size_y=384 \
--num_stages=${NUM_POSE_STAGES} \
--batch_size=128 \
--num_workers=20 \
--rank $X \
--world 128 &
# break
done

# # 2) nms, CPU-only
# # remove redundent persons by NMS with OKS
# echo "RUN NMS"
# PYTHONPATH=$PWD \
# python corrtrack/tools/pose_nms.py \
# --result_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${POSE_FOLDER}/ \
# --save_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${POSE_NMS_FOLDER}/ \
# --joint_threshold=0.05 \
# --oks_threshold=0.7

# # 3) pose warping, multi-GPU has some bug when batch size <= number of GPU
# # propagate all the poses from f-1 to f, using Siamese network, and use PoseNet to reestimate the pose
# echo "RUN WARPING"
# for X in {0..127}
# do
# printf -v pdd "%03d" $X
# if squeue -u keli22 | grep infer${pdd}
# then
# echo infer${pdd} have not been killed
# else
# srun --job-name=infer${pdd} --cpus-per-task=10 --gpus-per-node=1 --constraint volta32gb --nodes=1 --ntasks=1 --time=3-00:00:00 --partition=learnfair \
# python corrtrack/inference/warp_poses.py \
# --corr_ckpt_path=${CORR_MODEL_PATH} \
# --pose_ckpt_path=${POSE_ESTIMATION_MODEL_PATH} \
# --num_stages=${NUM_POSE_STAGES} \
# --sequences_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${POSE_NMS_FOLDER}/jt_0.05_oks_0.7/ \
# --save_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${WARPED_POSE_FOLDER} \
# --dataset_path=${DATASET_PATH} \
# --oks_threshold=0.8 \
# --corr_threshold=0.1 \
# --joint_threshold=0.1 \
# --rank $X \
# --world 128 &
# fi
# # break
# done

# # 4) recovering missed detections, CPU-only
# # for all new pose, if OKS being not too high to any old poses, add it
# echo "Run pose recovery"
# PYTHONPATH=$PWD \
# python corrtrack/inference/recover_missed_detections.py \
# --sequences_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${WARPED_POSE_FOLDER}/val_set_jt_0.1_with_corr_0.1_at_oks_0.8/ \
# --save_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${REFINED_POSES_FOLDER} \
# --oks=0.6 \
# --joint_th=0.1

# # 5) NMS again, CPU-only
# # NMS again for safety
# echo "RUN NMS AGAIN"
# PYTHONPATH=$PWD \
# python corrtrack/tools/pose_nms.py \
# --result_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${REFINED_POSES_FOLDER}/recover_missed_detections_jt_th_0.1_oks_0.6/ \
# --save_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${REFINED_POSE_NMS_FOLDER}/ \
# --joint_threshold=0.05 \
# --oks_threshold=0.7

# # 6) Corr Tracking, multi-GPU has some bug when batch size <= number of GPU
# # here is only the assosiation step. it uses the Siamese network to have better OKS
# echo "RUN TRACKING"
# for X in {0..127}
# do
# printf -v pdd "%03d" $X
# if squeue -u keli22 | grep infer${pdd}
# then
# echo infer${pdd} have not been killed
# else
# srun --job-name=infer${pdd} --cpus-per-task=10 --gpus-per-node=1 --constraint volta32gb --nodes=1 --ntasks=1 --time=3-00:00:00 --partition=learnfair \
# python corrtrack/inference/run_corrtrack.py \
# --save_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${CORR_TRACKING_FOLDER}/ \
# --sequences_path=${INFERENCE_FOLDER_PATH}/${EXPERIMENT_FOLDER_NAME}/${REFINED_POSE_NMS_FOLDER}/jt_0.05_oks_0.7/ \
# --dataset_path=${DATASET_PATH} \
# --joint_threshold=0.1 \
# --oks_threshold=0.2 \
# --corr_threshold=0.3 \
# --min_keypoints=2 \
# --min_track_len=3 \
# --duplicate_ratio=0.6 \
# --post_process_joint_threshold=0.3 \
# --break_tracks \
# --ckpt_path=${CORR_MODEL_PATH} \
# --rank $X \
# --world 128 &
# fi
# # break
# done
