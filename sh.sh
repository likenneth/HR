# python tools/train.py --exp coco20 --cfg experiments/coco/hrnet/coco.yaml DATASET.PARTIAL 0.2 &
# python tools/train.py --exp coco --cfg experiments/coco/hrnet/coco.yaml DATASET.PARTIAL 1.0 &
# srun --job-name=redo --mem 450G --gpus-per-node=8 --partition=pixar --constraint volta32gb --time=14-00:00:00 --cpus-per-task 80 --nodes=1 --ntasks=1 \ 
# python tools/train.py --exp coco_fgswcthr_redo --cfg experiments/coco/hrnet/coco_fgswcthr.yaml DATASET.PARTIAL 1.0 FINEGYM.PSEUDO_LABEL "smoother/COCO_wo_smooth" &
# python tools/train.py --exp coco20_fgswcthr --cfg experiments/coco/hrnet/coco_fgswcthr.yaml DATASET.PARTIAL 0.2 FINEGYM.PSEUDO_LABEL "smoother/COCO20_wo_smooth" &

# srun --job-name=71 --mem 450G --gpus-per-node=8 --partition=pixar --constraint volta32gb --time=14-00:00:00 --cpus-per-task 80 --nodes=1 --ntasks=1 \
# python tools/train.py --exp coco20_fgswcthr_kptthres71 --cfg experiments/coco/hrnet/coco_fgswcthr.yaml DATASET.PARTIAL 0.2 FINEGYM.KPT_CONF_THRES 0.71 &

# srun --job-name=wo0bb --mem 450G --gpus-per-node=8 --partition=pixar --constraint volta32gb --time=14-00:00:00 --cpus-per-task 80 --nodes=1 --ntasks=1 \
# python tools/train.py --exp coco20_fgswcthr_wo0bb --cfg experiments/coco/hrnet/coco_fgswcthr.yaml FINEGYM.PCT 32 FINEGYM.KPT_CONF_THRES 0.9 &

# srun --job-name=wo1000bb --mem 450G --gpus-per-node=8 --partition=pixar --constraint volta32gb --time=14-00:00:00 --cpus-per-task 80 --nodes=1 --ntasks=1 \
# python tools/train.py --exp coco20_fgswcthr_wo1000bb --cfg experiments/coco/hrnet/coco_fgswcthr.yaml FINEGYM.PCT 32 FINEGYM.KPT_CONF_THRES 0.9 FINEGYM.BB_SIZE_THRES 1000 &

# python tools/test.py --exp coco --cfg experiments/coco/hrnet/coco_scale2.yaml DATASET.PARTIAL 1.0 GPUS '(0,1)'
python tools/finetune_pt.py --exp posetrack --cfg experiments/coco/hrnet/coco_ptswcthr.yaml POSETRACK.PSEUDO_LABEL 'data/posetrack/posetrack_data' \
TRAIN.LR 0.0001
# GPUS '(0,1)' POSETRACK.PARTIAL 0.02