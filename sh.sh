# python tools/train.py --exp coco20 --cfg experiments/coco/hrnet/coco.yaml DATASET.PARTIAL 0.2 &
# python tools/train.py --exp coco --cfg experiments/coco/hrnet/coco.yaml DATASET.PARTIAL 1.0 &
# srun --job-name=redo --mem 450G --gpus-per-node=8 --partition=pixar --constraint volta32gb --time=14-00:00:00 --cpus-per-task 80 --nodes=1 --ntasks=1 \ 
# python tools/train.py --exp coco_fgswcthr_redo --cfg experiments/coco/hrnet/coco_fgswcthr.yaml DATASET.PARTIAL 1.0 FINEGYM.PSEUDO_LABEL "smoother/COCO_wo_smooth" &
# python tools/train.py --exp coco20_fgswcthr --cfg experiments/coco/hrnet/coco_fgswcthr.yaml DATASET.PARTIAL 0.2 FINEGYM.PSEUDO_LABEL "smoother/COCO20_wo_smooth" &

srun --job-name=t2 --mem 450G --gpus-per-node=8 --partition=pixar --constraint volta32gb --time=14-00:00:00 --cpus-per-task 80 --nodes=1 --ntasks=1 \
python tools/train.py --exp coco20_fgswcthr_pct32 --cfg experiments/coco/hrnet/coco_fgswcthr.yaml DATASET.PARTIAL 0.2 FINEGYM.PSEUDO_LABEL "smoother/COCO20_wo_smooth" FINEGYM.PCT 32 &
