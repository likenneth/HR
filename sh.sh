# srun --job-name=fgswcthr --cpus-per-task=80 --gpus-per-node=8 --constraint volta32gb --nodes=1 --ntasks=1 --time=14-00:00:00 --partition=pixar \

# python tools/train.py --exp coco20 --cfg experiments/coco/hrnet/coco.yaml DATASET.PARTIAL 0.2 &
# python tools/train.py --exp coco --cfg experiments/coco/hrnet/coco.yaml DATASET.PARTIAL 1.0 &
# python tools/train.py --exp coco_fgswcthr --cfg experiments/coco/hrnet/coco_fgswcthr.yaml DATASET.PARTIAL 1.0 &
