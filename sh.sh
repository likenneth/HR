srun --job-name=fgswcthr --cpus-per-task=80 --gpus-per-node=8 --constraint volta32gb --nodes=1 --ntasks=1 --time=14-00:00:00 --partition=pixar \
python tools/train.py --exp coco_fgswcthr --cfg experiments/coco/hrnet/coco_fgswhr.yaml DATASET.PARTIAL 1.0 &
# python tools/train.py --exp debug --cfg experiments/coco/hrnet/coco_fgswhr.yaml GPUS '(0,1)'