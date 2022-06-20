# for X in 0.7 0.8 0.9
# do
# for Y in 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# do

# echo findbbkptthres_bb_${X}_kpt_${Y}
# srun --job-name=t --cpus-per-task=80 --gpus-per-node=8 --constraint volta32gb --nodes=1 --ntasks=1 --time=3-00:00:00 --partition=learnfair python tools/train.py --exp findbbkptthres_bb_${X}_kpt_${Y} --cfg experiments/coco/hrnet/coco_fgswhr.yaml FINEGYM.BB_CONF_THRES $X FINEGYM.KPT_CONF_THRES $Y GPUS '(0,1,2,3,4,5,6,7)' &

# done
# done

# srun --job-name=t --cpus-per-task=80 --gpus-per-node=8 --constraint volta32gb --nodes=1 --ntasks=1 --time=3-00:00:00 --partition=pixar python tools/train.py --exp findbbkptthres_baseline --cfg experiments/coco/hrnet/baseline.yaml GPUS '(0,1,2,3,4,5,6,7)' &