for X in 0.7 0.8 0.9
do
for Y in 0.4 0.5 0.6 0.7 0.8
do

echo $X $Y
srun --job-name=t --cpus-per-task=80 --gpus-per-node=8 --nodes=1 --ntasks=1 --time=3-00:00:00 --partition=learnfair /private/home/keli22/.conda/envs/hr/bin/python tools/train.py --cfg experiments/coco/hrnet/coco_fgswhr.yaml FINEGYM.BB_CONF_THRES $X FINEGYM.KPT_CONF_THRES $Y GPUS '(0,1,2,3,4,5,6,7)' &

done
done