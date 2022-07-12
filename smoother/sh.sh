# this should be executed after 100% coco training done

for X in {0..127}
do
printf -v pdd "%03d" $X
if squeue -u keli22 | grep infer${pdd}
then
echo infer${pdd} has not been killed
else
srun --job-name=infer${pdd} --cpus-per-task=20 --gpus-per-node=2 --nodes=1 --ntasks=1 --time=3-00:00:00 --partition=learnfair \
python avg_smooth.py --exp coco20 --save_path COCO20_avg_smooth \
--rank $X \
--world 128 \
FINEGYM.KPT_CONF_THRES 0.0 GPUS '(0,1)' &
fi
# break
done

# python avg_smooth.py --exp coco20 --save_path debug \
# --rank 0 --world 128 --window 5 \
# FINEGYM.KPT_CONF_THRES 0.0 GPUS '(0,1)'