Y=8
for X in {0..7}
do
echo $Y
CUDA_VISIBLE_DEVICES=$X python mmdet_finegym.py --rank $Y --world-size 32 &
let Y++

echo $Y
CUDA_VISIBLE_DEVICES=$X python mmdet_finegym.py --rank $Y --world-size 32 &
let Y++

echo $Y
CUDA_VISIBLE_DEVICES=$X python mmdet_finegym.py --rank $Y --world-size 32 &
let Y++
done