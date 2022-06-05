Use `mm` environment for this sub-folder. This folder is used for extracting bounding boxes from FineGym as a degenerated case before getting CorrTrack code from Andreas. 

#### Installation of `mm`
+ Install MMDet and MMPose from official website 
+ `git clone https://github.com/kennymckormick/pyskl.git` and follow installation

#### Usage

```
cd openmmlab
CUDA_VISIBLE_DEVICES=1 python mmdet_finegym.py --rank 0 --world-size 128
```