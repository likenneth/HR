Alternative ways to install the environment:
```
conda create -n andoer python==3.8 -y
conda activate andoer
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch -y
conda install -c anaconda h5py==3.6.0 -y
conda install conda==4.11.0 -y 

pip install opencv-python

pip install cython==0.29.27 munkres==1.1.4 seaborn==0.11.2 networkx==2.6.3
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install scikit-learn==1.0.2 scipy==1.8.0 matplotlib==3.5.1
pip install tensorboard==2.8.0
pip install sacred gdown==3.12.2
Install https://github.com/KaiyangZhou/deep-person-reid from source
pip install kornia==0.1.4
```

<!-- conda install -c conda-forge opencv==4.5.5 -y  -->