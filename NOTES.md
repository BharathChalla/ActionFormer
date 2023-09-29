# Download the data


## Conda Environment Setup
conda create -n actionformer python=3.9
conda activate actionformer
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tensorboard pyyaml pandas h5py joblib

cd libs/utils
python setup.py install
cd ../..

### Install PyTorchVideo
pip install einops pytorchvideo timm

### Install the numba library at the end to force to use the 1.23.0 version
pip install numpy==1.23.0

# Example
## THUMOS
### Train the Model
python ./train.py ./configs/thumos_i3d.yaml --output reproduce 2>&1 > logs/train.out
### Evaluate the Model
python ./eval.py ./configs/thumos_i3d.yaml ./ckpt/thumos_i3d_reproduce > logs/eval.out

## Error Dataset
### Train the Model
python train.py configs/error_dataset_omnivore.yaml --output reproduce 2>&1 > logs/train.out

### Evaluate the Model
python eval.py configs/error_dataset_omnivore.yaml ckpt/error_dataset_omnivore_reproduce > logs/eval.out
