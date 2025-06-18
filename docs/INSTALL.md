
## Installation

Please refer to [data introduction](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html)
and [installation](https://opencood.readthedocs.io/en/latest/md_files/installation.html) guide to prepare data and install OpenCOOD.

### Install spconv v1.2.1
#### 1. Install conda
Please refer to https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

#### 2. Set up the conda environment and the dependancies
```
# conda create --name ACCO python=3.8.17 cmake=3.22.1 cudatoolkit=11.2 cudatoolkit-dev=11.2
conda create --name ACCO python=3.8.17
conda activate ACCO
conda install cudnn -c conda-forge
conda install boost

# install pytorch
# pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# option: if there is error or speed issues in install cudatoolkit
# could instead specify the PATH, CUDA_HOME, and LD_LIBRARY_PATH, using current cuda
# write it to ~/.bashrc, for example use Vim
vim ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda/bin:$CUDA_HOME
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# add head file search directories 
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/miniconda3/envs/ACCO/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/miniconda3/envs/ACCO/include
# add shared library searching directories
export LIBRARY_PATH=$LIBRARY_PATH:/miniconda3/envs/ACCO/lib
# add runtime library searching directories
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/miniconda3/envs/ACCO/lib

# go out of Vim and activate it in current shell
source ~/.bashrc

conda activate ACCO
```

#### 3. install spconv
```

https://github.com/traveller59/spconv

pip install spconv-cu116
# check if is successfully installed
python 
import spconv
```

### Install OpenCOOD
Please refer to [installation](https://opencood.readthedocs.io/en/latest/md_files/installation.html) guide to prepare.
```
# install requirements
# pip install -r requirements.txt   # Please install the mm series libraries separately.
python setup.py develop

python opencood/utils/setup.py build_ext --inplace
python opencood/pcdet_utils/setup.py build_ext --inplace
# if there is cuda version issue; ssh db92 -p 58122 and customize the cuda home
CUDA_HOME=/usr/local/cuda-11.1/ python opencood/pcdet_utils/setup.py build_ext --inplace
```


### mmdetection install

```
mmcv == 2.0.0

mmdet == 3.1.0

mmdet3d == 1.2.0

mmengine == 0.8.4
```
All of these libraries can be directly installed using pip.


### Compile the deformable_aggregation CUDA op

```
cd projects/mmdet3d_plugin/ops

python3 setup.py develop

cd ../../../
```

### Download pre-trained weights (Optional)

```
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```