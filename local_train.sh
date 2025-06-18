#!/usr/bin/env bash

# BUG for DDP
export PYTHONPATH=$PYTHONPATH:/home/ACCO
export CONFIG_FILE="/home/yangk/ACCO/opencood/hypes_yaml/opv2v/camera/opv2v_sparse4D_resnet50_multiloss_w_dir_depth_pointEncoderV6.yaml"
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env /home/yangk/ACCO/opencood/tools/train.py  \
--hypes_yaml ${CONFIG_FILE} > xxx.out 2>&1 &