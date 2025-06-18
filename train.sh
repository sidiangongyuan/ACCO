#!/usr/bin/env bash

# BUG for DDP
export PYTHONPATH=$PYTHONPATH:/home/ACCO
echo "Starting script"
nohup python -u ./opencood/tools/train.py \
--hypes_yaml "/home/yangk/ACCO/opencood/hypes_yaml/opv2v/camera/opv2v_sparse4D_resnet50_multiloss_w_dir_depth_pointEncoderV6.yaml" \
--cuda_device 1 > "xxx.out" 2>&1 &