#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:/home/xxx
echo "Starting script"
nohup python -u ./opencood/tools/inference.py \
--model_dir "/result/cosparse_logs/opv2v_sparse4D_resnet50_multiloss_w_dir_depth_V6_2024_04_21_13_25_49" \
--cuda_device 1 > "xxx.out" 2>&1 &