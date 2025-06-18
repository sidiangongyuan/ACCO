#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:/home/xxx
echo "Starting script"
python -u ./opencood/tools/train.py \ 
--hypes_yaml ./xxx.yaml \
--cuda_device 2 > xxx.out 2>&1 &
