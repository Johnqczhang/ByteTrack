#!/usr/bin/env zsh

ANNO_PATH="datasets/mot/annotations"
HT21_PATH="datasets/mot/ht21"
# HT21_PATH_TRAIN="datasets/mot/ht21/HT21-train/"
# HT21_PATH_TEST="datasets/mot/ht21/HT21-test/"

main="python tools/train.py"
# config file
cfg="-f exps/ht21/yolox_x_ht21.py"
# default of ByteTrack: 8 GPUs, batch-size: 48
batch_size="-d 2 -b 192"
# mix precision training
fp16="--fp16"
# occupy GPU memory first for training
gpu_mem="-o"
# checkpoint file
ckpt="-c pretrained/yolox_x.pth"

cmd="$main $cfg $batch_size $fp16 $gpu_mem $ckpt"
echo $cmd




# func=$1
# if [[ $func == "predet" ]] {
#     infer_with_pre_detection
# }
