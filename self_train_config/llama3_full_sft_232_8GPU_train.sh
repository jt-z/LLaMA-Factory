#!/bin/bash

# 设置显卡（根据实际情况修改）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 运行命令：指定 YAML 文件路径
llamafactory-cli train self_train_config/llama3_full_sft_232_8GPU.yaml