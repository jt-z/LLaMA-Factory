#!/bin/bash

set -x

MODEL_PATH=/home/ksa/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct


# 8卡3090上LoRA微调LLama3 8B，可以把单卡批次大小开到8，主要线性增长的是中间激活值和 微调的LoRA的权重矩阵里的少量参数 
llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target all \
    --dataset chatml_alpaca_gpt4_zh \
    --template llama3 \
    --cutoff_len 2048 \
    --max_samples 1000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir saves/llama3-8b/lora/sft_chatml_alpaca_gpt4_zh \
    --logging_steps 10 \
    --save_steps 500 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to none \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000
