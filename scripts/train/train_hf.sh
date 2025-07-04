#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD

python -m multi_intent_classification.main_hf \
    --dataloader_num_workers 2 \
    --seed 42 \
    --learning_rate 5e-5 \
    --num_train_epochs 15 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_length 512 \
    --optim adamw_torch_fused \
    --lr_scheduler_type linear \
    --model /data/vi-roberta-base-emoji/checkpoint-92520 \
    --pin_memory \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --test_batch_size 64 \
    --train_file datasets/UIT-VSMEC/train.jsonl \
    --valid_file datasets/UIT-VSMEC/val.jsonl \
    --test_file datasets/UIT-VSMEC/test.jsonl \
    --text_col sentence \
    --label_col label \
    --output_dir ./models \
    --record_output_file output.json \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 100 \
    --logging_dir ./models/logs \
    --fp16 \
    --metric_for_best_model eval_micro_f1 \
    --greater_is_better \
    --load_best_model_at_end \
    --report_to mlflow \
