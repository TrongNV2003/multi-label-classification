#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD

# Train
export RESULTS_FILE=results/test_code_results.txt
python -m multi_intent_classification.main \
    --dataloader_workers 2 \
    --seed 42 \
    --epochs 10 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --use_warmup_steps \
    --max_length 256 \
    --model vinai/phobert-base-v2 \
    --pin_memory \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --test_batch_size 16 \
    --train_file dataset/train.json \
    --val_file dataset/val.json \
    --test_file dataset/test.json \
    --text_col message \
    --label_col label \
    --output_dir ./models \
    --record_output_file output.json \
    --early_stopping_patience 5 \
    --early_stopping_threshold 0.001 \
    --evaluate_on_accuracy \
    --is_multi_label \
    --use_focal_loss \
    --focal_loss_alpha 0.25 \
    --focal_loss_gamma 2.0 \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target_modules "query, key, value, dense" \
    > $RESULTS_FILE
