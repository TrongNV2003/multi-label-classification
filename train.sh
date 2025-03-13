python -m services.train \
    --dataloader_workers 2 \
    --device cuda \
    --seed 42 \
    --epochs 10 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --max_length 256 \
    --pad_mask_id -100 \
    --model vinai/phobert-base-v2 \
    --train_batch_size 16 \
    --valid_batch_size 8 \
    --test_batch_size 8 \
    --train_file dataset/train.json \
    --valid_file dataset/val.json \
    --test_file dataset/test.json \
    --output_dir ./models/classification \
    --record_output_file output.json \
    --evaluate_on_accuracy True \

