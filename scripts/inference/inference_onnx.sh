#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD

python -m multi_intent_classification.onnx_cls.onnx_inference \
    --model vinai/phobert-base-v2 \
    --output_dir ./models \
    --train_set dataset/dataset/train.json \
    --test_set dataset/dataset/test.json \
    --is_multi_label \
    --output_file results/output.log \
    --max_length 256 \
    --test_batch_size 16 \
