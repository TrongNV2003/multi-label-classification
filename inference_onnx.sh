python -m multi_intent_classification.onnx_cls.onnx_inference \
    --model vinai/phobert-base-v2 \
    --output_dir ./models \
    --test_set dataset/test.json \
