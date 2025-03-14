python -m services.onnx_cls.onnx_inference \
    --tokenize_model vinai/phobert-base-v2 \
    --onnx_model models_onnx/classification-phobert-base-v2.onnx \
    --test_set dataset/test.json \
