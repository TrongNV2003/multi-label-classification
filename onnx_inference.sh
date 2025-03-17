python -m multi_intent_classification.services.onnx_cls.onnx_inference \
    --tokenize_model vinai/phobert-base-v2 \
    --onnx_model multi_intent_classification/models_onnx/classification-phobert-base-v2.onnx \
    --test_set multi_intent_classification/dataset/test.json \
