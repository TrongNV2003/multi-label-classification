# multi-intent-classification
Predict multiple intents for each message on conversations

## Installation
```sh
pip install -r requirements.txt
```

## Usage
training and evaluating models:
```sh
bash train.sh
```

## Example input for dataset
```json
[
    {
        "id": 1,
        "history": [
            "alo ạ"
        ],
        "message": "Xin chào bạn",
        "label_intent": [
            "Intent 1",
            "Intent 2"
        ]
    }
]
```

### Future plans:
    + Convert model sang ONNX để tăng tốc inference (done)
    + So sánh kết quả với LLM model (Qwen-72B) (done)
    