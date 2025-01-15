# multi-intent-classification
Predict multiple intents for each message on conversations

## Installation
```sh
pip install -r requirements.txt
```

## Usage
training and evaluating models:
```sh
python main.py
```

test single query (see Features testing):
```sh
python eval_query.py
```

### Features testing
- Có thể truyền từng messages vào theo thứ tự trong conversation:
    + input: single message -> output: current message, intent, history
    + lưu trữ tối đa 2 messages trước đó trong 1 conversation
    + với conversation mới cần 'clear' history của conversation trước đó

### Future plans:
    + Tự động 'clear' history khi sang conversation mới
    + Đóng thành API để testing
    + Convert model sang ONNX để tăng tốc inference (chưa biết -> tìm hiểu)
    + ...
    