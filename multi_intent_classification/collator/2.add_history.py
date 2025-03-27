import json
from collections import defaultdict


def create_history(input_path: str, output_path: str) -> None:

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    grouped_data = defaultdict(list)
    for item in data:
        grouped_data[item["id"]].append(item)

    train_data = []

    for conversation_id, messages in grouped_data.items():
        for idx, message in enumerate(messages):
            history = [msg["text"] for msg in messages[max(0, idx-999): idx]]
            current_message = message["text"]
            labeled_intent = message["label_intent"]

            if len(history) == 0:
                history = []
            elif len(history) == 1:
                history = history

            train_data.append({
                "id": conversation_id,
                "history": history,
                "current_message": current_message,
                "label_intent": labeled_intent
            })

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(train_data, file, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được xử lý và lưu tại: {output_path}")


def create_history_dify(input_path: str, output_path: str) -> None:
    # Đọc file JSON đầu vào
    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Nhóm các tin nhắn theo conversation_id
    grouped_data = defaultdict(list)
    for item in data:
        grouped_data[item["conversation_id"]].append(item)

    train_data = []

    # Xử lý từng cuộc hội thoại
    for conversation_id, messages in grouped_data.items():
        # Sắp xếp tin nhắn theo created_at (timestamp)
        sorted_messages = sorted(messages, key=lambda x: x["created_at"])

        # Tạo lịch sử cho từng tin nhắn
        for idx, message in enumerate(sorted_messages):
            # Lấy các tin nhắn trước làm lịch sử (từ đầu đến idx-1)
            history = [msg["current_message"] for msg in sorted_messages[:idx]]
            
            train_data.append({
                "conversation_id": conversation_id,
                "history": history,
                "current_message": message["current_message"],
                "label_intent": message["label_intent"]
            })

    # Lưu vào file JSON đầu ra
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(train_data, file, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được xử lý và lưu tại: {output_path}")
    print(f"Tổng số bản ghi: {len(train_data)}")

if __name__ == "__main__":
    input_path = "multi_intent_classification/dify_dataset/processed_data.json"
    output_path = "multi_intent_classification/dify_dataset/processed_data_history.json"
    create_history_dify(input_path, output_path)
