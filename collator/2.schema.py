import json
from collections import defaultdict


def create_history(input_path: str) -> None:
    """
    Function to make history for each message in a conversation
    history: min 0 message, max 2 messages
    """

    # input_path = "dataset/output_dedup.json"
    output_path = "dataset/output.json"

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    grouped_data = defaultdict(list)
    for item in data:
        grouped_data[item["id"]].append(item)

    train_data = []

    for conversation_id, messages in grouped_data.items():
        for idx, message in enumerate(messages):
            # Lấy 2 câu trước đó làm history
            history = [msg["text"] for msg in messages[max(0, idx - 2): idx]]
            current_message = message["text"]
            labeled_intent = message["labeled_intent"]

            # Xử lý trường hợp không có đủ history
            if len(history) == 0:
                history = []  # Message đầu tiên, không có history
            elif len(history) == 1:
                history = history  # Message thứ 2, chỉ có 1 lịch sử

            # Thêm vào dataset
            train_data.append({
                "id": conversation_id,
                "history": history,
                "current_message": current_message,
                "label_intent": labeled_intent
            })

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(train_data, file, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được xử lý và lưu tại: {output_path}")

if __name__ == "__main__":
    input_file = "dataset/output_dedup.json"
    create_history(input_file)