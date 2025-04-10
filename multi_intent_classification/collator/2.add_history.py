import json
from collections import defaultdict


def create_history(input_path: str, output_path: str) -> None:

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # nhóm dữ liệu theo conversation_id (nếu chưa được nhóm)
    # grouped_data = defaultdict(list)
    # for item in data:
    #     grouped_data[item["id"]].append(item)

    train_data = []

    for item in data:
        conversation_id = item["id"]
        messages = item["messages"]

        for idx, message in enumerate(messages):
            history = [msg["text"] for msg in messages[max(0, idx-999): idx]]
            current_message = message["text"]
            labeled_intent = message["label"]
            # labeled_intent = [label.split("|")[1] for label in message["labeled_intent"]]
            
            if len(history) == 0:
                history = []
            elif len(history) == 1:
                history = history

            train_data.append({
                "id": conversation_id,
                "history": history,
                "message": current_message,
                "label_intent": labeled_intent
            })

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(train_data, file, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được xử lý và lưu tại: {output_path}")


def create_history_dify(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    grouped_data = defaultdict(list)
    for item in data:
        grouped_data[item["conversation_id"]].append(item)

    train_data = []

    for conversation_id, messages in grouped_data.items():
        # Sắp xếp tin nhắn theo created_at (timestamp)
        sorted_messages = sorted(messages, key=lambda x: x["created_at"])

        for idx, message in enumerate(sorted_messages):
            history = [msg["current_message"] for msg in sorted_messages[:idx]]
            
            train_data.append({
                "conversation_id": conversation_id,
                "history": history,
                "current_message": message["current_message"],
                "label_intent": message["label_intent"]
            })

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(train_data, file, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được xử lý và lưu tại: {output_path}")
    print(f"Tổng số bản ghi: {len(train_data)}")

if __name__ == "__main__":
    input_path = "dataset_speech_analyse/test_raw.json"
    output_path = "dataset_speech_analyse/test.json"
    create_history(input_path, output_path)
