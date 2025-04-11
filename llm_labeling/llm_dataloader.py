import json
from typing import Tuple

class Dataset:
    def __init__(self, json_file: str) -> None:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, list]:
        item = self.data[index]
        id = item["id"]
        history = item["history"]
        current_message = item["message"]
        labels = item["label_intent"]

        history_text = "\n".join(history) if history else ""
        history = f"{history_text}"
        context = f"{current_message}"

        if not isinstance(labels, list):
            labels = [labels]

        return id, history, context, labels

# if __name__ == "__main__":
#     dataset = Dataset("dataset_speech_analyse/test.json")
#     for i in range(len(dataset)):
#         history, context, labels = dataset[i]
#         print(f"History: {history}")
#         print(f"Context: {context}")
#         print(f"Labels: {labels}")
#         print("-" * 20)