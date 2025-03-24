import json
from typing import Tuple

class Dataset:
    def __init__(self, json_file: str) -> None:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data = data
        self.sep_token = "<sep>"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, list]:
        item = self.data[index]
        history = item["history"]
        current_message = item["current_message"]
        labels = item["label_intent"]

        history_text = self.sep_token.join(history) if history else ""
        history = f"<history>{history_text}</history>"
        context = f"{current_message}"

        if not isinstance(labels, list):
            labels = [labels]

        return history, context, labels
