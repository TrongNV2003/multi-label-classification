import json
import numpy as np
from typing import Mapping, Tuple

import torch
from transformers import AutoTokenizer

class Dataset:
    def __init__(self, json_file: str, label_mapping: dict, tokenizer: AutoTokenizer) -> None:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data = data
        self.label_mapping = label_mapping
        self.sep_token = tokenizer.sep_token

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, list]:
        item = self.data[index]
        history = item["history"]
        current_message = item["current_message"]
        labels = item["label_intent"]

        if history:
            history_text = self.sep_token.join(history)
            context = f"<history>{history_text}</history><current>{current_message}</current>"
        else:
            context = f"<current>{current_message}</current>"

        label_vector = [0] * len(self.label_mapping)
        for label in labels:
            if label in self.label_mapping:
                label_vector[self.label_mapping[label]] = 1

        return context, label_vector


class LlmDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch: list) -> Mapping[str, torch.Tensor]:
        contexts, labels = zip(*batch)

        contexts_tensor = self.tokenizer(
            contexts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        label_tensor = torch.tensor(np.array(labels), dtype=torch.float)

        return {
            "input_ids": contexts_tensor["input_ids"],
            "attention_mask": contexts_tensor["attention_mask"],
            "labels": label_tensor
        }
