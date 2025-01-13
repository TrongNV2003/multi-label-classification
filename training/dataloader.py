import json
from typing import List, Mapping, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

class Dataset:
    def __init__(self, json_file: str, label_mapping: dict) -> None:
        """
        Args:
            json_file (str): Path to the JSON file.
            label_mapping (dict): Mapping of unique labels to indices.
        """
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data = data
        self.label_mapping = label_mapping
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        """
        Get the item at the given index

        Returns:
            text: the text of the item
            labels: Multi-label vector
        
        target input: [cls]<history>message(k-2)[sep]message(k-1)</history><current>message(k)</current>
        """

        item = self.data[index]
        history = item["history"]
        current_message = item["current_message"]
        labels = item["label_intent"]

        if history:
            history_text = "[SEP]".join(history)
            context = f"<history>{history_text}</history><current>{current_message}</current>"
        else:
            context = f"<current>{current_message}</current>"

        label_vector = [0] * len(self.label_mapping)
        for label in labels:
            if label in self.label_mapping:
                label_vector[self.label_mapping[label]] = 1

        return context, label_vector, current_message


class LlmDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch: list) -> Mapping[str, torch.Tensor]:
        """
        Tokenize the batch of data and convert tokenized data to tensor
        
        args:
            batch (tuple)

        returns:
            tensor (dict)
        """

        contexts, labels, current_message = zip(*batch)

        contexts_tensor = self.tokenizer(
            contexts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        label_tensor = torch.tensor(np.array(labels), dtype=torch.float)

        return {
            "text_input_ids": contexts_tensor["input_ids"],
            "text_attention_mask": contexts_tensor["attention_mask"],
            "labels": label_tensor,
            "current_message": current_message,
        }
