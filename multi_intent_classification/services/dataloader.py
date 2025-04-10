import json
import numpy as np
from typing import Mapping, Tuple
from underthesea import word_tokenize

import torch
from transformers import AutoTokenizer

class Dataset:
    def __init__(
        self,
        json_file: str,
        label_mapping: dict,
        tokenizer: AutoTokenizer,
        is_multi_label: bool = False,
        word_segment: bool = False,
    ) -> None:
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.data = data
        self.label_mapping = label_mapping
        self.sep_token = tokenizer.sep_token
        self.is_multi_label = is_multi_label
        self.word_segment = word_segment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, list]:
        item = self.data[index]
        if self.word_segment:
            history = [self._word_segment(text) for text in item.get("history", []) if text is not None]
            current_message = self._word_segment(item["message"])
        else:
            history = item.get("history", [])
            current_message = item["message"]
            
        labels = item["label_intent"]

        if history:
            history_text = self.sep_token.join(history)
            context = f"<history>{history_text}</history><current>{current_message}</current>"
        else:
            context = f"<current>{current_message}</current>"
        
        if self.is_multi_label:
            label_vector = [0] * len(self.label_mapping)
            for label in labels:
                if label in self.label_mapping:
                    label_vector[self.label_mapping[label]] = 1
        else:
            if len(labels) != 1:
                raise ValueError(f"Single-label mode nhưng mẫu {index} có {len(labels)} nhãn: {labels}")
            label_vector = self.label_mapping[labels[0]]

        return context, label_vector

    def _word_segment(self, text: str) -> str:
        tokens = word_tokenize(text)
        text_segment = " ".join(tokens)
        return text_segment

class LlmDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int, is_multi_label: bool = False) -> None:
        """
        Args:
            tokenizer (AutoTokenizer): Tokenizer từ transformers.
            max_length (int): Độ dài tối đa của chuỗi token.
            is_multi_label (bool): True nếu là multi-label, False nếu single-label.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_multi_label = is_multi_label
        
    def __call__(self, batch: list) -> Mapping[str, torch.Tensor]:
        contexts, labels = zip(*batch)

        contexts_tensor = self.tokenizer(
            contexts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if self.is_multi_label:
            label_tensor = torch.tensor(np.array(labels), dtype=torch.float)
        else:
            label_tensor = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": contexts_tensor["input_ids"],
            "attention_mask": contexts_tensor["attention_mask"],
            "labels": label_tensor
        }
