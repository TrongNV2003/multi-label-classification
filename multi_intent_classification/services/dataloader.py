import json
import numpy as np
from typing import Mapping, Tuple

import torch
from transformers import AutoTokenizer

from multi_intent_classification.services.preprocessing import word_normalize, word_segment


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
        history = item.get("history", [])
        query = item["current_message"]
        labels = item["label"]
        role = item["role"]
        
        if self.word_segment:
            history = [word_segment(text) for text in history if text is not None]
            query = word_segment(query)
        query = word_normalize(query)
        
        self.role_list = list({item["role"] for item in self.data})
        
        if role not in self.role_list:
            self.role_list = [role] + [r for r in self.role_list if r != role]
        num_turns = len(history) + 1
        last_role_idx = self.role_list.index(role)
        roles = []
        for i in range(num_turns):
            idx = (last_role_idx - (num_turns - 1 - i)) % len(self.role_list)
            roles.append(self.role_list[idx])
        
        history = [f"{r}: {t}" for r, t in zip(roles[:-1], history)]
        query = f"{role}: {query}"

        if history:
            history = history[::-1]
            history_text = self.sep_token.join(history)
        else:
            history_text = ""
        context = f"<current>{query}</current><history>{history_text}</history>"
        
        if self.is_multi_label:
            label_vector = [0] * len(self.label_mapping)
            for label in labels:
                if label in self.label_mapping:
                    label_vector[self.label_mapping[label]] = 1
        else:
            if len(labels) != 1:
                raise ValueError(f"Single-label mode nhưng mẫu {index} có {len(labels)} nhãn: {labels}")
            label_vector = self.label_mapping[labels[0]]

        return context, label_vector, role


class DataCollator:
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
        contexts, labels, roles = zip(*batch)

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
            "labels": label_tensor,
            "roles": roles
        }
