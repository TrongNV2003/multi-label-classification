import json
import torch
import numpy as np
from typing import Mapping, Tuple
from transformers import AutoTokenizer

from multi_intent_classification.services.preprocessing import word_normalize, word_segment


class Dataset:
    def __init__(
        self,
        json_file: str,
        label2id: dict,
        text_col: str,
        label_col: str,
        is_multi_label: bool = False,
        word_segment: bool = False,
    ) -> None:
        data = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
            
        self.data = data
        self.text_col = text_col
        self.label_col = label_col
        self.label_mapping = label2id
        self.is_multi_label = is_multi_label
        self.word_segment = word_segment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, list]:
        item = self.data[index]
        query = item[self.text_col]
        labels = item[self.label_col]

        if self.word_segment:
            history = [word_segment(text) for text in history if text is not None]
            query = word_segment(query)
        query = word_normalize(query)
        
        
        if self.is_multi_label:
            label_vector = [0] * len(self.label_mapping)
            for label in labels:
                if label in self.label_mapping:
                    label_vector[self.label_mapping[label]] = 1
        else:
            if isinstance(labels, str):
                labels = [labels]
            elif not isinstance(labels, list):
                raise ValueError(f"Single-label mode nhưng nhãn tại mẫu {index} không phải str hoặc list: {labels}")
            
            if len(labels) != 1:
                raise ValueError(f"Single-label mode nhưng mẫu {index} có {len(labels)} nhãn: {labels}")
            
            label_vector = self.label_mapping[labels[0]]

        return query, label_vector


class DataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int, is_multi_label: bool = False) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_multi_label = is_multi_label
        
    def __call__(self, batch: list) -> Mapping[str, torch.Tensor]:
        contexts, labels = zip(*batch)

        tensor = self.tokenizer(
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

        tensor['labels'] = label_tensor
        return tensor
