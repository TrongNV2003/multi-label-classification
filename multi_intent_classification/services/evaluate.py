import json
import time
import numpy as np
from loguru import logger
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Tester:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        output_file: str = None,
    ) -> None:
        self.test_loader = test_loader
        self.output_file = output_file

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)

    def evaluate(self):
        self.model.eval()
        latencies = []
        true_labels = []
        pred_labels = []
        total_loss = 0
        results = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                batch_start_time = time.time()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits

                batch_end_time = time.time()

                latency = batch_end_time - batch_start_time
                latencies.append(latency)

                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                pred_labels.extend(preds)
                true_labels.extend(labels.cpu().numpy())

                for i in range(len(preds)):
                    true_label_names = self._map_labels(labels.cpu().numpy()[i], self.model.config.id2label)
                    predicted_label_names = self._map_labels(preds[i], self.model.config.id2label)      # id2label từ checkpoint
                    results.append({
                        "true_labels": true_label_names,
                        "predicted_labels": predicted_label_names,
                        "latency": float(latency),
                    })
                    
        num_samples = len(results)
        
        if self.output_file:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"Results saved to {self.output_file}")

        self._print_metrics(true_labels, pred_labels, "micro")
        self._print_metrics(true_labels, pred_labels, "macro")
        self._print_metrics(true_labels, pred_labels, "weighted")
        self._calculate_accuracy(results)
        self._calculate_latency(latencies)

        print(f"num samples: {num_samples}")

    def _map_labels(self, label_indices: list, labels_mapping: dict) -> list:
        return [labels_mapping[idx] for idx, val in enumerate(label_indices) if val == 1.0]

    def _print_metrics(self, true_labels, pred_labels, average_type):
        precision = precision_score(true_labels, pred_labels, average=average_type, zero_division=0)
        recall = recall_score(true_labels, pred_labels, average=average_type, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, average=average_type, zero_division=0)
        print(f"\nMetrics ({average_type}):")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")

    def _calculate_accuracy(self, data):
        correct = 0
        total = len(data)
        for item in data:
            true_set = set(item["true_labels"])
            pred_set = set(item["predicted_labels"])
            if true_set == pred_set:
                correct += 1
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def _calculate_latency(self, latencies: list) -> None:
        p99_latency = np.percentile(latencies, 99)
        print(f"P99 Latency: {p99_latency * 1000:.2f} ms")