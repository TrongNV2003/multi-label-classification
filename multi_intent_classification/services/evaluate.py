import json
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
from torch.utils.data import DataLoader

class TestingArguments:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        output_file: str = None,
        is_multi_label: bool = False,
    ) -> None:
        self.test_loader = test_loader
        self.output_file = output_file
        self.is_multi_label = is_multi_label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def evaluate(self):
        self.model.eval()
        results = []
        latencies = []
        true_labels = []
        pred_labels = []
        
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

                if self.is_multi_label:
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float().cpu().numpy()
                else:
                    preds = torch.argmax(logits, dim=1).cpu().numpy()

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

        all_preds = np.concatenate(pred_labels)
        all_labels = np.concatenate(true_labels)

        self._print_metrics(all_preds, all_labels, "micro")
        self._print_metrics(all_labels, all_labels, "macro")
        self._print_metrics(all_labels, all_labels, "weighted")
        self._calculate_latency(latencies)

        print(f"num samples: {num_samples}")

    def _map_labels(self, label_data: list, labels_mapping: dict) -> list:
        if self.is_multi_label:
            return [labels_mapping[idx] for idx, val in enumerate(label_data) if val == 1]
        else:
            return [labels_mapping[label_data]]

    def _print_metrics(self, all_preds: np.ndarray, all_labels: np.ndarray, average_type: str) -> None:
        if self.is_multi_label:
            exact_match = np.all(all_preds == all_labels, axis=1).mean()
            accuracy = np.any(all_preds == all_labels, axis=1).mean() # Partial match ratio
            precision = precision_score(all_labels, all_preds, average=average_type, zero_division=0)
            recall = recall_score(all_labels, all_preds, average=average_type, zero_division=0)
            f1 = f1_score(all_labels, all_preds, average=average_type, zero_division=0)
        else:
            accuracy = np.mean(all_preds == all_labels)
            precision = precision_score(all_labels, all_preds, average=average_type, zero_division=0)
            recall = recall_score(all_labels, all_preds, average=average_type, zero_division=0)
            f1 = f1_score(all_labels, all_preds, average=average_type, zero_division=0)

        if self.is_multi_label:
            print(f"Exact Match: {exact_match * 100:.2f}%")
        print(f"\nMetrics ({average_type}):")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")

    def _calculate_latency(self, latencies: list) -> None:
        p99_latency = np.percentile(latencies, 99)
        print(f"\nP99 Latency: {p99_latency * 1000:.2f} ms")