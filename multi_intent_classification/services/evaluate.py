import json
import time
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Callable
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch
from torch.utils.data import DataLoader, Dataset

from multi_intent_classification.utils import constant

class TestingArguments:
    def __init__(
        self,
        dataloader_workers: int,
        device: str,
        model: torch.nn.Module,
        pin_memory: bool,
        test_set: Dataset,
        test_batch_size: int,
        collate_fn: Optional[Callable] = None,
        output_file: Optional[str] = None,
        is_multi_label: bool = False,
    ) -> None:
        self.output_file = output_file
        self.is_multi_label = is_multi_label
        self.device = device
        self.model = model.to(self.device)
        self.test_loader = DataLoader(
            test_set,
            batch_size=test_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def evaluate(self):
        self.model.eval()
        results = []
        latencies = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, total=len(self.test_loader), unit="batches"):
                try:
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
                        batch_preds = (probs > 0.5).float().cpu().numpy()
                    else:
                        probs = torch.softmax(logits, dim=1)
                        batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
                    
                    batch_labels = labels.cpu().numpy()
                    all_preds.append(batch_preds)
                    all_labels.append(batch_labels)

                    for i in range(len(batch_preds)):
                        true_label_names = self._map_labels(batch_labels[i], self.model.config.id2label)
                        predicted_label_names = self._map_labels(batch_preds[i], self.model.config.id2label)
                        probs_np = probs[i].cpu().numpy()

                        predicted_probs = {}
                        if self.is_multi_label:
                            for idx, val in enumerate(batch_preds[i]):
                                if val == 1:
                                    label_name = self.model.config.id2label[idx]
                                    predicted_probs[label_name] = float(probs_np[idx])
                                else: 
                                    label_name = "UNKNOWN|UNKNOWN"
                                    predicted_probs[label_name] = 0.0
                        else:
                            predicted_label_idx = batch_preds[i]
                            label_name = self.model.config.id2label[predicted_label_idx]
                            predicted_probs[label_name] = float(probs_np[predicted_label_idx])
                            
                        results.append({
                            "true_labels": true_label_names,
                            "predicted_labels": predicted_label_names,
                            "probabilities": predicted_probs,
                            "latency": float(latency) / len(batch_preds),
                        })

                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue

        all_preds = np.vstack(all_preds) if self.is_multi_label else np.concatenate(all_preds)
        all_labels = np.vstack(all_labels) if self.is_multi_label else np.concatenate(all_labels)
        
        if self.output_file:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"Results saved to {self.output_file}")

        metrics = {}
        for avg in ["micro", "macro", "weighted"]:
            metrics[avg] = self._calculate_metrics(all_preds, all_labels, avg)

        latency_stats = self._calculate_latency_stats(latencies)
        metrics["latency"] = latency_stats
        self._calculate_accuracy(results)
        
        num_samples = len(results)
        print(f"num samples: {num_samples}")
        return metrics

    def _map_labels(self, label_data, labels_mapping: Dict[int, str]) -> List[str]:
        if self.is_multi_label:
            vals = label_data.tolist() if hasattr(label_data, "tolist") else list(label_data)
            selected = [labels_mapping[idx] for idx, val in enumerate(vals) if val == 1]
            return selected if selected else [constant.UNKNOWN_LABEL]
        else:
            idx = int(label_data)
            return [labels_mapping.get(idx, constant.UNKNOWN_LABEL)]

    def _calculate_metrics(self, all_preds: np.ndarray, all_labels: np.ndarray, average_type: str) -> Dict[str, float]:
        metrics = {}
        metrics["precision"] = float(precision_score(all_labels, all_preds, average=average_type, zero_division=0))
        metrics["recall"] = float(recall_score(all_labels, all_preds, average=average_type, zero_division=0))
        metrics["f1"] = float(f1_score(all_labels, all_preds, average=average_type, zero_division=0))
        print(f"\nMetrics ({average_type}):")
        print(f"Precision: {metrics['precision'] * 100:.2f}")
        print(f"Recall: {metrics['recall'] * 100:.2f}")
        print(f"F1 Score: {metrics['f1'] * 100:.2f}")
        return metrics

    def _calculate_accuracy(self, results):
        correct = 0
        correct_one = 0
        total = len(results)
        for item in results:
            true_set = set(item["true_labels"])
            pred_set = set(item["predicted_labels"])
            if true_set == pred_set:
                correct += 1
            if true_set & pred_set:
                correct_one += 1
        accuracy = correct / total if total > 0 else 0
        accuracy_one = correct_one / total if total > 0 else 0
        print(f"\nAccuracy (Match one): {accuracy_one * 100:.2f}%")
        print(f"Accuracy (Match all): {accuracy * 100:.2f}%")


    def _calculate_latency_stats(self, latencies: List[float]) -> Dict[str, float]:
        stats = {
            "p95_ms": float(np.percentile(latencies, 95) * 1000),
            "p99_ms": float(np.percentile(latencies, 99) * 1000),
            "mean_ms": float(np.mean(latencies) * 1000),
        }
        print("\nLatency Statistics:")
        print(f"P95 Latency: {stats['p95_ms']:.2f} ms")
        print(f"P99 Latency: {stats['p99_ms']:.2f} ms")
        print(f"Mean Latency: {stats['mean_ms']:.2f} ms")
        return stats