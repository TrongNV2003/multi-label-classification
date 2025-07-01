import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from sklearn.metrics import precision_score, recall_score, f1_score


def _partial_accuracy(all_preds: np.ndarray, all_labels: np.ndarray) -> float:
    if len(all_preds) == 0 or len(all_labels) == 0:
        raise ValueError("Predictions or labels cannot be empty")
    if all_preds.shape != all_labels.shape:
        raise ValueError(f"Shape mismatch: all_preds {all_preds.shape}, all_labels {all_labels.shape}")

    correct_one = 0
    total = len(all_preds)
    for pred, label in zip(all_preds, all_labels):
        pred_set = set(np.where(pred)[0])
        true_set = set(np.where(label)[0])
        if pred_set & true_set:
            correct_one += 1
    accuracy_one = correct_one / total if total > 0 else 0

    return accuracy_one


def calculate_metrics(all_preds: np.ndarray, all_labels: np.ndarray, average_type: str, is_multi_label: bool) -> Dict[str, float]:
    if is_multi_label:
        accuracy = np.mean((all_preds == all_labels).all(axis=-1))
        accuracy_one = _partial_accuracy(all_preds, all_labels)
    else:
        accuracy = np.mean(all_preds == all_labels)
    precision = precision_score(all_labels, all_preds, average=average_type, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=average_type, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=average_type, zero_division=0)
    metrics = {
        "average_type": average_type,
        "accuracy": accuracy * 100,
        "partial_accuracy": accuracy_one * 100 if is_multi_label else None,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1_score": f1 * 100
    }
    print(f"\n=== Metrics ({average_type}) ===")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1 Score: {metrics['f1_score']:.2f}")
    return metrics


def calculate_latency(latencies: List[float]) -> Dict[str, float]:
    p95 = np.percentile(latencies, 95) * 1000
    p99 = np.percentile(latencies, 99) * 1000
    mean_latency = np.mean(latencies) * 1000
    metrics = {
        "p95": p95,
        "p99": p99,
        "mean_latency": mean_latency
    }
    print("\nLatency Statistics:")
    print(f"P95 Latency: {metrics['p95']:.2f} ms")
    print(f"P99 Latency: {metrics['p99']:.2f} ms")
    print(f"Mean Latency: {metrics['mean_latency']:.2f} ms")
    return metrics


def save_metrics_to_json(metrics: Dict, output_file: str) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_file}")
    