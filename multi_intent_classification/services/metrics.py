import numpy as np
from typing import List, Dict
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_metrics(all_preds: np.ndarray, all_labels: np.ndarray, average_type: str, is_multi_label: bool) -> None:
    if is_multi_label:
        accuracy = np.mean((all_preds == all_labels).all(axis=-1))
    else:
        accuracy = np.mean(all_preds == all_labels)
    precision = precision_score(all_labels, all_preds, average=average_type, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=average_type, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=average_type, zero_division=0)
    print(f"\n=== Metrics ({average_type}) ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")


def partial_accuracy(results: List[Dict[str, List[str]]]) -> None:
    correct_one = 0
    total = len(results)
    for item in results:
        true_set = set(item["true_labels"])
        pred_set = set(item["predicted_labels"])
        if true_set & pred_set:
            correct_one += 1
    accuracy_one = correct_one / total if total > 0 else 0
    print(f"\nAccuracy (Match one): {accuracy_one * 100:.2f}%")


def calculate_latency(latencies: List[float]) -> Dict[str, float]:
    p95 = np.percentile(latencies, 95) * 1000
    p99 = np.percentile(latencies, 99) * 1000
    mean_latency = np.mean(latencies) * 1000
    print("\nLatency Statistics:")
    print(f"P95 Latency: {p95:.2f} ms")
    print(f"P99 Latency: {p99:.2f} ms")
    print(f"Mean Latency: {mean_latency:.2f} ms")