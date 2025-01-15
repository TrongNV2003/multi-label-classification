import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader
import json

class Tester:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        output_file: str,
        id2label: list,
    ) -> None:
        self.test_loader = test_loader
        self.output_file = output_file
        self.id2label = id2label

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def evaluate(self):
        """
        This function will eval the model on test set and return the accuracy, F1-score and latency

        Parameters:
            None

        Returns:
            None
        """

        self.model.eval()
        latencies = []
        all_labels = []
        all_preds = []
        total_loss = 0
        results = []
        
        start_time = time.time()    #throughput
        with torch.no_grad():
            for batch in self.test_loader:
                text_input_ids = batch["input_ids"].to(self.device)
                text_attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                text_samples = batch["current_message"]

                batch_start_time = time.time()
                outputs = self.model(
                    input_ids=text_input_ids,
                    attention_mask=text_attention_mask,
                )
                logits = outputs.logits
                batch_end_time = time.time()
                latency = batch_end_time - batch_start_time
                latencies.append(latency)

                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

                for i in range(len(text_samples)):
                    true_label_names = self._map_labels(labels.cpu().numpy()[i], self.id2label)
                    predicted_label_names = self._map_labels(preds[i], self.id2label)
                    results.append({
                        "text": text_samples[i],
                        "true_labels": true_label_names,
                        "predicted_labels": predicted_label_names,
                        "latency": float(latency),
                    })
        total_time = time.time() - start_time
        num_samples = len(results)

        
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {self.output_file}")

        self.score(all_labels, all_preds, results)
        self.calculate_latency(latencies)

        throughput = num_samples / total_time
        print(f"num samples: {num_samples}")
        print(f"Throughput: {throughput:.2f} samples/s")

    def _map_labels(self, label_indices: list, labels_mapping: dict) -> list:
        """
        Map label indices to their corresponding names.

        Parameters:
            label_indices: List of binary labels (0 or 1).
            labels_mapping: Dictionary mapping indices to label names.

        Returns:
            List of label names.
        """
        return [labels_mapping[idx] for idx, val in enumerate(label_indices) if val == 1.0]


    def score(self, label: list, predict: list, output: list) -> None:
        """
        This function will calculate the F1-score

        Parameters:
            label: list
            predict: list

        Returns:
            f1 score
        """

        precision = precision_score(label, predict, average="weighted", zero_division=0)
        recall = recall_score(label, predict, average="weighted", zero_division=0)
        f1_score = 2 * (precision * recall) / (precision + recall)
        accuracy = self._accuracy(output)

        print(f"Accuracy: {accuracy * 100:.2f}")
        print(f"Precision: {precision * 100:.2f}")
        print(f"Recall: {recall * 100:.2f}")
        print(f"F1 score: {f1_score * 100:.2f}")

    def calculate_latency(self, latencies: list) -> None:
        """
        This function will calculate the latency for each sample

        Parameters:
            latencies: list

        Returns:
            P95 latency
        """

        p99_latency = np.percentile(latencies, 99)
        print(f"P99 Latency: {p99_latency * 1000:.2f} ms")
    

    def _accuracy(self, output_data: list) -> float:
        """
        Calculate accuracy for multi-label predictions where a sample is correct
        if at least one predicted label matches the true labels.

        Parameters:
            output_data (list): List of dictionaries containing `true_labels` and `predicted_labels`.

        Returns:
            float: Accuracy score.
        """

        correct = 0
        total = len(output_data)

        for sample in output_data:
            true_labels = set(sample["true_labels"])
            predicted_labels = set(sample["predicted_labels"])
            
            if true_labels & predicted_labels:  # Giao của true_labels và predicted_labels không rỗng
                correct += 1

        return correct / total if total > 0 else 0.0
