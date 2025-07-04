import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_metrics(eval_pred: tuple, is_multi_label: bool):
    logits, labels = eval_pred
    
    if is_multi_label:
        sigmoid_preds = sigmoid(logits)
        predictions = (sigmoid_preds > 0.5).astype(int)
        references = labels.astype(int)

        results = {}
        for avg in ["micro", "macro", "weighted"]:
                results[f"{avg}_precision"] = precision_score(references, predictions, average=avg, zero_division=0)
                results[f"{avg}_recall"] = recall_score(references, predictions, average=avg, zero_division=0)
                results[f"{avg}_f1"] = f1_score(references, predictions, average=avg, zero_division=0)
            
    else:
        predictions = np.argmax(logits, axis=1).astype(int)
        references = labels.astype(int)

        results = {
            "precision": precision_score(references, predictions, average="weighted", zero_division=0) * 100,
            "recall": recall_score(references, predictions, average="weighted", zero_division=0) * 100,
            "f1": f1_score(references, predictions, average="weighted", zero_division=0) * 100,
            "accuracy": accuracy_score(references, predictions) * 100
        }
    
    return results
