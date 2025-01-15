from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader

class Tester:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
    ) -> None:
        self.test_loader = test_loader

        unique_labels = ["Cung cấp thông tin", "Tương tác", "Hỏi thông tin giao hàng", "Hỗ trợ, hướng dẫn", "Yêu cầu", "Phản hồi", "Sự vụ", "UNKNOWN"]
        self.id2label = {idx: label for idx, label in enumerate(unique_labels)}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
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

    def evaluate_query(self, query: str, history: list = None) -> None:
        """
        Evaluate a single query input from the user and print the predicted labels.
        Nhập message theo thứ tự conversation, lưu trữ tối đa 2 messages gần nhất
        Parameters:
            query (str): The input text query.

        Returns:
            None
        """
        self.model.eval()
        tokenizer = AutoTokenizer.from_pretrained("bert-classification")

        if history:
            # Giới hạn tối đa 2 tin nhắn lịch sử
            history = history[-2:]
            sep_token = tokenizer.sep_token
            history_text = sep_token.join(history)
            input_text = f"<history>{history_text}</history><current>{query}</current>"
        else:
            input_text = f"<current>{query}</current>"

        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()

        predicted_labels = self._map_labels(preds[0], self.id2label)
        if history:
            print(f"History: {history}")
        else:
            print("History: []")
        print(f"Predicted Labels: {predicted_labels}")

if __name__ == "__main__":
    MODEL = "bert-classification"
    tuned_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    tester = Tester(
        model=tuned_model,
        test_loader=None,
    )

    history = []
    while True:
        query = input("Enter a query ('exit': quit, 'clear': clear history): ")
        if query.lower() == "exit":
            break

        if query.lower() == "clear":
            history = []
            print("History has been cleared.")
            continue
        
        tester.evaluate_query(query, history=history)

        history.append(query)
        if len(history) > 2:  # Giữ tối đa 2 tin nhắn gần nhất
            history.pop(0)
