import torch
import logging
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from multi_intent_classification.services.dataloader import Dataset, LlmDataCollator

class Inference:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        output_file: str,
    ) -> None:
        self.model = model
        self.test_loader = test_loader
        self.output_file = output_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logging.basicConfig(
            filename=output_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger()

    def run(self):
        """
        Chạy inference từng mẫu, mô phỏng cuộc tấn công và ghi log giống thực tế.
        """
        self.model.eval()
        print("Bắt đầu mô phỏng inference...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                
                for i in range(len(preds)):
                    true_label_names = self._map_labels(labels.cpu().numpy()[i], self.model.config.id2label)
                    predicted_label_names = self._map_labels(preds[i], self.model.config.id2label)

                    self.logger.info(f"Sample {batch_idx * self.test_loader.batch_size + i}: {true_label_names} -> {predicted_label_names}")
                    logger.info(f"Sample {batch_idx * self.test_loader.batch_size + i}: {true_label_names} -> {predicted_label_names}")

    def _map_labels(self, label_indices: list, labels_mapping: dict) -> list:
        return [labels_mapping[idx] for idx, val in enumerate(label_indices) if val == 1.0]

if __name__ == "__main__":
    MODEL = "models/classification-phobert-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    unique_labels = ["Cung cấp thông tin", "Tương tác", "Hỏi thông tin giao hàng", "Hỗ trợ, hướng dẫn", "Yêu cầu", "Phản hồi", "Sự vụ", "UNKNOWN"]
    print(unique_labels)
    print(model.config.id2label)

    test_set = Dataset(
        json_file="multi_intent_classification/dataset/test.json",
        label_mapping=model.config.label2id,
        tokenizer=tokenizer
    )

    collator = LlmDataCollator(tokenizer=tokenizer, max_length=256)

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=True,
        collate_fn=collator
    )

    infer = Inference(
        model=model,
        test_loader=test_loader,
        output_file="attack_demo.log"
    )

    infer.run()