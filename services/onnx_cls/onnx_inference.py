import torch
import logging
import argparse
import onnxruntime
from loguru import logger
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from services.dataloader import Dataset, LlmDataCollator

class ONNXInference:
    def __init__(
            self,
            onnx_model_path: str,
            test_loader: DataLoader,
            output_file: str,
            id2label: dict
    ) -> None:
        self.onnx_model_path = onnx_model_path
        self.test_loader = test_loader
        self.id2label = id2label
        self.output_file = output_file

        self.session = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        )
        self.sess_output = [out.name for out in self.session.get_outputs()]

        logging.basicConfig(
            filename=output_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger()

    def run(self):
        for batch_idx, batch in enumerate(self.test_loader):
            input_ids = batch["input_ids"].numpy()  # Chuyển sang NumPy cho ONNX
            attention_mask = batch["attention_mask"].numpy()

            sess_input = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

            outputs = self.session.run(self.sess_output, sess_input)
            logits = torch.from_numpy(outputs[0])

            pred = torch.argmax(logits, dim=1).item()
            predicted_label_name = self.id2label[pred]
            confidence = torch.softmax(logits, dim=1).max().item()

            log_message = (
                f"Query {batch_idx + 1}: "
                f"Category type: {predicted_label_name} | "
                f"Confidence: {confidence:.4f}"
            )
            logger.info(log_message)
            self.logger.info(log_message)


if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenize_model", type=str, default="vinai/phobert-base-v2", help="Tokenizer model", required=True)
    parser.add_argument("--onnx_model", type=str, default="models_onnx/classification-phobert-base-v2.onnx", help="ONNX model path", required=True)
    parser.add_argument("--test_set", type=str, default="dataset/test.json", help="Test dataset file", required=True)
    parser.add_argument("--output_file", type=str, default="output.log", help="Output log file")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenize_model)
    unique_labels = ["Cung cấp thông tin", "Tương tác", "Hỏi thông tin giao hàng", "Hỗ trợ, hướng dẫn", "Yêu cầu", "Phản hồi", "Sự vụ", "UNKNOWN"]
    id2label = {idx: label for idx, label in enumerate(unique_labels)}
    label2id = {label: idx for idx, label in enumerate(unique_labels)}

    test_set = Dataset(
        json_file=args.test_set,
        label_mapping=label2id,
        tokenizer=tokenizer
    )

    collator = LlmDataCollator(tokenizer=tokenizer, max_length=256)

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=True,
        collate_fn=collator
    )

    infer = ONNXInference(
        onnx_model_path=args.onnx_model,
        test_loader=test_loader,
        output_file=args.output_file,
        id2label=id2label
    )

    start_time = time.time()
    infer.run()
    finish_time = time.time() - start_time
    print(f"Finish process in: {finish_time:.2f} sec")
