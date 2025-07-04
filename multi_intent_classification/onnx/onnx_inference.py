import os
import time
import argparse
from loguru import logger

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from optimum.onnxruntime import ORTModelForSequenceClassification

from multi_intent_classification.onnx.onnx_converter import OnnxConverter
from multi_intent_classification.services.dataloader import Dataset, DataCollator

from multi_intent_classification.utils.get_labels import get_unique_labels

class ONNXInference:
    def __init__(
            self,
            test_loader: DataLoader,
            device: str,
            model: ORTModelForSequenceClassification,
            id2label: dict,
            output_file: str,
    ) -> None:
        self.test_loader = test_loader
        self.device = device
        self.model = model
        self.id2label = id2label
        self.output_file = output_file
        

    def run(self):
        for batch_idx, batch in enumerate(self.test_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            pred_indices = preds[0].nonzero(as_tuple=True)[0].tolist()
            predicted_label_names = [self.id2label[idx] for idx in pred_indices] or ["UNKNOWN|UNKNOWN"]

            confidence = torch.softmax(logits, dim=-1)[0].max().item()

            log_message = (
                f"Query {batch_idx + 1}: "
                f"Labels: {predicted_label_names} | "
                f"Confidence: {confidence:.4f}"
            )
            logger.info(log_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vinai/phobert-base-v2", help="Torch model", required=True)
    parser.add_argument("--output_dir", type=str, default="./models", help="Output directory to save ONNX model")
    parser.add_argument("--train_set", type=str, default="dataset/train.json", help="Train dataset file", required=True)
    parser.add_argument("--test_set", type=str, default="dataset/test.json", help="Test dataset file", required=True)
    parser.add_argument("--is_multi_label", action="store_true", help="Whether the task is multi-label classification")
    parser.add_argument("--output_file", type=str, default="output.log", help="Output log file")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Batch size for testing")
    args = parser.parse_args()

    model_name = args.model.split('/')[-1]  # phobert-base-v2
    torch_model_dir = f"{args.output_dir}/{model_name}"   #./models/phobert-base-v2
    onnx_model_dir = f"{args.output_dir}/{model_name}_onnx" # ./models/phobert-base-v2_onnx
    if not os.path.exists(onnx_model_dir):
        OnnxConverter.convert_to_onnx(model_name=torch_model_dir, save_dir=onnx_model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    providers = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
    model = ORTModelForSequenceClassification.from_pretrained(
        onnx_model_dir, provider=providers, use_io_binding=True, file_name="model_optimized.onnx"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    unique_labels = get_unique_labels(args.train_set)
    id2label = {idx: label for idx, label in enumerate(unique_labels)}
    label2id = {label: idx for idx, label in enumerate(unique_labels)}

    test_set = Dataset(
        json_file=args.test_set,
        label2id=label2id,
        tokenizer=tokenizer,
        is_multi_label=args.is_multi_label,
    )

    collator = DataCollator(tokenizer=tokenizer, max_length=args.max_length)

    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=collator
    )

    infer = ONNXInference(
        test_loader=test_loader,
        model=model,
        id2label=id2label,
        device=device,
        output_file=args.output_file,
    )

    start_time = time.time()
    infer.run()
    finish_time = time.time() - start_time
    print(f"Finish process in: {finish_time:.2f} sec")
