import random
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from services.evaluate import Tester
from services.trainer import LlmTrainer
from services.dataloader import Dataset, LlmDataCollator

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument("--dataloader_workers", type=int, default=2, required=True)
parser.add_argument("--device", type=str, default="cuda", required=True)
parser.add_argument("--seed", type=int, default=42, required=True)
parser.add_argument("--epochs", type=int, default=10, required=True)
parser.add_argument("--learning_rate", type=float, default=3e-5, required=True)
parser.add_argument("--weight_decay", type=float, default=0.01, required=True)
parser.add_argument("--max_length", type=int, default=256, required=True)
parser.add_argument("--pad_mask_id", type=int, default=-100, required=True)
parser.add_argument("--model", type=str, default="vinai/phobert-base-v2", required=True)
parser.add_argument("--train_batch_size", type=int, default=16, required=True)
parser.add_argument("--valid_batch_size", type=int, default=8, required=True)
parser.add_argument("--test_batch_size", type=int, default=8, required=True)
parser.add_argument("--warmup_steps", type=int, default=50, required=True)
parser.add_argument("--train_file", type=str, default="dataset/train.json", required=True)
parser.add_argument("--valid_file", type=str, default="dataset/val.json", required=True)
parser.add_argument("--test_file", type=str, default="dataset/test.json", required=True)
parser.add_argument("--output_dir", type=str, default="./models/classification", required=True)
parser.add_argument("--record_output_file", type=str, default="output.json")
parser.add_argument("--evaluate_on_accuracy", type=bool, default=True, required=True)
parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
args = parser.parse_args()


def get_tokenizer(checkpoint: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    """
    Input format: <history>{history_1}<sep>{history_2}<sep>...<sep>{history_n}</history><current>{context}</current>
    """

    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<history>', '</history>', '<current>', '</current>']}
    )
    return tokenizer


def get_model(
    checkpoint: str, device: str, tokenizer: AutoTokenizer, num_labels: str, id2label: list, label2id: list
    ) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model

def count_parameters(model: torch.nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

if __name__ == "__main__":
    set_seed(args.seed)

    unique_labels = ["Cung cấp thông tin", "Tương tác", "Hỏi thông tin giao hàng", "Hỗ trợ, hướng dẫn", "Yêu cầu", "Phản hồi", "Sự vụ", "UNKNOWN"]
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for idx, label in enumerate(unique_labels)}

    tokenizer = get_tokenizer(args.model)
    
    train_set = Dataset(json_file=args.train_file, label_mapping=label2id, tokenizer=tokenizer)
    valid_set = Dataset(json_file=args.valid_file, label_mapping=label2id, tokenizer=tokenizer)
    test_set = Dataset(json_file=args.test_file, label_mapping=label2id, tokenizer=tokenizer)

    collator = LlmDataCollator(tokenizer=tokenizer, max_length=args.max_length)

    model = get_model(args.model, args.device, tokenizer, num_labels=len(unique_labels), label2id=label2id, id2label=id2label)
    print(model.config.id2label)
    count_parameters(model)

    model_name = args.model.split('/')[-1]
    save_dir = f"{args.output_dir}-{model_name}"

    trainer = LlmTrainer(
        dataloader_workers=args.dataloader_workers,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        model=model,
        pin_memory=args.pin_memory,
        save_dir=save_dir,
        tokenizer=tokenizer,
        train_set=train_set,
        valid_set=valid_set,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        collator_fn=collator,
        evaluate_on_accuracy=args.evaluate_on_accuracy,

    )
    trainer.train()

    # Test model on test set
    tuned_model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, collate_fn=collator)
    tester = Tester(model=tuned_model, test_loader=test_loader, output_file=args.record_output_file)
    tester.evaluate()

    print(f"\nmodel: {args.model}")
    print(f"params: lr {args.learning_rate}, epoch {args.epochs}")

