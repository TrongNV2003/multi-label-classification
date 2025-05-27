import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import time
import random
import argparse
import numpy as np
from collections import Counter

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from multi_intent_classification.services.trainer import TrainingArguments
from multi_intent_classification.services.evaluate import TestingArguments
from multi_intent_classification.services.dataloader import Dataset, DataCollator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def get_vram_usage(device):
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)

def count_parameters(model: torch.nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

def get_unique_labels(input_file: str) -> list:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    labels_count = Counter()
    for record in data:
        for label in record["label"]:
            labels_count.update([label])
    outputs_labels_list = [label for label, _ in labels_count.most_common()]

    print(f"\nDanh sách nhãn: {outputs_labels_list}")
    print(f"Tổng số lượng nhãn: {len(outputs_labels_list)}")
    return outputs_labels_list

parser = argparse.ArgumentParser()
parser.add_argument("--dataloader_workers", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=10, required=True)
parser.add_argument("--learning_rate", type=float, default=3e-5, required=True)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_steps", type=int, default=50)
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--pad_mask_id", type=int, default=-100)
parser.add_argument("--model", type=str, default="vinai/phobert-base-v2", required=True)
parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
parser.add_argument("--train_batch_size", type=int, default=16, required=True)
parser.add_argument("--val_batch_size", type=int, default=16, required=True)
parser.add_argument("--test_batch_size", type=int, default=16, required=True)
parser.add_argument("--train_file", type=str, default="dataset/train.json", required=True)
parser.add_argument("--val_file", type=str, default="dataset/val.json", required=True)
parser.add_argument("--test_file", type=str, default="dataset/test.json", required=True)
parser.add_argument("--output_dir", type=str, default="./models", required=True)
parser.add_argument("--record_output_file", type=str, default="output.json")
parser.add_argument("--early_stopping_patience", type=int, default=5, required=True)
parser.add_argument("--early_stopping_threshold", type=float, default=0.001)
parser.add_argument("--evaluate_on_accuracy", action="store_true", default=False)
parser.add_argument("--is_multi_label", action="store_true", default=False)

parser.add_argument("--use_focal_loss", action="store_true", default=False)
parser.add_argument("--focal_loss_gamma", type=float, default=2.0)
parser.add_argument("--focal_loss_alpha", type=float, default=0.25)

parser.add_argument("--use_lora", action="store_true", default=False, help="Whether to use LoRA for fine-tuning")
parser.add_argument("--lora_rank", type=int, default=16, help="Rank for LoRA adaptation")
parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha parameter for LoRA")
parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout probability for LoRA layers")
parser.add_argument("--lora_target_modules", type=str, default=None,
                    help="Comma-separated list of target modules for LoRA. If None, defaults to q_proj,v_proj")
args = parser.parse_args()

def get_tokenizer(checkpoint: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<history>', '</history>', '<current>', '</current>']}
    )
    return tokenizer

def get_model(
        checkpoint: str,
        device: str,
        tokenizer: AutoTokenizer,
        num_labels: int,
        id2label: dict,
        label2id: dict,
        resize_token_embeddings: bool = True,
    ) -> AutoModelForSequenceClassification:
    config = AutoConfig.from_pretrained(
        checkpoint,
        problem_type="multi_label_classification" if args.is_multi_label else "single_label_classification",
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        config=config
    )
    if resize_token_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model


if __name__ == "__main__":
    set_seed(args.seed)

    unique_labels = get_unique_labels(args.train_file)

    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for idx, label in enumerate(unique_labels)}

    tokenizer = get_tokenizer(args.model)
    
    train_set = Dataset(json_file=args.train_file, label_mapping=label2id, tokenizer=tokenizer, is_multi_label=args.is_multi_label)
    val_set = Dataset(json_file=args.val_file, label_mapping=label2id, tokenizer=tokenizer, is_multi_label=args.is_multi_label)
    test_set = Dataset(json_file=args.test_file, label_mapping=label2id, tokenizer=tokenizer, is_multi_label=args.is_multi_label)

    collator = DataCollator(tokenizer=tokenizer, max_length=args.max_length, is_multi_label=args.is_multi_label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    
    model = get_model(
        args.model, device, tokenizer, num_labels=len(unique_labels), label2id=label2id, id2label=id2label
    )
    
    print(f"\nLabel: {model.config.id2label}")
    print(f"\nEval_on_accuracy: {args.evaluate_on_accuracy}")
    count_parameters(model)

    model_name = args.model.split('/')[-1]
    if args.use_lora:
        save_dir = f"{args.output_dir}/{model_name}-lora"
    else:
        save_dir = f"{args.output_dir}/{model_name}"

    start_time = time.time()
    trainer = TrainingArguments(
        dataloader_workers=args.dataloader_workers,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        model=model,
        pin_memory=args.pin_memory,
        save_dir=save_dir,
        tokenizer=tokenizer,
        train_set=train_set,
        valid_set=val_set,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.val_batch_size,
        collator_fn=collator,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        evaluate_on_accuracy=args.evaluate_on_accuracy,
        is_multi_label=args.is_multi_label,
        use_focal_loss=args.use_focal_loss,
        focal_loss_alpha=args.focal_loss_alpha,
        focal_loss_gamma=args.focal_loss_gamma,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules.split(',') if args.lora_target_modules else None,
    )
    trainer.train()
    
    end_time = time.time()
    print(f"Training time: {(end_time - start_time) / 60} mins")

    if torch.cuda.is_available():
        max_vram = get_vram_usage(device)
        print(f"VRAM tối đa tiêu tốn khi huấn luyện: {max_vram:.2f} GB")


    # Evaluate model
    if args.use_lora:
        from peft import PeftModel
        base_model = get_model(
            args.model,
            device,
            tokenizer,
            num_labels=len(unique_labels),
            label2id=label2id,
            id2label=id2label,
        )
        tuned_model = PeftModel.from_pretrained(base_model, save_dir)
    else:
        tuned_model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    tester = TestingArguments(
        dataloader_workers=args.dataloader_workers,
        device=device,
        model=tuned_model,
        pin_memory=args.pin_memory,
        test_set=test_set,
        test_batch_size=args.test_batch_size,
        collate_fn=collator,
        output_file=args.record_output_file,
        is_multi_label=args.is_multi_label,
    )
    tester.evaluate()

    print(f"\nmodel: {args.model}")
    print(f"params: lr {args.learning_rate}, epoch {args.epochs}")

