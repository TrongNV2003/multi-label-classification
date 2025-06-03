import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, get_scheduler

import numpy as np
from tqdm import tqdm
from loguru import logger
from typing import Optional, Callable
from sklearn.metrics import precision_score, recall_score, f1_score

from multi_intent_classification.utils.utils import AverageMeter
from multi_intent_classification.services.loss import LossFunctionFactory

class TrainingArguments:
    def __init__(
        self,
        dataloader_workers: int,
        device: str,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        pin_memory: bool,
        save_dir: str,
        train_batch_size: int,
        train_set: Dataset,
        valid_batch_size: int,
        valid_set: Dataset,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.001,
        evaluate_on_accuracy: bool = True,
        is_multi_label: bool = False,
        collator_fn: Optional[Callable] = None,
        use_focal_loss: bool = False,
        focal_loss_gamma: float = 2.0,
        focal_loss_alpha: float = 0.25,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: list = None,
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.is_multi_label = is_multi_label
        
        self.use_focal_loss = use_focal_loss
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha

        
        self.train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=collator_fn,
        )
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=valid_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=collator_fn,
        )
        self.use_lora = use_lora

        if self.use_lora:
            lora_target_modules = ["query", "value"] if lora_target_modules is None else lora_target_modules
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="SEQ_CLS",
                use_rslora=False
            )
            self.model = get_peft_model(model, lora_config)
            self.model.to(self.device)

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
            logger.info("Using LoRA for model training.")

            train_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Training {train_params:,d} parameters out of {all_params:,d} parameters ({train_params/all_params:.2%})")
        else:
            self.model = model.to(self.device)

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        self.loss_factory = LossFunctionFactory()
        if self.is_multi_label:
            if self.use_focal_loss:
                self.loss_fn = self.loss_factory.get_loss("focal_multi", num_labels=len(self.model.config.id2label), gamma=self.focal_loss_gamma, alpha=self.focal_loss_alpha)
                logger.info(f"Using Focal Loss for multi-label classification with {len(self.model.config.id2label)} labels.")
            else:
                self.loss_fn = self.loss_factory.get_loss("bce")
                logger.info("Using BCE Loss for multi-label classification.")
        else:
            self.loss_fn = self.loss_factory.get_loss("ce")
            logger.info("Using CE Loss for single-label classification.")

        num_training_steps = len(self.train_loader) * epochs
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )

        self.evaluate_on_accuracy = evaluate_on_accuracy
        self.best_valid_score = 0 if evaluate_on_accuracy else float("inf")
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_counter = 0
        self.best_epoch = 0

    def train(self) -> None:
        print(f"Task type: {'Multi-label' if self.is_multi_label else 'Single-label'}")
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = AverageMeter()

            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for data in self.train_loader:
                    input_ids = data["input_ids"].to(self.device)
                    attention_mask = data["attention_mask"].to(self.device)
                    labels = data["labels"].to(self.device)

                    self.optimizer.zero_grad()
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = outputs.logits
                    
                    loss = self.loss_fn(logits, labels)
                    loss.backward()
                    
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()

                    train_loss.update(loss.item(), input_ids.size(0))
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    tepoch.set_postfix({"train_loss": train_loss.avg, "lr": current_lr})
                    tepoch.update(1)

            valid_score = self._validate(self.valid_loader)
            improved = False

            if self.evaluate_on_accuracy:
                if valid_score > self.best_valid_score + self.early_stopping_threshold:
                    print(f"Validation accuracy improved from {self.best_valid_score:.4f} to {valid_score:.4f}. Saving...")
                    self.best_valid_score = valid_score
                    self.best_epoch = epoch
                    self._save()
                    self.early_stopping_counter = 0
                    improved = True
                else:
                    self.early_stopping_counter += 1
                    print(f"No improvement in val accuracy. Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")

            else:
                if valid_score < self.best_valid_score - self.early_stopping_threshold:
                    print(f"Validation loss decreased from {self.best_valid_score:.4f} to {valid_score:.4f}. Saving...")
                    self.best_valid_score = valid_score
                    self.best_epoch = epoch
                    self._save()
                    self.early_stopping_counter = 0
                    improved = True
                else:
                    self.early_stopping_counter += 1
                    print(f"No improvement in validation loss. Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")

            if improved:
                print(f"Saved best model at epoch {self.best_epoch}.")
            
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement.")
                break

    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        eval_loss = AverageMeter()
        all_preds = []
        all_labels = []

        with tqdm(total=len(dataloader), unit="batches") as tepoch:
            tepoch.set_description("validation")
            for data in dataloader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                labels = data["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                loss = self.loss_fn(logits, labels)
                eval_loss.update(loss.item(), input_ids.size(0))

                if self.is_multi_label:
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float().cpu().numpy()
                else:
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

                tepoch.set_postfix({"valid_loss": eval_loss.avg})
                tepoch.update(1)

        all_preds = np.vstack(all_preds) if self.is_multi_label else np.concatenate(all_preds)
        all_labels = np.vstack(all_labels) if self.is_multi_label else np.concatenate(all_labels)

        if self.is_multi_label:
            accuracy = np.mean((all_preds == all_labels).all(axis=-1))
            self._metrics(all_preds, all_labels, average_type="micro")
        else:
            accuracy = np.mean(all_preds == all_labels)
            self._metrics(all_preds, all_labels, average_type="weighted")

        return accuracy if self.evaluate_on_accuracy else eval_loss.avg


    def _metrics(self, all_preds: np.ndarray, all_labels: np.ndarray, average_type: str) -> None:
        if self.is_multi_label:
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


    def _save(self) -> None:
        self.tokenizer.save_pretrained(self.save_dir)
        if self.use_lora:
            self.model.save_pretrained(self.save_dir, save_embedding_layers=True)   # save LoRA weights
        else:
            self.model.save_pretrained(self.save_dir)   # save full model