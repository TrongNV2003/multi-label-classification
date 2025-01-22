import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from training.utils import AverageMeter
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LlmTrainer:
    def __init__(
        self,
        dataloader_workers: int,
        device: str,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        pin_memory: bool,
        save_dir: str,
        train_batch_size: int,
        train_set: Dataset,
        valid_batch_size: int,
        valid_set: Dataset,
        collator_fn=None,
        evaluate_on_accuracy: bool = False,
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

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
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        # self.optimizer = AdamW(
        #     self.model.parameters(),
        #     lr=learning_rate,
        #     weight_decay=weight_decay,
        # )

        # Không áp dụng weight decay cho bias và LayerNorm.weight
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

        self.train_loss = AverageMeter()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=2, factor=0.5)
        self.evaluate_on_accuracy = evaluate_on_accuracy
        if evaluate_on_accuracy:
            self.best_valid_score = 0
        else:
            self.best_valid_score = float("inf")

    def train(self) -> None:
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.train_loss.reset()

            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for data in self.train_loader:
                    text_input_ids = data["input_ids"].to(self.device)
                    text_attention_mask = data["attention_mask"].to(self.device)
                    labels = data["labels"].to(self.device)

                    outputs = self.model(
                        input_ids=text_input_ids,
                        attention_mask=text_attention_mask,
                    )
                    logits = outputs.logits
                    loss = self.loss_fn(logits, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.train_loss.update(loss.item(), text_input_ids.size(0))
                    tepoch.set_postfix({"train_loss": self.train_loss.avg})
                    tepoch.update(1)

            valid_loss = self.evaluate(self.valid_loader)
            self.scheduler.step(valid_loss)

            if valid_loss < self.best_valid_score:
                print(
                    f"Validation loss decreased from {self.best_valid_score:.4f} to {valid_loss:.4f}. Saving."
                )
                self.best_valid_score = valid_loss
                self._save()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        eval_loss = AverageMeter()
        with tqdm(total=len(dataloader), unit="batches") as tepoch:
            tepoch.set_description("validation")
            for data in dataloader:
                text_input_ids = data["input_ids"].to(self.device)
                text_attention_mask = data["attention_mask"].to(self.device)
                labels = data["labels"].to(self.device)

                outputs = self.model(
                    input_ids=text_input_ids,
                    attention_mask=text_attention_mask,
                )
                logits = outputs.logits
                loss = self.loss_fn(logits, labels)
                
                eval_loss.update(loss.item(), text_input_ids.size(0))
                tepoch.set_postfix({"valid_loss": eval_loss.avg})
                tepoch.update(1)
        return eval_loss.avg

    def _save(self) -> None:
        self.tokenizer.save_pretrained(self.save_dir)
        self.model.save_pretrained(self.save_dir)
