from loguru import logger
from typing import Optional
from transformers import Trainer

from multi_intent_classification.trainer_hf.loss import LossFunctionFactory

class HFTrainer(Trainer):
    def __init__(self, *args, is_multi_label: bool, id2label = dict, alpha: Optional[float] = None, gamma: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_multi_label = is_multi_label
        self.id2label = id2label
        self.alpha = alpha
        self.gamma = gamma
        if self.is_multi_label:
            if self.alpha is not None and self.gamma is not None:
                logger.info(f"Using Focal Loss with gamma={self.gamma}, alpha={self.alpha} for multi-label classification")
                self.loss_fn = LossFunctionFactory.get_loss("focal_multi", num_labels=len(self.id2label), gamma=self.gamma, alpha=self.alpha)
            else:
                logger.info("Using Binary Cross Entropy Loss for multi-label classification")
                self.loss_fn = LossFunctionFactory.get_loss("bce")
        else:
            logger.info("Using CrossEntropyLoss for single-label classification")
            self.loss_fn = LossFunctionFactory.get_loss("ce")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        if self.is_multi_label:
            loss = self.loss_fn(logits, labels.float())
        else:
            loss = self.loss_fn(logits, labels.long())
        return (loss, outputs) if return_outputs else loss
