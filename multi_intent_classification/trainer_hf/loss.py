import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits, targets, weight=self.weight, reduction=self.reduction
        )


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
            
        return F.cross_entropy(
            logits, targets, weight=self.class_weights, reduction=self.reduction
        )


class FocalLossForSingleLabel(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLossForSingleLabel, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)

        labels_one_hot = F.one_hot(labels, num_classes=logits.size(1))
        p_t = (probs * labels_one_hot).sum(dim=1)
        modulating_factor = (1.0 - p_t) ** self.gamma

        alpha_t = torch.where(labels_one_hot == 1, self.alpha, 1 - self.alpha).sum(dim=1)

        ce_loss = -torch.log(p_t + 1e-8)

        focal_loss = alpha_t * modulating_factor * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, weight=self.weight, reduction=self.reduction
        )


class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights, reduction='mean'):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)

        sample_weights = self.class_weights.unsqueeze(0)  # Shape (1, num_classes)

        loss = F.binary_cross_entropy_with_logits(
            inputs, targets, weight=sample_weights, reduction=self.reduction
        )
        return loss


class FocalLossForMultiLabel(nn.Module):
    def __init__(self, num_labels: int, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLossForMultiLabel, self).__init__()
        self.num_labels = num_labels
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        l = logits.reshape(-1)
        t = targets.reshape(-1)

        p = torch.sigmoid(l)
        p = torch.where(t >= 0.5, p, 1 - p)
        logp = -torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))
        loss = self.alpha * ((1 - p) ** self.gamma) * logp
        loss = self.num_labels * loss.mean()
        return loss


class LossFunctionFactory:
    @staticmethod
    def get_loss(loss_name, **kwargs):
        loss_dict = {
            "ce": CrossEntropyLoss,
            "weighted_ce": WeightedCrossEntropyLoss,
            "focal_single": FocalLossForSingleLabel,
            
            "bce": BinaryCrossEntropyLoss,
            "weighted_bce": WeightedBinaryCrossEntropyLoss,
            "focal_multi": FocalLossForMultiLabel,
        }
        
        if loss_name not in loss_dict:
            raise ValueError(f"Loss function '{loss_name}' not found. Available losses: {list(loss_dict.keys())}")
        
        return loss_dict[loss_name](**kwargs)