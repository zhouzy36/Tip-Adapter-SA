# coding=utf-8
# Implementation of some commonly used single positive multi-label learning loss functions
import torch
import torch.nn as nn
from torch import Tensor


class IULoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        """Ignore unobserved (IU) loss: set the loss terms corresponding to unobserved labels to zero.
        Args:
            reduction (str): Specifies the reduction to apply to the output (default: "mean").
        """
        super(IULoss, self).__init__()
        assert reduction in ["sum", "mean"]
        self.reduction = reduction
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction="none")
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape == target.shape
        positive_loss = self.BCEWithLogitsLoss(input[target == 1], target[target == 1])
        loss_sum = torch.sum(positive_loss)
        if self.reduction == "sum":
            return loss_sum
        else:
            return loss_sum / torch.numel(input)



class ANLoss(nn.Module):
    def __init__(self, epsilon: float = 0.0, reduction: str = "mean"):
        """Assume negative (AN) loss: assume unobserved labels are negative.
        Args:
            reduction (str): Specifies the reduction to apply to the output (default: "mean").
            epsilon (float): Hyperparameters for label smoothing (default: 0.).
        """
        super(ANLoss, self).__init__()
        assert reduction in ["sum", "mean"]
        assert epsilon >= 0. and epsilon <= 1.
        self.reduction = reduction
        self.epsilon = epsilon
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction="none")
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape == target.shape
        smoothed_target = torch.where(target == 1, 1 - self.epsilon, self.epsilon).to(target.device)
        positive_loss = self.BCEWithLogitsLoss(input[target == 1], smoothed_target[target == 1])
        negative_loss = self.BCEWithLogitsLoss(-input[target != 1], 1 - smoothed_target[target != 1])
        loss_sum = torch.sum(positive_loss) + torch.sum(negative_loss)
        if self.reduction == "sum":
            return loss_sum
        else:
            return loss_sum / torch.numel(input)
    


class WANLoss(nn.Module):
    def __init__(self, gamma: float, reduction: str = "mean"):
        """Weak assume negative (WAN) loss: Assume unobserved labels are negative and down-weight terms in the loss corresponding to negative labels.
        Args:
            gamma (float): The weight of negative item.
            reduction (str): Specifies the reduction to apply to the output (default: "mean").
        """
        super(WANLoss, self).__init__()
        assert gamma >= 0 and gamma <= 1
        assert reduction in ["sum", "mean"]
        self.gamma = gamma
        self.reduction = reduction
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction="none")
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape == target.shape
        positive_loss = self.BCEWithLogitsLoss(input[target == 1], target[target == 1])
        negative_loss = self.gamma * self.BCEWithLogitsLoss(-input[target != 1], 1 - target[target != 1])
        loss_sum = torch.sum(positive_loss) + torch.sum(negative_loss)
        if self.reduction == "sum":
            return loss_sum
        else:
            return loss_sum / torch.numel(input)



class EMLoss(nn.Module):
    def __init__(self, alpha: float, reduction: str = "mean"):
        """Entropy-Maximization (EM) Loss
        Args:
            alpha (float): The hyperparameter to down-weight the strength of entropy maximization.
            reduction (str): Specifies the reduction to apply to the output (default: "mean").
        """
        super(EMLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape == target.shape
        positive_loss = self.BCEWithLogitsLoss(input[target == 1], target[target == 1])
        unknown_loss = -self.alpha * self.entropy(F.sigmoid(input[target == 0]))
        loss_sum = torch.sum(positive_loss) + torch.sum(unknown_loss)
        if self.reduction == "sum":
            return loss_sum
        else:
            return loss_sum / torch.numel(input)

    def entropy(self, p: Tensor):
        return -(p * self.log(p) + (1 - p) * self.log(1 - p))
    
    def log(self, x: Tensor) -> Tensor:
        return torch.clamp(torch.log(x), min=-100)