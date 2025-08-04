# https://github.com/qubvel/segmentation_models.pytorch?tab=readme-ov-file
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb

import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
from typing import Optional
from torch.nn.modules.loss import _Loss
from sklearn.metrics import precision_score, recall_score, f1_score

class BaseModel(nn.Module):
    def __init__(self, cfg, run_type="train"):
        super().__init__()    
        if run_type not in cfg.MODEL.RUN_TYPES: 
            raise ValueError("Invalid run type. Expected one of: %s" % cfg.MODEL.RUN_TYPES)
        
        self.run_type = run_type
        self.cfg = cfg
        
    def forward(self, x):
        return x
        
    def calculate_metrics(self, targets, preds):    
        flat_mask = targets.squeeze().flatten().detach().cpu().numpy()    
        
        flat_pred = preds.squeeze().flatten().detach().cpu().numpy()
        flat_pred = np.where(flat_pred > self.cfg.DATASETS.BINARY_TRESHOLD, 1, 0).astype(float)
            
        recall = recall_score(flat_mask, flat_pred)
        precision = precision_score(flat_mask, flat_pred)
        f1 = f1_score(flat_mask, flat_pred)
                        
        return recall, precision, f1

# slightly dfferent implementation beacuase of bug in source code
class FocalLoss(_Loss):
    def __init__(
        self,
        mode: str,
        alpha: Optional[float] = None,
        gamma: Optional[float] = 2.0,
        ignore_index: Optional[int] = None,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.mode = mode
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            self.focal_loss,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )
        
    def focal_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        reduction: str = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        
        target = target.type(output.type())

        logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
        pt = torch.exp(-logpt)

        # compute the loss
        if reduced_threshold is None:
            focal_term = (1.0 - pt).pow(gamma)
        else:
            focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
            focal_term[pt < reduced_threshold] = 1

        loss = focal_term * logpt

        if alpha is not None:
            loss *= alpha * target + (1 - alpha) * (1 - target)

        if normalized:
            norm_factor = focal_term.sum().clamp_min(eps)
            loss /= norm_factor

        if reduction == "mean":
            loss = loss.mean()
        if reduction == "sum":
            loss = loss.sum()
        if reduction == "batchwise_mean":
            loss = loss.sum(0)

        return loss

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:        
        y_true = y_true.squeeze().flatten()
        y_pred = y_pred.squeeze().flatten()
    
        if self.ignore_index is not None:
            not_ignored = y_true != self.ignore_index
            y_pred = y_pred[not_ignored]
            y_true = y_true[not_ignored]

        loss = self.focal_loss_fn(y_pred, y_true)

        return loss
