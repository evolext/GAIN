import numpy as np
import torch
from dataclasses import dataclass, field


def get_mask(x: torch.Tensor) -> torch.Tensor:
    """
    :param x:
    :return: 0 indicating missing values, 1 indicating not missing values
    """
    return torch.logical_not(torch.isnan(x)).long().float()


@dataclass
class TrainHistory:
    G_loss: list = field(default_factory=list)
    D_loss: list = field(default_factory=list)
    MSE_train: list = field(default_factory=list)
    MSE_test: list = field(default_factory=list)


class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, loss: float) -> None:
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        elif loss > (self.best_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
