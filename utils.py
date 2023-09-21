"""
    Support classes and functions for the model
"""

from dataclasses import dataclass, field
import numpy as np
import torch


def get_mask(matrix: torch.Tensor) -> torch.Tensor:
    """
    Gets a indicator mask of missing values for the given matrix

    @param matrix: two-dimensional data tensor
    @return:       a tensor with 1.0 values for non-NaN values and 0.0 for NaN values according to matrix
    """
    return torch.logical_not(torch.isnan(matrix)).long().float()


@dataclass
class TrainHistory:
    """
        Class to store the training history
    """
    generator_loss: list = field(default_factory=list)
    discriminator_loss: list = field(default_factory=list)
    mse_train: list = field(default_factory=list)
    mse_test: list = field(default_factory=list)


class EarlyStopper:
    """
        Class to implement early stopping for GAIN model
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, loss: float) -> None:
        """
        Calls the early stopping mechanism

        @param loss: error value at the current epoch
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        elif loss > (self.best_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def reset(self) -> None:
        """
            Reset the early stopper
        """
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
