"""
    Description of the GAIN model generator and discriminator units
"""

import torch
from torch import nn


class Generator(nn.Module):
    """
        Generator model unit
    """
    def __init__(self, input_size: int) -> None:
        """
        @param input_size: number of features in the processed data set
        """
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(in_features=2 * input_size, out_features=input_size),
            nn.ReLU(),
            nn.Linear(in_features=input_size, out_features=input_size),
            nn.ReLU(),
            nn.Linear(in_features=input_size, out_features=input_size),
            nn.Sigmoid()
        )

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator model

        @param x_batch: processing data tensor
        """
        return self.main(x_batch)


class Discriminator(nn.Module):
    """
        Discriminator model unit
    """
    def __init__(self, input_size: int) -> None:
        """
        @param input_size: number of features in the processed data set
        """
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(in_features=2 * input_size, out_features=input_size),
            nn.ReLU(),
            nn.Linear(in_features=input_size, out_features=input_size),
            nn.ReLU(),
            nn.Linear(in_features=input_size, out_features=input_size),
            nn.Sigmoid()
        )

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator model

        @param x_batch: processing data tensor
        """
        return self.main(x_batch)
