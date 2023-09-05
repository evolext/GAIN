import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(in_features=2 * input_size, out_features=input_size),
            nn.ReLU(),
            nn.Linear(in_features=input_size, out_features=input_size),
            nn.ReLU(),
            nn.Linear(in_features=input_size, out_features=input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(in_features=2 * input_size, out_features=input_size),
            nn.ReLU(),
            nn.Linear(in_features=input_size, out_features=input_size),
            nn.ReLU(),
            nn.Linear(in_features=input_size, out_features=input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
