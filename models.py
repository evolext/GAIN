import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(in_features=2*input_size, out_features=input_size)
        self.fc2 = nn.Linear(in_features=input_size, out_features=input_size)
        self.fc3 = nn.Linear(in_features=input_size, out_features=input_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(in_features=2*input_size, out_features=input_size)
        self.fc2 = nn.Linear(in_features=input_size, out_features=input_size)
        self.fc3 = nn.Linear(in_features=input_size, out_features=input_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
