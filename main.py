import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset

from sklearn.preprocessing import MinMaxScaler
from GAIN import GAIN
import matplotlib.pyplot as plt


def add_missings(x: np.array, miss_rate: float) -> np.array:
    x_missed = np.copy(x)
    n, m = x.shape

    for j in range(m):
        nan_indexes = np.random.choice(n, int(n * miss_rate), replace=False)
        x_missed[nan_indexes, j] = np.nan

    return x_missed


def get_mask(x: torch.Tensor) -> torch.Tensor:
    """
    :param x:
    :return: 0 indicating missing values, 1 indicating not missing values
    """
    return torch.logical_not(torch.isnan(x)).long().float()


def sample_z(shape: list[int]) -> torch.tensor:
    return torch.tensor(np.random.uniform(low=0, high=0.01, size=shape)).float()


def sample_b(shape: list[int], p: float) -> torch.Tensor:
    b = np.random.uniform(low=0, high=1, size=shape)
    return torch.tensor(b > p).float()


def distriminator_step(m, x_new, h, generator, discriminator):
    x_imputed = generator(torch.cat(tensors=[x_new, m], dim=1))
    x_hat = m * x_new + (1 - m) * x_imputed
    m_hat = discriminator(torch.cat(tensors=[x_hat, h], dim=1))
    loss = -torch.mean(m * torch.log(m_hat + 1e-8) + (1 - m) * torch.log(1 - m_hat + 1e-8))
    return loss


def generator_step(x_old, x_new, m, h, generator, discriminator, alpha):
    x_imputed = generator(torch.cat(tensors=[x_new, m], dim=1))
    x_hat = m * x_new + (1 - m) * x_imputed
    m_hat = discriminator(torch.cat(tensors=[x_hat, h], dim=1))

    mse_train = torch.mean(torch.pow(m * x_new - m * x_imputed, 2)) / torch.mean(m)
    loss = -torch.mean((1 - m) * torch.log(m_hat + 1e-8)) + alpha * mse_train
    mse_test = torch.mean(torch.pow((1 - m) * x_old - (1 - m) * x_imputed, 2)) / torch.mean(m)
    return loss, mse_train, mse_test


if __name__ == '__main__':

    seed = 1
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = pd.read_csv('./data/Spam.csv').values
    data_missed = add_missings(data, miss_rate=0.2)

    # Normalization
    scaler = MinMaxScaler()
    data_std = scaler.fit_transform(data_missed)

    # col = np.arange(len(data)).reshape(-1, 1)
    # data_std = np.append(data_std, col, axis=1)

    # train\test split
    train_cutoff = int(data_std.shape[0] * 0.8)
    X_train, X_test = data_std[:train_cutoff], data_std[train_cutoff:]

    X_train_tensor = torch.tensor(X_train).float()
    M_train_tensor = get_mask(X_train_tensor)
    train_dataset = TensorDataset(X_train_tensor, M_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)

    model = GAIN(train_loader=train_loader)

    optimizer_G = torch.optim.Adam(model.G.parameters())
    optimizer_D = torch.optim.Adam(model.D.parameters())
    model.set_optimizer(optimizer=optimizer_G, generator=True)
    model.set_optimizer(optimizer=optimizer_D, generator=False)

    model.train(n_epoches=20, verbose=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.plot(model.history['RMSE_train'], label='train')
    ax.plot(model.history['RMSE_test'], label='test')

    ax.legend()

    plt.show()


