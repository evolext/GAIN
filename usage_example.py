import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from GAIN import GAIN
from utils import get_mask, EarlyStopper


def add_missings(x: np.array, miss_rate: float) -> np.array:
    x_missed = np.copy(x).astype(float)
    n, m = x.shape

    for j in range(m):
        nan_indexes = np.random.choice(n, int(n * miss_rate), replace=False)
        x_missed[nan_indexes, j] = np.nan

    return x_missed


if __name__ == '__main__':

    seed = 13
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stopper = EarlyStopper(patience=2, min_delta=0.001)

    data = pd.read_csv('./data/Letter.csv').values

    np.random.seed(seed)
    data_missed = add_missings(data, miss_rate=0.2)

    # Normalization
    scaler = MinMaxScaler()
    data_std = scaler.fit_transform(data_missed)

    # train\test split
    train_cutoff = int(data_std.shape[0] * 0.8)
    X_train, X_test = data_std[:train_cutoff], data_std[train_cutoff:]
    X_actual = data[train_cutoff:]

    X_train_tensor = torch.tensor(X_train).float()
    M_train_tensor = get_mask(X_train_tensor)
    train_dataset = TensorDataset(X_train_tensor, M_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)

    X_test_tensor = torch.tensor(X_test).float()
    M_test_tensor = get_mask(X_test_tensor)
    X_actual_tensor = torch.tensor(X_actual).float()
    test_dataset = TensorDataset(X_test_tensor, M_test_tensor, X_actual_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Model initialization and training
    model = GAIN(train_loader=train_loader, seed=seed)

    optimizer_G = torch.optim.Adam(model.G.parameters())
    optimizer_D = torch.optim.Adam(model.D.parameters())
    model.set_optimizer(optimizer=optimizer_G, generator=True)
    model.set_optimizer(optimizer=optimizer_D, generator=False)

    model.to(device)
    model.train(n_epoches=100, verbose=True, stopper=stopper)

    # Model evaluation
    rmse_batch = []

    for x_test_batch, m_batch, x_actual_batch in test_loader:
        x_batch_imputed = model.evaluation(x=x_test_batch, m=m_batch)
        x_batch_imputed = scaler.inverse_transform(x_batch_imputed.cpu().numpy())

        rmse = np.sqrt(mean_squared_error(y_true=x_actual_batch.numpy(), y_pred=x_batch_imputed))
        rmse_batch.append(rmse)

    print(np.mean(rmse_batch))

    # fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    # ax.plot(model.history.MSE_train, label='train')
    # ax.plot(model.history.MSE_test, label='test')
    # ax.legend()
    # plt.show()
