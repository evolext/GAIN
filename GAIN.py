import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import Generator, Discriminator


class GAIN:
    def __init__(self, train_loader: DataLoader, hint_rate: float = 0.1, alpha: float = 10) -> None:
        if hint_rate > 1 or hint_rate < 0:
            raise ValueError('@hint_rate must be between 0 and 1 inclusive')

        self.hint_rate = hint_rate
        self.alpha = alpha
        self.train_loader = train_loader

        x_data, m_data = iter(train_loader).next()
        if x_data.shape[1] != m_data.shape[1]:
            raise ValueError('@x_data and @m_data must have the same number of features')

        self.n_features = x_data.shape[1]

        self.G = Generator(self.n_features)
        self.D = Discriminator(self.n_features)

        self.optimizer_G = None
        self.optimizer_D = None
        self.history = {
            'G_loss': [],
            'D_loss': [],
            'RMSE_train': [],
            'RMSE_test': []
        }

    def set_optimizer(self, optimizer: torch.optim.Optimizer, generator: bool = True) -> None:
        if generator:
            self.optimizer_G = optimizer
        else:
            self.optimizer_D = optimizer

    def train(self, n_epoches: int, verbose=True) -> None:
        if self.optimizer_G is None or self.optimizer_D is None:
            return

        for epoch in range(n_epoches):
            D_mb = []
            G_mb = []
            RMSE_train_mb = []
            RMSE_test_mb = []

            t_epoch = tqdm(self.train_loader, unit='batch', disable=(not verbose))
            for X, M in t_epoch:
                t_epoch.set_description(f'Epoch {epoch}')
                t_epoch.refresh()

                X_old = torch.nan_to_num(X, nan=0)

                Z = self._sample_z(shape=M.shape)
                B = self._sample_b(shape=M.shape, p=self.hint_rate)
                H = B * M + 0.5 * (1 - B)

                X_new = M * X_old + (1 - M) * Z

                # 1-й этап: оптимизация дискриминатора при фиксированном генераторе
                D_loss = self._discriminator_step(x_new=X_new, m=M, h=H)
                D_loss.backward()
                self.optimizer_D.step()
                self.optimizer_D.zero_grad()
                D_mb.append(D_loss.item())

                # 2-й этап: оптимизация генератора при фиксированном дискриминаторе
                G_loss, MSE_train, MSE_test = self._generator_step(x_old=X_old, x_new=X_new, m=M, h=H)
                G_loss.backward()
                self.optimizer_G.step()
                self.optimizer_G.zero_grad()

                RMSE_train_mb.append(np.sqrt(MSE_train.item()))
                RMSE_test_mb.append(np.sqrt(MSE_test.item()))
                G_mb.append(G_loss.item())

                t_epoch.set_postfix(rmse_train=np.mean(RMSE_train_mb), rmse_test=np.mean(RMSE_test_mb))

            D_loss_epoch = np.mean(D_mb)
            G_loss_epoch = np.mean(G_mb)
            RMSE_train_epoch = np.mean(RMSE_train_mb)
            RMSE_test_epoch = np.mean(RMSE_test_mb)

            self.history['D_loss'].append(D_loss_epoch)
            self.history['G_loss'].append(G_loss_epoch)
            self.history['RMSE_train'].append(RMSE_train_epoch)
            self.history['RMSE_test'].append(RMSE_test_epoch)

    def _discriminator_step(self, x_new, m, h):
        x_imputed = self.G(torch.cat(tensors=[x_new, m], dim=1))
        x_hat = m * x_new + (1 - m) * x_imputed
        m_hat = self.D(torch.cat(tensors=[x_hat, h], dim=1))
        loss = -torch.mean(m * torch.log(m_hat + 1e-8) + (1 - m) * torch.log(1 - m_hat + 1e-8))
        return loss

    def _generator_step(self, x_old, x_new, m, h):
        x_imputed = self.G(torch.cat(tensors=[x_new, m], dim=1))
        x_hat = m * x_new + (1 - m) * x_imputed
        m_hat = self.D(torch.cat(tensors=[x_hat, h], dim=1))

        absolute_error_train = m * x_new - m * x_imputed
        mse_train = torch.mean(absolute_error_train * absolute_error_train) / torch.mean(m)
        loss = -torch.mean((1 - m) * torch.log(m_hat + 1e-8)) + self.alpha * mse_train

        m_inv = 1 - m
        absolute_error_test = m_inv * x_old - m_inv * x_imputed
        mse_test = torch.mean(absolute_error_test * absolute_error_test) / torch.mean(m_inv)

        return loss, mse_train, mse_test

    @staticmethod
    def _sample_z(shape: list[int]) -> torch.tensor:
        return torch.tensor(np.random.uniform(low=0, high=0.01, size=shape)).float()

    @staticmethod
    def _sample_b(shape: list[int], p: float) -> torch.Tensor:
        b = np.random.uniform(low=0, high=1, size=shape)
        return torch.tensor(b > p).float()
 