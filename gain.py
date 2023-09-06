"""
    Implementation of the GAIN model of missing data imputation
"""

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from models import Generator, Discriminator
from utils import TrainHistory, EarlyStopper


class GAIN:
    """
        GAIN model for missing data imputation
    """
    def __init__(self,
                 train_loader: DataLoader,
                 hint_rate: float = 0.1,
                 alpha: float = 10,
                 device: str = 'cpu',
                 seed: int = None
                 ) -> None:
        """
        @param train_loader: torch dataloader for data and mask matrices
        @param hint_rate:    the proportion of missing values in the hint matrix (from 0 to 1))
        @param alpha:        penalty coefficient
        @param device:       device for training
        @param seed:         value for reproducibility of train
        """
        if hint_rate > 1 or hint_rate < 0:
            raise ValueError('@hint_rate must be between 0 and 1 inclusive')

        self.hint_rate = hint_rate
        self.alpha = alpha
        self.device = device
        self.train_loader = train_loader

        x_data, m_data = iter(train_loader).next()
        if x_data.shape[1] != m_data.shape[1]:
            raise ValueError('@x_data and @m_data must have the same number of features')

        if seed is not None:
            self._set_seed(seed)

        n_features = x_data.shape[1]
        self.G = Generator(n_features).to(self.device)
        self.D = Discriminator(n_features).to(self.device)

        self.optimizer_G = None
        self.optimizer_D = None
        self.history = TrainHistory()

    def set_optimizer(self, optimizer: Optimizer, generator: bool = True) -> None:
        """
        Set the optimizer for the generator and discriminator

        @param optimizer: torch optimizer object
        @param generator: if True, set the generator optimizer, else set the discriminator optimizer
        """
        if generator:
            self.optimizer_G = optimizer
        else:
            self.optimizer_D = optimizer

    def to(self, device: str) -> None:
        """
        Set the device for GAIN model

        @param device: 'cpu' or 'cuda'
        """
        self.device = device
        self.G.to(self.device)
        self.D.to(self.device)

    def train(self, n_epoches: int, stopper: EarlyStopper = None,  verbose=True) -> None:
        """
        Train the GAIN model

        @param n_epoches: maximum number of epochs for training
        @param stopper:   EarlyStopper object for control the duration of training
        @param verbose:   print information in the learning process if True
        """
        if self.optimizer_G is None or self.optimizer_D is None:
            return

        self.G.train()
        self.D.train()

        for epoch in range(n_epoches):
            history_epoch = TrainHistory()

            t = tqdm(self.train_loader, unit='batch', disable=not verbose)
            for X_batch, M_batch in t:
                t.set_description(f'Epoch {epoch}')
                t.refresh()

                M = M_batch.to(self.device)
                X_old = torch.nan_to_num(X_batch, nan=0).to(self.device)

                Z = self._sample_z(shape=M.shape).to(self.device)
                B = self._sample_b(shape=M.shape, p=self.hint_rate).to(self.device)
                H = B * M + 0.5 * (1 - B)

                X_new = M * X_old + (1 - M) * Z

                # 1-й этап: оптимизация дискриминатора при фиксированном генераторе
                D_loss = self._discriminator_step(x_new=X_new, m=M, h=H)
                D_loss.backward()
                self.optimizer_D.step()
                self.optimizer_D.zero_grad()
                history_epoch.discriminator_loss.append(D_loss.item())

                # 2-й этап: оптимизация генератора при фиксированном дискриминаторе
                G_loss, MSE_train, MSE_test = self._generator_step(x_old=X_old, x_new=X_new, m=M, h=H)
                G_loss.backward()
                self.optimizer_G.step()
                self.optimizer_G.zero_grad()

                history_epoch.mse_train.append(MSE_train.item())
                history_epoch.mse_test.append(MSE_test.item())
                history_epoch.generator_loss.append(G_loss.item())

                t.set_postfix(mse_train=np.mean(history_epoch.mse_train), mse_test=np.mean(history_epoch.mse_test))

            self.history.generator_loss.append(np.mean(history_epoch.generator_loss))
            self.history.discriminator_loss.append(np.mean(history_epoch.discriminator_loss))
            self.history.mse_train.append(np.mean(history_epoch.mse_train))
            self.history.mse_test.append(np.mean(history_epoch.mse_test))

            if stopper is not None:
                stopper(loss=self.history.mse_test[-1])
                if stopper.early_stop:
                    break

    def imputation(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Imputes data using a generator

        @param x: data tensor with NaN values
        @param m: mask with indicators of NaN values (1 for non-NaN values and 0.0 for NaN values)
        @return:  the original data tensor with imputed NaN values
        """
        x = torch.nan_to_num(x, nan=0).to(self.device)
        m = m.to(self.device)

        self.G.eval()
        with torch.no_grad():
            x_imputed = self.G(torch.cat(tensors=[x, m], dim=1))
            x_hat = m * x + (1 - m) * x_imputed
            return x_hat

    def _set_seed(self, seed: int) -> None:
        """
        Sets the seed for all

        @param seed: initializing value of randomness generators
        """
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _discriminator_step(self, x_new: torch.Tensor, m: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Updates the discriminator parameters

        @param x_new: a data matrix where the NaN`s are replaced by random uniformly distributed values
        @param m:     mask with indicators of NaN values of the original tensor
        @param h:     hint matrix
        @return:      the value of the discriminator error
        """
        x_imputed = self.G(torch.cat(tensors=[x_new, m], dim=1))
        x_hat = m * x_new + (1 - m) * x_imputed
        m_hat = self.D(torch.cat(tensors=[x_hat, h], dim=1))
        loss = -torch.mean(m * torch.log(m_hat + 1e-8) + (1 - m) * torch.log(1 - m_hat + 1e-8))
        return loss

    def _generator_step(self,
                        x_old: torch.Tensor,
                        x_new: torch.Tensor,
                        m: torch.Tensor,
                        h: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Updates the generator parameters

        @param x_old: a data matrix where the NaN`s are replaced by zeros
        @param x_new: a data matrix where the NaN`s are replaced by random uniformly distributed values
        @param m:     mask with indicators of NaN values of the original tensor
        @param h:     hint matrix
        @return:      the value of the generator error
        """
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
        """
        Generates a tensor of uniformly distributed random numbers

        @param shape: dimension of the generated tensor
        @return:
        """
        return torch.tensor(np.random.uniform(low=0, high=0.01, size=shape)).float()

    @staticmethod
    def _sample_b(shape: list[int], p: float) -> torch.Tensor:
        """
        Generates a tensor with randomly distributed zeros and ones

        @param shape: dimension of the generated tensor
        @param p:     the proportion of zeros in the generated tensor
        @return:
        """
        b = np.random.uniform(low=0, high=1, size=shape)
        return torch.tensor(b > p).float()
 