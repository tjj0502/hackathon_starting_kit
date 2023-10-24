from dataclasses import dataclass
from typing import List, Tuple
from functools import partial
from src.utils import to_numpy
import torch
from torch import nn
import numpy as np

class BaseAugmentation:
    pass

    def apply(self, *args: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError('Needs to be implemented by child.')


@dataclass
class Scale(BaseAugmentation):
    scale: float = 1
    dim: int = None

    def apply(self, x: torch.Tensor):
        if self.dim == None:
            return self.scale * x
        else:
            x[...,self.dim] = self.scale * x[...,self.dim]
            return x

def apply_augmentations(x: torch.Tensor, augmentations: Tuple) -> torch.Tensor:
    y = x.clone()
    for augmentation in augmentations:
        y = augmentation.apply(y)
    return y


def cov_torch(x, rowvar=False, bias=True, ddof=None, aweights=None):
    device = x.device
    x = to_numpy(x)
    _, L, C = x.shape
    x = x.reshape(-1, L*C)
    return torch.from_numpy(np.cov(x, rowvar=False)).to(device).float()


def acf_torch(x: torch.Tensor, max_lag: int, dim: Tuple[int] = (0, 1)) -> torch.Tensor:
    """
    :param x: torch.Tensor [B, S, D]
    :param max_lag: int. specifies number of lags to compute the acf for
    :return: acf of x. [max_lag, D]
    """
    acf_list = list()
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)


def non_stationary_acf_torch(X, symmetric=False):
    """
    Compute the correlation matrix between any two time points of the time series
    Parameters
    ----------
    X (torch.Tensor): [B, T, D]
    symmetric (bool): whether to return the upper triangular matrix of the full matrix

    Returns
    -------
    Correlation matrix of the shape [T, T, D] where each entry (t_i, t_j, d_i) is the correlation between the d_i-th coordinate of X_{t_i} and X_{t_j}
    """
    # Get the batch size, sequence length, and input dimension from the input tensor
    B, T, D = X.shape

    # Create a tensor to hold the correlations
    correlations = torch.zeros(T, T, D)

    # Loop through each time step from lag to T-1
    for t in range(T):
        # Loop through each lag from 1 to lag
        for tau in range(t, T):
            # Compute the correlation between X_{t, d} and X_{t-tau, d}
            correlation = torch.sum(X[:, t, :] * X[:, tau, :], dim=0) / (
                torch.norm(X[:, t, :], dim=0) * torch.norm(X[:, tau, :], dim=0))
            # print(correlation)
            # Store the correlation in the output tensor
            correlations[t, tau, :] = correlation
            if symmetric:
                correlations[tau, t, :] = correlation

    return correlations


def cacf_torch(x, lags: list, dim=(0, 1)):
    """
    Computes the cross-correlation between feature dimension and time dimension
    Parameters
    ----------
    x
    lags
    dim

    Returns
    -------

    """
    # Define a helper function to get the lower triangular indices for a given dimension
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    # Get the lower triangular indices for the input tensor x
    ind = get_lower_triangular_indices(x.shape[2])

    # Standardize the input tensor x along the given dimensions
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)

    # Split the input tensor into left and right parts based on the lower triangular indices
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]

    # Compute the cross-correlation at each lag and store in a list
    cacf_list = list()
    for i in range(lags):
        # Compute the element-wise product of the left and right parts, shifted by the lag if i > 0
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r

        # Compute the mean of the product along the time dimension
        cacf_i = torch.mean(y, (1))

        # Append the result to the list of cross-correlations
        cacf_list.append(cacf_i)

    # Concatenate the cross-correlations across lags and reshape to the desired output shape
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)


def acf_diff(x): return torch.sqrt(torch.pow(x, 2).sum(0))
def cc_diff(x): return torch.abs(x).sum(0)
def cov_diff(x): return torch.abs(x).mean()


class ACFLoss(Loss):
    def __init__(self, x_real, max_lag=64, stationary=True, **kwargs):
        super(ACFLoss, self).__init__(norm_foo=acf_diff, **kwargs)
        self.max_lag = min(max_lag, x_real.shape[1])
        self.stationary = stationary
        if stationary:
            self.acf_real = acf_torch(self.transform(
                x_real), self.max_lag, dim=(0, 1))
        else:
            self.acf_real = non_stationary_acf_torch(self.transform(
                x_real), symmetric=False)  # Divide by 2 because it is symmetric matrix

    def compute(self, x_fake):
        if self.stationary:
            acf_fake = acf_torch(self.transform(x_fake), self.max_lag)
        else:
            acf_fake = non_stationary_acf_torch(self.transform(
                x_fake), symmetric=False).to(x_fake.device)
        return self.norm_foo(acf_fake - self.acf_real.to(x_fake.device))


class MeanLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(MeanLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.mean = x_real.mean((0, 1))

    def compute(self, x_fake, **kwargs):
        return self.norm_foo(x_fake.mean((0, 1)) - self.mean)


class StdLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(StdLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.std_real = x_real.std((0, 1))

    def compute(self, x_fake, **kwargs):
        return self.norm_foo(x_fake.std((0, 1)) - self.std_real)


class CrossCorrelLoss(Loss):
    def __init__(self, x_real, max_lag=64, **kwargs):
        super(CrossCorrelLoss, self).__init__(norm_foo=cc_diff, **kwargs)
        self.cross_correl_real = cacf_torch(
            self.transform(x_real), max_lag).mean(0)[0]
        self.max_lag = max_lag

    def compute(self, x_fake):
        cross_correl_fake = cacf_torch(
            self.transform(x_fake), self.max_lag).mean(0)[0]
        loss = self.norm_foo(
            cross_correl_fake - self.cross_correl_real.to(x_fake.device)).unsqueeze(0)
        return loss


def histogram_torch(x, n_bins, density=True):
    a, b = x.min().item(), x.max().item()
    b = b+1e-5 if b == a else b
    # delta = (b - a) / n_bins
    bins = torch.linspace(a, b, n_bins+1)
    delta = bins[1]-bins[0]
    # bins = torch.arange(a, b + 1.5e-5, step=delta)
    count = torch.histc(x, bins=n_bins, min=a, max=b).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins


class HistoLoss(Loss):

    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            tmp_densities = list()
            tmp_locs = list()
            tmp_deltas = list()
            # Exclude the initial point
            for t in range(x_real.shape[1]):
                x_ti = x_real[:, t, i].reshape(-1, 1)
                d, b = histogram_torch(x_ti, n_bins, density=True)
                tmp_densities.append(nn.Parameter(d).to(x_real.device))
                delta = b[1:2] - b[:1]
                loc = 0.5 * (b[1:] + b[:-1])
                tmp_locs.append(loc)
                tmp_deltas.append(delta)
            self.densities.append(tmp_densities)
            self.locs.append(tmp_locs)
            self.deltas.append(tmp_deltas)

    def compute(self, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            tmp_loss = list()
            # Exclude the initial point
            for t in range(x_fake.shape[1]):
                loc = self.locs[i][t].view(1, -1).to(x_fake.device)
                x_ti = x_fake[:, t, i].contiguous(
                ).view(-1, 1).repeat(1, loc.shape[1])
                dist = torch.abs(x_ti - loc)
                counter = (relu(self.deltas[i][t].to(
                    x_fake.device) / 2. - dist) > 0.).float()
                density = counter.mean(0) / self.deltas[i][t].to(x_fake.device)
                abs_metric = torch.abs(
                    density - self.densities[i][t].to(x_fake.device))
                loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise


class CovLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(CovLoss, self).__init__(norm_foo=cov_diff, **kwargs)
        self.covariance_real = cov_torch(
            self.transform(x_real))

    def compute(self, x_fake):
        covariance_fake = cov_torch(self.transform(x_fake))
        loss = self.norm_foo(covariance_fake -
                             self.covariance_real.to(x_fake.device))
        return loss


class cross_correlation(Loss):
    def __init__(self, x_real, **kwargs):
        super(cross_correlation).__init__(**kwargs)
        self.x_real = x_real

    def compute(self, x_fake):
        fake_corre = torch.from_numpy(np.corrcoef(
            x_fake.mean(1).permute(1, 0))).float()
        real_corre = torch.from_numpy(np.corrcoef(
            self.x_real.mean(1).permute(1, 0))).float()
        return torch.abs(fake_corre-real_corre)

def rmse(x, y):
    return (x - y).pow(2).sum().sqrt()

def diff(x): return x[:, 1:] - x[:, :-1]


test_metrics = {
    # 'Sig_mmd': partial(Sig_MMD_loss, name='Sig_mmd', depth=4),
    # 'SigW1': partial(SigW1Loss, name='SigW1', augmentations=[], normalise=False, depth=4),
    'marginal_distribution': partial(HistoLoss, n_bins=50, name='marginal_distribution'),
    'cross_correl': partial(CrossCorrelLoss, name='cross_correl'),
    'covariance': partial(CovLoss, name='covariance'),
    'auto_correl': partial(ACFLoss, name='auto_correl')}


def is_multivariate(x: torch.Tensor):
    """ Check if the path / tensor is multivariate. """
    return True if x.shape[-1] > 1 else False


