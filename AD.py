import numpy as np
import math
import torch
import torch.nn as nn
import torch.distributions as dists
import torch.nn.functional as F

from numbers import Number
from neural_network import NeuralNetwork, FNeuralNetwork
from pdb import set_trace


def flatten(param_lst):
    tensor_lst = list()
    for p in param_lst:
        tensor_lst.append(p.reshape((1, -1)))
    tensor_lst = torch.cat(tensor_lst, dim=1)
    return tensor_lst


class AlphaDivergenceLoss(nn.Module):
    def __init__(self, alpha, lam, gam, N, K):
        super(AlphaDivergenceLoss, self).__init__()
        self.alpha = alpha
        # Prior variance for weights.
        self.sigma_0 = lam
        # Prior variance for z.
        self.gam = gam
        # Num total training datapoints.
        self.N = N
        # Number of neural nets to approximate expectations over q.
        self.K = K

    def logistic(self, x):
        #return x
        return torch.exp(x)
        #return 1.0 / (1.0 + torch.exp(-x))

    def f_theta(self, w, mu, sigma, sigma_0, N=1):
        out = (sigma - sigma_0) / (sigma_0 * sigma) * (w * w) + mu / sigma * w
        return torch.sum(out/N, dim=1)

    def logZ(self, w_mu, w_sigma, log_w_sigma=None):
        log_two_pi = torch.log(torch.tensor(2 * math.pi, dtype=torch.float32))
        if log_w_sigma is None:
            log_w_sigma = torch.log(w_sigma)
        z = 0.5 * (log_two_pi + log_w_sigma) + w_mu * w_mu / w_sigma
        return torch.sum(z, dim=1)

    # Trick to avoid underflow/overflow issues.
    def log_sum_exp(self, quantity, dim=0):
        maximum = torch.max(quantity, dim=dim)[0]
        out = torch.exp(quantity - maximum)
        out = torch.sum(out, dim=dim) / self.K
        out = torch.log(out) + maximum
        return out

    def forward(self, w_mu, log_w_sigma, Sigma, loglik, x, y):
        sigma_0 = self.sigma_0
        alpha = self.alpha
        N = self.N

        w_sigma = [self.logistic(x) for x in log_w_sigma]
        w_dists = [dists.normal.Normal(mu, sigma) for mu, sigma in zip(w_mu, w_sigma)]

        flat_w_mu = flatten(w_mu)
        flat_w_sigma = flatten(w_sigma)
        flat_log_w_sigma = flatten(log_w_sigma)

        # 1/alpha * sum_n log E_q [p(y_n|w) / f(w)]^alpha
        lls = torch.zeros((self.K, 1))
        f_w = torch.zeros((self.K, 1))
        for k in range(self.K):
            w_k = [w_dist.rsample() for w_dist in w_dists]
            flat_w_k = flatten(w_k)

            # Log-likelihood
            lls[k] = loglik(y, x, w_k, Sigma)
            f_w[k] = self.f_theta(flat_w_k, flat_w_mu, flat_w_sigma, sigma_0, N=N)

        exponent = alpha * (lls - f_w)
        loss = self.log_sum_exp(exponent) / alpha
        loss += self.logZ(flat_w_mu, flat_w_sigma, flat_log_w_sigma)
        return loss


def normal_log_prob(mu, sigma, val):
    var = sigma ** 2
    log_scale = np.log(sigma) if isinstance(sigma, Number) else sigma.log()
    return -((val - mu) ** 2) / (2 * var) - log_scale - np.log(np.sqrt(2 * np.pi))


def linear_loglik(y, x, w, Sigma):
    """

    Args:
      y (batch_size, ydim):
      x (batch_size, xdim):
      w [(ydim, xdim)]:
      Sigma: scalar
    """
    prediction = F.linear(x, w[0])
    probs = normal_log_prob(prediction, Sigma, y)
    return torch.sum(probs)

def gaussian_loglik(y, x, w, Sigma):


