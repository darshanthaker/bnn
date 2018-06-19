import pytest
import numpy as np
import torch
import AD
from AD import AlphaDivergenceLoss

from pdb import set_trace

@pytest.fixture
def loss():
    alpha = 1
    lam = 1
    gam = 1
    N = 100
    K = 25
    return AlphaDivergenceLoss(alpha, lam, gam, N, K)

def test_linear_loglik_shape():
    y_dim = 5 
    x_dim = 10
    batch_size = 100
    x = torch.randn((batch_size, x_dim))
    y = torch.randn((batch_size, y_dim))
    w = torch.randn((y_dim, x_dim))
    Sigma = torch.ones(1)
    output = AD.linear_loglik(y, x, w, Sigma) 
    assert output.shape == torch.Size([])

def test_AD_shape(loss):
    y_dim = 5 
    x_dim = 10
    batch_size = 100
    x = torch.randn((batch_size, x_dim))
    y = torch.randn((batch_size, y_dim))
    Sigma = torch.ones(1)
    w_mu = [torch.zeros((y_dim, x_dim))]
    log_w_sigma = [torch.zeros((y_dim, x_dim))]
    loglik = AD.linear_loglik

    loss_val = loss.forward(w_mu, log_w_sigma, Sigma, loglik, x, y)
    assert loss_val.shape == torch.Size([1])

def test_1D_example(loss):
    x_dim = 5
    y_dim = 10
    batch_size = 100
    pass 
