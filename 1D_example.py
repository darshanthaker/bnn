import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.distributions as dists

from neural_network import NeuralNetwork
from AD_loss import AlphaDivergenceLoss
from pdb import set_trace

def build_toy_dataset2(N, plot_data=True):
    X = np.random.uniform(low=-4, high=4, size=N)
    sorted_X = np.sort(X)
    noise = np.random.normal(0, 0.7, size=N)
    y = noise
    ground_truth_fn = lambda x: np.zeros(x.shape)
    ground_truth = ground_truth_fn(sorted_X)
    X = np.reshape(X, (N, 1))
    sorted_X = np.reshape(sorted_X, (N, 1))
    y = np.reshape(y, (N, 1))
    if plot_data:
        plt.plot(X, y, 'ro', label='Generated Data')
        plt.plot(sorted_X, ground_truth, 'b', label='Ground truth')
        plt.legend()
        plt.show()
    X = torch.tensor(X).type(torch.Tensor)
    y = torch.tensor(y).type(torch.Tensor)
    data = (X, y)
    return data, ground_truth_fn

class BayesianRegression(nn.Module):


    def __init__(self, N, p):
        super(BayesianRegression, self).__init__()
        input_size = p
        self.N = N
        self.w_var = 0.1 # TODO(dbthaker): Change this back to 1 at some point.
        self.z_var = 0 # Irrelevant
        self.z_mu = [torch.zeros(N)] # Irrelevant.
        self.z_sigma = [torch.zeros(N)] # Irrelevant.
        noise_lst = nn.ParameterList()
        self.additive_noise = nn.Parameter(torch.ones(1))
        noise_lst.append(self.additive_noise)

        self.w_mu = [torch.zeros((1, 1))]
        self.w_sigma = nn.ParameterList()
        w_sigma = nn.Parameter(torch.ones(1, 1))
        self.w_sigma.append(w_sigma)

        self.trainable_params = list(self.w_sigma) + list(noise_lst)
        #self.trainable_params = list(self.w_sigma)
        self.optimizer = torch.optim.Adam(self.trainable_params, lr=1e-2)

    def _zero_out_sigma(self):
        for (i, sigma) in enumerate(self.w_sigma):
            self.w_sigma[i] = nn.Parameter(torch.zeros(sigma.shape))

    # Run inference.
    def forward(self, X, y, alpha=1):
        self.loss = AlphaDivergenceLoss(alpha, self.w_var, self.z_var, self.N, 25, \
                None, use_biases=False)

        all_losses = list()
        num_epochs = 1000
        for i in range(num_epochs):
            loss, ll = self.loss(self.w_mu, self.w_sigma, self.z_mu, self.z_sigma, \
                    self.additive_noise, X, y)
            all_losses.append(loss)
            self.optimizer.zero_grad()
            loss.backward() 
            #print(self.w_sigma[0])
            #for group in self.optimizer.param_groups:
            #    for p in group['params']:
            #        if p.grad is not None:
            #            print("YAY NON NONE GRADIENT: {}".format(p.grad))
            self.optimizer.step()
            if i % 1 == 0:
                print("[{}] Loss: {}, Sigma: {}, AN: {}".format(i, loss.item(), self.w_sigma[0].item(), self.additive_noise[0]))
        plt.plot(range(num_epochs), all_losses)
        plt.show()

    def predict(self, domain, ground_truth_fn):
        assert len(domain) == 2 # [Start, end]
        num_points = 100
        X = np.linspace(domain[0], domain[1], num=num_points).reshape((num_points, 1))
        ground_truth = ground_truth_fn(X)
        #min_max_scaler = MinMaxScaler()
        #X = min_max_scaler.fit_transform(X)
        X = torch.tensor(X).type(torch.Tensor)
        num_nns = 100
        pred_ys = list()
        #self._zero_out_sigma()
        for i in range(num_nns):
            w = dists.Normal(self.w_mu[0], self.w_sigma[0]).sample()
            sampled_noise = dists.Normal(0, self.additive_noise).sample()
            y = F.linear(X, w) + sampled_noise
            pred_ys.append(y)
        stacked_ys = np.stack(pred_ys)
        mean_pred = np.mean(stacked_ys, axis=0)
        sd = np.std(stacked_ys, axis=0)
        lb = mean_pred - sd
        ub = mean_pred + sd
        lb = np.reshape(lb, (num_nns))
        ub = np.reshape(ub, (num_nns))
        #tiled_xs = np.tile(X[:, 0], num_nns)
        #tiled_ys = np.concatenate(pred_ys)
        plt.plot(X.numpy(), ground_truth, 'b', label='Ground Truth')
        plt.plot(X.numpy(), mean_pred, 'r', label='Prediction mean')
        plt.fill_between(X.squeeze(1).numpy(), ub, lb)
        #plt.scatter(tiled_xs, tiled_ys, s=0.1, label='Predicted')
        #plt.ylim(-10, 10)
        plt.legend()
        plt.show()

def main():
    N = 1000
    (X, y), ground_truth_fn = build_toy_dataset2(N, plot_data=True)
    p = X.shape[1]
    bnn = BayesianRegression(N, p)
    #bnn.predict((-4, 4), ground_truth_fn)
    bnn.forward(X, y)
    bnn.predict((-4, 4), ground_truth_fn)

if __name__ == '__main__':
    main()


