import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dists
from pyro.distributions import Normal
from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO
from pyro.optim import Adam
from pdb import set_trace

"""
NOTE: This is incomplete and BNN inference does not yield good results,
      presumably due to some bug. Refer to bnn.py for a working version.
"""

# Plot 2 from [Depeweg et al. 2017] Figure 5.
# Ground truth function: y = 7sin(x)
# Noisy estimate (heteroskedastic noise): y = 7sin(x) + 3|cos(x/2)|*noise
# Domain: [-4, 4)
# N: Number of data points to sample.
def build_toy_dataset2(N, plot_data=False):
    X = np.random.uniform(low=-4, high=4, size=N)
    sorted_X = np.sort(X)
    sin_scale = 7
    noise = np.random.normal(size=N) # Unit Gaussian noise.
    #y = 7 * np.sin(X) + 3 * np.multiply(np.abs(np.cos(X / 2)), noise)
    y = sin_scale * np.sin(X) + noise # Homoskedastic noise vs. heteroskedastic noise for now.
    ground_truth_fn = lambda x: sin_scale * np.sin(x)
    ground_truth = ground_truth_fn(sorted_X)
    X = np.reshape(X, (N, 1))
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

class NeuralNetwork(nn.Module):


    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential( \
                nn.Linear(input_size, 50), \
                nn.ReLU(), \
                nn.Linear(50, 50), \
                nn.ReLU(), \
                nn.Linear(50, 50), \
                nn.ReLU(), \
                nn.Linear(50, 1))

    def forward(self, x):
        return self.network(x)

class VanillaSGDTrainer(object):


    def __init__(self, p):
        self.neural_net = NeuralNetwork(p)

    def train(self, X, y):
        batch_size = 250
        optimizer = torch.optim.Adam(self.neural_net.parameters(), lr=0.002)
        loss_fn = nn.MSELoss(size_average=False)
        for i in range(10000):
            shuffled_indices = np.random.permutation(range(X.shape[0]))
            for start in range(0, len(shuffled_indices), batch_size):
                end = min(start + batch_size + 1, len(shuffled_indices))
                indices = shuffled_indices[start:end]
                batch_X = X[indices, :]
                batch_y = y[indices, :]
                output = self.neural_net.forward(batch_X)
                loss = loss_fn(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if i % 100 == 0:
                print("[{}] Loss: {}".format(i, loss.item()))
        for name, param in self.neural_net.named_parameters():
            print("{}: {}".format(name, param.data.numpy())) 

    def validate(self, domain, ground_truth_fn):
        assert len(domain) == 2 # [Start, end]
        num_points = 100
        X = np.linspace(domain[0], domain[1], num=num_points).reshape((num_points, 1))
        X = torch.tensor(X).type(torch.Tensor)
        ground_truth = ground_truth_fn(X).numpy()
        y = self.neural_net(X).squeeze(-1).detach().numpy()
        plt.plot(X.numpy(), ground_truth, 'b', label='Ground truth')
        plt.scatter(X, y, s=10, label='Predicted')
        plt.legend()
        plt.show()

class BayesianNeuralNetwork(object):


    def __init__(self, p):
        input_size = p + 1 # Additional feature for stochastic disturbance.
        self.neural_net = NeuralNetwork(input_size)
        self.softplus = nn.Softplus()
        self.weight_var = 20
        self.z_var = p

    def set_up_model_priors(self, det_model, N):
        prior_dict = dict()
        for name, param in det_model.named_parameters():
            mu = torch.zeros(param.shape)
            sigma = (2.0 / param.shape[-1]) * torch.ones(param.shape)
            prior_dict[name] = dists.Normal(mu, sigma)
        # TODO(dbthaker): This should be batchified eventually!
        z = torch.zeros((N, 1))
        for i in range(N):
            z_i = pyro.sample("z_{}".format(i), dists.Normal(0, self.z_var))
            z[i, :] = z_i
        return prior_dict, z

    def set_up_variational_parameters(self, det_model, N):
        prior_dict = dict()
        z_prior_lst = list()
        for name, param in det_model.named_parameters():
            mu = 0.1 * torch.randn(param.shape)
            log_sigma = -0.9 * torch.ones(param.shape) + 0.05 * torch.randn(param.shape)
            mu_param = pyro.param("guide_{}_mu".format(name), mu)
            sigma_param = self.softplus(pyro.param("guide_{}_logsigma".format(name), \
                                                   log_sigma))
            prior_dict[name] = dists.Normal(mu_param, sigma_param)
        # TODO(dbthaker): This should be batchified eventually!
        for i in range(N): 
            mu = torch.zeros(1)
            log_sigma = self.z_var * torch.ones(1)
            # TODO(dbthaker): Uncomment this when q(z) needs to be learned as well!
            #                 and change above to be V.P. initialization and not prior.
            #mu_param = pyro.param("guide_z{}_mu".format(i), mu)
            #sigma_param = self.softplus(pyro.param("guide_z{}_logsigma".format(i), \
            #                                       log_sigma))
            #z_prior_lst.append(dists.Normal(mu_param, sigma_param))
            z_prior_lst.append(dists.Normal(mu, log_sigma))
        return prior_dict, z_prior_lst
            
    # P(x|z)P(z)
    def model(self, X, y):
        N = X.shape[0]
        priors, z = self.set_up_model_priors(self.neural_net, N)
        lifted_module = pyro.random_module("module", self.neural_net, priors)
        lifted_reg_model = lifted_module()

        # TODO(dbthaker): Batchify this.
        #noise = pyro.sample("noise", dists.Normal(0, 1))
        disturbed_X = torch.cat([X, z], dim=1)
        # Below line assumes independence across N datapoints.
        with pyro.iarange("map", N):
            prediction = lifted_reg_model(disturbed_X).squeeze(-1)
            pyro.sample("obs", dists.Normal(prediction, torch.ones(N)).independent(1), \
                 obs=y)

    # q_v(z|x) where v are variational parameters.
    def guide(self, X, y):
        N = X.shape[0] 

        # TODO(dbthaker): Deal with eps additive Gaussian noise?
        dist, z_lst = self.set_up_variational_parameters(self.neural_net, N)
        lifted_module = pyro.random_module("module", self.neural_net, dist)
        sampled_model = lifted_module()
        set_trace()
        sampled_z = np.zeros(N)
        for i in pyro.irange("z_loop", len(z_lst)):
            sampled_z[i] = pyro.sample("z_{}".format(i), z_lst[i])
        return sampled_model, torch.from_numpy(sampled_z)

    def run_inference(self, X, y):
        N = X.shape[0]

        pyro.clear_param_store()
        optim = Adam({"lr": 0.002})
        svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())
        for j in range(1000):
            epoch_loss = svi.step(X, y)
            #set_trace()
            if j % 100 == 0:
                print("[{}] Loss: {}".format(j, epoch_loss / float(N)))
        for name in pyro.get_param_store().get_all_param_names():
            print("[{}]: {}".format(name, pyro.param(name).data.numpy()))

    def validate(self, domain, ground_truth_fn):
        assert len(domain) == 2 # [Start, end]
        num_points = 100
        X = np.linspace(domain[0], domain[1], num=num_points).reshape((num_points, 1))
        X = torch.tensor(X).type(torch.Tensor)
        ground_truth = ground_truth_fn(X)
        num_nns = 100
        pred_ys = list()
        for i in range(num_nns):
            sampled_model, z = self.guide(X, None)
            z = z.view(num_points, 1).type(torch.Tensor)
            disturbed_X = torch.cat([X, z], dim=1).type(torch.Tensor)
            y = sampled_model(disturbed_X).squeeze(-1).detach().numpy()
            #plt.scatter(X, y, s=0.1)
            #plt.show()
            pred_ys.append(y)
        tiled_xs = np.tile(X[:, 0], num_nns)
        tiled_ys = np.concatenate(pred_ys)
        plt.plot(X.numpy(), ground_truth.numpy(), 'b', label='Ground Truth')
        plt.scatter(tiled_xs, tiled_ys, s=0.1, label='Predicted')
        plt.ylim(-10, 10)
        plt.legend()
        plt.show()

def main():
    N = 1000
    (X, y), ground_truth_fn = build_toy_dataset2(N, plot_data=True)
    p = X.shape[1]
    model = "vanilla"
    if model == "vanilla":
        trainer = VanillaSGDTrainer(p)
        trainer.train(X, y)
        trainer.validate((-4, 4), ground_truth_fn)
    elif model == "bnn":
        bnn = BayesianNeuralNetwork(p)
        bnn.run_inference(X, y)
        bnn.validate((-4, 4), ground_truth_fn)

if __name__ == '__main__':
    main()
