import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dists
from pyro.distributions import Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pdb import set_trace

# Plot 2 from [Depeweg et al. 2017] Figure 5.
# Ground truth function: y = 7sin(x)
# Noisy estimate (heteroskedastic noise): y = 7sin(x) + 3|cos(x/2)|*noise
# Domain: [-4, 4)
# N: Number of data points to sample.
def build_toy_dataset2(N, plot_data=False):
    X = np.random.uniform(low=-4, high=4, size=N)
    sorted_X = np.sort(X)
    noise = np.random.normal(size=N) # Unit Gaussian noise.
    y = 7 * np.sin(X) + 3 * np.multiply(np.abs(np.cos(X / 2)), noise)
    ground_truth = 7 * np.sin(sorted_X)
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
    return data

class NeuralNetwork(nn.Module):


    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential( \
                nn.Linear(input_size, 5), \
                nn.ReLU(), \
                #nn.Linear(5, 5), \
                #nn.ReLU(), \
                nn.Linear(5, 1))

    def forward(self, x):
        return self.network(x)

class BayesianNeuralNetwork(object):


    def __init__(self, p):
        input_size = p # Additional feature for stochastic disturbance.
        self.neural_net = NeuralNetwork(input_size)
        self.softplus = nn.Softplus()
        self.weight_var = 1
        self.z_var = p

    def set_up_model_priors(self, det_model):
        prior_dict = dict()
        for name, param in det_model.named_parameters():
            mu = torch.zeros(param.shape)
            sigma = self.weight_var * torch.ones(param.shape)
            prior_dict[name] = dists.Normal(mu, sigma)
        return prior_dict

    def set_up_variational_parameters(self, det_model):
        prior_dict = dict()
        for name, param in det_model.named_parameters():
            # TODO(dbthaker): Better initialization of V.P.?
            mu = torch.zeros(param.shape)
            sigma = 10 * torch.ones(param.shape)
            mu_param = pyro.param("guide_{}_mu".format(name), mu)
            sigma_param = self.softplus(pyro.param("guide_{}_sigma".format(name), sigma))
            prior_dict[name] = dists.Normal(mu_param, sigma_param)
        return prior_dict
            
    # P(x|z)P(z)
    def model(self, X, y):
        N = X.shape[0]
        priors = self.set_up_model_priors(self.neural_net)
        lifted_module = pyro.random_module("module", self.neural_net, priors)
        lifted_reg_model = lifted_module()

        # TODO(dbthaker): Introduce noise variables z.
        # TODO(dbthaker): Batchify this.
        prediction = lifted_reg_model(X).squeeze(-1)
        noise = pyro.sample("noise", dists.Normal(0, 1))
        return prediction + noise

        # Sample latent variable z here?
        #z = pyro.sample("disturbance", dists.Normal(0, self.z_var))
        #set_trace()
        #with pyro.iarange("map", N, subsample=data):
        #with pyro.iarange("map", N, subsample_size=250) as ind:
        #with pyro.iarange("map", N, subsample=X):
        #batch_X = X.index_select(0, ind)
        #batch_X = X
        #batch_y = y
        #batch_y = y.index_select(0, ind)
        #prediction = lifted_reg_model(batch_X).squeeze(-1)
        #return prediction
        #return prediction + pyro.sample("noise", dists.Normal(0, 1))

    # q_v(z|x) where v are variational parameters.
    def guide(self, X, y):
        # TODO(dbthaker): How can we use z here?
        dist = self.set_up_variational_parameters(self.neural_net)
        # Instead of returning lifted_module(), just sample it
        # and then also sample the z's? Should we also sample noise?
        lifted_module = pyro.random_module("module", self.neural_net, dist)
        return lifted_module()

    def run_inference(self, X, y):
        N = X.shape[0]

        pyro.clear_param_store()
        optim = Adam({"lr": 0.002})
        svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())
        for j in range(10000):
            epoch_loss = svi.step(X, y)
            if j % 100 == 0:
                print("[{}] Loss: {}".format(j, epoch_loss / float(N)))
        for name in pyro.get_param_store().get_all_param_names():
            print("[{}]: {}".format(name, pyro.param(name).data.numpy()))

    def validate(self, domain):
        assert len(domain) == 2 # [Start, end]
        num_points = 100
        X = np.linspace(domain[0], domain[1], num=num_points).reshape((num_points, 1))
        X = torch.tensor(X).type(torch.Tensor)
        num_nns = 100
        pred_ys = list()
        for i in range(num_nns):
            sampled_model = self.guide(None, None)
            y = sampled_model(X).squeeze(-1).detach().numpy()
            #plt.scatter(X, y, s=0.1)
            #plt.show()
            pred_ys.append(y)
        tiled_xs = np.tile(X[:, 0], num_nns)
        tiled_ys = np.concatenate(pred_ys)
        plt.scatter(tiled_xs, tiled_ys, s=0.1)
        plt.show()

def main():
    N = 1000
    X, y = build_toy_dataset2(N, plot_data=True)
    p = X.shape[1]
    bnn = BayesianNeuralNetwork(p)
    bnn.run_inference(X, y)
    bnn.validate((-4, 4))


if __name__ == '__main__':
    main()
