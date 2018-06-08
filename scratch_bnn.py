import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.distributions as dists

from pdb import set_trace

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
                nn.Linear(50, 1))

    def forward(self, x):
        return self.network(x)


class BayesianNeuralNetwork(object):


    def __init__(self, p):
        input_size = p + 1 # Additional feature for stochastic disturbance.
        self.neural_net = NeuralNetwork(input_size)
        self.weight_var = 10
        self.z_var = p

    def set_up_model_priors(self, det_model, use_xavier=False):
        prior_dict = dict()
        for name, param in det_model.named_parameters():
            mu = torch.zeros(param.shape, requires_grad=True)
            if use_xavier:
                sigma = (2.0 / param.shape[-1]) * torch.ones(param.shape)
            else:
                sigma = torch.tensor(self.weight_var * torch.ones(param.shape), \
                                     requires_grad=True)
            # TODO(dbthaker): How to ensure independent sampling for each weight?
            # Answer: Should be fine. See:
            #     https://pytorch.org/docs/master/_modules/torch/distributions/normal.html
            prior_dict[name] = dists.normal.Normal(mu, sigma)
        return prior_dict

    def set_up_z(self, N):
        z = dict()
        # TODO(dbthaker): This should be batchified eventually!
        for i in range(N):
            sigma = torch.tensor(self.z_var, requires_grad=True)
            z_i = dists.normal.Normal(0, sigma)
            z[i] = z_i
        return z

    def sample_from_dists(self, dists):
        sampled_values = dict()
        for (k, v) in dists.items():
            sampled_values[k] = v.rsample()
        return sampled_values

    # DANGER: In-place modification of det_model weights.
    def sample_bnn(self, det_model, weight_distributions):
        sampled_weights = self.sample_from_dists(weight_distributions)
        det_model.load_state_dict(sampled_weights)

    def _print_weights(self, model):
        print("-----------------------------------------------")
        for name, param in model.named_parameters():
            print("{}: {}".format(name, param.data.numpy()))
        print("-----------------------------------------------")

    def run_inference(self):
        priors = self.set_up_model_priors(self.neural_net)
        self.sample_bnn(self.neural_net, priors)
        set_trace()

def main():
    N = 1000
    (X, y), ground_truth_fn = build_toy_dataset2(N, plot_data=False)
    p = X.shape[1]
    bnn = BayesianNeuralNetwork(p)
    bnn.run_inference()

if __name__ == '__main__':
    main()
