import numpy as np
import math
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


class AlphaDivergenceLoss(nn.Module):


    def __init__(self, alpha, lam, gam, N, K):
        super(AlphaDivergenceLoss, self).__init__()
        self.alpha = alpha
        # Prior variance for weights.
        self.lam = lam
        # Prior variance for z.
        self.gam = gam
        # Num total training datapoints.
        self.N = N
        # Number of neural nets to approximate expectations over q.
        self.K = K

    def set_up_distributions(self, params):
        assert 'mus' in params and 'sigmas' in params
        out_dists = list()
        for (mu, sigma) in zip(params['mus'], params['sigmas']):
            out_dists.append(dists.normal.Normal(mu, sigma))
        return out_dists

    def sample_from_dists(self, inp_dists):
        sampled_values = list()
        for v in inp_dists:
            sampled_values.append(v.rsample())
        return torch.from_numpy(np.array(sampled_values))

    def sample_bnn_weights(self, weight_dists):
        P = len(weight_dists)
        out = torch.zeros((self.K, P))
        for i in range(self.K):
            sample = self.sample_from_dists(weight_dists)
            out[i, :] = sample
        return out

    """
        Computes log normalization constant of exponential Gaussian form of q.

        Notation:
            B: Mini-batch size
        Args: 
            w_mu     : (1, P) - mean for each weight in BNN.
            w_sigma  : (1, P) - variance for each weight in BNN. 
            z_mu     : (B, 1) - mean for each z value in mini-batch.
            z_sigma  : (B, 1) - variance for each z value in mini-batch.
    """
    def log_normalizer(self, w_mu, w_sigma, z_mu, z_sigma):
        w_normalizer = 0.5 * torch.log(2 * math.pi * w_sigma) + \
                    torch.div(w_mu * w_mu, w_sigma)
        w_normalizer = torch.sum(w_normalizer, dim=1)
        z_normalizer = 0.5 * torch.log(2 * math.pi * z_sigma) + \
                    torch.div(z_mu * z_mu, z_sigma)
        z_normalizer = torch.sum(z_normalizer, dim=0)
        return w_normalizer + z_normalizer

    """
        Computes f(W), which is in exponential Gaussian form and proportional to
        [q(W) / p(W)]^{1/N}.

        Notation:
            K: Number of NNs to approximate expectations over q.
            P: Total number of parameters in fully-connected network (L layers and V_l
               neurons in each layer)
        Args: 
            W     : (K, P) - weights for each sampled NN.
            mu    : (1, P) - mean for each weight in BNN. 
            sigma : (1, P) - variance for each weight in BNN.
    """
    def f_w(self, W, mu, sigma):
        K = W.shape[0]
        # Convert mu and sigma to tiled (K, P) matrices for easier computations below.
        mu = torch.cat([mu for i in range(K)], dim=0)
        sigma = torch.cat([sigma for i in range(K)], dim=0)
    
        out = torch.mul(torch.div(sigma - self.lam, self.lam * sigma), W * W) + \
                    torch.mul(torch.div(mu, sigma), W)
        out /= N
        out = torch.sum(out, dim=1)
        out = torch.exp(out)
        return out

    """
        Computes f(z), which is in exponential Gaussian form and proportional to
        [q(z_n) / p(z_n)].

        Notation:
            B: Mini-batch size
        Args: 
            Z     : (B, 1) - sampled z value for each point in mini-batch.
            mu    : (B, 1) - mean for each z value. 
            sigma : (B, 1) - variance for each z value.
    """
    def f_z(self, Z, mu, sigma):
        out = torch.mul(torch.div(sigma - self.gam, self.gam * sigma), Z * Z) + \
                    torch.mul(torch.div(mu, sigma), Z)
        out = torch.exp(out)

    def forward(self, weight_params, z_params, true_labs):
        batch_size = true_labs.shape[0]
        weight_dists = self.set_up_distributions(weight_params)
        z_dists = self.set_up_distributions(z_params)
        W = self.sample_bnn_weights(weight_dists)
        loss = -self.log_normalizer() - (1.0 / self.alpha)


class BayesianNeuralNetwork(object):


    def __init__(self, p):
        input_size = p + 1 # Additional feature for stochastic disturbance.
        self.neural_net = NeuralNetwork(input_size)
        self.weight_var = 10
        self.z_var = p

    def set_up_model_priors(self, det_model, use_xavier=False):
        train_params = map()
        train_params['mus'] = list()
        train_params['sigmas'] = list()
        for name, param in det_model.named_parameters():
            mu = torch.zeros(param.shape, requires_grad=True)
            train_params['mus'].append(mu)
            if use_xavier:
                sigma = (2.0 / param.shape[-1]) * torch.ones(param.shape)
            else:
                sigma = torch.tensor(self.weight_var * torch.ones(param.shape), \
                                     requires_grad=True)
            train_params['sigmas'].append(sigma)
            # TODO(dbthaker): How to ensure independent sampling for each weight?
            # Answer: Should be fine. See:
            #     https://pytorch.org/docs/master/_modules/torch/distributions/normal.html
            #prior_dict[name] = dists.normal.Normal(mu, sigma)
        return train_params

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

    def run_inference(self, data, alpha=0.5):
        N = data.shape[0]
        weights_params = self.set_up_model_priors(self.neural_net)
        self.sample_bnn(self.neural_net, priors)

        self.optimizer = torch.optim.Adam(weights_params, lr=1e-3)
        self.loss = AlphaDivergenceLoss(alpha, self.w_var, self.z_var, N)

        self.optimizer.zero_grad() 
        self.optimizer.step()

def main():
    N = 1000
    (X, y), ground_truth_fn = build_toy_dataset2(N, plot_data=False)
    p = X.shape[1]
    bnn = BayesianNeuralNetwork(p)
    bnn.run_inference()

if __name__ == '__main__':
    main()
