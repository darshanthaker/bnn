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

    def set_up_distributions(self, mus, sigmas):
        out_dists = list()
        for (mu, sigma) in zip(mus, sigmas):
            out_dists.append(dists.normal.Normal(mu, sigma))
        return out_dists

    def sample_from_dists(self, inp_dists):
        sampled_values = list()
        for v in inp_dists:
            # Flatten if matrices.
            sampled_values.append(v.rsample().reshape((1, -1)))
        sampled_values = torch.cat(sampled_values, dim=1)
        return sampled_values

    def sample_batches(self, weight_dists, num_rows, P):
        out = torch.zeros((num_rows, P))
        for i in range(num_rows):
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
            z_mu     : (1, B) - mean for each z value in mini-batch.
            z_sigma  : (1, B) - variance for each z value in mini-batch.
    """
    def negative_log_normalizer(self, w_mu, w_sigma, z_mu, z_sigma):
        w_normalizer = 0.5 * torch.log(2 * math.pi * w_sigma) + \
                    torch.div(w_mu * w_mu, w_sigma)
        w_normalizer = torch.sum(w_normalizer, dim=1)
        z_normalizer = 0.5 * torch.log(2 * math.pi * z_sigma) + \
                    torch.div(z_mu * z_mu, z_sigma)
        z_normalizer = torch.sum(z_normalizer, dim=1)
        return -torch.log(w_normalizer + z_normalizer)

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
        Output:
            out   : (1, K) - vector of f(W) for each sampled W ~ q.
    """
    def calc_f_w(self, W, mu, sigma):
        K = W.shape[0]
        # Convert mu and sigma to tiled (K, P) matrices for easier computations.
        mu = torch.cat([mu for i in range(K)], dim=0)
        sigma = torch.cat([sigma for i in range(K)], dim=0)
    
        out = torch.mul(torch.div(sigma - self.lam, self.lam * sigma), W * W) + \
                    torch.mul(torch.div(mu, sigma), W)
        out /= self.N
        out = torch.sum(out, dim=1)
        out = torch.exp(out).reshape((1, K))
        return out

    """
        Computes f(z), which is in exponential Gaussian form and proportional to
        [q(z_n) / p(z_n)].

        Notation:
            B: Mini-batch size
        Args: 
            Z     : (1, B) - sampled z value for each point in mini-batch.
            mu    : (1, B) - mean for each z value. 
            sigma : (1, B) - variance for each z value.
        Output:
            out   : (1, B) - vector of f_i(z_i) for each z_i value in mini-batch.
    """
    def calc_f_z(self, Z, mu, sigma):
        out = torch.mul(torch.div(sigma - self.gam, self.gam * sigma), Z * Z) + \
                    torch.mul(torch.div(mu, sigma), Z)
        out = torch.exp(out)
        return out

    def flatten(self, param_lst):
        tensor_lst = list()
        for p in param_lst:
            tensor_lst.append(p.reshape((1, -1)))
        tensor_lst = torch.cat(tensor_lst, dim=1)
        return tensor_lst

    # Trick to avoid underflow/overflow issues.
    def log_sum_exp(self, quantity, dim=0):
        maximum = torch.max(quantity, dim=dim)[0]
        out = torch.exp(quantity - maximum)
        out = torch.sum(out, dim=dim)
        out = torch.log(out) + maximum
        return out

    def calc_local_alpha_divs(self, f_w, f_z, ll):
        # Calculate product of each f(W)*f_i(z_i) for all z_i and W ~ q.
        # TODO(dbthaker): Make this log_f_w and log_f_z for numerical stability?
        prod = torch.mm(f_w.transpose(0, 1), f_z)
        # Average across K values of W ~ q to approximate expectation.
        exponent = self.alpha * ll - prod 
        out = self.log_sum_exp(exponent) / self.K
        out = torch.sum(out) / self.alpha
        return out

    def forward(self, w_mu, w_sigma, z_mu, z_sigma, ll, true_labs):
        batch_size = true_labs.shape[0]
        weight_dists = self.set_up_distributions(w_mu, w_sigma)
        z_dists = self.set_up_distributions(z_mu, z_sigma)
        flat_w_mu = self.flatten(w_mu)
        flat_w_sigma = self.flatten(w_sigma)
        flat_z_mu = self.flatten(z_mu)
        flat_z_sigma = self.flatten(z_sigma)

        W = self.sample_batches(weight_dists, self.K, flat_w_mu.shape[1])
        Z = self.sample_batches(z_dists, 1, batch_size)
        f_w = self.calc_f_w(W, flat_w_mu, flat_w_sigma)
        f_z = self.calc_f_z(Z, flat_z_mu, flat_z_sigma)
        lad = self.calc_local_alpha_divs(f_w, f_z, ll)
        
        loss = self.negative_log_normalizer(flat_w_mu, flat_w_sigma, \
                flat_z_mu, flat_z_sigma) - lad
        return loss

class BayesianNeuralNetwork(nn.Module):


    def __init__(self, N, p):
        super(BayesianNeuralNetwork, self).__init__()
        input_size = p # Additional feature for stochastic disturbance.
        self.N = N
        self.neural_net = NeuralNetwork(input_size)
        self.w_var = 10
        self.z_var = p
        self.additive_noise = nn.Parameter(20 * torch.ones(1))

        self.w_mu, self.w_sigma = self.set_up_model_priors(self.neural_net)
        self.z_mu, self.z_sigma = self.set_up_z_priors(self.N)
        self.trainable_params = list(self.w_mu) + list(self.w_sigma) + \
                list(self.z_mu) + list(self.z_sigma)

    def set_up_model_priors(self, det_model, use_xavier=False):
        train_mus = nn.ParameterList()
        train_sigmas = nn.ParameterList()
        for name, param in det_model.named_parameters():
            mu = nn.Parameter(torch.zeros(param.shape))
            train_mus.append(mu)
            if use_xavier:
                sigma = (2.0 / param.shape[-1]) * torch.ones(param.shape)
            else:
                sigma = nn.Parameter(self.w_var * torch.ones(param.shape))
            train_sigmas.append(sigma)
            # TODO(dbthaker): How to ensure independent sampling for each weight?
            # Answer: Should be fine. See:
            #     https://pytorch.org/docs/master/_modules/torch/distributions/normal.html
            #prior_dict[name] = dists.normal.Normal(mu, sigma)
        return train_mus, train_sigmas

    def set_up_z_priors(self, N):
        train_mus = nn.ParameterList()
        train_sigmas = nn.ParameterList()
        for i in range(N):
            mu = nn.Parameter(torch.zeros(1))
            sigma = nn.Parameter(self.z_var * torch.ones(1))
            train_mus.append(mu)
            train_sigmas.append(sigma)
        return train_mus, train_sigmas

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

    def calc_log_likelihood(self, X, y):
        prediction = self.neural_net(X)
        predictive_dist = dists.normal.Normal(prediction, self.additive_noise)
        probs = predictive_dist.log_prob(y)
        log_likelihood = torch.sum(probs, dim=0)
        return log_likelihood

    def calc_loss(self, X, y):
        log_likelihood = self.calc_log_likelihood(X, y)
        return self.loss(self.w_mu, self.w_sigma, self.z_mu, self.z_sigma, \
            log_likelihood, y)

    def forward(self, X, y, alpha=0.5):
        self.optimizer = torch.optim.Adam(self.trainable_params, lr=1e-2)
        self.loss = AlphaDivergenceLoss(alpha, self.w_var, self.z_var, self.N, 25)

        for i in range(1000):
            loss = self.calc_loss(X, y)
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
            print("[{}] Loss: {}".format(i, loss.item()))

def main():
    N = 1000
    (X, y), ground_truth_fn = build_toy_dataset2(N, plot_data=False)
    p = X.shape[1]
    bnn = BayesianNeuralNetwork(N, p)
    bnn.forward(X, y)

if __name__ == '__main__':
    main()
