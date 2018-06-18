import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.distributions as dists
import pickle

from sklearn.preprocessing import MinMaxScaler
from neural_network import NeuralNetwork
from AD_loss import AlphaDivergenceLoss
from pdb import set_trace

# Plot 2 from [Depeweg et al. 2017] Figure 5.
# Ground truth function: y = 7sin(x)
# Noisy estimate (heteroskedastic noise): y = 7sin(x) + 3|cos(x/2)|*noise
# Domain: [-4, 4)
# N: Number of data points to sample.
def build_toy_dataset2(N, plot_data=True):
    X = np.random.uniform(low=-4, high=4, size=N)
    sorted_X = np.sort(X)
    sin_scale = 7
    noise = np.random.normal(0, 0.5, size=N) # Unit Gaussian noise.
    #y = 7 * np.sin(X) + 3 * np.multiply(np.abs(np.cos(X / 2)), noise)
    y = sin_scale * np.sin(X) + noise # Homoskedastic noise vs. heteroskedastic noise for now.
    ground_truth_fn = lambda x: sin_scale * np.sin(x)
    ground_truth = ground_truth_fn(sorted_X)
    X = np.reshape(X, (N, 1))
    sorted_X = np.reshape(sorted_X, (N, 1))
    #min_max_scaler = MinMaxScaler()
    #X = min_max_scaler.fit_transform(X)
    #sorted_X = min_max_scaler.fit_transform(sorted_X)
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


class BayesianNeuralNetwork(nn.Module):


    def __init__(self, N, p):
        super(BayesianNeuralNetwork, self).__init__()
        input_size = p # Additional feature for stochastic disturbance.
        self.N = N
        self.neural_net = NeuralNetwork(input_size, model='model1')
        self.w_var = 1 # TODO(dbthaker): Change this back to 1 at some point.
        self.z_var = 0 # TODO(dbthaker): Change this back to p at some point.
        noise_lst = nn.ParameterList()
        self.additive_noise = nn.Parameter(0.85 * torch.ones(1))
        noise_lst.append(self.additive_noise)

        self.w_mu, self.log_w_sigma = self.set_up_model_priors(self.neural_net)
        self.z_mu, self.z_sigma = self.set_up_z_priors(self.N)
        #self.trainable_params = list(self.w_mu) + list(self.log_w_sigma) + \
        #        list(self.z_mu) + list(self.z_sigma) + list(noise_lst)
        #self.trainable_params = list(self.w_mu) + list(self.log_w_sigma)
        #self.trainable_params = list(self.w_mu)
        self.trainable_params = list(self.w_mu) + list(self.log_w_sigma) + list(noise_lst)
        #self.trainable_params = list(self.log_w_sigma) + list(noise_lst)
        #self.trainable_params = list(self.w_mu) + list(noise_lst)
        self.optimizer = torch.optim.Adam(self.trainable_params, lr=1e-2)

    def set_up_model_priors(self, det_model, use_xavier=False):
        train_mus = nn.ParameterList()
        train_log_sigmas = nn.ParameterList()
        mus = pickle.load(open('saved_mu.pickle', 'rb'))
        #for name, param in det_model.named_parameters():
        for param in mus:
            #mu = nn.Parameter(torch.zeros(param.shape))
            #sample_mu = torch.zeros(param.shape)
            #if 'bias' not in name:
            #    sample_sigma = (2.0 / param.shape[-1]) * torch.ones(param.shape)
            #else:
            #    sample_sigma = torch.zeros(param.shape)
            #mu = nn.Parameter(dists.normal.Normal(sample_mu, sample_sigma).sample())
            mu = nn.Parameter(param.clone())
            train_mus.append(mu)
            if use_xavier:
                sigma = (2.0 / param.shape[-1]) * torch.ones(param.shape)
            else:
                log_sigma = nn.Parameter(self.w_var * torch.ones(param.shape))
            train_log_sigmas.append(log_sigma)
        return train_mus, train_log_sigmas

    def set_up_z_priors(self, N):
        train_mus = nn.ParameterList()
        train_sigmas = nn.ParameterList()
        for i in range(N):
            mu = nn.Parameter(torch.zeros(1))
            sigma = nn.Parameter(self.z_var * torch.ones(1))
            train_mus.append(mu)
            train_sigmas.append(sigma)
        return train_mus, train_sigmas

    def logistic(self, x):
        return 1.0 / (1.0 + torch.exp(-x))

    def get_distributions(self, det_model, w_mu, w_sigma):
        out_dists = dict()
        for ((name, _), mu, sigma) in zip(det_model.named_parameters(), \
                   w_mu, w_sigma):
            sigma = self.logistic(sigma)
            out_dists[name] = dists.normal.Normal(mu, sigma)
        return out_dists

    def sample_from_dists(self, w_dists):
        sampled_values = dict()
        for (k, v) in w_dists.items():
            sampled_values[k] = v.sample()
        return sampled_values

    # DANGER: In-place modification of det_model weights.
    def sample_bnn(self, det_model, w_mu, w_sigma):
        w_dists = self.get_distributions(det_model, w_mu, w_sigma)
        sampled_weights = self.sample_from_dists(w_dists)
        det_model.load_state_dict(sampled_weights)

    def sample_z(self, X):
        z = torch.zeros((X.shape[0], 1))
        i = 0
        for i in range(X.shape[0]):
            z[i, :] = dists.normal.Normal(0, self.z_var).sample()
            i += 1
        return z

    def _print_parameters(self):
        print("--------------- PARAMETERS ------------------------")
        for i in self.trainable_params:
            print(i)
        print("--------------- PARAMETERS ------------------------")

    def _print_weights(self, model):
        print("---------------- WEIGHTS -------------------------")
        for name, param in model.named_parameters():
            print("{}: {}".format(name, param.data.numpy()))
        print("---------------- WEIGHTS -------------------------")

    def _zero_out_sigma(self):
        for (i, sigma) in enumerate(self.log_w_sigma):
            self.log_w_sigma[i] = nn.Parameter(-10000 * torch.ones(sigma.shape))

    # Run inference.
    def forward(self, X, y, alpha=-10):
        self.sample_bnn(self.neural_net, self.w_mu, self.log_w_sigma)
        self._print_weights(self.neural_net)
        self.loss = AlphaDivergenceLoss(alpha, self.w_var, self.z_var, self.N, 25, \
                self.neural_net)

        all_losses = list()
        all_ans = list()
        num_epochs = 600
        for i in range(num_epochs):
            loss, ll = self.loss(self.w_mu, self.log_w_sigma, self.z_mu, self.z_sigma, \
                    self.additive_noise, X, y)
            all_losses.append(loss)
            all_ans.append(self.additive_noise[0].item())
            self.optimizer.zero_grad()
            loss.backward()
            #print(self.log_w_sigma[0])
            #for group in self.optimizer.param_groups:
            #    for p in group['params']:
            #        if p.grad is not None:
            #            print("YAY NON NONE GRADIENT: {}".format(p.grad))
            self.optimizer.step()
            if i % 1 == 0:
                print("[{}] Loss: {}, LL: {}, AN: {}".format(i, loss.item(), "N/A", self.additive_noise[0]))
        set_trace()
        plt.plot(range(num_epochs), all_losses)
        plt.show()
        plt.plot(range(num_epochs), all_ans)
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
        self.sample_bnn(self.neural_net, self.w_mu, self.log_w_sigma)
        self._print_weights(self.neural_net)
        self._print_parameters()
        for i in range(num_nns):
            self.sample_bnn(self.neural_net, self.w_mu, self.log_w_sigma)
            z = self.sample_z(X)
            disturbed_X = torch.cat([X, z], dim=1).type(torch.Tensor)
            #y = self.neural_net(disturbed_X).squeeze(-1).detach().numpy()
            y = self.neural_net(X).squeeze(-1).detach().numpy()
            sampled_noise = dists.Normal(0, self.additive_noise).sample()
            y += sampled_noise
            pred_ys.append(y)
        stacked_ys = np.stack(pred_ys)
        #mean_pred = np.mean(stacked_ys, axis=0)
        self._zero_out_sigma()
        self.sample_bnn(self.neural_net, self.w_mu, self.log_w_sigma)
        mean_pred = self.neural_net(X).squeeze(-1).detach().numpy()
        sd = np.std(stacked_ys, axis=0)
        lb = mean_pred - sd
        ub = mean_pred + sd
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
    bnn = BayesianNeuralNetwork(N, p)
    #bnn.predict((-4, 4), ground_truth_fn)
    bnn.forward(X, y)
    bnn.predict((-4, 4), ground_truth_fn)

if __name__ == '__main__':
    main()
