import numpy as np
import math
import torch
import torch.nn as nn
import torch.distributions as dists

from neural_network import NeuralNetwork
from pdb import set_trace

class AlphaDivergenceLoss(nn.Module):


    def __init__(self, alpha, lam, gam, N, K, neural_net):
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
        self.nn = neural_net

    def set_up_distributions(self, mus, sigmas):
        out_dists = list()
        for (mu, sigma) in zip(mus, sigmas):
            out_dists.append(dists.normal.Normal(mu, sigma))
        return out_dists

    def sample_from_dists(self, inp_dists, use_nn=False):
        sampled_values = dict()
        flattened_values = list()
        if use_nn:
            for ((name, _), v) in zip(self.nn.named_parameters(), inp_dists):
                # rsample() uses reparameterization trick (Kingma et al. 2014)
                # to give differentiable sample.
                sample = v.rsample()
                sampled_values[name] = sample
                flattened_values.append(sample.reshape((1, -1)))
        else:
            for (i, v) in enumerate(inp_dists):
                sample = v.rsample()
                sampled_values[i] = sample
                flattened_values.append(sample.reshape((1, -1)))
        flattened_values = torch.cat(flattened_values, dim=1)
        return flattened_values, sampled_values

    def sample_batches(self, weight_dists, num_rows, P, use_nn=False):
        out = torch.zeros((num_rows, P))
        w_out = list()
        for i in range(num_rows):
            sample, s = self.sample_from_dists(weight_dists, use_nn=use_nn)
            out[i, :] = sample
            w_out.append(s)
        return out, w_out

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
        #return -torch.log(w_normalizer + z_normalizer)
        return -torch.log(w_normalizer)

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
        out = torch.sum(out, dim=dim) / self.K
        out = torch.log(out) + maximum
        return out

    def calc_local_alpha_divs(self, f_w, f_z, ll):
        # Calculate product of each f(W)*f_i(z_i) for all z_i and W ~ q.
        # TODO(dbthaker): Make this log_f_w and log_f_z for numerical stability?
        prod = torch.mm(f_w.transpose(0, 1), f_z)
        # Average across K values of W ~ q to approximate expectation.
        exponent = self.alpha * ll - torch.log(prod)
        out = self.log_sum_exp(exponent)
        out = torch.sum(out) / self.alpha
        return out

    def calc_log_likelihood(self, X, Z, ws, y, det_model, an):
        lls = torch.zeros((len(ws), 1))
        for (i, w) in enumerate(ws):
            det_model.load_state_dict(w)
            Zt = Z.transpose(0, 1)
            disturbed_X = torch.cat([X, Zt], dim=1)
            prediction = det_model(disturbed_X)
            predictive_dist = dists.normal.Normal(prediction, an)
            probs = predictive_dist.log_prob(y)
            #set_trace()
            log_likelihood = torch.sum(probs, dim=0)
            lls[i] = log_likelihood
        mean_lls = torch.sum(lls, dim=0) / len(ws)
        return mean_lls

    def tmp_calc_log_likelihood(self, X, Z, w, y, det_model, an):
        det_model.load_state_dict(w)
        Zt = Z.transpose(0, 1)
        disturbed_X = torch.cat([X, Zt], dim=1)
        prediction = det_model(disturbed_X)
        predictive_dist = dists.normal.Normal(prediction, an)
        probs = predictive_dist.log_prob(y)
        #set_trace()
        log_likelihood = torch.sum(probs, dim=0)
        return log_likelihood

    def forward(self, w_mu, w_sigma, z_mu, z_sigma, an, X, true_labs):
        batch_size = true_labs.shape[0]
        #weight_dists = self.set_up_distributions(w_mu, w_sigma)
        #z_dists = self.set_up_distributions(z_mu, z_sigma)
        #flat_w_mu = self.flatten(w_mu)
        #flat_w_sigma = self.flatten(w_sigma)
        #flat_z_mu = self.flatten(z_mu)
        #flat_z_sigma = self.flatten(z_sigma)

        #W, nets_w = self.sample_batches(weight_dists, self.K, flat_w_mu.shape[1], \
        #        use_nn=True)
        #Z, _ = self.sample_batches(z_dists, 1, batch_size)
        Z = torch.zeros((1, batch_size))
        #ll = self.calc_log_likelihood(X, Z, nets_w, true_labs, self.nn, an)
        nets_w = dict()
        for ((name, _), mu) in zip(self.nn.named_parameters(), w_mu):
            nets_w[name] = mu
        ll = self.tmp_calc_log_likelihood(X, Z, nets_w, true_labs, self.nn, an)
        #f_w = self.calc_f_w(W, flat_w_mu, flat_w_sigma)
        #f_z = self.calc_f_z(Z, flat_z_mu, flat_z_sigma)
        f_w = torch.ones((1, self.K))
        f_z = torch.ones((1, batch_size))
        lad = self.calc_local_alpha_divs(f_w, f_z, ll)
        #set_trace()
        
        #loss = self.negative_log_normalizer(flat_w_mu, flat_w_sigma, \
        #        flat_z_mu, flat_z_sigma) - lad
        loss = -lad
        return loss, ll

def main():
    net = NeuralNetwork(2)
    w_mu = list()
    w_sigma = list()
    z_mu = list()
    z_sigma = list()
    an = 10
    w = dict()
    for (name, param) in net.named_parameters():
        w[name] = torch.ones(param.shape)
        w_mu.append(torch.ones(param.shape))
        w_sigma.append(1e-4 * torch.ones(param.shape))

    for i in range(4):
        z_mu.append(torch.ones(1))
        z_sigma.append(1e-4 * torch.ones(1))

    net.load_state_dict(w)

    X = torch.from_numpy(np.array([[0, 1, 2, 3]])).type(torch.Tensor).transpose(0, 1)
    tmp_z = torch.ones((4, 1))
    disturbed_X = torch.cat([X, tmp_z], dim=1)
    y = net(disturbed_X)
    loss = AlphaDivergenceLoss(0.5, 1e-4, 1e-4, 4, 1, net)
    l = loss(w_mu, w_sigma, z_mu, z_sigma, an, X, y)
    print("Loss: {}".format(l))

if __name__=='__main__':
    main()