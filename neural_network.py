import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO(dbthaker): Change this to use torch.functional instead of torch.nn
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

class FNeuralNetwork(nn.Module):


    def __init__(self):
        super(FNeuralNetwork, self).__init__()
    
    def forward(self, x, params):
        out = x
        # Assume params is list of (weight, bias) tuples.
        assert len(params) % 2 == 0
        for i in range(0, len(params), 2):
            w = params[i]
            b = params[i + 1]
            if i == len(params) - 2:
                out = F.linear(out, w, b)
            else:
                out = F.relu(F.linear(out, w, b))
        return out
