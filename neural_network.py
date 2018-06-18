import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace

# TODO(dbthaker): Change this to use torch.functional instead of torch.nn
class NeuralNetwork(nn.Module):


    def __init__(self, input_size, model='model1'):
        super(NeuralNetwork, self).__init__()
        if model == 'model1':
            self.network = nn.Sequential( \
                    nn.Linear(input_size, 50), \
                    nn.ReLU(), \
                    nn.Linear(50, 50), \
                    nn.ReLU(), \
                    nn.Linear(50, 1))
        elif model == 'model2':
            self.network = nn.Sequential( \
                    nn.Linear(input_size, 1))

    def forward(self, x):
        return self.network(x)

class FNeuralNetwork(nn.Module):


    def __init__(self, use_biases=True):
        super(FNeuralNetwork, self).__init__()
        self.use_biases = use_biases
    
    def forward(self, x, params):
        out = x
        if self.use_biases:
            # Assume params is list of (weight, bias) tuples.
            assert len(params) % 2 == 0
            for i in range(0, len(params), 2):
                w = params[i]
                b = params[i + 1]
                if i == len(params) - 2:
                    out = F.linear(out, w, b)
                else:
                    out = F.relu(F.linear(out, w, b))
        else:
            for i in range(len(params)):
                w = params[i]
                out = F.linear(out, w) # Assume only one weight for now, so no ReLU.
        return out
