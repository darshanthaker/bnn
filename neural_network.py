import torch
import torch.nn.functional as F

# TODO(dbthaker): Change this to use torch.functional instead of torch.nn
class VanillaNeuralNetwork(nn.Module):


    def __init__(self, input_size):
        super(VanillaNeuralNetwork, self).__init__()
        self.network = nn.Sequential( \
                nn.Linear(input_size, 5), \
                nn.ReLU(), \
                nn.Linear(5, 5), \
                nn.ReLU(), \
                nn.Linear(5, 1))

    def forward(self, x):
        return self.network(x)

class NeuralNetwork(nn.Module):


    # Can we introduce a ref model and copy its weights?
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
    
    def forward(self, x, weights, biases):
        pass
