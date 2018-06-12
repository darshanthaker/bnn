import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):


    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential( \
                nn.Linear(input_size, 5), \
                nn.ReLU(), \
                nn.Linear(5, 5), \
                nn.ReLU(), \
                nn.Linear(5, 1))

    def forward(self, x):
        return self.network(x)
