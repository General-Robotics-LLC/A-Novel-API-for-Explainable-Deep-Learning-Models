import torch
import torch.nn as nn

class SimpleMLP(torch.nn.Module):
    def __init__(self, num_input_nodes, num_hidden_nodes):
        super(SimpleMLP, self).__init__()
        self.lin1 = nn.Linear(num_input_nodes, num_hidden_nodes)
        self.activation = nn.Sigmoid()
        self.lin2 = nn.Linear(num_hidden_nodes, 10)  # Output layer for two classes

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        return x