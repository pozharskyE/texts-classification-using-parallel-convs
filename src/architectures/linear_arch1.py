from typing import List

from torch import nn
from submodules.linear_layers_list import LinearLayersList


class LinearBinaryClass(nn.Module):
    def __init__(self, input_size, hidden_sizes: List[int] = [256, 256, 1]):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_sigmoid = nn.Sequential(
            LinearLayersList(input_size, hidden_sizes=hidden_sizes),
            nn.Sigmoid()
        )

    def forward(self, X):
        X = self.flatten(X)
        return self.linear_relu_sigmoid(X)
