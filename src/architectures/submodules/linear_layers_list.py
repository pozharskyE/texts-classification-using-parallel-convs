from typing import List
from torch import nn

class LinearLayersList(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int]):
        super().__init__()

        # Type checking
        if type(hidden_sizes) != list:
            raise ValueError(f'hidden_sizes must be python list (not {type(hidden_sizes)}). Even if you want only 1 hidden layer, make it [num_units]')
        for hs in hidden_sizes:
            if type(hs) != int:
                raise ValueError(f'At least one of given values in hidden_sizes is not python int type (incorrect element: {hs} of type {type(hs)})')



        self.linear_layers_list = nn.ModuleList()

        # First layer
        self.linear_layers_list.append(nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU()
            ))

        # Other layers
        if len(hidden_sizes) > 1:

          for i in range(1, len(hidden_sizes)):

              self.linear_layers_list.append(
                  nn.Sequential(
                      nn.Linear(hidden_sizes[i-1], hidden_sizes[i]),
                      nn.ReLU()
                  )
              )


    def forward(self, inputs):
        var = inputs
        for layer in self.linear_layers_list:
            var = layer(var)
        return var