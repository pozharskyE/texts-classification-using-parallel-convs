from typing import List
from torch import nn

from submodules.parallel_1dconvs_layer import Parallel1DConvsLayer
from submodules.linear_layers_list import LinearLayersList


class Parallel1DConvsLinearClass(nn.Module):
    def __init__(self, input_channels: int, kernel_sizes: List[int] = [3, 4, 5], out_channels_per_kernel: int = 64, linear_hidden_sizes: List[int] = [256, 256, 1]):
        super().__init__()

        self.input_channels = input_channels 
        self.kernel_sizes = kernel_sizes
        self.out_channels_per_kernel = out_channels_per_kernel
        self.linear_hidden_sizes = linear_hidden_sizes

        self.paral_1dconvs_max_linear_relu = nn.Sequential(

            Parallel1DConvsLayer(
                input_channels=input_channels,
                kernel_sizes=kernel_sizes,
                out_channels_per_kernel=out_channels_per_kernel),


            nn.Flatten(),


            LinearLayersList(
                input_size=(out_channels_per_kernel * len(kernel_sizes)),
                hidden_sizes=linear_hidden_sizes,
                inter_activation=nn.ReLU,
                output_activation=None
            ),

        )

        #  Set output activation function (for binary - sigmoid, for multiclass - softmax)
        if linear_hidden_sizes[-1] == 1:
            self.output_activation = nn.Sigmoid()
        elif linear_hidden_sizes[-1] > 1:
            self.output_activation = nn.Softmax()
        else:
            raise ValueError(
                f'This error was raised because last hidden size is not >= 1. Please ensure, that the last element in linear_hidden_sizes list, that you gave, is >= 1 (this is aslo an output size)')

    def forward(self, X):
        logits = self.paral_1dconvs_max_linear_relu(X)
        return self.output_activation(logits)
