from typing import List
import torch
from torch import nn


class Parallel1DConvsLayer(nn.Module):
    def __init__(self, input_channels: int, kernel_sizes: List[int] = [3, 4, 5], out_channels_per_kernel: int = 64):
        super().__init__()

        self.multiple_1d_convs = nn.ModuleList()

        for kernel_size in kernel_sizes:

            self.multiple_1d_convs.append(
                nn.Sequential(
                    nn.Conv1d(input_channels, out_channels_per_kernel,
                              kernel_size=kernel_size, padding=(kernel_size // 2)),
                    nn.AdaptiveMaxPool1d(1)
                )
            )

    def forward(self, inputs):
        outputs = torch.concat([conv_1d_max(inputs)
                               for conv_1d_max in self.multiple_1d_convs], dim=1)
        return outputs
