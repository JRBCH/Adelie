# June 2021
# Author: Julian Rossbroich

"""
inputs.py

Implements a simple input layer that can be connected to an AdelieConnection object
"""

from ..core.module import AdelieModule
import torch


class Input(AdelieModule):

    def __init__(self, n: int, **kwargs):

        super(Input, self).__init__(**kwargs)

        self.n = n
        self.y = torch.zeros(n)

    def forward(self, x, *args) -> torch.Tensor:
        self.y = x
        return self.y

    def reset_state(self, batchsize=None) -> None:

        if batchsize is not None:
            self.y = torch.zeros(batchsize, self.n)
        else:
            self.y = torch.zeros(self.n)
