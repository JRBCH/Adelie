# June 2021
# Author: Julian Rossbroich

"""
module.py

Implements the Adelie base module, inherits from torch.nn.Module
"""

import torch


class AdelieModule(torch.nn.Module):
    """
    Abstract base class for all ApproxGroup objects.
    Inherits from torch.nn.Module
    """

    def __init__(self, device: str = "cpu", batchsize: int = 1, **kwargs):
        super(AdelieModule, self).__init__()

        self.device = device
        self.batchsize = batchsize

    def forward(self, x, *args) -> torch.Tensor:
        """ forward pass """
        pass

    def reset_state(self, batchsize: int = 1) -> None:
        """ Resets state variables in all Sub-modules """

        self.batchsize = batchsize

        for module in self.children():
            module.reset_state(batchsize)

    def save_state(self, path):
        """
        Saves state dict at path

        Args:
            path: (string) Absolute or relative path
        """
        torch.save(self.state_dict(), path)

        # TODO
        # Log saving state and path

    def load_state(self, path):
        """
        Loads state dict from file

        Args:
            path: (string) Absolute or relative path
        """
        self.load_state_dict(torch.load(path))

    def to(self, *args, **kwargs):
        """ Added to method of torch.nn.Module class """

        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        for module in self.children():
            module.device = device

        self.device = device

        super().to(*args, **kwargs)

    def precompute_decays(self, dt):
        """
        Precomputes decays and time constants

        Args:
            dt: timestep (in seconds)
        """

        for module in self.children():
            module.precompute_decays(dt)

