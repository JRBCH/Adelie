# June 2021
# Author: Julian Rossbroich

"""
module.py

Implements the Adelie base module, inherits from torch.nn.Module
"""

import torch
import numpy as np


class AdelieModule(torch.nn.Module):
    """
    Abstract base class for all ApproxGroup objects.
    Inherits from torch.nn.Module
    """

    def __init__(self, device: str = "cpu", batchsize: int = 1, mode: str = 'online', **kwargs):
        super(AdelieModule, self).__init__()

        self.device = device
        self.batchsize = batchsize

        self.mode = mode

    def forward(self, x, *args) -> torch.Tensor:
        """ forward pass """
        pass

    def converge(self,
                 x,
                 target_delta: float = 1e-2,
                 stability_iterations: int = None,
                 *args) -> torch.Tensor:
        """
        Calculates the converged state of the network given an input.

        If target_delta is supplied, converges the network until the
            total change in output is less than target_delta

        Alternatively, if stability_iterations is supplied, iterates
            for that many iterations and returns the output
        """

        # if number of iterations is supplied, iterate
        if isinstance(stability_iterations, int):
            for i in range(stability_iterations):
                out = self.forward(x)

        # else if target stability criterion is supplied, iterate until convergence
        elif isinstance(target_delta, float):
            iteration_count = 1
            delta = np.inf
            out = self.forward(x).clone()

            while delta > target_delta:
                out_new = self.forward(x).clone()
                delta = (out - out_new).abs().max()
                out = out_new
                iteration_count += 1

        else:
            raise ValueError('Either stability_iterations (int) or target_delta (float) must be supplied')

        return out

    def _test_convergence(self,
                         x,
                         stability_iterations: int = 10):

        state = list()

        for i in range(stability_iterations):
            state.append(self.forward(x).clone())

        return state

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

    def set_mode(self, new_mode):
        """
        Sets mode to either ONLINE or CONVERGED
        :param mode: 'online' or 'converged'
        """

        if new_mode in ('online', 'converged', 'o', 'c'):
            self.mode = new_mode

            for module in self.children():
                module.set_mode(new_mode)

        else:
            raise ValueError('Invalid mode. must be "online" or "converged"')

