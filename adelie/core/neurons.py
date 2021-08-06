import torch
from .module import AdelieModule
from typing import Union, Callable


# Custom Activation functions

def relu_squared(x):
    return torch.relu(x)**2

# Neuron Class

class Neurons(AdelieModule):
    """
    Group of Neurons
    """

    def __init__(self,
                 n: int,
                 bias: Union[float, torch.tensor] = 0.0,
                 tau: float = 1,
                 activation: Union[str, Callable] = "relu",
                 **kwargs):
        """
        Constructor for Neuron class

        :param n:           Number of neurons
        :param bias:        Bias for neurons (float or vector of size n)
        :param tau:         Integration time constant
        :param activation:  Activation function. Can be one of:
                                - string ('relu', 'sigmoid', 'tanh', 'linear', 'relu_squared')
                                - callable function
        :param kwargs:      Keyword parameters
        """

        super(Neurons, self).__init__(**kwargs)

        self.n = n
        self.tau = tau

        if isinstance(bias, torch.Tensor):
            assert len(bias) == n, 'bias dimensionality does not match number of neurons'

        # Register state variables
        # # # # # # # # # # # # # #

        self.register_buffer('bias', torch.tensor(bias, device=self.device), persistent=True)
        self.register_buffer('y', torch.zeros((self.batchsize, self.n), device=self.device), persistent=False)

        # Register activation function
        if callable(activation):
            self._activation = activation

        elif isinstance(activation, str):

            if activation.lower() == 'relu':
                self._activation = torch.relu
            if activation.lower() == 'relu_squared':
                self._activation = relu_squared
            if activation.lower() == 'sigmoid':
                self._activation = torch.sigmoid
            if activation.lower() == 'tanh':
                self._activation = torch.tanh
            if activation.lower() == 'linear':
                self._activation = lambda x: x

        else:
            raise ValueError('Invalid activation function. Must be a callable or a known string.')

    def __repr__(self):
        return self.y.__repr__()

    def precompute_decays(self, dt):
        """
        Precomputes decays and time constants

        Args:
            dt: timestep (in seconds)
        """

        self._dt_over_tau = dt / self.tau
        super().precompute_decays(dt)

    def reset_state(self, batchsize: int = 1):
        """
        Resets all state variables
        """

        self.y = torch.zeros((batchsize, self.n), device=self.device)

        super().reset_state(batchsize)

    def forward(self, x, *args):
        """
        A single integration step

        Args:
            x: input current

        Returns:
            y: The output of the population
        """
        self.y += self._dt_over_tau * (self._activation(x) - self.y)

        return self.y
