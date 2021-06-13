"""
connection.py module

Implements the `layer` class, representing a single input-output layer.
"""

import torch
from typing import Union
from .module import AdelieModule


class AdelieConnection(AdelieModule):

    def __init__(
        self,
        source,
        target,
        sign: int,
        gain: float = 'auto',
        w: Union[torch.Tensor, float, str] = None,
        wmin: float = None,
        wmax: float = None,
        zero_diag: bool = False,
        **kwargs
    ) -> None:
        """
        Constructor for`AdelieConnection` class.

        :param source: A population of neurons or inputs from which the connection originates.
        :param target: A population of neurons to which the connection connects.
        :param w: synaptic weight matrix
        :param gain: modulator for synaptic strength
        :param wmin: minimum weight
        :param wmax: maximum weight
        :param zero_diag: weather to set the diagonal of the weight matrix to zero
                            (weather to allow for autapses if source==target)
        :param plastic: weather synapse is plastic
        """

        super(AdelieConnection, self).__init__(**kwargs)

        self.source = source
        self.target = target

        # Sign of synapse
        # # # # # # # # # #

        assert sign in [1, -1], 'Sign must be 1 (exc) or -1 (inh)'
        self.sign = sign

        # Calculate gain
        # # # # # # # # # #

        if gain == 'auto':
            self.gain = self.target.n / self.source.n
        elif isinstance(gain, (float, int)):
            self.gain = gain
        else:
            raise ValueError('invalid gain. Must be "auto", float or int')

        self._output_modifier = self.sign * self.gain

        # Hard-bounds for weights
        # # # # # # # # # #

        self.wmin = wmin
        self.wmax = wmax

        if self.wmin is not None or self.wmax is not None:
            self._clamp_weights = True
        else:
            self._clamp_weights = False

        # Allow or don't allow for zero diagonals
        # # # # # # # # # #

        if self.source.n == self.target.n:
            self.zero_diag = zero_diag
        else:
            self.zero_diag = False

        # Weight initialization
        # # # # # # # # # #

        if w == "random" or w is None:
            if self.wmin is not None and self.wmax is not None:
                w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin)
            else:
                w = torch.rand(source.n, target.n)

        elif isinstance(
            w, (int, float)
        ):  # if w is a single number, scale all synapses by w
            w = torch.ones(source.n, target.n) * w

        assert w.shape == (source.n, target.n), "ERROR: Faulty weight matrix shape"

        # Initialize state variables
        # # # # # # # # # # # # # #

        self.register_parameter(
            "w", torch.nn.Parameter(w.to(self.device), requires_grad=False)
        )

        self.register_buffer(
            'out', torch.zeros(self.batchsize, self.target.n, device=self.device), persistent=False
        )

    def __repr__(self):
        return self.w.__repr__()

    def reset_state(self, batchsize: int = 1) -> None:
        """
        Resets state variables
        """

        self.out = torch.zeros(self.batchsize, self.target.n, device=self.device)
        super().reset_state(batchsize)

    def forward(self, *args) -> torch.Tensor:
        """
        Forward pass
        Returns: X % W
        """

        self.out = torch.matmul(self.source.y, self.w) * self._output_modifier
        return self.out

    def update(self) -> None:
        """ Update weights """
        torch.clamp(self.w, self.wmin, self.wmax, out=self.w)

        if self.zero_diag:
            self.w.fill_diagonal_(0)
