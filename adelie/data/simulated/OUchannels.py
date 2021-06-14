import torch
import math

from torch.utils.data import IterableDataset
from typing import Union, Optional, Iterable, Dict, Tuple

# Local
from adelie.functions.initialize import generate_gaussian_tuning_curves


class OUprocess:

    """
    Provides independent inputs in form of an Ornstein Uhlenbeck process.

    :param n:           number or shape of signals
    :param tau:         time constant of OU process
    :param amp:         Fluctuation amplitude
    :param dt:          timestep
    """

    def __init__(
        self,
        n: Union[int, Tuple],
        tau: Union[float, Iterable] = 0.2,      # seconds
        mu: Union[float, Iterable] = 0,         # A.U.
        sigma: Union[float, Iterable] = 1,      # A.U.
        dt: float = 0.001,
    ) -> None:

        self.n = n
        self.tau = tau
        self.mu = mu
        self.sigma = sigma

        self.dt = dt

        # prep work
        self._precompute_constants()

        # State variables
        self.state = torch.ones(self.n) * self.mu

    def _precompute_constants(self) -> None:
        self._dt_over_tau = self.dt / self.tau
        self._sd_sqrtdt = self.sigma * math.sqrt(self.dt)

    def _sample(self) -> torch.Tensor:
        return torch.randn(self.n)

    def step(self) -> torch.Tensor:

        self.state += (-(self.state - self.mu) * self._dt_over_tau + self._sd_sqrtdt * self._sample())

        return self.state

    def reset_state(self):
        """
        Resets state
        """
        self.state = torch.ones(self.n) * self.mu


class MixedChannelInputs(IterableDataset):

    def __init__(self,
                 n_signals,
                 n_inputs,
                 batchsize: int = 1,
                 cutoff: float = 0.0,
                 amp: float = 1.0,
                 dt: float = 0.001,
                 tuning_width: float = 0.01,
                 **kwargs
                 ):

        """
        Simulated data to mimic different input channels (e.g. auditory frequencies).
        The data can implement correlations between input channels by reading out the independent channels through
        a weight matrix of overlapping gaussian functions.

        :param n_signals:       number of independent signals
        :param n_inputs:        number of input neurons
        :param c:               constant c to subtract from OU process (controls sparseness)
        :param amp:             amplitude of OU processes
        :param dt:              timestep
        :param batchsize:       batchsize
        :param tuning_width:    tuning width of the input neurons.

        """

        self.n = n_inputs
        self.dt = dt
        self.batchsize = batchsize

        self.cutoff = cutoff
        self.amp = amp

        self.Channels = OUprocess((batchsize, n_signals), dt=dt, **kwargs)
        self.tuning = generate_gaussian_tuning_curves(n_signals, n_inputs, tuning_width)

        # normalize tuning curves
        self.tuning *= 1 / self.tuning.sum(0)

    def __iter__(self):
        """
        Returns an iterator over the dataset
        :return:
        """

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = self.batchsize

        else:  # in a worker process split workload
            per_worker = int(math.ceil(self.batchsize / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.batchsize)

        return iter(self._data[:, iter_start:iter_end, :])

    def simulate(self, T, device='cpu') -> torch.Tensor:
        """
        Generates output for time T
        """

        # generate
        timesteps = int(T/self.dt)
        out = torch.zeros(timesteps, self.batchsize, self.n)

        for ix in range(timesteps):
            out[ix, :] = self.Channels.step()

        self._data = torch.matmul((out - self.cutoff).clip(min=0) * self.amp, self.tuning).to(device)

        return self._data


    def get_output(self, s):
        """
        Given a predetermined signal, get x
        """
        return torch.matmul(s, self.tuning)

    def reset_state(self):
        """
        reset state
        """
        self.Channels.reset_state()
