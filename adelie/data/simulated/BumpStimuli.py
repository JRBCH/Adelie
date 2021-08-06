import torch
import math

from torch.utils.data import Dataset
from typing import Union, Optional, Iterable, Dict, Tuple


class MixedChannelInputs(Dataset):

    def __init__(self,
                 n_inputs,
                 amp_min: float = 0.0,
                 amp_max: float = 1.0,
                 width: Union[float, str] = 'auto',
                 batchsize: int = 1,
                 **kwargs
                 ):

        """
        Simulated data to mimic a continuous input dimension (e.g. Frequency).
        A moving gaussian bump with fixed amplitude is moving to a new random
        location every `duration / dt` timesteps.

        :param n_inputs:        number of input neurons
        :param amp_floor:       baseline firing rate
        :param amp_peak:        peak firing rate
        :param width:           width of moving bump
        :param batchsize:       batchsize
        """
        self.amp_floor = amp_floor
        self.amp_peak = amp_peak
        self.n = n_inputs

        # Width of the gaussian peak
        if width is 'auto':
            self.width = int(self.n / 10)
        else:
            self.width = int(width)



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

        # cut off OU process
        cutoff_signal = (out - self.cutoff).clip(min=0)

        # calculate lifetime sparseness
        self._lifetime_sparseness = (cutoff_signal.mean(0)**2 / (cutoff_signal**2).mean(0)).mean()

        # normalize each channel to a mean output of amp
        cutoff_signal *= self.amp / cutoff_signal.mean(0)

        # compute data
        self._data = torch.matmul(cutoff_signal, self.tuning).to(device)

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

    @property
    def lifetime_sparseness(self):
        """
        returns Lifetime sparseness of each independent signal
        :return: E[s]**2 / E[s**2]
        """
        return self._lifetime_sparseness


