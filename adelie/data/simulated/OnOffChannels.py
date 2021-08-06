import torch

from torch.utils.data import Dataset
from typing import Union, Optional, Iterable, Dict, Tuple

# Local
from adelie.functions.initialize import generate_gaussian_tuning_curves


class MixedOnOffChannelInputs(Dataset):

    def __init__(self,
                 n: int,
                 datapoints: int,
                 amp_min: float = 1,
                 amp_max: float = 1,
                 amp_noise: float = 0.0,
                 prob: float = 0.1,
                 tuning_width: float = 0.01,
                 device: str = 'cpu'
                 ):

        """
        Simulated data to mimic different input channels (e.g. auditory frequencies).
        The data can implement correlations between input channels by reading out the independent channels through
        a weight matrix of overlapping gaussian functions.

        :param n:               number of inputs
        :param datapoints       number of datapoints to simulate
        :param amp_min:         minimum amplitude
        :param amp_max:         maximum amplitude
        :param prob:            probability that each channel is active
        :param tuning_width:    tuning width of the input neurons. Set to 0.01 for independent channels
        :param device:          device to store data on
        """

        self.n = n

        self.amp_min = amp_min
        self.amp_max = amp_max
        self.amp_noise = amp_noise

        self.prob = prob

        # create mixing matrix
        self.tuning = generate_gaussian_tuning_curves(n, n, tuning_width)

        # normalize mixing matrix
        self.tuning *= 1 / self.tuning.sum(0)

        # simulate data
        self.simulate(datapoints=datapoints, device=device)

    def simulate(self, datapoints, device='cpu') -> torch.Tensor:
        """
        Generates X datapoints
        """

        # Create [T x N] matrix of active neurons at each timestep
        active = torch.rand(datapoints, self.n) <= self.prob

        # Sample amplitudes for peaks and noise
        amps = self.amp_min + torch.rand(datapoints, self.n) * (self.amp_max - self.amp_min)
        noise = torch.rand(datapoints, self.n) * self.amp_noise

        # Assemble signals
        signals = active * amps + noise

        # calculate lifetime sparseness
        self._lifetime_sparseness = (signals.mean(0)**2 / (signals**2).mean(0)).mean()

        # Multiply with mixing matrix
        self._data = torch.matmul(signals, self.tuning).to(device)

        return self._data

    def get_output(self, s):
        """
        Given a predetermined signal, get x
        """
        return torch.matmul(s, self.tuning)

    @property
    def lifetime_sparseness(self):
        """
        returns Lifetime sparseness of each independent signal
        :return: E[s]**2 / E[s**2]
        """
        return self._lifetime_sparseness

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self._data[idx]

