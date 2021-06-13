# June 2021
#
# Author: Julian Rossbroich

from progress.bar import FillingCirclesBar as Bar
from torch.utils.data import DataLoader, Dataset
from typing import Union, Optional, Iterable, Dict, Tuple

# Local
from ..core.module import AdelieModule

"""
Trainer for online learning paradigms
"""


class OnlineTrainer:

    def __init__(self, network, data, batchsize, dt):

        self.dt = dt
        self.batchsize = batchsize

        # Prepare network object
        # # # # # # # # # # # # # #

        assert isinstance(AdelieModule, network), "network must be an adelie.Module instance"
        network.precompute_decays(dt)

        self.network = network

        # Prepare dataset object
        # # # # # # # # # # # # # #

        assert isinstance(Dataset, data), "dataset must be an adelie.data.OnlineDataset instance"
        self.data = data

    def train(self, time, update_interval=1):
        """
        Trains the network for a given time

        :param time: how long to run the simulation for (usually in seconds)
        :param update_interval: interval for weight updates (in timesteps!)
        """

        # PREPARE COUNTERS
        # # # # # # # # #

        update_counter = update_interval
        timesteps = int(time / self.dt)

        # ASSERTION CHECKS
        # # # # # # # # #
        assert self.network.dt == self.dt
        assert self.network.batchsize == self.batchsize

        assert self.data.dt == self.dt
        assert self.data.batchsize == self.batchsize

        # PREPARE DATA LOADER
        # # # # # # # # #

        # Pin memory if CPU
        pin_memory = False if self.network.device == "cpu" else True

        loader = iter(
            DataLoader(
                self.data,
                batch_size=self.batchsize,
                shuffle=False,
                pin_memory=pin_memory,
            )
        )

        # TRAINING LOOP
        # # # # # # # # #

        bar = Bar(
            "Training network...",
            max=timesteps,
            suffix="%(percent)d%% simulated."
                   " Estimated time until completion: %(eta_td)s",
        )

        for step in range(timesteps):
            try:
                x = next(loader)

            except (StopIteration, IndexError):
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                loader = iter(
                    DataLoader(
                        self.data,
                        batch_size=self.batchsize,
                        shuffle=False,
                        pin_memory=pin_memory,
                    )
                )
                x = next(loader)

            # forward pass
            x = x.to(self.network.device)
            self.network.forward(x)

            # weight update
            if update_counter == update_interval:
                self.network.update()
                update_counter = 0

            bar.next()
            update_counter += 1

        # FINISH TRAINING
        # # # # # # # # #

        bar.finish()
        print("Total runtime: {}".format(bar.elapsed_td))
        print("Average runtime per timestep: {} ms".format(np.round(bar.avg * 1000, 3)))





