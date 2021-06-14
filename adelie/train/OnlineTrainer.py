# June 2021
#
# Author: Julian Rossbroich

import numpy as np
from progress.bar import FillingCirclesBar as Bar
from torch.utils.data import DataLoader, IterableDataset
from typing import Union, Optional, Iterable, Dict, Tuple

# Local
from ..core.module import AdelieModule
from ..core.monitor import OnlineTrainingMonitor

"""
Trainer for online learning paradigms
"""


class OnlineTrainer:

    def __init__(self, model, data, batchsize, dt, monitors=()):

        self.dt = dt
        self.batchsize = batchsize
        self.monitors = list()

        # add variables monitors
        for m in monitors:
            self.add_monitor(m)

        # Prepare network object
        # # # # # # # # # # # # # #

        assert isinstance(model, AdelieModule), "model must be an adelie.Module instance"
        self.model = model

        # Prepare dataset object
        # # # # # # # # # # # # # #

        assert isinstance(data, IterableDataset), "dataset must be an IterableDataset instance"
        self.data = data

    def add_monitor(self, monitor):
        """
        Adds a variable monitor to the Trainer
        :param monitor: monitor object
        """

        assert isinstance(monitor, OnlineTrainingMonitor), "Not a valid monitor object"
        assert monitor.dt == self.dt, "Monitor timestep does not match trainer timestep"

        self.monitors.append(monitor)

    def train(self, time, update_interval=1, datagen_chunksize='auto', max_chunksize=1e6):
        """
        Trains the network for a given time

        :param time: how long to run the simulation for (usually in seconds)
        :param update_interval: interval for weight updates (in timesteps!)
        """

        # PREPARE MODEL AND MONITORS
        # # # # # # # # #

        # Pre-compute decays
        self.model.precompute_decays(self.dt)

        # prepare monitors
        for m in self.monitors:
            m.prepare_recording(time)

        # PREPARE COUNTERS
        # # # # # # # # #

        update_counter = update_interval
        timesteps = int(time / self.dt)

        # ASSERTION CHECKS
        # # # # # # # # #
        assert self.model.batchsize == self.batchsize

        assert self.data.dt == self.dt
        assert self.data.batchsize == self.batchsize

        # PREPARE DATA LOADER
        # # # # # # # # #

        # automate datageneration chunksize
        if datagen_chunksize == 'auto':

            if timesteps * self.batchsize <= max_chunksize:
                datagen_chunksize = time
            else:
                datagen_chunksize = max_chunksize * self.dt / self.batchsize


        #simulate first batch of data
        self.data.simulate(datagen_chunksize)

        # Pin memory if CPU
        pin_memory = False if self.model.device == "cpu" else True

        loader = iter(
            DataLoader(
                self.data,
                batch_size=None,  # must be None, because batchsize is supplied by dataset
                pin_memory=pin_memory
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
                x = next(loader).to(self.model.device)

            except (StopIteration, IndexError):
                # StopIteration is thrown if dataset ends

                # resample data
                time_remaining = (timesteps - step) * self.dt

                # for remaining time if smaller than chunksize
                if time_remaining <= datagen_chunksize:
                    self.data.simulate(time_remaining)

                # for chunksize otherwise
                else:
                    self.data.simulate(datagen_chunksize)

                # reinitialize data loader
                loader = iter(
                    DataLoader(
                        self.data,
                        batch_size=None,  # must be None, because batchsize is supplied by dataset
                        pin_memory=pin_memory
                    )
                )
                x = next(loader).to(self.model.device)

            self.model.forward(x)

            # weight update
            if update_counter == update_interval:
                self.model.update()
                update_counter = 0

            # recording
            for m in self.monitors:
                m.record()

            bar.next()
            update_counter += 1

        # FINISH TRAINING
        # # # # # # # # #

        bar.finish()
        print("Total runtime: {}".format(bar.elapsed_td))
        print("Average runtime per timestep: {} ms".format(np.round(bar.avg * 1000, 3)))





