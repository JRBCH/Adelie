# Aug 2021
#
# Author: Julian Rossbroich

import numpy as np
from progress.bar import FillingCirclesBar as Bar
from torch.utils.data import DataLoader, IterableDataset, Dataset
from typing import Union, Optional, Iterable, Dict, Tuple

# Local
from ..core.module import AdelieModule
from ..core.monitor import OnlineTrainingMonitor

"""
Trainer for online learning paradigms
"""


class ConvergedTrainer:

    def __init__(self, model, data, batchsize, monitors=()):

        self.batchsize = batchsize
        self.monitors = list()

        # add variables monitors
        for m in monitors:
            self.add_monitor(m)

        # Prepare network object
        # # # # # # # # # # # # # #

        assert isinstance(model, AdelieModule), "model must be an adelie.Module instance"
        self.model = model
        self.model.reset_state(batchsize)

        # Prepare dataset object
        # # # # # # # # # # # # # #

        assert isinstance(data, Dataset), "dataset must be a torch.utils.data.Dataset instance"
        self.data = data

    def add_monitor(self, monitor):
        """
        Adds a variable monitor to the Trainer
        :param monitor: monitor object
        """

        # Needs assertion check

        #assert isinstance(monitor, EpochMonitor), "Not a valid monitor object"

        self.monitors.append(monitor)

    def train(self, epochs=None, steps=None, shuffle_batches=True,
              stability_target_delta=1e-2, stability_iterations=None):
        """
        Either ``epochs`` or ``steps`` have to be provided to train the model.
		If ``epochs`` are provided, the training dataset will be iterated over that many times.
		If ``steps`` are provided instead, that many training batches will be iterated over.
        """

        # PREPARE MODEL AND MONITORS
        # # # # # # # # #

        # Set to converged mode
        self.model.set_mode('converged')

        # Pre-compute decays
        self.model.precompute_decays(1)

        # ASSERTION CHECKS
        # # # # # # # # #
        assert self.model.batchsize == self.batchsize

        # PREPARE DATA LOADER
        # # # # # # # # #

        # Pin memory if CPU
        pin_memory = False if self.model.device == "cpu" else True

        loader = iter(
            DataLoader(
                self.data,
                shuffle=shuffle_batches,
                batch_size=self.batchsize,
                pin_memory=pin_memory
            )
        )

        # TRAINING LOOP
        # # # # # # # # #

        if epochs is None and steps is None:
            raise ValueError("provide either `epochs` or `steps` for training")

        # TRAINING FOR X EPOCHS
        elif isinstance(epochs, (int, float)):
            assert steps is None, "If `epochs` are provided, do not provide `steps`."
            epochs = int(epochs)

            # prepare monitors
            for m in self.monitors:
                m.prepare_recording(epochs)

            bar = Bar(
                "Training network...",
                max=epochs,
                suffix="%(remaining)d Epochs remaining."
                " Estimated time until completion: %(eta_td)s",
            )

            for e in range(epochs):

                for x in loader:

                    x = x.to(self.model.device)

                    self.model.converge(x,
                                        target_delta=stability_target_delta,
                                        stability_iterations=stability_iterations)

                    self.model.update()

                # recording
                for m in self.monitors:
                    m.record()

                bar.next()

            bar.finish()
            print("Total runtime: {}".format(bar.elapsed_td))
            print("Average runtime per epoch: {} ms".format(np.round(bar.avg * 1000, 3)))


        # TRAINING FOR X STEPS
        elif isinstance(steps, (int, float)):
            assert epochs is None, "If `steps` are provided, do not provide `epochs`."
            steps = int(steps)

            # prepare monitors
            for m in self.monitors:
                m.prepare_recording(steps)

            bar = Bar(
                "Training network...",
                max=steps,
                suffix="%(remaining)d Steps remaining."
                " Estimated time until completion: %(eta_td)s",
            )

            for s in range(steps):
                try:
                    x = next(loader).to(self.model.device)

                except (StopIteration, IndexError):
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    loader = iter(
                        DataLoader(
                            self.data,
                            shuffle=shuffle_batches,
                            batch_size=self.batchsize,
                            pin_memory=pin_memory
                        )
                    )
                    x = next(loader).to(self.model.device)

                self.model.converge(x,
                                    target_delta=stability_target_delta,
                                    stability_iterations=stability_iterations)
                self.model.update()

                # recording
                for m in self.monitors:
                    m.record()

                bar.next()

            bar.finish()
            print("Total runtime: {}".format(bar.elapsed_td))
            print("Average runtime per step: {} ms".format(np.round(bar.avg * 1000, 3)))

        else:
            raise ValueError("Invalid input for `epochs` or `steps`")
