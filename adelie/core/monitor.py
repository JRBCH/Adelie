import torch

from ..functions.misc import getattr_deep

class OnlineTrainingMonitor():
    """
    Monitors variables throughout online training.
    """
    def __init__(self,
                 target,
                 variables,
                 rec_stepsize: int = None,
                 rec_dt: float = None,
                 dt: float = 0.001
                 ):
        """
        :param target:          target module to record state variables from.
        :param variables:       Iterable of variable names to record.
                                Can also be a tuple consisting of a name and function
                                Examples:
                                    - ['he', 'hi', 'y']
                                    - ['he', ('rate', func(y))]
        :param rec_stepsize:    Record every _th timestep
        :param rec_dt:          Record with timestep dt
        :param dt:              The timestep of the simulation
        """

        assert isinstance(dt, float), "invalid timestep. must be float."
        self.dt = dt

        if rec_stepsize is None and rec_dt is None:
            Warning("Auto-setting recording stepsize to every timestep.")
            self.rec_stepsize = 1
            self.rec_dt = dt

        elif rec_stepsize is not None and rec_dt is not None:
            raise ValueError("Must provide only one of rec_stepsize and rec_dt")

        elif isinstance(rec_stepsize, int):
            self.rec_stepsize = rec_stepsize
            self.rec_dt = rec_stepsize * self.dt

        elif isinstance(rec_dt, (int, float)):
            self.rec_dt = rec_dt
            self.rec_stepsize = int(rec_dt / self.dt)

        else:
            raise ValueError("Invalid Value for rec_stepsize or rec_dt")

        self.target = target
        self.variables = variables if variables is not None else []

        self.recording = {}

        self.t = 0
        self._recording_times = torch.tensor([0.0])
        self._rec_array_index = 0
        self._step_count = 0

        self.reset_state()

    def prepare_recording(self, T):
        """
        Prepares (creates or appends to) recording tensors

        :param T: how long the simulation runs for (in units of time)
        """
        self._step_count = 0

        # append recording times to time vector
        new_recording_times = torch.arange(self.t, self.t + T, self.rec_dt) + self.rec_dt
        self._recording_times = torch.cat([self._recording_times, new_recording_times])

        # update time to after simulation
        # (ensures that future training is properly appended)
        self.t = T

        # Make recording tensors
        for v in self.variables:
            if isinstance(v, str):
                self.recording[v] = torch.cat(
                    [
                        self.recording[v],
                        torch.zeros(len(new_recording_times), *getattr_deep(self.target, v).size())
                    ]
                )

            elif isinstance(v, tuple):
                self.recording[v] = torch.cat(
                    [
                        self.recording[v[0]],
                        torch.zeros(len(new_recording_times), *v[1](self.target).size())
                    ]
                )
                
    def reset_state(self) -> None:
        """
        Resets recording dictionary
        """

        self.recording = {}
        self.t = 0
        self._recording_times = torch.tensor([0.0])
        self._step_count = 0
        
        # Record the baseline (t=0.0 timepoint)
        for v in self.variables:
            if isinstance(v, str):
                self.recording[v] = torch.zeros(1, *getattr_deep(self.target, v).size())
                self.recording[v][0] = getattr_deep(self.target, v)

            elif isinstance(v, tuple):
                self.recording[v] = torch.zeros(1, *v[1](self.target).size())
                self.recording[v[0]][0] = v[1](self.target)

        self._rec_array_index = 1

    def get(self):
        """ Returns recording dictionary """

        self.recording['t'] = self._recording_times
        self.to("cpu")
        return self.recording

    def to(self, device) -> None:
        """ Moves recording dictionary to device"""

        if self.recording != {}:

            for v in self.variables:
                if isinstance(v, str):
                    self.recording[v] = self.recording[v].to(device)

                elif isinstance(v, tuple):
                    self.recording[v[0]] = self.recording[v[0]].to(device)

    def record(self) -> None:
        """ Call at every timestep"""

        self._step_count += 1

        # If timepoint i should be recorded:
        if self._step_count == self.rec_stepsize:

            for v in self.variables:
                if isinstance(v, str):
                    self.recording[v][self._rec_array_index] = getattr_deep(self.target, v)

                elif isinstance(v, tuple):
                    self.recording[v[0]][self._rec_array_index] = v[1](self.target)

            self._rec_array_index += 1
            self._step_count = 0
