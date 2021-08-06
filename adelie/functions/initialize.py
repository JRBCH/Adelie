import torch
import numpy as np

def generate_gaussian_tuning_curves(n_inputs, n_outputs, sigma):
    """
    Generates input weights that correspond to gaussian tuning curves

    :param n_inputs:    Nr. presynaptic neurons
    :param n_outputs:   Nr. postsynaptic neurons
    :param sigma:       Gaussian STD
    """

    # make gaussian
    gaussian = torch.exp((-(int(n_inputs/2) - torch.arange(n_inputs)) ** 2) / (2 * sigma))

    # distribute peaks equally
    peaks = np.arange(0, n_inputs, n_inputs/n_outputs)

    # empty weight matrix
    w = torch.zeros(n_inputs, n_outputs)

    # iterate over neurons
    for i in range(n_outputs):
        w[:, i] = gaussian.roll(int(peaks[i]))

    return w

def generate_stacked_identity(Npre, Npost):

    if Npre < Npost:
        # if pre < post, concatenate matrices and form [Npre x Nost] tensor
        return torch.cat([torch.eye(Npre)] * int(np.ceil(Npost / Npre))).T[
            :Npre, :Npost
        ]

    elif Npost < Npre:
        # if post < pre, concatenate matrices and form [Npre x Nost] tensor
        return torch.cat([torch.eye(Npost)] * int(np.ceil(Npre / Npost)))[:Npre, :Npost]

    else:
        # else the two are the same. return identity
        return torch.eye(Npre)