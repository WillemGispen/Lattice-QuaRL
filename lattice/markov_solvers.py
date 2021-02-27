""" Defines class WaitingTime. """

import torch
import torch.nn as nn


class WaitingTime(nn.Module):
    """ Does one step of the Waiting Time method. """
    def __init__(self, hparams):
        super(WaitingTime, self).__init__()
        self.ising = hparams.potential.startswith('ising')
        self.holding = hparams.holding

    def forward(self, state, rates, k=1):
        batch_size, L = state.shape[0], state.shape[-1]
        rates = rates.view(batch_size, -1, L, L)

        # generate waiting times
        if self.holding:  # transition indices decided by multinomial
            dt = torch.distributions.exponential.Exponential(rates.sum(dim=(-1, -2, -3))).sample()
            action = rates.view(batch_size, -1).multinomial(1).squeeze()
        else:  # transition indices decided by taking minimum
            times = torch.distributions.exponential.Exponential(rates).sample()
            dt, action = times.view(batch_size, -1).min(axis=1)

        # make transition
        if self.ising:  # flip spin i.e. 1->0 and 0->1
            state = make_ising_transition(state, action[:,0], k=k)
        else:
            state = make_particle_hop(state, action, rates.shape)

        return state, dt, action


class MarkovChain(nn.Module):
    """ Does one step of a discrete time Markov chain. """
    def __init__(self, hparams):
        super(MarkovChain, self).__init__()
        self.ising = hparams.potential.startswith('ising')
        if self.ising:
            self.out_channels = 1
        elif hparams.potential.startswith('j1'):
            self.out_channels = 4
        else:
            self.out_channels = 2

    def forward(self, state, action, k=1):
        # make transition
        if self.ising:  # flip spin i.e. 1->0 and 0->1
            state = make_ising_transition(state, action, k=k)
        else:
            policy_shape = (state.shape[0], self.out_channels) + state.shape[-2:]
            state = make_particle_hop(state, action, policy_shape)

        return state


def make_adjacent_states(state, ising, out_channels):
    """ Makes all adjacent states from a given state. """
    B = state.shape[0]
    L = state.shape[-1]
    states = state.repeat_interleave(out_channels * L**2, dim=0)
    actions = torch.arange(out_channels * L**2).repeat(B)
    if ising:
        states = make_ising_transition(states, actions)
    else:
        policy_shape = (states.shape[0], out_channels) + state.shape[-2:]
        states = make_particle_hop(states, actions, policy_shape)
        state = state * torch.le(state, 1.0) * torch.ge(state, 0.0)

    return states.view(B, -1, L, L)


def make_ising_transition(state, action, k=1):
    """  Flips one spin. """
    batch_size = state.shape[0]
    y = state.clone().view(batch_size, -1)
    for i in range(k):
        y[range(batch_size), action[:, i]] -= 1
        y[range(batch_size), action[:, i]] *= -1
    y = y.view_as(state)
    return y


def make_particle_hop(state, action, rates_shape):
    """ Makes one particle hop by flipping the spins of both sites"""
    batch_size = state.shape[0]

    if rates_shape[-3] == 2:  # if nearest neighbours
        l, m, l_, m_ = find_particle_position(action, rates_shape)
    # else:  # if also next nearest neighbors
    #     l, m, l_, m_ = j1j2_find_particle_position(action, rates_shape)

    if torch.lt(state, 0.0).any():
        raise ValueError('negative rates')

    y = state.clone()
    y[range(batch_size), l, m] -= 1
    y[range(batch_size), l, m] *= -1

    y[range(batch_size), l_, m_] -= 1
    y[range(batch_size), l_, m_] *= -1

    return y


def find_particle_position(action, rates_shape, return_d=False):
    """ Uses unravel_index to find the site the particle leaves and enters. """
    batch_size, out_channels, lattice_size, _ = rates_shape
    batch_size = action.shape[0]

    # convert linear indices to indices appropriate for state.shape
    d, l, m = unravel_index(action, (out_channels, lattice_size, lattice_size))

    # convert d from range(2) to (N, E)
    directions = torch.tensor(
        [[-1, 0], [0, 1]],
        device=action.device).repeat(batch_size, 1, 1)
    dl, dm = directions[range(batch_size), d].chunk(2, dim=-1)
    dl, dm = dl.squeeze(dim=-1), dm.squeeze(dim=-1)

    l_ = torch.fmod((l + dl), lattice_size)
    m_ = torch.fmod((m + dm), lattice_size)

    if return_d:
        return l, m, l_, m_, d
    else:
        return l, m, l_, m_


def unravel_index(index, shape):
    """ Analog of numpy.unravel_index. """
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))
