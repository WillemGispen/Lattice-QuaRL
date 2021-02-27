"""
Module for lattice potentials
"""
import torch


def ising_potential(hparams):
    """
    Potential for square Ising model with periodic boundary conditions
    """
    J = getattr(hparams, 'J', 1.0)
    def potential(state, passive_rates):
        x = state
        x_up = torch.roll(x, 1, dims=-2)
        x_right = torch.roll(x, -1, dims=-1)

        pot = (2*x - 1) * (2*x_up - 1)  # vertical interactions
        pot += (2*x - 1) * (2*x_right - 1)  # horizontal interactions
        pot = -J * pot.sum(dim=(-1, -2)) # sum over all sites

        pot -= passive_rates.sum(dim=(-1, -2, -3))  # sum over all sites and directions
        return pot

    return potential


def ising_passive_rates(hparams):
    """
    Passive rates for square Ising model with periodic boundary conditions
    """
    g = getattr(hparams, 'g', 1.0)
    def passive_rates_fn(state):
        x = state.unsqueeze(dim=-3)
        rates = g * torch.ones_like(x)
        return rates

    return passive_rates_fn


def ising_terminal(hparams):
    """
    Compares state with terminal state for square Ising model with periodic boundary conditions
    """
    def terminal_fn(state):
        state_ = 2 * state - 1  # convert to -1, 1
        L = hparams.lattice_size
        done = 1.0 * torch.abs(state_.sum((-1, -2))).gt(L**2 - 0.5)

        return done

    return terminal_fn


def xy_potential(_):
    """
    Potential for square XY model with periodic boundary conditions
    """
    def potential(_, passive_rates):

        pot = -passive_rates.sum(dim=(-1, -2, -3))  # sum over all sites and directions
        return pot

    return potential


def xy_passive_rates(_):
    """
    Passive rates for square XY model with periodic boundary conditions
    """
    def passive_rates_fn(state):
        x = state.unsqueeze(dim=-3)  # to be broadcastable
        x_up = torch.roll(x, 1, dims=-2)
        x_right = torch.roll(x, -1, dims=-1)
        x_shift = torch.cat((x_up, x_right), dim=-3)

        rates = x * (1 - x_shift) + (1 - x) * x_shift  # if different
        return rates

    return passive_rates_fn


def heisenberg_antiferromagnet_potential(hparams):
    """
    Potential for square Heisenberg model with periodic boundary conditions
    """
    J = getattr(hparams, 'J', 1.0)
    L = hparams.lattice_size

    def potential(state, passive_rates):
        x = state
        x_up = torch.roll(x, 1, dims=-2)
        x_right = torch.roll(x, -1, dims=-1)

        pot = x * x_up  # vertical interactions
        pot += x * x_right  # horizontal interactions
        pot = J * torch.sum(pot, dim=(-1, -2))  # sum over all sites and directions

        pot -= 0.5 * J * L**2  # add energy of classical Neel state
        pot -= passive_rates.sum(dim=(-1, -2, -3))  # sum over all sites and directions

        return pot

    return potential


def heisenberg_antiferromagnet_passive_rates(hparams):
    """
    Passive rates for square Heisenberg antiferromagnet with periodic boundary conditions
    """
    J = getattr(hparams, 'J', 1.0)
    xy_passive_rates_fn = xy_passive_rates(hparams)

    def passive_rates_fn(state):
        rates = xy_passive_rates_fn(state)
        rates *= 0.5 * J
        return rates

    return passive_rates_fn


def bh_potential(hparams):
    """
    Potential for square Bose-Hubbard model with periodic boundary conditions
    """
    U = getattr(hparams, 'U', 1)

    def potential(state, passive_rates):
        pot = - U / 2 * state * (state - 1)
        pot = -pot.sum(dim=(-1, -2))
        pot -= passive_rates.sum(dim=(-1, -2, -3))  # sum over all sites and directions

        return pot

    return potential


def bh_passive_rates(hparams):
    """
    Passive rates for square Bose-Hubbard model with periodic boundary conditions
    """
    
    t = getattr(hparams, 't', 1)

    def passive_rates_fn(state):
        # Create lattices that are 'rolled' one lattice separation in each direction
        x = state.unsqueeze(dim=-3)  # to be broadcastable
        x_up = torch.roll(x, 1, dims=-2)
        x_right = torch.roll(x, -1, dims=-1)
        x_down = torch.roll(x, -1, dims=-2)
        x_left = torch.roll(x, 1, dims=-1)
        x_shift = torch.cat((x_up, x_right, x_down, x_left), dim=-3)

        # Hopping terms in potential
        epsilon = 1e-16  # to avoid NaN gradient of sqrt when x is zero.
        rates = t * torch.sqrt((x + epsilon) * (x_shift + 1))  # hopping
        rates *= torch.gt(x, 0.0)  # makes rates 0.0 exactly if x is zero

        return rates

    return passive_rates_fn
