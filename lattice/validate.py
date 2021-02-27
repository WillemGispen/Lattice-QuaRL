""" Defines the class Validator."""

import torch
import lattice


class Validator:
    """
    Validates the energy of model with variance approximately below the given tolerance.

    Args (of __init__):
      model (lattice.Model): the drift to validate

    Args (of __call__):
      atol_var (Float): absolute tolerance for the variance.
      state (Tensor): an initial state.
          A good initial state is the latest state computed during training.

    Returns:
      avg_energy (Float): the validated energy
      var (Float): estimated variance of the validated energy
    """

    def __init__(self, model, atol_var):
        self.model = model
        self.h = model.hparams
        self.num_steps = self.h.num_steps
        # self.atol_var = torch.tensor(atol_var)

        if self.h.validate_fn.startswith('local'):
            self.energy_fn = globals()[self.h.validate_fn](
                model.env.potential_fn, model.agent.get_wavefunction, model.env.passive_rates_fn, self.h)
        else:
            self.energy_fn = globals()[self.h.validate_fn](
                model.env.potential_fn, model.agent.get_rates, model.env.passive_rates_fn)

        self.state = model.env.initial_state()

    def __call__(self, state):
        state = self._preprocess(state)
        var = torch.tensor(float('inf'))

        # while var > 0.0:
        if True:
            energy, var = self._calculate_energy(state)
            sstate = 2*(state - 0.5)
            magnetization = torch.abs(sstate.sum(dim=(-1, -2))).mean() / sstate.shape[-1]**2

            # avg_energy = energy.mean()
            # var = self._calculate_var(energy)

            # # change batch size
            # batch_size_multiplier = 10  #var / self.atol_var
            # if batch_size_multiplier > 1.1:
            #     state = state.repeat([math.ceil(batch_size_multiplier), 1, 1])

        return energy, magnetization

    def _preprocess(self, state):
        """
        Ensures correct shape of state.
        """
        state = state.squeeze(dim=0)
        if state.shape[0] % 32 != 0:
            state = state.repeat([32, 1, 1])
        
        traj, _, _ = self.model.agent.get_trajectory(self.h.num_steps, state=state)
        state = traj[-1]

        return state

    @staticmethod
    def _calculate_var(energy):
        """ Calculates variance of mean in robust way.
        Divides batch in 32 pieces and uses variance between those 32 pieces.
        """
        energies = energy.chunk(32)
        means = torch.tensor([chunk.mean() for chunk in energies])
        return means.var() / 32

    def _calculate_energy(self, state):
        traj, dts, transition_indices = self.model.agent.get_trajectory(self.h.num_steps, state=state)
        state = traj[-2]
        dt = dts[-1]
        energy, var = self.energy_fn(state, dt, transition_indices)
        self.state = traj[-1]
        return energy, var


def kl_energy(potential_fn, rates_fn, passive_rates_fn):
    """ Calculates the mean log-likelihood ell of a trajectory.
    Compares Feynman-Kac measure with parametrized measure.
    Normalizes mean log-likelihood by dividing by total time and number of sites.
    """

    def validate_fn(traj, dts, transition_indices):
        # pre-process
        traj = traj[:-1]
        s = traj.shape

        # compute rates, view as (TxB, L, L) to use net, then view as (T, B, DxLxL)
        rates = rates_fn((traj
                         ).view((-1,)+s[2:])).view(s[0:2]+(-1,))
        passive_rates = passive_rates_fn(traj)

        potential = potential_fn(traj, passive_rates)

        potential = torch.sum(potential * dts, dim=0)  # integrate over time

        passive_rates = passive_rates.view(s[0:2]+(-1,))

        def get_log_rn():
            """ Calculate log_rn between parametrized and passive dynamics. """
            # Calculate time integral (continuous) part of log_rn
            # sum over lattice sites and directions of rates
            kinetic = (passive_rates - rates).sum(dim=-1)

            continuous_log_rn = torch.sum(kinetic * dts, dim=0)  # integrate over time

            # Calculate transition part of log_rn
            transition_indices_ = transition_indices.unsqueeze(dim=-1)
            chosen_rates = rates.gather(dim=-1, index=transition_indices_)

            chosen_passive_rates = passive_rates.gather(dim=-1, index=transition_indices_)
            transition_log_rn = torch.sum(  # over time
                torch.log(chosen_rates / chosen_passive_rates), dim=0).squeeze()

            return continuous_log_rn + transition_log_rn

        log_rn = get_log_rn()

        T = dts.sum(dim=0).mean(dim=0)  # sum over time, average over batch
        N = s[-1] * s[-2]  # number of lattice sitess
        energy = (potential + log_rn) / (N * T)

        return energy, log_rn

    return validate_fn


def local_stoquastic_energy(potential_fn, wave_fn, passive_rates_fn, hparams):
    """ Calculates the mean local energy using the Hamiltonian and all ajacent states.
    """

    ising = hparams.potential.startswith('ising')
    out_channels = 1 if ising else 2

    def validate_fn(state, dt, _):
        # pre-process
        s = state.shape
        adjacent_states = lattice.markov_solvers.make_adjacent_states(state, ising, out_channels)

        passive_rates = passive_rates_fn(state)
        Hss = potential_fn(state, passive_rates) + passive_rates.sum(dim=(-1, -2, -3))
        passive_rates = passive_rates.view(passive_rates.shape)
        wave = wave_fn(state)
        L = s[-1]
        adjacent_waves = wave_fn(adjacent_states.view(-1, L, L)).view((s[0], out_channels, L, L))

        energy = -torch.sum(passive_rates * adjacent_waves, dim=(1, 2, 3)) / wave + Hss # local energy
        N = s[-1] * s[-2]  # number of lattice sites
        energy = energy.mean() / N
        var = energy.var() / N**2
        # T = dt.sum(dim=0) # sum over batch
        # energy = torch.sum(energy * dt) / (N * T)

        return energy, var

    return validate_fn
