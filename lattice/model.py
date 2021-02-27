"""
Module for lattice models
"""

from argparse import Namespace
import subprocess

import torch
import pytorch_lightning as pl

import lattice.markov_solvers
import lattice.nets
import lattice.potentials


try:
    git_commit = subprocess.check_output(
        ["git", "describe", "--always"]).strip().decode()
except:
    git_commit = 'None'


DEFAULTS = {
    'git_commit': git_commit,
    ## training parameters
    'epoch_size': 10,
    'batch_size': 64,
    'lr': 0.001,
    'gamma_lr': 0.95,
    'hidden_channels': 64,
    'num_steps': 256,
    'lattice_size': 5,
    'kernel_size': 3,
    'holding': True,
    'loss_fn': 'reinforce',
    'validate_fn': 'kl_energy'
}

DEFAULT_HPARAMS = Namespace(**DEFAULTS)

ALLOWED_HPARAMS = set(DEFAULTS.keys()).union(
    {'net',  # class from lattice.nets
     # energy related functions
     'potential',  # function from lattice.potentials
     'passive_rates',  # function from lattice.potentials
     'loss_fn',  # function from lattice.losses
     'validate_fn',  # function from lattice.validate
     # initial state settings
     'concentration',  # initial average concentration of particles
     # coupling constants
     'J',  # for Ising
     't',  # for BH
     'U',  # for BH
     'Z',  # for nets.PassiveRates
    }  
)


class Model(pl.LightningModule):
    def __init__(self, hparams):

        super(Model, self).__init__()
        for key in vars(hparams):
            if key not in ALLOWED_HPARAMS:
                raise AttributeError(f"Hparam {key} is not in ALLOWED_HPARAMS")

        self.hparams = hparams

        self.potential_fn = getattr(lattice.potentials, hparams.potential)(hparams)
        self.passive_rates_fn = getattr(lattice.potentials, hparams.passive_rates)(hparams)

        self.rates_fn = getattr(lattice.nets, hparams.net)(hparams, self.passive_rates_fn)

        self.markov = lattice.markov_solvers.WaitingTime(self.hparams)

        self.loss_fn = getattr(lattice.losses, hparams.loss_fn)(
            self.potential_fn, self.rates_fn, self.passive_rates_fn)

        self.validator = lattice.validate.Validator(self, 1e-1)

        self.state = initial_state(hparams)

    def forward(self, state, num_steps):
        traj = torch.empty((num_steps + 1,) + state.shape, device=state.device)
        dts = torch.empty((num_steps, state.shape[0]), device=state.device)
        transition_indices = torch.empty((num_steps, state.shape[0]),
                                         device=state.device, dtype=torch.long)

        for i in range(num_steps):
            rates = self.rates_fn(state)
            new_state, dts[i], transition_indices[i] = self.markov(state, rates)
            traj[i] = state.unsqueeze(dim=0)
            state = new_state

        traj[-1] = state
        return traj, dts, transition_indices

    def train_dataloader(self):
        return torch.utils.data.DataLoader(ModelState(self, self.hparams.epoch_size))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.markov.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, self.hparams.gamma_lr)

        return [optimizer], [scheduler]

    def training_step(self, state, _):
        state = state.detach_().squeeze(dim=0)
        state.requires_grad = True

        if self.hparams.loss_fn == 'reinforce':
            with torch.no_grad():
                state_traj, dts, transition_indices = self.forward(state, self.hparams.num_steps)
        else:
            state_traj, dts, transition_indices = self.forward(state, self.hparams.num_steps)

        self.state = state_traj[-1]

        loss, display_loss = self.loss_fn(state_traj, dts, transition_indices)

        train_log = {'loss': display_loss}
        return {'loss': loss, 'log': train_log}

    def validation_step(self, state, _):
        """ Computes the validation energy. """
        val_energy = self.validator(state)
        return {'val_energy': val_energy}

    def validation_epoch_end(self, outputs):
        """ Averages and logs the validation energy. """
        val_energy_mean = torch.stack([x['val_energy'] for x in outputs]).mean()
        val_log = {'val_energy': val_energy_mean}

        return {'val_energy': val_energy_mean, 'log': val_log}

    def val_dataloader(self):
        return torch.utils.data.DataLoader(ModelState(self, 1))


def initial_state(hparams):
    """ Samples a random initial state. """
    B = hparams.batch_size
    L = hparams.lattice_size
    num_particles = round(hparams.concentration * L**2)

    # choose sites that get a particle. Bose-Hubbard allows multiple particles per site.
    if hparams.potential.startswith('bh'):
        indices = torch.multinomial(torch.ones(B, L**2), num_particles, replacement=True)
    else:
        indices = torch.multinomial(torch.ones(B, L**2), num_particles, replacement=False)
    
    state = torch.zeros(B, L**2)
    state = state.scatter_add_(1, indices, torch.ones_like(indices, dtype=torch.float))
    state = state.view(B, L, L)

    return state


class ModelState(torch.utils.data.Dataset):
    """ Loads attribute state from argument model.

    The 'data' for the model lattice.Model is a batch of initial states for the sde.
    The model lattice.Model does not load data from disk,
    but these initial states are equal to the final states of the previous episode.
    Therefore, we use a Dataset that simply returns these previous final states.

    Args:
      model (lattice.Model)
      epoch_size (int): number of episodes in one epoch.
    """
    def __init__(self, model, epoch_size):
        super(ModelState, self).__init__()
        self.model = model
        self.epoch_size = epoch_size

    def __getitem__(self, key):
        return self.model.state

    def __len__(self):
        return self.epoch_size
