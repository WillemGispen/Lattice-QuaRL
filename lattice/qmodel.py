"""
Module for lattice models
"""

from argparse import Namespace
import subprocess

import math
import torch
import pytorch_lightning as pl

import lattice.markov_solvers
from lattice.markov_solvers import find_particle_position
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
    'batch_size': 1024,
    'buffer_capacity': 10240,
    'sync_rate': 10,
    'tau': 0.01,  # = C for infinite discrete time
    'lr': 0.001,
    'gamma_lr': 0.95,
    'hidden_channels': 64,
    'num_steps': 256,
    'lattice_size': 5,
    'kernel_size': 3,
    'holding': True,
    'passive_mcmc': False,
    'loss_fn': 'bellman_continuous',
    'validate_fn': 'local_stoquastic_energy',
    'net': 'QPeriodicCNN',
}

Q_DEFAULT_HPARAMS = Namespace(**DEFAULTS)

ALLOWED_HPARAMS = set(DEFAULTS.keys()).union(
    {'net',  # class from lattice.nets
     # energy related functions
     'potential',  # function from lattice.potentials
     'passive_rates',  # function from lattice.potentials
     'passive_mcmc', # Boolean: if passive or learnt policy is used for MCMC
     'terminal', # function from latice.potentials
     'loss_fn',  # function from lattice.losses
     'validate_fn',  # function from lattice.validate
     'E',  # initial guess for ground state energy
     # initial state settings
     'concentration',  # initial average concentration of particles
     'buffer_capacity',
     # coupling constants
     'J',  # for Ising (correlation)
     'g',  # for Ising (transverse field)
     't',  # for BH
     'U',  # for BH
     'Z',  # for nets.PassiveRates
    }  
)


class Environment:
    def __init__(self, model, hparams):
        self.model = model
        self.hparams = hparams
        self.potential_fn = getattr(lattice.potentials, hparams.potential)(hparams)
        self.passive_rates_fn = getattr(lattice.potentials, hparams.passive_rates)(hparams)
        if self.hparams.loss_fn == 'bellman_terminal':
            self.terminal_fn = getattr(lattice.potentials, hparams.terminal)(hparams)

        # Markov solvers
        self.markov = lattice.markov_solvers.MarkovChain(hparams)
        self.markov_continuous_time = lattice.markov_solvers.WaitingTime(hparams)

        self.state = self.initial_state()

    def normalization_fn(self, state):
        """ Normalization for e^Q(a_0, s)."""
        passive_rates = self.passive_rates_fn(state)
        rates_sum = passive_rates.sum(dim=(-1, -2, -3))
        potential = self.potential_fn(state, passive_rates)
        Hss = potential + rates_sum
        L = self.hparams.lattice_size

        if torch.le(Hss, self.model.E * L**2).any():
            minH = torch.min(Hss).detach()
            eps = 1e-3
            self.model.E = float(minH / L**2 - eps)
            self.model.minHss = float(minH / L**2)

        if self.hparams.loss_fn == 'bellman_continuous':
            normalization = Hss - self.model.E * L**2
        elif self.hparams.loss_fn == 'bellman_discrete':
            normalization = self.hparams.tau - potential
        elif self.hparams.loss_fn == 'bellman_terminal':
            normalization = rates_sum

        return normalization

    def reward_fn(self, state):
        """ Reward of a state (independent of action)."""
        passive_rates = self.passive_rates_fn(state)
        potential = self.potential_fn(state, passive_rates)
        normalization = self.normalization_fn(state)

        if self.hparams.loss_fn == 'bellman_continuous':
            reward = potential * self.hparams.tau
        elif self.hparams.loss_fn == 'bellman_discrete':
            reward = torch.log(normalization)
        elif self.hparams.loss_fn == 'bellman_terminal':
            Hss = potential + passive_rates.sum(dim=(-1, -2, -3))
            L = self.hparams.lattice_size
            reward = torch.log(normalization / (Hss - self.model.E * L**2))

        if torch.isnan(reward).any():
            raise Exception(f"nan")

        return reward

    def initial_state(self):
        """ Samples a random initial state. """
        B = self.hparams.batch_size
        L = self.hparams.lattice_size
        num_particles = round(self.hparams.concentration * L**2)

        # choose sites that get a particle. Bose-Hubbard allows multiple particles per site.
        if self.hparams.potential.startswith('bh'):
            indices = torch.multinomial(torch.ones(B, L**2), num_particles, replacement=True)
        else:
            indices = torch.multinomial(torch.ones(B, L**2), num_particles, replacement=False)
        
        state = torch.zeros(B, L**2)
        state = state.scatter_add_(1, indices, torch.ones_like(indices, dtype=torch.float))
        state = state.view(B, L, L)

        return state


class Agent:
    def __init__(self, model, hparams):
        self.model = model
        self.env = model.env
        self.hparams = hparams

    def get_policy(self, state, Q=None):
        """ Policy following from Q. Not normalized! """
        passive_rates = self.env.passive_rates_fn(state)
        Q = self.model.Q_fn(state) if Q is None else Q
        exp_Q = torch.exp(Q)

        return passive_rates * exp_Q

    def get_action(self, state, Q=None, k=1):
        batch_size = state.shape[0]
        transition_probabilities = self.get_policy(state, Q=Q)
        action = transition_probabilities.view(batch_size, -1).multinomial(k, replacement=True)
        return action

    def get_rates(self, state, Q=None):
        """ Rates_fn following from Q. """
        passive_rates = self.env.passive_rates_fn(state)
        Q = self.model.Q_fn(state) if Q is None else Q
        exp_Q = torch.exp(Q)

        # wavefunction is expectation of exp_Q under passive policy
        normalization = self.env.normalization_fn(state).view((exp_Q.shape[0], 1, 1, 1))
        exp_Q0 = torch.sum(passive_rates * exp_Q, dim=(-1, -2, -3), keepdim=True) / normalization
        rates = passive_rates * exp_Q / exp_Q0

        return rates

    def get_wavefunction(self, state, Q=None):
        """
        Wavefunction following from Q. 
        Essentially expectation of exp_Q under passive policy
        """
        batch_size = state.shape[0]
        normalization = self.env.normalization_fn(state)
        tau = self.hparams.tau

        # Compute exp(Q)
        Q = self.model.Q_fn(state) if Q is None else Q
        passive_rates = self.env.passive_rates_fn(state).view(batch_size, -1)
        exp_Q = torch.exp(Q).view(batch_size, -1)

        # Compute wave
        wave = (exp_Q * passive_rates).sum(-1)
        if self.hparams.loss_fn == 'bellman_continuous':
            wave *= torch.exp(tau * (normalization - passive_rates.sum(-1)))
        elif self.hparams.loss_fn == 'bellman_discrete':
            L = self.hparams.lattice_size
            potential = self.env.potential_fn(state, passive_rates.view(batch_size, -1, L, L))
            Hss = potential + passive_rates.sum(-1)
            exp_Q0 = wave / (Hss - self.model.E * L**2)
            wave = wave + (tau - Hss) * exp_Q0

        # apply normalization
        wave /= normalization

        return wave

    def get_experience(self, num_steps, device):
        """ Takes num_steps steps of the discrete time Markov chain.
        Return the last experience. """

        if self.env.state.device != device:
            self.env.state = self.env.state.to(device)

        traj, _, _ = self.get_trajectory(num_steps, state=self.env.state, metropolis=True)
        state = traj[-1]
        action = self.get_action(state)
        reward = self.env.reward_fn(state)
        if self.hparams.loss_fn == 'bellman_terminal':
            done = self.env.terminal_fn(state)
        else:
            done = torch.zeros_like(reward)

        new_state = self.env.markov(state, action)

        self.env.state = new_state

        return state, action, reward, done, new_state

    def get_trajectory(self, num_steps, state=None, metropolis=True, passive=False, k=1):
        """ Samples trajectories with continuous time Markov process """

        if state is None:
            state = self.env.state

        traj = torch.empty((num_steps + 1,) + state.shape, device=state.device)
        dts = torch.empty((num_steps, state.shape[0]), device=state.device)
        actions = torch.empty((num_steps, state.shape[0], k),
                              device=state.device, dtype=torch.long)

        B = state.shape[0]

        for i in range(num_steps):
            Q = self.model.Q_fn(state)
            rates = self.get_rates(state, Q=Q)
            rates_shape = rates.shape
            # new_state, dts[i], actions[i] = self.env.markov_continuous_time(state, rates)
            
            if passive:
                actions[i] = self.get_action(state, Q=0*Q, k=k)
            else:
                actions[i] = self.get_action(state, Q=Q, k=k)
            policy = self.get_policy(state, Q=Q)
            rates = policy

            if torch.isclose(policy, torch.tensor(0.0)).any():
                raise Exception('nan')

            new_state = self.env.markov(state, actions[i], k=k)

            new_Q = self.model.Q_fn(new_state)
            traj[i] = state.unsqueeze(dim=0)

            if metropolis:
                B = state.shape[0]
                wave = self.get_wavefunction(state, Q=Q)
                new_wave = self.get_wavefunction(new_state, Q=new_Q)

                # compute rates
                if passive:
                    rates = self.env.passive_rates_fn(state).view(B, -1)
                    new_rates = self.env.passive_rates_fn(new_state).view(B, -1)
                else:
                    rates = rates.view(B, -1)
                    new_rates = self.get_rates(new_state, Q=new_Q).view(B, -1)

                rates_prob = rates / rates.sum(-1, keepdim=True)  # normalized transition probabilities
                new_rates_prob = new_rates / new_rates.sum(-1, keepdim=True)

                log_select_prob = 0.0
                for kk in range(k):
                    log_select_prob += torch.log(rates_prob[range(B), actions[i, :, kk]])  # select prob of chosen action

                # unravel inverse action
                ising = self.hparams.potential.startswith('ising')
                if ising:
                    # new_rates_prob = new_rates_prob[range(B), actions[i, :, 0]]
                    new_log_select_prob = 0.0
                    for kk in range(k):
                        new_log_select_prob += torch.log(new_rates_prob[range(B), actions[i, :, kk]])  # select prob of chosen action
                    # new_select_prob = 1
                    # for kk in range(k):
                    #     new_select_prob = new_select_prob * new_rates_prob[range(B), actions[i, :, kk]]  # select prob of chosen action
                else:
                    l, m, l_, m_, d = find_particle_position(actions[i], rates_shape, return_d=True)
                    # d = torch.fmod(d + 2, 4)  # inverse action
                    new_rates_prob = new_rates_prob.view(rates_shape)
                    new_rates_prob = new_rates_prob[range(B), d, l, m]

                # compute accepted transitions
                # print(prob_accept.shape, new)
                prob_accept = new_wave**2 / wave**2
                prob_accept *= torch.exp(new_log_select_prob - log_select_prob)
                # prob_accept *= new_rates_prob / rates_prob

                # prob_accept *= new_select_prob / select_prob

                if torch.isnan(prob_accept).any():
                    raise Exception('nan')

                prob_accept = torch.min(prob_accept, torch.ones_like(prob_accept))
                self.model.prob_accept = prob_accept.mean()

                if i % 10 == 0:
                    print(f"{i}: prob_accept = {self.model.prob_accept}")

                random = torch.rand_like(prob_accept)
                accept = torch.gt(prob_accept, random).view(B, 1, 1).float()

                # accept/reject step
                new_state = accept * new_state + (1 - accept) * state

            state = new_state

        traj[-1] = state
        return traj, dts, actions


class ReplayBuffer(torch.utils.data.Dataset):
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    
    Args:
        capacity: size of the buffer
        batch_size: 
        state_shape: shape of a state of the environment
    """

    def __init__(self, capacity, batch_size, state_shape, epoch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.state_buffer = torch.empty((capacity,) + state_shape)
        self.action_buffer = torch.empty(capacity, dtype=torch.long)
        self.reward_buffer = torch.empty(capacity)
        self.done_buffer = torch.empty(capacity)
        self.new_state_buffer = torch.empty((capacity,) + state_shape)

        self.populated = False

    def __len__(self):
        return self.epoch_size

    def save(self, experience):
        """
        Add experience to the buffer. Overwrite old experience if full.
        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        state, action, reward, done, new_state = experience

        self.state_buffer = self.state_buffer.roll(self.batch_size, 0)
        self.state_buffer[0:self.batch_size] = state
        
        self.action_buffer = self.action_buffer.roll(self.batch_size, 0)
        self.action_buffer[0:self.batch_size] = action
        
        self.reward_buffer = self.reward_buffer.roll(self.batch_size, 0)
        self.reward_buffer[0:self.batch_size] = reward

        self.done_buffer = self.done_buffer.roll(self.batch_size, 0)
        self.done_buffer[0:self.batch_size] = done
        
        self.new_state_buffer = self.new_state_buffer.roll(self.batch_size, 0)
        self.new_state_buffer[0:self.batch_size] = new_state

    def sample(self, batch_size):
        indices = torch.multinomial(torch.ones(self.capacity), batch_size)
        state = self.state_buffer[indices]
        action = self.action_buffer[indices]
        reward = self.reward_buffer[indices]
        done = self.done_buffer[indices]
        new_state = self.new_state_buffer[indices]

        return state, action, reward, done, new_state

    def __getitem__(self, key):
        state, action, reward, done, new_state = self.sample(self.batch_size)
        return state, action, reward, done, new_state


class ValStateDataSet(torch.utils.data.Dataset):
    """ Loads attribute state from argument model.

    The 'data' for the model lattice.Model is a batch of initial states for the sde.
    The model lattice.Model does not load data from disk,
    but these initial states are equal to the final states of the previous episode.
    Therefore, we use a Dataset that simply returns these previous final states.

    Args:
      model (lattice.Model)
    """
    def __init__(self, validator):
        super(ValStateDataSet, self).__init__()
        self.validator = validator

    def __getitem__(self, key):
        return self.validator.state

    def __len__(self):
        return 1


class QModel(pl.LightningModule):
    def __init__(self, hparams):

        super(QModel, self).__init__()
        for key in vars(hparams):
            if key not in ALLOWED_HPARAMS:
                print(f"Warning: Hparam {key} is not in ALLOWED_HPARAMS")

        self.hparams = hparams

        # Learnable neural networks and parameter
        self.Q_fn = getattr(lattice.nets, hparams.net)(hparams)
        self.Q_target_fn = getattr(lattice.nets, hparams.net)(hparams)

        self.E = hparams.E
        self.minHss = hparams.E

        self.env = Environment(self, hparams)
        self.agent = Agent(self, hparams)

        self.loss_fn = getattr(lattice.losses, hparams.loss_fn)(
            self.Q_fn, self.Q_target_fn, self.env.passive_rates_fn, hparams.tau)
        self.check_loss_fn = getattr(lattice.losses, hparams.loss_fn)(
            self.Q_fn, self.Q_fn, self.env.passive_rates_fn, hparams.tau)
        self.validator = lattice.validate.Validator(self, 1e-1)

        self.buffer = ReplayBuffer(
            self.hparams.buffer_capacity, hparams.batch_size,
            self.env.state.shape[1:], self.hparams.epoch_size)

        if False:
        # if True:
            self.populate('cpu')
    
    def forward(self, state):
        return self.Q_fn(state)

    def populate(self, device):
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences
        Args:
            steps: number of random steps to populate the buffer with
        """
        steps = math.ceil(self.buffer.capacity / self.hparams.batch_size)
        for i in range(steps):
            print("Populate: step {i} of {steps}")
            experience = self.agent.get_experience(1, device)
            self.buffer.save(experience)

    def configure_optimizers(self):
        parameters = list(self.Q_fn.parameters())# + [self.E]
        optimizer = torch.optim.Adam(parameters, lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, self.hparams.gamma_lr)

        # return optimizer
        return [optimizer], [scheduler]

    ###
    ### TRAINING 
    ###

    def training_step(self, batch, _):
        device = batch[0].device

        with torch.no_grad():
            # Populate if necessary
            # if not self.buffer.populated:
            #     self.populate(device)
            #     self.buffer.populated = True

            # Update target network
            if self.global_step % self.hparams.sync_rate == 0:
                self.Q_target_fn.load_state_dict(self.Q_fn.state_dict())

            # Get and save experience
            experience = self.agent.get_experience(self.hparams.num_steps, device)
            state, action, reward, done, new_state = experience
            self.buffer.save(experience)

        # Unpack batch of experience sampled from replay memory
        for i, exp in enumerate(batch):
            exp = exp.detach().squeeze(dim=0)
            try:
                exp.requires_grad = True
            except:
                pass
            batch[i] = exp
            
        state, action, reward, done, new_state = batch
        normalization = self.env.normalization_fn(state).detach()

        # Compute loss
        loss = self.loss_fn(state, action, reward, done, new_state, normalization, self.E)
        check_loss = self.check_loss_fn(state, action, reward, done, new_state, normalization, self.E)
        train_log = {'loss': loss, 'check_loss': check_loss, 'E': self.E}
        return {'loss': loss, 'log': train_log}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.buffer)

    ###
    ### VALIDATION
    ###

    def validation_step(self, state, _):
        """ Computes the validation energy. """
        val_energy, var = self.validator(state)
        return {'val_energy': val_energy, 'val_energy_var': var, 'prob_accept': self.prob_accept}

    def validation_epoch_end(self, outputs):
        """ Averages and logs the validation energy. """
        val_energy_mean = torch.stack([x['val_energy'] for x in outputs]).mean()
        val_energy_var = torch.stack([x['val_energy_var'] for x in outputs]).mean()
        prob_accept = torch.stack([x['prob_accept'] for x in outputs]).mean()
        
        if not torch.isnan(val_energy_mean):
            if self.E < self.minHss or val_energy_mean < self.minHss:
                self.E = val_energy_mean
            else:
                self.E = self.minHss

        val_log = {'val_energy': val_energy_mean, 'val_energy_magn': val_energy_var, 'prob_accept': prob_accept}

        return {'val_energy': val_energy_mean, 'log': val_log}

    def val_dataloader(self):
        return torch.utils.data.DataLoader(ValStateDataSet(self.validator), 1)
