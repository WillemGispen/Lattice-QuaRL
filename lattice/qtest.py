""" Tests for lattice models with Q-learning
Usage: python -m unittest lattice.test
"""

import unittest
import os
import shutil
import argparse

import torch

import lattice
from trainer import Trainer

torch.autograd.set_detect_anomaly(True)


class TestQPeriodicCNN(unittest.TestCase):
    """ Test class QPeriodicCNN of lattice.nets """

    def setUp(self):
        self.hparams = lattice.Q_DEFAULT_HPARAMS
        self.hparams.concentration = 0.4
        self.hparams.potential = 'ising_potential'

        self.net = lattice.nets.QPeriodicCNN(self.hparams)
        self.state = lattice.model.initial_state(self.hparams)

    def test_nonzero(self):
        out = self.net(self.state)
        self.assertFalse(torch.allclose(out, 0*out))

    def test_correctshape(self):
        out = self.net(self.state)

        shape = list(out.shape)
        L = self.hparams.lattice_size
        required_shape = [self.hparams.batch_size, 1, L, L]

        self.assertEqual(shape, required_shape)

    def test_isequivariant(self):
        out = self.net(self.state)

        state_2 = self.state.roll((3, 2), dims=(-1, -2))
        out_2 = self.net(state_2)
        out_2 = out_2.roll((-3, -2), dims=(-1, -2))

        self.assertTrue(torch.allclose(out, out_2))


class TestQTraining(unittest.TestCase):
    """ Test PyTorch-Lightning Training routine for soft Q-learning.
    Does the pre-training routine, which includes a validation
    sanity check, as well as one training step. """

    def setUp(self):
        self.hparams = lattice.Q_DEFAULT_HPARAMS
        self.hparams.concentration = 0.5
        self.hparams.epoch_size = 1
        self.hparams.num_steps = 16
        self.hparams.batch_size = 64
        self.hparams.buffer_capacity = 2*64
        self.hparams.lattice_size = 5

    def test_no_exceptions_ising_gpu(self):
        # Make hparams from default_hparams
        self.hparams.potential = 'ising_potential'
        self.hparams.passive_rates = 'ising_passive_rates'
        self.hparams.loss_fn = 'bellman_continuous'
        self.hparams.J = 1
        self.hparams.E = -2.0

        model = lattice.QModel(self.hparams)

        if torch.cuda.is_available():
            trainer = Trainer(name='test', max_epochs=1, gpus=[0])
            trainer.fit(model)

    def test_no_exceptions_ising_discrete(self):
        # Make hparams from default_hparams
        self.hparams.potential = 'ising_potential'
        self.hparams.passive_rates = 'ising_passive_rates'
        self.hparams.loss_fn = 'bellman_discrete'
        self.hparams.tau = 100.0
        self.hparams.J = 1
        self.hparams.E = -0.5

        model = lattice.QModel(self.hparams)
        trainer = Trainer(name='test', max_epochs=1)
        trainer.fit(model)

    def test_no_exceptions_ising_terminal(self):
        # Make hparams from default_hparams
        self.hparams.potential = 'ising_potential'
        self.hparams.passive_rates = 'ising_passive_rates'
        self.hparams.loss_fn = 'bellman_terminal'
        self.hparams.terminal = 'ising_terminal'
        self.hparams.J = 1
        self.hparams.E = -0.5

        model = lattice.QModel(self.hparams)
        trainer = Trainer(name='test', max_epochs=1)
        trainer.fit(model)

    def test_no_exceptions_xy(self):
        # Make hparams from default_hparams
        self.hparams.potential = 'xy_potential'
        self.hparams.passive_rates = 'xy_passive_rates'
        self.hparams.E = -2.0
        self.hparams.loss_fn = 'bellman_continuous'

        model = lattice.QModel(self.hparams)
        trainer = Trainer(name='test', max_epochs=1)
        trainer.fit(model)

    def tearDown(self):
        if os.path.exists('results/test'):
            shutil.rmtree('results/test')
        if os.path.exists('results'):
            os.rmdir('results')  # if empty


class TestMarkovChainXY(unittest.TestCase):
    """ Tests markov_solvers.MarkovChain for the XY model."""

    def setUp(self):
        self.hparams = lattice.Q_DEFAULT_HPARAMS
        self.hparams.concentration = 0.4
        self.hparams.potential = 'xy_potential'

        self.state = lattice.model.initial_state(self.hparams)

        passive_rates_fn = lattice.potentials.xy_passive_rates(self.hparams)
        passive_rates = passive_rates_fn(self.state)
        self.action = passive_rates.view(self.hparams.batch_size, -1).multinomial(1).squeeze()

        self.markov = lattice.markov_solvers.MarkovChain(self.hparams)

    def test_correctshapes(self):
        new_state = self.markov(self.state, self.action)
        self.assertEqual(new_state.shape, self.state.shape)
        self.assertEqual(list(self.action.shape), [self.hparams.batch_size])

    def test_one_particle_hop(self):
        """ Tests whether exactly one particle has hopped. """
        new_state = self.markov(self.state, self.action)
        dx = (new_state - self.state)
        dx_norm = dx.view(self.hparams.batch_size, -1).norm(dim=-1)

        self.assertTrue(torch.allclose(dx_norm**2, 2*torch.ones(self.hparams.batch_size)))

    def test_range(self):
        new_state = self.markov(self.state, self.action)

        self.assertTrue(torch.ge(new_state, 0.0).all())
        self.assertTrue(torch.le(new_state, 1.0).all())



class TestBellmanLoss(unittest.TestCase):
    """ Tests the method `bellman` of lattice.losses. """
    
    def setUp(self):
        hparams = lattice.Q_DEFAULT_HPARAMS
        hparams.J = 1.0
        hparams.concentration = 0.5
        hparams.potential = 'ising_potential'
        hparams.tau = 0.01

        self.passive_rates_fn = lattice.potentials.ising_passive_rates(hparams)
        self.Q_fn = lattice.nets.QPeriodicCNN(hparams)
        self.Q_target_fn = lattice.nets.QPeriodicCNN(hparams)
        self.loss_fn = lattice.losses.bellman_continuous(self.Q_fn, self.Q_target_fn, self.passive_rates_fn, hparams.tau)

        self.state = torch.randint(
            0, 2,
            (hparams.batch_size, hparams.lattice_size, hparams.lattice_size),
            dtype=torch.float, requires_grad=True)
        self.action = torch.randint(
            0, hparams.lattice_size**2,
            (hparams.batch_size,),
            dtype=torch.long)
        self.new_state = torch.randint(
            0, 2,
            (hparams.batch_size, hparams.lattice_size, hparams.lattice_size),
            dtype=torch.float, requires_grad=True)
        rand = torch.randn(hparams.batch_size)
        self.normalization = rand * torch.sign(rand)

    def test_bellman_range(self):
        """ Tests whether Bellman's loss returns a finite positive float. """
        rand = torch.randn(self.state.shape[0]) 
        reward = rand * torch.sign(rand)  # random positive reward
        reward.requires_grad = True
        loss = self.loss_fn(self.state, self.action, reward, None, self.new_state, self.normalization, None)

        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.ge(loss, 0.0).all())
        self.assertTrue(torch.le(loss, 10.0).all())

    def test_backward(self):
        """ Tests whether Bellman's loss is differentiable with backward """
        rand = torch.randn(self.state.shape[0]) 
        reward = rand * torch.sign(rand)  # random positive reward
        reward.requires_grad = True
        
        loss = self.loss_fn(self.state, self.action, reward, None, self.new_state, self.normalization, None)

        loss.backward()


class TestAgent(unittest.TestCase):
    """ Tests qmodel.Agent """

    def setUp(self):
        self.hparams = lattice.Q_DEFAULT_HPARAMS
        self.hparams.potential = 'xy_potential'
        self.hparams.passive_rates = 'xy_passive_rates'
        self.hparams.concentration = 0.5
        self.hparams.epoch_size = 1
        self.hparams.num_steps = 16
        self.hparams.batch_size = 2
        self.hparams.buffer_capacity = 4

        Q_fn = lattice.nets.QPeriodicCNN(self.hparams)
        E = -2.0
        model = argparse.Namespace(E=E, Q_fn=Q_fn)
        self.env = lattice.qmodel.Environment(model, self.hparams)
        model = argparse.Namespace(E=E, Q_fn=Q_fn, env=self.env)
        self.agent = lattice.qmodel.Agent(model, self.hparams)

    def test_policy(self, prob_or_rates=None):
        """ Tests get_policy. Can also be used to test rates. """
        if prob_or_rates is None:
            prob = self.agent.get_policy(self.env.state)
        else:
            prob = prob_or_rates

        # test range
        self.assertTrue(torch.ge(prob, 0.0).all())
        self.assertTrue(torch.isfinite(prob).all())
        # test shape
        L = self.hparams.lattice_size
        required_shape = (self.hparams.batch_size, 2, L, L)
        self.assertEqual(prob.shape, required_shape)

    def test_action(self, action=None, batch_size=None):
        if action is None:
            action = self.agent.get_action(self.env.state)

        # test range
        self.assertEqual(action.dtype, torch.long)
        action = action.double()
        self.assertTrue(torch.ge(action, 0.0).all())
        L = self.hparams.lattice_size
        self.assertTrue(torch.le(action, 4*L*L).all())

        # test shape
        if batch_size is not None:
            required_shape = (batch_size, )
        else:
            required_shape = (self.hparams.batch_size,)
        self.assertEqual(action.shape, required_shape)

    def test_rates(self):
        rates = self.agent.get_rates(self.env.state)
        self.test_policy(prob_or_rates=rates)

    def _test_state(self, state=None, batch_size=None):
        if state is None:
            state = self.env.state

        self.assertTrue(torch.ge(state, 0.0).all())
        self.assertTrue(torch.le(state, 1.0).all())

        L = self.hparams.lattice_size
        if batch_size is not None:
            required_shape = (batch_size, L, L)
        else:
            required_shape = (self.hparams.batch_size, L, L)
        self.assertEqual(state.shape, required_shape)

    def test_experience(self):
        num_steps = 8
        experience = self.agent.get_experience(num_steps, 'cpu')
        state, action, reward, done, new_state = experience

        self._test_state(state=state)
        self.test_action(action=action)
        self.assertEqual(reward.shape, (self.hparams.batch_size,))
        self.assertTrue(torch.isfinite(reward).all())
        self._test_state(state=new_state)

    def test_trajectory(self):
        num_steps = 8
        traj, dts, actions = self.agent.get_trajectory(num_steps)

        self._test_state(state=traj.view((-1,) + traj.shape[2:]), batch_size=(1+num_steps)*self.hparams.batch_size)
        self.test_action(action=actions.view((-1,) + actions.shape[2:]), batch_size=num_steps*self.hparams.batch_size)
        self.assertEqual(dts.shape, (num_steps, self.hparams.batch_size))
        self.assertTrue(torch.gt(dts, 0.0).all())


class TestEnvironment(unittest.TestCase):
    """ Tests qmodel.Environment """

    def setUp(self):
        self.hparams = lattice.Q_DEFAULT_HPARAMS
        self.hparams.potential = 'ising_potential'
        self.hparams.passive_rates = 'ising_passive_rates'
        self.hparams.concentration = 0.5
        self.hparams.epoch_size = 1
        self.hparams.num_steps = 16
        self.hparams.tau = 0.01
        self.hparams.batch_size = 2
        self.hparams.buffer_capacity = 4

        E = -2.0
        model = argparse.Namespace(E=E)
        self.env = lattice.qmodel.Environment(model, self.hparams)
    
    def test_reward_fn(self):
        reward = self.env.reward_fn(self.env.state)
        self.assertEqual(reward.shape, (self.hparams.batch_size,))
        self.assertTrue(torch.isfinite(reward).all())

    def test_state(self):
        self.assertTrue(torch.ge(self.env.state, 0.0).all())
        self.assertTrue(torch.le(self.env.state, 1.0).all())
        required_shape = [self.hparams.batch_size, self.hparams.lattice_size, self.hparams.lattice_size]
        self.assertEqual(list(self.env.state.shape), required_shape)


class TestReplayBuffer(unittest.TestCase):
    """ Tests lattice.qmodel.ReplayBuffer. """

    def setUp(self):
        self.capacity, self.batch_size, self.state_shape, self.epoch_size = 4, 2, (5, 5), 10
        self.buffer = lattice.qmodel.ReplayBuffer(self.capacity, self.batch_size, self.state_shape, self.epoch_size)

    def test_save(self):
        state = torch.randn((self.batch_size,) + self.state_shape)
        action = torch.randint(0, 10, (self.batch_size,), dtype=torch.long)
        reward = torch.randn(self.batch_size)
        done = torch.zeros(self.batch_size)
        new_state = state

        experience = state, action, reward, done, new_state
        self.buffer.save(experience)
        self.assertTrue(torch.allclose(self.buffer.state_buffer[0:self.batch_size], state))
        self.assertTrue(torch.allclose(self.buffer.action_buffer[0:self.batch_size], action))
        self.assertTrue(torch.allclose(self.buffer.reward_buffer[0:self.batch_size], reward))
        self.assertTrue(torch.allclose(self.buffer.new_state_buffer[0:self.batch_size], new_state))


class TestLocalStoquasticEnergy(unittest.TestCase):
    """ Tests lattice.validate.local_stoquastic_energy. """

    def setUp(self):
        self.hparams = lattice.Q_DEFAULT_HPARAMS
        self.hparams.lattice_size = 5
        self.hparams.potential = 'xy_potential'
        self.hparams.concentration = 0.50001
        self.hparams.E = -0.5

        potential_fn = lattice.potentials.xy_potential(self.hparams)
        passive_rates_fn = lattice.potentials.xy_passive_rates(self.hparams)

        def wavefunction(state):
            rand = torch.randn(state.shape[0])
            wave = rand * torch.sign(rand)  # random positive times
            return wave

        self.energy_fn = lattice.validate.local_stoquastic_energy(potential_fn, wavefunction, passive_rates_fn, self.hparams)

    def test_no_exceptions(self):
        state = lattice.model.initial_state(self.hparams)
        rand = torch.randn(state.shape[0])
        dt = rand * torch.sign(rand)  # random positive times
        energy, _ = self.energy_fn(state, dt, None)

        self.assertTrue(torch.isfinite(energy).all())


class TestComplexQPeriodicCNN(unittest.TestCase):
    """ Test complex variant of class QPeriodicCNN of lattice.nets """

    def setUp(self):
        self.hparams = lattice.Q_DEFAULT_HPARAMS
        self.hparams.concentration = 0.4
        self.hparams.potential = 'ising_potential'
        self.hparams.lattice_size = 6

        self.net = lattice.nets.HeisenbergPeriodicCNN(self.hparams)
        self.state = lattice.model.initial_state(self.hparams)

    def test_nonzero(self):
        re_out, im_out = self.net(self.state)
        self.assertFalse(torch.allclose(re_out, 0*re_out))
        self.assertFalse(torch.allclose(im_out, 0*im_out))

    def test_correctshape(self):
        re_out, im_out = self.net(self.state)

        re_shape = list(re_out.shape)
        im_shape = list(im_out.shape)
        L = self.hparams.lattice_size
        required_shape = [self.hparams.batch_size, 1, L, L]

        self.assertEqual(re_shape, required_shape)
        self.assertEqual(im_shape, required_shape)

    def test_isequivariant(self):
        re_out, im_out = self.net(self.state)

        state_2 = self.state.roll((3, 2), dims=(-1, -2))
        re_out_2, im_out_2 = self.net(state_2)
        re_out_2 = re_out_2.roll((-3, -2), dims=(-1, -2))
        im_out_2 = im_out_2.roll((-3, -2), dims=(-1, -2))

        self.assertTrue(torch.allclose(re_out, re_out_2))
        self.assertTrue(torch.allclose(im_out, im_out_2))


class TestMakeAdjacentStates(unittest.TestCase):

    def setUp(self):
        self.hparams = lattice.Q_DEFAULT_HPARAMS
        self.hparams.concentration = 0.5001
        self.hparams.potential = 'ising_potential'
        self.hparams.batch_size = 2
        self.hparams.lattice_size = 3
        self.state = lattice.model.initial_state(self.hparams)
        self.out_channels = 1
        self.ising = True

    def test_correct_shape(self):
        s = self.state.shape
        states = lattice.markov_solvers.make_adjacent_states(self.state, self.ising, self.out_channels)
        required_shape = [s[0], self.out_channels * s[-1]**2, s[-1], s[-1]]

        self.assertEqual(list(states.shape), required_shape)

    def test_one_spin_flip(self):
        s = self.state.shape
        states = lattice.markov_solvers.make_adjacent_states(self.state, self.ising, self.out_channels)
        B, L = s[0], s[-1]
        states = states.view(B, L**2, L**2)
        state = self.state.view(B, 1, L**2)
        dx = (states - state).norm(dim=-1)
        self.assertTrue(torch.allclose(dx, torch.ones_like(dx)))
    
    def test_all_different(self):
        s = self.state.shape
        states = lattice.markov_solvers.make_adjacent_states(self.state, self.ising, self.out_channels)
        B, L = s[0], s[-1]
        states = states.view(B, L**2, L**2)
        state = self.state.view(B, 1, L**2)
        dx = states - state
        sum_dx = torch.abs(dx.sum(-2))
        self.assertTrue(torch.allclose(sum_dx, torch.ones_like(sum_dx)))
