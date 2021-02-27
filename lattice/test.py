""" Tests for lattice models.
Usage: python -m unittest lattice.test
"""

import unittest
import os
import shutil
import math

import torch

import lattice
from trainer import Trainer

torch.autograd.set_detect_anomaly(True)


class TestPeriodicCNN(unittest.TestCase):
    """ Test class PeriodicCNN of lattice.nets """

    def setUp(self):
        self.hparams = lattice.DEFAULT_HPARAMS
        self.hparams.concentration = 0.4
        self.hparams.potential = 'ising_potential'

        passive_rates_fn = lattice.potentials.ising_passive_rates(self.hparams)
        self.net = lattice.nets.PeriodicCNN(self.hparams, passive_rates_fn)
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


class TestIsingPotential(unittest.TestCase):
    """ Test method ising_potential of lattice.potentials. """

    def setUp(self):
        self.hparams = lattice.DEFAULT_HPARAMS
        self.hparams.concentration = 0.4
        self.hparams.J = 1
        self.hparams.potential = 'ising_potential'

        self.state = lattice.model.initial_state(self.hparams)

        self.potential_fn = lattice.potentials.ising_potential(self.hparams)
        self.passive_rates_fn = lattice.potentials.ising_passive_rates(self.hparams)

    def test_correctshape(self):
        passive_rates = self.passive_rates_fn(self.state)
        potential = self.potential_fn(self.state, passive_rates)

        shape = list(potential.shape)
        required_shape = [self.hparams.batch_size]

        self.assertEqual(shape, required_shape)

    def test_correctvalue(self):
        state_0 = torch.zeros_like(self.state[0])
        passive_rates_0 = self.passive_rates_fn(state_0)
        potential_0 = self.potential_fn(state_0, passive_rates_0)

        state_1 = torch.ones_like(self.state[0])
        passive_rates_1 = self.passive_rates_fn(state_1)
        potential_1 = self.potential_fn(state_1, passive_rates_1)

        N = self.hparams.lattice_size ** 2
        required_potential = -N - 2 * self.hparams.J * N

        self.assertEqual(potential_0, required_potential)
        self.assertEqual(potential_1, required_potential)

    def test_isequivariant(self):
        """ Potential should be translation invariant """
        passive_rates = self.passive_rates_fn(self.state)
        potential = self.potential_fn(self.state, passive_rates)

        state_2 = self.state.roll((3, 2), dims=(-1, -2))
        passive_rates_2 = self.passive_rates_fn(state_2)
        potential_2 = self.potential_fn(state_2, passive_rates_2)

        self.assertTrue(torch.allclose(potential, potential_2))


class TestXYPotential(unittest.TestCase):
    """ Test method xy_potential of lattice.potentials. """

    def setUp(self):
        self.hparams = lattice.DEFAULT_HPARAMS
        self.hparams.concentration = 0.4
        self.hparams.J = 1
        self.hparams.potential = 'xy_potential'

        self.state = lattice.model.initial_state(self.hparams)
        self.potential_fn = lattice.potentials.xy_potential(self.hparams)
        self.passive_rates_fn = lattice.potentials.xy_passive_rates(self.hparams)

    def test_correctshape(self):
        passive_rates = self.passive_rates_fn(self.state)
        potential = self.potential_fn(self.state, passive_rates)

        shape = list(potential.shape)
        required_shape = [self.hparams.batch_size]

        self.assertEqual(shape, required_shape)

    def test_correctvalue(self):
        state_0 = torch.zeros_like(self.state[0])
        passive_rates_0 = self.passive_rates_fn(state_0)
        potential_0 = self.potential_fn(state_0, passive_rates_0)

        state_1 = torch.ones_like(self.state[0])
        passive_rates_1 = self.passive_rates_fn(state_1)
        potential_1 = self.potential_fn(state_1, passive_rates_1)

        self.assertEqual(potential_0, 0)
        self.assertEqual(potential_1, 0)

    def test_isequivariant(self):
        """ Potential should be translation invariant """
        passive_rates = self.passive_rates_fn(self.state)
        potential = self.potential_fn(self.state, passive_rates)

        state_2 = self.state.roll((3, 2), dims=(-1, -2))
        passive_rates_2 = self.passive_rates_fn(state_2)
        potential_2 = self.potential_fn(state_2, passive_rates_2)

        self.assertTrue(torch.allclose(potential, potential_2))


class TestAntiHeisenbergPotential(unittest.TestCase):
    """ Test method heisenberg_antiferromagnet_potential of lattice.potentials. """

    def setUp(self):
        self.hparams = lattice.DEFAULT_HPARAMS
        self.hparams.concentration = 0.4
        self.hparams.J = 1
        self.hparams.potential = 'heisenberg_antiferromagnet_potential'

        self.state = lattice.model.initial_state(self.hparams)
        self.potential_fn = lattice.potentials.heisenberg_antiferromagnet_potential(self.hparams)
        self.passive_rates_fn = lattice.potentials.heisenberg_antiferromagnet_passive_rates(self.hparams)

    def test_correctshape(self):
        passive_rates = self.passive_rates_fn(self.state)
        potential = self.potential_fn(self.state, passive_rates)

        shape = list(potential.shape)
        required_shape = [self.hparams.batch_size]

        self.assertEqual(shape, required_shape)

    def test_correctvalue(self):
        state_0 = torch.zeros_like(self.state[0])
        passive_rates_0 = self.passive_rates_fn(state_0)
        potential_0 = self.potential_fn(state_0, passive_rates_0)

        state_1 = torch.ones_like(self.state[0])
        passive_rates_1 = self.passive_rates_fn(state_1)
        potential_1 = self.potential_fn(state_1, passive_rates_1)

        N = self.hparams.lattice_size ** 2
        self.assertEqual(potential_0, -0.5 * N)
        self.assertEqual(potential_1, 1.5 * N)

    def test_isequivariant(self):
        """ Potential should be translation invariant """
        passive_rates = self.passive_rates_fn(self.state)
        potential = self.potential_fn(self.state, passive_rates)

        state_2 = self.state.roll((3, 2), dims=(-1, -2))
        passive_rates_2 = self.passive_rates_fn(state_2)
        potential_2 = self.potential_fn(state_2, passive_rates_2)

        self.assertTrue(torch.allclose(potential, potential_2))


class TestBHPotential(unittest.TestCase):
    """ Test method bh_potential of lattice.potentials. """

    def setUp(self):
        self.hparams = lattice.DEFAULT_HPARAMS
        self.hparams.concentration = 0.4
        self.hparams.t = 1
        self.hparams.J = 1
        self.hparams.potential = 'bh_potential'

        self.state = lattice.model.initial_state(self.hparams)
        self.potential_fn = lattice.potentials.bh_potential(self.hparams)
        self.passive_rates_fn = lattice.potentials.bh_passive_rates(self.hparams)

    def test_correctshape(self):
        passive_rates = self.passive_rates_fn(self.state)
        potential = self.potential_fn(self.state, passive_rates)

        shape = list(potential.shape)
        required_shape = [self.hparams.batch_size]

        self.assertEqual(shape, required_shape)

    def test_correctvalue(self):
        state_0 = torch.zeros_like(self.state[0])
        passive_rates_0 = self.passive_rates_fn(state_0)
        potential_0 = self.potential_fn(state_0, passive_rates_0)

        state_1 = torch.ones_like(self.state[0])
        passive_rates_1 = self.passive_rates_fn(state_1)
        potential_1 = self.potential_fn(state_1, passive_rates_1)

        N = self.hparams.lattice_size ** 2
        required_potential_0 = torch.tensor(0.0)
        required_potential_1 = -torch.tensor(self.hparams.t * N * 4 * math.sqrt(2))

        self.assertTrue(torch.allclose(potential_0, required_potential_0))
        self.assertTrue(torch.allclose(potential_1, required_potential_1))

    def test_isequivariant(self):
        """ Potential should be translation invariant """
        passive_rates = self.passive_rates_fn(self.state)
        potential = self.potential_fn(self.state, passive_rates)

        state_2 = self.state.roll((3, 2), dims=(-1, -2))
        passive_rates_2 = self.passive_rates_fn(state_2)
        potential_2 = self.potential_fn(state_2, passive_rates_2)

        self.assertTrue(torch.allclose(potential, potential_2))


class TestInitialState(unittest.TestCase):
    """ Tests model.initial_state. """

    def test_bh_initial_state(self):
        """ Tests range, size and concentration of Bose-Hubbard initial state. """
        h = lattice.DEFAULT_HPARAMS
        h.concentration = 1.2
        h.potential = 'bh_potential'
        state = lattice.model.initial_state(h)
        
        self.assertEqual(state.shape, (h.batch_size, h.lattice_size, h.lattice_size))
        self.assertTrue(torch.ge(state, 0.0).all())
        self.assertTrue(torch.allclose(state.mean(), torch.tensor(h.concentration)))

    def test_xy_initial_state(self):
        """ Tests range, size and concentration of XY initial state. """
        h = lattice.DEFAULT_HPARAMS
        h.concentration = 0.6
        h.potential = 'xy_potential'
        state = lattice.model.initial_state(h)
        
        self.assertEqual(state.shape, (h.batch_size, h.lattice_size, h.lattice_size))
        self.assertTrue(torch.ge(state, 0.0).all())
        self.assertTrue(torch.le(state, 1.0).all())
        self.assertTrue(torch.allclose(state.mean(), torch.tensor(h.concentration)))


class TestWaitingTimeIsing(unittest.TestCase):
    """ Tests markov_solvers.WaitingTime for the Ising model."""

    def setUp(self):
        self.hparams = lattice.DEFAULT_HPARAMS
        self.hparams.concentration = 0.4
        self.hparams.potential = 'ising_potential'

        self.state = lattice.model.initial_state(self.hparams)

        passive_rates_fn = lattice.potentials.ising_passive_rates(self.hparams)
        self.rates_fn = lattice.nets.PeriodicCNN(self.hparams, passive_rates_fn)

        self.markov = lattice.markov_solvers.WaitingTime(self.hparams)

    def test_correctshapes(self):
        rates = self.rates_fn(self.state)
        new_state, dt, transition_indices = self.markov(self.state, rates)
        self.assertEqual(new_state.shape, self.state.shape)
        self.assertEqual(list(dt.shape), [self.hparams.batch_size])
        self.assertEqual(list(transition_indices.shape), [self.hparams.batch_size])

    def test_one_spin_flip(self):
        """ Tests whether exactly one spin has been flipped. """
        rates = self.rates_fn(self.state)
        new_state, _, _ = self.markov(self.state, rates)
        dx = (new_state - self.state).view(self.hparams.batch_size, -1)
        self.assertTrue(torch.equal(dx.norm(dim=-1), torch.ones(self.hparams.batch_size)))

    def test_range(self):
        rates = self.rates_fn(self.state)
        new_state, dt, _ = self.markov(self.state, rates)
        self.assertTrue(torch.ge(new_state, 0.0).all())
        self.assertTrue(torch.le(new_state, 1.0).all())
        self.assertTrue(torch.gt(dt, 0.0).all())

    def test_holding_false(self):
        """ Perform the same tests again but now with holding=False. """
        self.markov = lattice.markov_solvers.WaitingTime(self.hparams)
        self.test_correctshapes()
        self.test_one_spin_flip()
        self.test_range()


class TestWaitingTimeXY(unittest.TestCase):
    """ Tests markov_solvers.WaitingTime for the XY model."""

    def setUp(self):
        self.hparams = lattice.DEFAULT_HPARAMS
        self.hparams.concentration = 0.4
        self.hparams.potential = 'xy_potential'

        self.state = lattice.model.initial_state(self.hparams)

        passive_rates_fn = lattice.potentials.xy_passive_rates(self.hparams)
        self.rates_fn = lattice.nets.PeriodicCNN(self.hparams, passive_rates_fn)
        self.rates = self.rates_fn(self.state)

        self.markov = lattice.markov_solvers.WaitingTime(self.hparams)

    def test_correctshapes(self):
        new_state, dt, transition_indices = self.markov(self.state, self.rates)
        self.assertEqual(new_state.shape, self.state.shape)
        self.assertEqual(list(dt.shape), [self.hparams.batch_size])
        self.assertEqual(list(transition_indices.shape), [self.hparams.batch_size])

    def test_one_particle_hop(self):
        """ Tests whether exactly one particle has hopped. """
        new_state, _, _ = self.markov(self.state, self.rates)
        dx = (new_state - self.state)
        dx_norm = dx.view(self.hparams.batch_size, -1).norm(dim=-1)

        self.assertTrue(torch.allclose(dx_norm**2, 2*torch.ones(self.hparams.batch_size)))

    def test_range(self):
        new_state, dt, _ = self.markov(self.state, self.rates)

        self.assertTrue(torch.ge(new_state, 0.0).all())
        self.assertTrue(torch.le(new_state, 1.0).all())
        self.assertTrue(torch.gt(dt, 0.0).all())

    def test_holding_false(self):
        """ Perform the same tests again but now with holding=False. """
        self.hparams.holding = False
        self.markov = lattice.markov_solvers.WaitingTime(self.hparams)
        self.test_correctshapes()
        self.test_one_particle_hop()
        self.test_range()
