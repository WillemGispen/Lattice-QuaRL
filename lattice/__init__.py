""" Package for lattice models.

Usage:
  Create custom hparams from lattice.DEFAULT_HPARAMS,
  and pass this into lattice.Model.

Submodules:
  evaluate: calculates evaluated losses for models.
  model: defines lattice.Model
  potentials: defines potential energy functions that can be used by Model.
  markov_solvers: defines sde discrete time approximation schemes
  test: testing suite

Subpackage:
  nets: contains neural nets appropriate for different lattice models.
"""

import lattice.losses
import lattice.markov_solvers
import lattice.nets
import lattice.potentials
import lattice.validate

from .model import Model, DEFAULT_HPARAMS
from .qmodel import QModel, Q_DEFAULT_HPARAMS
