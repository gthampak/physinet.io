# !pip install --upgrade pip
# !pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

# !pip install moviepy

# +
import jax
import jax.numpy as jnp
import numpy as np
import argparse
from jax import jit
from jax.experimental.ode import odeint
from functools import partial

from jax.experimental import stax
from jax.experimental import optimizers

import os, sys, time
sys.path.append('..')
# -
sys.path.append('../experiment_dblpend/')

print(sys.path)

from lagrangian_nns.lnn import lagrangian_eom_rk4, lagrangian_eom, unconstrained_eom, raw_lagrangian_eom
from lagrangian_nns.experiment_dblpend.data import get_dataset
from lagrangian_nns.models import mlp as make_mlp
from lagrangian_nns.utils import wrap_coords

sys.path.append('../hyperopt')


from lagrangian_nns.hyperopt.HyperparameterSearch import learned_dynamics
from lagrangian_nns.hyperopt.HyperparameterSearch import extended_mlp