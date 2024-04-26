import jax
import jax.numpy as jnp
import equinox as eqx
from model import VAE
from crafter_dataset import get_crafter_dataset
from utils import symlog, symexp, mse, kl_divergence





