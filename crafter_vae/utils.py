import jax
import numpy as np
import jax.numpy as jnp

# input normalisation

def symlog(x):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)

def symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

# loss functions

@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.vmap
def mse(recon, obs):
    return jnp.sum((recon-obs)**2)


# computing util(precision casting)

def cast_to_compute(values, compute_dtype):
    return jax.tree_util.tree_map(lambda x: x if x.dtype == compute_dtype else x.astype(compute_dtype), values)

