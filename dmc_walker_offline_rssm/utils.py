import jax
import numpy as np
import jax.numpy as jnp


def symlog(x):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

def cast_to_compute(values, compute_dtype):
    return jax.tree_util.tree_map(lambda x: x if x.dtype == compute_dtype else x.astype(compute_dtype), values)

