import jax
import equinox as eqx
import jax.numpy as jnp
from optax._src import base
from optax._src.clipping import unitwise_norm, unitwise_clip


# normalising function

def symlog(x):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


# computing function

def cast_to_compute(values, compute_dtype):
    return jax.tree_util.tree_map(lambda x: x if x.dtype == compute_dtype else x.astype(compute_dtype), values)


# adaptive_gradient_clip for equinox

AdaptiveGradClipState = base.EmptyState

def eqx_adaptive_grad_clip(clipping: float, eps: float = 1e-3):
    def init_fn(params):
        del params
        return AdaptiveGradClipState()
    
    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)
        params = eqx.filter(params, eqx.is_array) # parameter filtering for eqx module
        g_norm, p_norm = jax.tree_util.tree_map(unitwise_norm, (updates, params))
        # Maximum allowable norm.
        max_norm = jax.tree_util.tree_map(lambda x: clipping * jnp.maximum(x, eps), p_norm)
        # If grad norm > clipping * param_norm, rescale.
        updates = jax.tree_util.tree_map(unitwise_clip, g_norm, max_norm, updates)
        return updates, state
    
    return base.GradientTransformation(init_fn, update_fn)

