import equinox as eqx
import jax.numpy as jnp


def symlog(x):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def initialise(module, kernel_init, bias_init=False):
    
    if isinstance(module, eqx.nn.Linear):
        module = eqx.tree_at(lambda l: l.weight, module, replace_fn=kernel_init)
        if module.use_bias and bias_init:
            module = eqx.tree_at(lambda l:l.bias, module, replace_fn=bias_init)
        return module
    elif isinstance(module, eqx.nn.Conv):
        module = eqx.tree_at(lambda l: l.weight, module, replace_fn=kernel_init)
        if module.use_bias and bias_init:
            module = eqx.tree_at(lambda l:l.bias, module, replace_fn=bias_init)
        return module
    elif isinstance(module, eqx.nn.MLP):
        where_kernel = lambda m: [lin.weight for lin in m.layers]
        where_bias = lambda m: [lin.bias for lin in m.layers]
        module = eqx.tree_at(where_kernel, module, kernel_init)
        if module.use_bias and bias_init:
            module = eqx.tree_at(where_bias, module, replace_fn=bias_init)
        return module
    else:
        raise NotImplementedError("Only for Convolution, Linear, and MLP layers at current moment")