import jax
import einops
import equinox as eqx
import jax.numpy as jnp
from typing import Tuple
from jax.nn.initializers import variance_scaling


from utils import *



class LinearandNorm(eqx.Module):
    pass



class Norm(eqx.Module):
    scale: jax.Array 
    offset: jax.Array
    _impl: str
    _eps: float
    act: str = 'none'

    def __init__(self, num_features: int, impl, eps=1e-4, act='none', param_dtype='float32'):
        if '1em' in impl:
            impl, exponent = impl.split('1em')
            eps = 10 ** -int(exponent)
        
        self._impl = impl
        self._eps = eps
        self.act = act
        self.scale = jnp.ones((num_features,), dtype=param_dtype)
        self.offset = jnp.zeros((num_features,), dtype=param_dtype)
    
    def __call__(self, x):
        x = self._norm(x)
        x = get_act(self.act)(x)
        return x
    
    def _norm(self, x):
        if self._impl == 'none':
            return x
        elif self._impl == 'layer':
            x = x.astype('float32')
            mean = x.mean(-1)[..., None]
            mean2 = jnp.square(x).mean(-1)[..., None]
            var = jnp.maximum(0, mean2 - jnp.square(mean))
            mult = self.scale * jax.lax.rsqrt(var + self._eps)
            x = (x - mean) * mult + self.offset
            return x # should include precision control
        elif self._impl == 'rms':
            dtype = x.dtype
            x = x.astype('float32') if x.dtype == jnp.float16 else x
            mult = jax.lax.rsqrt((x * x).mean(-1)[..., None] + self._eps) * self.scale
            return (x * mult).astype(dtype)
        elif self._impl == 'rms_instance':
            x = x.astype('float32')
            mult = jax.lax.rsqrt((x * x).mean((-3, -2), keepdims=True) + self._eps)
            mult = mult * self.scale
            return x * mult # should include precision control
        elif self._impl == 'grn':
            assert len(x.shape) >= 4, x.shape
            x = x.astype('float32')
            norm = jnp.linalg.norm(x, 2, (-3, -2), keepdims=True)
            norm /= (norm.mean(-1, keepdims=True) + self._eps)
            x = (norm * self.scale + 1) * x + self.offset
            return x # should include precision control
        elif self._impl == 'instance':
            x = x.astype('float32')
            mean = x.mean(axis=(-3, -2), keepdims=True)
            var = x.var(axis=(-3, -2), keepdims=True)
            x = (self.scale * jax.lax.rsqrt(var + self._eps)) * (x - mean) + self.offset
            return x # should include precision control
        else:
            raise NotImplementedError(self._impl)




def get_act(name):
    if callable(name):
        return name
    elif name == 'none':
        return lambda x: x
    elif name == 'cswiglu':
        def fn(x):
            x, y = jnp.split(x, 2, -1)
            y1, y2 = jnp.split(y, 2, -1)
            pad = jnp.ones_like(y1)
            x = jax.nn.swish(jnp.concatenate([x, -x], -1))
            y = jnp.concatenate([y1, pad, y2, pad], -1)
            return x * y
        return fn
    elif name == 'mish':
        return lambda x: x * jnp.tanh(jax.nn.softplus(x))
    elif hasattr(jax.nn, name):
        return getattr(jax.nn, name)
    else:
        raise NotImplementedError(name)



if __name__ == '__main__':
    import optax
    from jax import random
    norm_module = Norm(num_features=32, impl='instance')
    random_x_array = random.normal(random.key(0), shape=(16, 64, 32, 32), dtype='float32')
    random_y_array = random.bernoulli(random.key(0), shape=(16, 64, 32, 32)).astype('float32')
    print(norm_module)
    print(random_x_array.shape)
    print(norm_module(random_x_array).dtype)

    def loss(model, x, y):
        return jnp.mean((jax.vmap(model)(x) - y)**2)
    
    optim = optax.rmsprop(learning_rate=4e-5)
    opt_state = optim.init(eqx.filter(norm_module, eqx.is_array))
    print(opt_state)

    @eqx.filter_jit
    def make_step(loss_fn, model, opt_state, inputs, label):
        value, grads = eqx.filter_value_and_grad(loss_fn)(model, inputs, label)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, value
    
    model_scale = 0
    for i in range(100):
        norm_module, opt_state, value = make_step(loss, norm_module, opt_state, random_x_array, random_y_array)
        model_scale = jnp.sum(model_scale - norm_module.scale)
        print(model_scale)
        model_scale = norm_module.scale