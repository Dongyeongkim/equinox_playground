import jax
import einops
import equinox as eqx
import jax.numpy as jnp
from typing import List, Tuple
from jax.nn.initializers import zeros
from jax.nn.initializers import variance_scaling


from utils import *


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    _norm: eqx.Module
    use_bias: bool
    act: str = "none"
    pdtype: str = "float32"
    cdtype: str = "float32"

    def __init__(
        self,
        key,
        in_features: int,
        out_features: int,
        act: str = "none",
        norm: str = "none",
        use_bias=True,
        outscale=1.0,
        winit: str = "normal",
        fan: str = "fan_in",
        pdtype="float32",
        cdtype="float32",
    ):
        self.pdtype = pdtype
        self.cdtype = cdtype
        assert isinstance(
            in_features, int
        ), "num_features should be the type of integer"
        assert isinstance(
            out_features, int
        ), "out_features should be the type of integer"
        wkey, bkey = jax.random.split(key, 2)
        self.weight = variance_scaling(scale=outscale, mode=fan, distribution=winit)(
            wkey, (in_features, out_features), dtype=self.pdtype
        )
        if use_bias:
            self.bias = zeros(bkey, (out_features,), dtype=self.pdtype)
            self.use_bias = True

        self._norm = Norm(num_features=out_features, impl=norm, cdtype=self.cdtype)
        self.act = act

    def __call__(self, x):
        x = cast_to_compute(x, self.cdtype)
        x = self._layer(x)
        x = self._norm(x)
        x = get_act(self.act)(x)
        return x

    def _layer(self, x):
        x = x @ self.weight.astype(self.cdtype)
        if self.use_bias:
            x += self.bias.astype(self.cdtype)
        return x


class BlockLinear(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    _norm: eqx.Module | List
    use_bias: bool
    act: str = "none"
    pdtype: str = "float32"
    cdtype: str = "float32"

    def __init__(
        self,
        key,
        in_features: int,
        out_features: int,
        num_groups: int,
        act: str = "none",
        norm: str = "none",
        use_bias=True,
        outscale=1.0,
        winit: str = "normal",
        fan: str = "fan_in",
        block_fans: bool = False,
        block_norm: bool = False,
        pdtype="float32",
        cdtype="float32",
    ):
        self.pdtype = pdtype
        self.cdtype = cdtype
        assert isinstance(in_features, int), "The type of in_features should be int"
        assert isinstance(out_features, int), "The type of out_features should be int"
        assert isinstance(num_groups, int), "The type of num_groups should be int"
        assert (
            num_groups <= out_features
        ), "The type of num_groups cannot be larger than out_features"
        wkey, bkey = jax.random.split(key, 2)
        self.weight = 0
        if use_bias:
            self.bias = zeros(bkey, (out_features,), dtype=self.pdtype)
            self.use_bias = True
        if block_norm:
            pass
        else:
            pass

    def __call__(self, x):
        pass

    def _layer(self, x):
        pass


class Norm(eqx.Module):
    scale: jax.Array
    offset: jax.Array
    _impl: str
    _eps: float
    act: str = "none"
    pdtype: str = (
        "float32"  # JUST FOR PLACEHOLDER; IT ALWAYS SHOULD BE PRECISER OR EQUAL THAN FLOAT32(PRECISION ISSUE)
    )
    cdtype: str = "float32"

    def __init__(
        self,
        num_features: int,
        impl,
        eps=1e-4,
        act="none",
        pdtype="float32",
        cdtype="float32",
    ):
        self._impl = impl
        self._eps = eps
        self.act = act
        self.cdtype = cdtype
        assert isinstance(
            num_features, int
        ), "num_features should be the type of integer"
        self.scale = jnp.ones(
            (num_features,), dtype="float32"
        )  # IT ALWAYS SHOULD BE FLOAT32 (PRECISION ISSUE)
        self.offset = jnp.zeros(
            (num_features,), dtype="float32"
        )  # IT ALWAYS SHOULD BE FLOAT32 (PRECISION ISSUE)

    def __call__(self, x):
        x = self._norm(x)
        x = get_act(self.act)(x)
        return x

    def _norm(self, x):
        if self._impl == "none":
            return x
        elif self._impl == "layer":
            x = x.astype("float32")
            mean = x.mean(-1)[..., None]
            mean2 = jnp.square(x).mean(-1)[..., None]
            var = jnp.maximum(0, mean2 - jnp.square(mean))
            mult = self.scale * jax.lax.rsqrt(var + self._eps)
            x = (x - mean) * mult + self.offset
            return cast_to_compute(x, self.cdtype)
        elif self._impl == "rms":
            dtype = x.dtype
            x = x.astype("float32") if x.dtype == jnp.float16 else x
            mult = jax.lax.rsqrt((x * x).mean(-1)[..., None] + self._eps) * self.scale
            return (x * mult).astype(dtype)
        elif self._impl == "rms_instance":
            x = x.astype("float32")
            mult = jax.lax.rsqrt((x * x).mean((-3, -2), keepdims=True) + self._eps)
            mult = mult * self.scale
            return cast_to_compute(x * mult, self.cdtype)
        elif self._impl == "grn":
            assert len(x.shape) >= 4, x.shape
            x = x.astype("float32")
            norm = jnp.linalg.norm(x, 2, (-3, -2), keepdims=True)
            norm /= norm.mean(-1, keepdims=True) + self._eps
            x = (norm * self.scale + 1) * x + self.offset
            return cast_to_compute(x, self.cdtype)
        elif self._impl == "instance":
            x = x.astype("float32")
            mean = x.mean(axis=(-3, -2), keepdims=True)
            var = x.var(axis=(-3, -2), keepdims=True)
            x = (self.scale * jax.lax.rsqrt(var + self._eps)) * (x - mean) + self.offset
            return cast_to_compute(x, self.cdtype)
        else:
            raise NotImplementedError(self._impl)


def get_act(name):
    if callable(name):
        return name
    elif name == "none":
        return lambda x: x
    elif name == "cswiglu":

        def fn(x):
            x, y = jnp.split(x, 2, -1)
            y1, y2 = jnp.split(y, 2, -1)
            pad = jnp.ones_like(y1)
            x = jax.nn.swish(jnp.concatenate([x, -x], -1))
            y = jnp.concatenate([y1, pad, y2, pad], -1)
            return x * y

        return fn
    elif name == "mish":
        return lambda x: x * jnp.tanh(jax.nn.softplus(x))
    elif hasattr(jax.nn, name):
        return getattr(jax.nn, name)
    else:
        raise NotImplementedError(name)


if __name__ == "__main__":
    import optax
    from jax import random

    """
    norm_module = Norm(num_features=32, impl="rms")
    random_x_array = random.normal(
        random.key(0), shape=(16, 64, 32, 32), dtype="bfloat16"
    )
    random_y_array = random.bernoulli(random.key(0), shape=(16, 64, 32, 32)).astype(
        "bfloat16"
    )
    print(norm_module)
    print(random_x_array.shape)
    print(norm_module(random_x_array).dtype)

    def loss(model, x, y):
        return jnp.mean((jax.vmap(model)(x) - y) ** 2)

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
    import time

    N = 10000
    a = time.time()
    for i in range(N):
        norm_module, opt_state, value = make_step(
            loss, norm_module, opt_state, random_x_array, random_y_array
        )
        model_scale = jnp.sum(model_scale - norm_module.scale)
        # print(model_scale)
        model_scale = norm_module.scale
    print((time.time() - a) / N)
    """

    CDTYPE = "bfloat16"

    norm_module = Linear(
        jax.random.key(0),
        in_features=1024,
        out_features=32,
        act="sigmoid",
        norm="rms",
        cdtype=CDTYPE,
    )
    random_x_array = random.normal(
        random.key(0), shape=(16, 64, 32, 1024), dtype=CDTYPE
    )
    random_y_array = random.bernoulli(random.key(0), shape=(16, 64, 32, 32)).astype(
        CDTYPE
    )
    print(norm_module)
    print(random_x_array.shape)
    print(norm_module(random_x_array).dtype)

    def loss(model, x, y):
        return jnp.mean((jax.vmap(model)(x) - y) ** 2)

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
    import time

    N = 10000
    a = time.time()
    for i in range(N):
        norm_module, opt_state, value = make_step(
            loss, norm_module, opt_state, random_x_array, random_y_array
        )
        # print(value)
    print((time.time() - a) / N)
