import jax
from utils import *
import numpy as np
import equinox as eqx
import jax.numpy as jnp
from typing import List
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class Dist(eqx.Module):
    _proj: dict
    _dist: list
    out_shape: tuple
    padding: int
    num_units: int
    outscale: float
    minstd: float
    maxstd: float
    unimix: float
    bins: int
    pdtype: str = "float32"
    cdtype: str = "float32"

    def __init__(
        self,
        key,
        in_features: int,
        out_shape: tuple,
        dist="mse",
        use_bias=True,
        minstd=1.0,
        maxstd=1.0,
        unimix=0.0,
        bins=255,
        outscale=0.1,
        winit: str = "normal",
        fan="in",
        fanin=0,
        pdtype="float32",
        cdtype="float32",
    ):
        self.minstd = minstd
        self.maxstd = maxstd
        self.unimix = unimix

        self.bins = bins
        self.padding = 0
        if "twohot" in dist or dist == "softmax":
            self.padding = int(self.bins % 2)
            self.out_shape = (*out_shape, self.bins + self.padding)

        self.num_units = int(np.prod(self.out_shape))
        self.pdtype = pdtype
        self.cdtype = cdtype
        main_key, param_key = random.split(key, num=2)

        if "normal" in dist:
            param_key1, param_key2 = random.split(param_key)
            self._proj = {
                "mean": Linear(
                    param_key1,
                    in_features,
                    self.num_units,
                    use_bias=use_bias,
                    outscale=outscale,
                    winit=winit,
                    fan=fan,
                    fanin=fanin,
                    pdtype=self.pdtype,
                    cdtype=self.cdtype,
                ),
                "std": Linear(
                    param_key2,
                    in_features,
                    self.num_units,
                    use_bias=use_bias,
                    outscale=outscale,
                    winit=winit,
                    fan=fan,
                    fanin=fanin,
                    pdtype=self.pdtype,
                    cdtype=self.cdtype,
                ),
            }
        else:
            self._proj = {
                "mean": Linear(
                    param_key1,
                    in_features,
                    self.num_units,
                    use_bias=use_bias,
                    outscale=outscale,
                    winit=winit,
                    fan=fan,
                    fanin=fanin,
                    pdtype=self.pdtype,
                    cdtype=self.cdtype,
                )
            }

        last_axis_of_output_shape = self.out_shape[-1] - self.padding

        if dist == "symlog_mse":
            fwd, bwd = symlog, symexp
            self._dist = [
                lambda out: TransformedMseDist(
                    out["mean"], len(self.out_shape), fwd, bwd
                )
            ]

        if dist == "hyperbolic_mse":
            fwd = lambda x, eps=1e-3: (
                jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + eps * x
            )
            bwd = lambda x, eps=1e-3: jnp.sign(x) * (
                jnp.square(
                    jnp.sqrt(1 + 4 * eps * (eps + 1 + jnp.abs(x))) / 2 / eps
                    - 1 / 2 / eps
                )
                - 1
            )
            self._dist = [
                lambda out: TransformedMseDist(
                    out["mean"], len(self.out_shape), fwd, bwd
                )
            ]

        if dist == "symlog_and_twohot":
            bins = np.linspace(-20, 20, last_axis_of_output_shape)
            self._dist = [lambda out: TwoHotDist(out["mean"], bins, symlog, symexp)]

        if dist == "symexp_twohot":
            if last_axis_of_output_shape % 2 == 1:
                half = jnp.linspace(
                    -20, 0, (last_axis_of_output_shape - 1) // 2 + 1, dtype="float32"
                )
                half = symexp(half)
                bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
            else:
                half = jnp.linspace(
                    -20, 0, last_axis_of_output_shape // 2, dtype="float32"
                )
                half = symexp(half)
                bins = jnp.concatenate([half, -half[::-1]], 0)
            self._dist = [
                lambda out: TwoHotDist(out["mean"], bins, len(self.out_shape))
            ]

        if dist == "hyperbolic_tangent":
            eps = 0.001
            f = lambda x: np.sign(x) * (
                np.square(
                    np.sqrt(1 + 4 * eps * (eps + 1 + np.abs(x))) / 2 / eps - 1 / 2 / eps
                )
                - 1
            )
            bins = f(np.linspace(-300, 300, last_axis_of_output_shape))
            self._dist = [
                lambda out: TwoHotDist(out["mean"], bins, len(self.out_shape))
            ]

        if dist == "mse":
            self._dist = [lambda out: MSEDist(out["mean"], len(self.out_shape), "sum")]

        if dist == "huber":
            self._dist = [
                lambda out: HuberDist(out["mean"], len(self.out_shape), "sum")
            ]

        # Should add entropy term
        if dist == "normal":
            self._dist = [
                lambda out: tfd.Independent(
                    tfd.Normal(
                        jnp.tanh(out["mean"]),
                        (
                            (self.maxstd - self.minstd)
                            * jax.nn.sigmoid(out["std"] + 2.0)
                            + self.minstd
                        ),
                    ),
                    len(self.out_shape),
                )
            ]

        # Should add entropy term
        if dist == "trunc_normal":
            self._dist = [
                lambda out: tfd.Independent(
                    tfd.TruncatedNormal(
                        jnp.tanh(out["mean"]),
                        (
                            (self.maxstd - self.minstd)
                            * jax.nn.sigmoid(out["std"] + 2.0)
                            + self.minstd
                        ),
                    ),
                    len(self.out_shape),
                )
            ]

        if dist == "binary":
            if len(self.shape) > 1:
                self._dist = [
                    lambda out: tfd.Independent(
                        tfd.Bernoulli(out["mean"]), len(self.out_shape) - 1
                    )
                ]
            else:
                self._dist = [lambda out: tfd.Bernoulli(out["mean"])]

        if dist == "softmax":
            if len(self.shape) > 1:
                self._dist = [
                    lambda out: tfd.Independent(
                        tfd.Categorical(out["mean"]), len(self.out_shape) - 1
                    )
                ]
            else:
                self._dist = [lambda out: tfd.Categorical(out["mean"])]

        # Should add entropy term
        if dist == "onehot":
            if self.unimix:
                self._dist = [
                    lambda out: jax.nn.softmax(out["mean"], -1),
                    lambda out: (1 - self.unimix) * out
                    + self.unimix * (jnp.ones_like(out) / out.shape[-1]),
                    lambda out: jnp.log(out),
                ]
            else:
                self._dist = [lambda out: jax.nn.log_softmax(out["mean"], -1)]
            if len(self.shape) > 1:
                self._dist.append(
                    lambda out: tfd.Independent(
                        OneHotDist(out), len(self.out_shape) - 1
                    )
                )

            else:
                self._dist.append(lambda out: OneHotDist(out))

        else:
            raise NotImplementedError(dist)

    def __call__(self, x):
        dist = {k: v(x) for k, v in self._proj.items()}
        for func in self._dist:
            dist = func(dist)
        return x


class Conv2D(eqx.Module):
    kernel: jax.Array
    bias: jax.Array
    _norm: eqx.Module
    stride: int
    pad: str
    num_groups: int
    transposed: bool
    use_bias: bool
    act: str = "none"
    pdtype: str = "float32"
    cdtype: str = "float32"

    def __init__(
        self,
        key,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        transpose: bool = False,
        act: str = "none",
        norm: str = "none",
        pad: str = "same",
        use_bias: bool = True,
        outscale: float = 1.0,
        winit: str = "normal",
        binit: bool = False,
        fan: str = "in",
        pdtype="float32",
        cdtype="float32",
    ):
        self.transposed = transpose
        self.pad = pad
        self.act = act
        self.pdtype = pdtype
        self.cdtype = cdtype
        assert isinstance(in_channels, int), "in channels should be the type of integer"
        assert isinstance(
            out_channels, int
        ), "out channels should be the type of integer"
        assert isinstance(kernel_size, int), "kernel_size should be the type of integer"
        assert isinstance(stride, int), "stride should be the type of integer"
        assert isinstance(groups, int), "groups should be the type of integer"
        assert groups > 0, "Number of Groups muste be a positive integer"
        assert (
            in_channels % groups == 0
        ), "Number of in channels must be divisible by Number of Groups"
        assert not (
            transpose and (groups > 1)
        ), "In ConvTranspose the Number of Groups must be 1"
        wkey, bkey = jax.random.split(key, num=2)
        self.stride = stride
        self.num_groups = groups
        self.kernel = Initializer(dist=winit, scale=outscale, mode=fan)(
            wkey,
            (kernel_size, kernel_size, in_channels, out_channels),
            None,
            dtype=self.pdtype,
        )
        if use_bias:
            if binit:
                self.bias = Initializer(dist=winit, scale=outscale, mode=fan)(
                    bkey,
                    (out_channels,),
                    (kernel_size, kernel_size, in_channels, out_channels),
                    dtype=self.pdtype,
                )
            else:
                self.bias = Initializer(dist="zeros")(
                    bkey,
                    (out_channels,),
                    (kernel_size, kernel_size, in_channels, out_channels),
                    dtype=self.pdtype,
                )
            self.use_bias = True
        self._norm = Norm(num_features=out_channels, impl=norm, cdtype=self.cdtype)

    def __call__(self, x):
        x = cast_to_compute(x, self.cdtype)
        x = self._layer(x)
        x = self._norm(x)
        x = get_act(self.act)(x)
        return x

    def _layer(self, x):
        if self.transposed:
            x = jax.lax.conv_transpose(
                x,
                self.kernel.astype(self.cdtype),
                (self.stride, self.stride),
                self.pad.upper(),
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
        else:
            x = jax.lax.conv_general_dilated(
                x,
                self.kernel.astype(self.cdtype),
                (self.stride, self.stride),
                self.pad.upper(),
                feature_group_count=self.num_groups,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
        if self.use_bias:
            x += self.bias.astype(self.cdtype)
        return x


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
        binit: bool = False,
        fan: str = "in",
        fanin=0,
        pdtype="float32",
        cdtype="float32",
    ):
        self.act = act
        self.pdtype = pdtype
        self.cdtype = cdtype
        assert isinstance(
            in_features, int
        ), "num_features should be the type of integer"
        assert isinstance(
            out_features, int
        ), "out_features should be the type of integer"
        assert isinstance(fanin, int), "fanin should be the type of integer"

        wkey, bkey = jax.random.split(key, num=2)
        fan_shape = (fanin, out_features) if fanin else None
        self.weight = Initializer(dist=winit, scale=outscale, mode=fan)(
            wkey, (in_features, out_features), fan_shape, dtype=self.pdtype
        )

        if use_bias:
            if binit:
                self.bias = Initializer(dist=winit, scale=outscale, mode=fan)(
                    bkey, (out_features,), fan_shape, dtype=self.pdtype
                )
                pass
            else:
                self.bias = Initializer(dist="zeros")(
                    bkey, (out_features,), fan_shape, dtype=self.pdtype
                )
            self.use_bias = True

        self._norm = Norm(num_features=out_features, impl=norm, cdtype=self.cdtype)

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
    in_features: int
    out_features: int
    use_bias: bool
    num_groups: int
    block_norm: bool = False
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
        binit: bool = False,
        fan: str = "in",
        block_fans: bool = False,
        block_norm: bool = False,
        pdtype="float32",
        cdtype="float32",
    ):
        self.act = act
        self.pdtype = pdtype
        self.cdtype = cdtype
        assert isinstance(in_features, int), "The type of in_features should be int"
        assert isinstance(out_features, int), "The type of out_features should be int"
        assert isinstance(num_groups, int), "The type of num_groups should be int"
        assert (
            num_groups <= out_features
        ), "The type of num_groups cannot be larger than out_features"
        assert (
            in_features % num_groups == 0
        ), "The number of in_features should be the multiplier of number of groups"
        assert (
            out_features % num_groups == 0
        ), "The number of out_features should be the multiplier of number of groups"

        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        wkey, bkey = jax.random.split(key, num=2)
        self.weight = Initializer(
            dist=winit, scale=outscale, mode=fan, block_fans=block_fans
        )(
            wkey,
            (num_groups, in_features // num_groups, out_features // num_groups),
            None,
            dtype=self.pdtype,
        )

        if use_bias:
            if binit:
                self.bias = Initializer(dist=winit, scale=outscale, mode=fan)(
                    bkey, (out_features,), None, dtype=self.pdtype
                )
            else:
                self.bias = Initializer(dist="zeros")(
                    bkey, (out_features,), None, dtype=self.pdtype
                )
            self.use_bias = True
        if block_norm:
            self._norm = [
                Norm(
                    num_features=out_features // self.num_groups,
                    impl=norm,
                    cdtype=self.cdtype,
                )
                for _ in range(self.num_groups)
            ]
            self.block_norm = True
        else:
            self._norm = Norm(num_features=out_features, impl=norm, cdtype=self.cdtype)

    def __call__(self, x):
        x = cast_to_compute(x, self.cdtype)
        x = self._layer(x)
        if self.block_norm and self._norm != "none":
            x = jnp.concatenate(
                [f(y) for f, y in zip(self._norm, jnp.split(x, self.num_groups, -1))],
                -1,
            )
        else:
            x = self._norm(x)
        x = get_act(self.act)(x)
        return x

    def _layer(self, x):
        bdims = x.shape[:-1]
        x = x.reshape(*bdims, self.num_groups, self.in_features // self.num_groups)
        x = jnp.einsum("...ki,kio->...ko", x, self.weight.astype(self.cdtype))
        x = x.reshape(*bdims, self.out_features)
        if self.use_bias:
            x += self.bias.astype(self.cdtype)
        return x


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


class Initializer:
    VARIANCE_FACTOR = 1.0

    def __init__(self, dist="normal", scale=1.0, mode="in", block_fans=False):
        self.dist = dist
        self.scale = scale
        self.fan = mode
        self.block_fans = block_fans

    def __call__(self, key, shape, fan_shape=None, dtype="float32"):
        shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
        assert all(x > 0 for x in shape), shape
        fanin, fanout = self._fans(fan_shape or shape)
        fan = {"avg": (fanin + fanout) / 2, "in": fanin, "out": fanout}[self.fan]
        if self.dist == "zeros":
            value = jnp.zeros(shape, dtype)
        elif self.dist == "uniform":
            limit = np.sqrt(self.VARIANCE_FACTOR / fan)
            value = jax.random.uniform(key, shape, dtype, -limit, limit)
        elif self.dist == "normal":
            value = jax.random.truncated_normal(key, -2, 2, shape)
            value *= 1.1368 * np.sqrt(self.VARIANCE_FACTOR / fan)
            value = value.astype(dtype)
        elif self.dist == "normed":
            value = jax.random.uniform(key, shape, dtype, -1, 1)
            value /= jnp.linalg.norm(value.reshape((-1, shape[-1])), 2, 0)
        elif self.dist == "complex":
            assert jnp.issubdtype(dtype, jnp.complexfloating), dtype
            realdt = jnp.finfo(dtype).dtype
            value = jax.random.truncated_normal(key, -2, 2, (2, *shape), realdt)
            value = value[0] + 1j * value[1]
            value *= jax.lax.convert_element_type(1.137 * np.sqrt(1 / fan), realdt)
        elif self.dist == "ortho":
            nrows, ncols = shape[-1], np.prod(shape) // shape[-1]
            matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
            mat = jax.random.normal(key, matshape, dtype)
            qmat, rmat = jnp.linalg.qr(mat)
            qmat *= jnp.sign(jnp.diag(rmat))
            qmat = qmat.T if nrows < ncols else qmat
            qmat = qmat.reshape(nrows, *shape[:-1])
            value = jnp.moveaxis(qmat, 0, -1)
        else:
            raise NotImplementedError(self.dist)
        value *= self.scale
        return value

    def _fans(self, shape):
        if len(shape) == 0:
            return (1, 1)
        elif len(shape) == 1:
            return (1, shape[0])
        elif len(shape) == 2:
            return shape
        elif len(shape) == 3 and self.block_fans:
            return shape[1:]
        else:
            space = int(np.prod(shape[:-2]))
            return (shape[-2] * space, shape[-1] * space)


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

    linear_module = Linear(
        jax.random.key(0),
        in_features=1536,
        out_features=12288,
        act="sigmoid",
        norm="rms",
        cdtype=CDTYPE,
    )
    blocklinear_module = BlockLinear(
        jax.random.key(0),
        in_features=1536,
        out_features=12288,
        num_groups=8,
        act="sigmoid",
        norm="rms",
        block_fans=False,
        block_norm=False,
        cdtype=CDTYPE,
    )
    random_x_array = random.normal(random.key(0), shape=(16, 64, 1536), dtype=CDTYPE)
    random_y_array = random.bernoulli(random.key(0), shape=(16, 64, 12288)).astype(
        CDTYPE
    )

    print("starting on linear")
    print(linear_module)
    print(random_x_array.shape)
    print(linear_module(random_x_array).dtype)

    def loss(model, x, y):
        return jnp.mean((jax.vmap(model)(x) - y) ** 2)

    optim = optax.rmsprop(learning_rate=4e-5)
    opt_state = optim.init(eqx.filter(linear_module, eqx.is_array))
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
        linear_module, opt_state, value = make_step(
            loss, linear_module, opt_state, random_x_array, random_y_array
        )
        # print(value)
    print((time.time() - a) / N)

    print("sleeping...")
    time.sleep(5)
    print("starting on blocklinear")

    print(blocklinear_module)
    print(random_x_array.shape)
    print(blocklinear_module(random_x_array).dtype)

    optim = optax.rmsprop(learning_rate=4e-5)
    opt_state = optim.init(eqx.filter(blocklinear_module, eqx.is_array))
    print(opt_state)

    N = 10000
    a = time.time()
    for i in range(N):
        blocklinear_module, opt_state, value = make_step(
            loss, blocklinear_module, opt_state, random_x_array, random_y_array
        )
        # print(value)
    print((time.time() - a) / N)
