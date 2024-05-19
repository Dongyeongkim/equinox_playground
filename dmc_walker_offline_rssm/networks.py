import jax
import einops
import numpy as np
import equinox as eqx
from jax import random
import jax.numpy as jnp
from typing import List
from utils import symlog, symexp, cast_to_compute
from utils import OneHotDist, MSEDist, HuberDist
from utils import TransformedMseDist, TwoHotDist
from utils import traj_reset, tensorstats
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)


class RSSM(eqx.Module):
    dynlayers: dict
    imglayers: dict
    obslayers: dict

    deter: int
    hidden: int
    latent_dim: int
    latent_cls: int
    action_dim: int
    channel_depth: int
    channel_mults: tuple

    minres: int = 4
    norm: str = "rms"
    act: str = "silu"
    winit: str = "normal"
    unimix: float = 0.01
    outscale: float = 1.0

    num_imglayer: int = 2
    num_obslayer: int = 1
    num_dynlayer: int = 1

    blocks: int = 8
    block_fans: bool = False
    block_norm: bool = False
    pdtype: str = "float32"
    cdtype: str = "float32"

    def __init__(
        self,
        key,
        deter,
        hidden,
        action_dim,
        latent_dim,
        latent_cls,
        channel_depth,
        channel_mults,
        minres=4,
        norm="rms",
        act="silu",
        winit="normal",
        unimix=0.01,
        outscale=1.0,
        num_imglayer=2,
        num_obslayer=1,
        num_dynlayer=1,
        blocks=8,
        block_fans=False,
        block_norm=False,
        pdtype="float32",
        cdtype="float32",
    ):

        # Basic hyperparameters (architecture)

        self.deter = deter
        self.hidden = hidden
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.latent_cls = latent_cls
        self.channel_depth = channel_depth
        self.channel_mults = channel_mults

        # dynamics predictor(img), encoder(obs - except CNN), BlockGRU(dyn) hyperparameters

        self.num_imglayer = num_imglayer
        self.num_obslayer = num_obslayer
        self.num_dynlayer = num_dynlayer

        # minres, normalisation, activation

        self.minres = minres
        self.norm = norm
        self.act = act
        self.winit = winit
        self.unimix = unimix
        self.outscale = outscale

        # block gru hyperparameters

        self.blocks = blocks
        self.block_fans = block_fans
        self.block_norm = block_norm

        # parameter dtype and compute dtype

        self.pdtype = pdtype
        self.cdtype = cdtype

        # parameter key generation

        img_key, obs_key, dyn_key = random.split(key, num=3)
        img_key, imglogit_key, imginp_key = random.split(img_key, num=3)
        obs_key, obslogit_key, paraminp_key = random.split(obs_key, num=3)
        dyn_key, dyn_i1_key, dynh_key, dyn_deter_key, dyn_stoch_key, dyn_action_key = (
            random.split(dyn_key, num=6)
        )

        self.imglayers = {
            "imglogit": Linear(
                imglogit_key,
                in_features=self.hidden,
                out_features=self.latent_dim * self.latent_cls,
                act=self.act,
                norm=self.norm,
                outscale=self.outscale,
                winit=self.winit,
                binit=False,
                pdtype=self.pdtype,
                cdtype=self.cdtype,
            ),
            "img": [
                Linear(
                    imginp_key,
                    in_features=self.deter,
                    out_features=self.hidden,
                    act=self.act,
                    norm=self.norm,
                    winit=self.winit,
                    pdtype=self.pdtype,
                    cdtype=self.cdtype,
                ),
            ],
        }

        self.stack_module(img_key, "imglayers", "img", self.num_imglayer - 1)

        self.obslayers = {
            "obslogit": Linear(
                obslogit_key,
                in_features=self.hidden,
                out_features=self.latent_dim * self.latent_cls,
                act=self.act,
                norm=self.norm,
                outscale=self.outscale,
                winit=self.winit,
                binit=False,
                pdtype=self.pdtype,
                cdtype=self.cdtype,
            ),
            "obs": [
                Linear(
                    paraminp_key,
                    in_features=self.deter
                    + (self.minres**2) * self.channel_depth * self.channel_mults[-1],
                    out_features=self.hidden,
                    act=self.act,
                    norm=self.norm,
                    winit=self.winit,
                    pdtype=self.pdtype,
                    cdtype=self.cdtype,
                ),
            ],
        }

        self.stack_module(obs_key, "obslayers", "obs", self.num_obslayer - 1)

        self.dynlayers = {
            "dyn_h": BlockLinear(
                dynh_key,
                in_features=self.hidden,
                out_features=3 * self.deter,
                num_groups=self.blocks,
                winit=self.winit,
                pdtype=self.pdtype,
                cdtype=self.cdtype,
            ),
            "dyn_i": [
                BlockLinear(
                    dyn_i1_key,
                    in_features=3 * self.blocks * self.hidden + self.deter,
                    out_features=self.hidden,
                    num_groups=self.blocks,
                    act=self.act,
                    norm=self.norm,
                    winit=self.winit,
                    pdtype=self.pdtype,
                    cdtype=self.cdtype,
                )
            ],
            "dyn_in1": Linear(
                key=dyn_deter_key,
                in_features=self.deter,
                out_features=self.hidden,
                act=self.act,
                norm=self.norm,
                winit=self.winit,
                pdtype=self.pdtype,
                cdtype=self.cdtype,
            ),
            "dyn_in2": Linear(
                key=dyn_stoch_key,
                in_features=self.latent_dim * self.latent_cls,
                out_features=self.hidden,
                act=self.act,
                norm=self.norm,
                winit=self.winit,
                pdtype=self.pdtype,
                cdtype=self.cdtype,
            ),
            "dyn_in3": Linear(
                key=dyn_action_key,
                in_features=self.action_dim,
                out_features=self.hidden,
                act=self.act,
                norm=self.norm,
                winit=self.winit,
                pdtype=self.pdtype,
                cdtype=self.cdtype,
            ),
        }

        for _ in range(self.num_dynlayer - 1):
            dyn_key, dyn_i_key = random.split(dyn_key, num=2)
            self.dynlayers["dyn_i"].append(
                BlockLinear(
                    dyn_i_key,
                    in_features=self.hidden,
                    out_features=self.hidden,
                    num_groups=self.blocks,
                    act=self.act,
                    norm=self.norm,
                    winit=self.winit,
                    pdtype=self.pdtype,
                    cdtype=self.cdtype,
                )
            )

    def initial(self, bsize):
        carry = dict(
            deter=jnp.zeros([bsize, self.deter], self.pdtype),
            stoch=jnp.zeros([bsize, self.latent_dim, self.latent_cls], self.pdtype),
        )
        return cast_to_compute(carry, self.cdtype)

    def observe(self, key, carry, actions, embeds, resets, tdim=1):

        # input as (B, T, C), calculates in (T, B, C), and output as (B, T, C)
        actions = actions.swapaxes(
            0, tdim
        )  # change carry and action swapaxes (B, T, C) -> (T, B, C)
        embeds = embeds.swapaxes(
            0, tdim
        )  # change carry and action swapaxes (B, T, C) -> (T, B, C)
        resets = resets.swapaxes(
            0, tdim
        )  # change carry and action swapaxes (B, T) -> (T, B)

        carry["key"] = key
        carry, outs = jax.lax.scan(
            f=lambda *a, **kw: self.obs_step(*a, **kw),
            init=carry,
            xs=(actions, embeds, resets),
        )  # https://github.com/patrick-kidger/equinox/issues/558
        outs = {k: v.swapaxes(tdim, 0) for k, v in outs.items()}
        return carry, outs

    def imagine(self, key, carry, actions, tdim=1):
        # input as (B, T, C), calculates in (T, B, C), and output as (B, T, C)
        actions = actions.swapaxes(
            0, tdim
        )  # change carry and action swapaxes (B, T, C) -> (T, B, C)
        actions = cast_to_compute(actions, self.cdtype)

        carry["key"] = key
        carry, states = jax.lax.scan(
            f=lambda *a, **kw: self.img_step(*a, **kw), init=carry, xs=actions
        )  # https://github.com/patrick-kidger/equinox/issues/558
        states = {k: v.swapaxes(tdim, 0) for k, v in states.items()}
        return carry, states

    def obs_step(self, carry, action_embed_reset):
        key, step_key = random.split(carry["key"], num=2)
        action, embed, reset = action_embed_reset
        action, embed, reset = cast_to_compute((action, embed, reset), self.cdtype)
        deter, stoch, action = traj_reset(
            (
                cast_to_compute(carry["deter"], self.cdtype),
                cast_to_compute(carry["stoch"], self.cdtype),
                action,
            ),
            reset,
        )

        deter, feat = self._blockgru(deter, stoch, action)
        priorlogit = self._prior(feat)
        priorlogit = cast_to_compute(priorlogit, self.cdtype)

        post = jnp.concatenate([feat, embed], -1)
        for layer in self.obslayers["obs"]:
            post = layer(post)
        post = self.obslayers["obslogit"](post)
        postlogit = self._logit(post)
        postlogit = cast_to_compute(postlogit, self.cdtype)

        deterst = cast_to_compute(deter, self.cdtype)
        stochst = cast_to_compute(
            self._dist(postlogit).sample(seed=step_key), self.cdtype
        )
        carry = dict(
            key=key,
            deter=deterst,
            stoch=stochst,
        )
        outs = dict(deter=deterst, stoch=stochst, prior=priorlogit, post=postlogit)

        return carry, outs

    def img_step(self, carry, action):
        key, step_key = random.split(carry["key"], num=2)
        deter, feat = self._blockgru(carry["deter"], carry["stoch"], action)
        logit = self._prior(feat)
        deter_st = cast_to_compute(deter, self.cdtype)
        stoch_st = cast_to_compute(self._dist(logit).sample(seed=step_key), self.cdtype)

        carry = dict(
            key=key,
            deter=deter_st,
            stoch=stoch_st,
        )
        outs = dict(deter=deter_st, stoch=stoch_st, logit=logit)
        return carry, cast_to_compute(outs, self.cdtype)

    def loss(self, key, outs, free=1.0):
        metrics = {}
        dyn = self._dist(sg(outs["post"])).kl_divergence(self._dist(outs["prior"]))
        rep = self._dist(outs["post"]).kl_divergence(self._dist(sg(outs["prior"])))
        if free:
            dyn = jnp.maximum(dyn, free)
            rep = jnp.maximum(rep, free)

        metrics.update(
            tensorstats(key, self._dist(outs["prior"]).entropy(), "prior_ent")
        )
        metrics.update(tensorstats(key, self._dist(outs["post"]).entropy(), "post_ent"))
        return {"dyn": dyn, "rep": rep}, metrics

    def get_feat(self, state):
        return jnp.concatenate(
            [state["stoch"].reshape(*state["stoch"].shape[:-2], -1), state["deter"]], -1
        )

    def _prior(self, feat):
        x = feat
        for layer in self.imglayers["img"]:
            x = layer(x)
        x = self.imglayers["imglogit"](x)
        return self._logit(x)

    def _blockgru(self, deter, stoch, action):
        stoch = stoch.reshape((stoch.shape[0], -1))
        action /= sg(jnp.maximum(1, jnp.abs(action)))
        flat2group = lambda x: einops.rearrange(
            x, "... (g h) -> ... g h", g=self.blocks
        )
        group2flat = lambda x: einops.rearrange(
            x, "... g h -> ... (g h)", g=self.blocks
        )
        x0 = self.dynlayers["dyn_in1"](deter)
        x1 = self.dynlayers["dyn_in2"](stoch)
        x2 = self.dynlayers["dyn_in3"](action)
        x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(self.blocks, -2)
        x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
        for layer in self.dynlayers["dyn_i"]:
            x = layer(x)
        x = self.dynlayers["dyn_h"](x)
        gates = jnp.split(flat2group(x), 3, -1)
        reset, cand, update = [group2flat(x) for x in gates]
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deter
        out = deter
        return deter, out

    def _logit(self, x):
        logit = x.reshape(x.shape[:-1] + (self.latent_dim, self.latent_cls))
        if self.unimix:
            probs = jax.nn.softmax(logit, -1)
            uniform = jnp.ones_like(probs) / probs.shape[-1]
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logit = jnp.log(probs)
        return logit

    def _dist(self, logit):
        return tfd.Independent(OneHotDist(logit.astype("float32")), 1)

    def stack_module(self, key, module_name, submodule_name, num):
        layer = []
        for _ in range(num):
            key, param_key = random.split(key, num=2)
            layer.append(
                Linear(
                    param_key,
                    in_features=self.hidden,
                    out_features=self.hidden,
                    act=self.act,
                    norm=self.norm,
                    winit=self.winit,
                    pdtype=self.pdtype,
                    cdtype=self.cdtype,
                )
            )
        getattr(self, module_name)[submodule_name].extend(layer)


class ImageEncoder(eqx.Module):
    _conv_layers: list
    pdtype: str = "float32"
    cdtype: str = "float32"

    def __init__(
        self,
        key,
        debug_outer,
        channel_depth,
        channel_mults,
        kernel_size,
        stride,
        norm="rms",
        act="silu",
        winit="normal",
        minres=4,
        use_rgb=True,
        pdtype="float32",
        cdtype="float32",
    ):

        channels = (3 if use_rgb else 1,) + tuple(
            [channel_depth * mult for mult in channel_mults]
        )

        self._conv_layers = []
        for i in range(len(channel_mults)):
            stride_ = 1 if (debug_outer and (i == 0)) else stride
            key, param_key = random.split(key, num=2)
            self._conv_layers.append(
                Conv2D(
                    param_key,
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    stride=stride_,
                    act=act,
                    norm=norm,
                    winit=winit,
                    pdtype=pdtype,
                    cdtype=cdtype,
                )
            )
        self.pdtype = pdtype
        self.cdtype = cdtype

    def __call__(self, x):
        x = cast_to_compute(x, self.cdtype)
        x -= 0.5
        for layer in self._conv_layers:
            x = layer(x)
        x = x.reshape(x.shape[0], -1)
        return x


class ImageDecoder(eqx.Module):
    _convtr_layers: list
    _linear_proj: eqx.Module
    minres: int
    pdtype: str = "float32"
    cdtype: str = "float32"

    def __init__(
        self,
        key,
        deter,
        latent_dim,
        latent_cls,
        debug_outer,
        channel_depth,
        channel_mults,
        kernel_size,
        stride,
        norm="rms",
        act="silu",
        winit="normal",
        minres=4,
        use_rgb=True,
        pdtype="float32",
        cdtype="float32",
    ):
        channels = (3 if use_rgb else 1,) + tuple(
            [channel_depth * mult for mult in channel_mults]
        )

        key, param_key = random.split(key, num=2)
        self._linear_proj = Linear(
            param_key,
            in_features=latent_dim * latent_cls + deter,
            out_features=(minres**2) * channels[-1],
            act=act,
            norm=norm,
            pdtype=pdtype,
            cdtype=cdtype,
        )
        self._convtr_layers = []
        for i in reversed(range(1, len(channels))):
            stride_ = 1 if (debug_outer and (i == 1)) else stride
            key, param_key = random.split(key, num=2)
            if i == len(channels) - 1:
                self._convtr_layers.append(
                    Conv2D(
                        param_key,
                        in_channels=channels[i],
                        out_channels=channels[i - 1],
                        kernel_size=kernel_size,
                        stride=stride_,
                        transpose=True,
                        norm=norm,
                        winit=winit,
                        pdtype=pdtype,
                        cdtype=cdtype,
                    )
                )
            else:
                self._convtr_layers.append(
                    Conv2D(
                        param_key,
                        in_channels=channels[i],
                        out_channels=channels[i - 1],
                        kernel_size=kernel_size,
                        stride=stride_,
                        transpose=True,
                        act=act,
                        norm=norm,
                        winit=winit,
                        pdtype=pdtype,
                        cdtype=cdtype,
                    )
                )
        self.minres = minres
        self.pdtype = pdtype
        self.cdtype = cdtype

    def __call__(self, x):
        x = cast_to_compute(x, self.cdtype)
        x = self._linear_proj(x)
        x = x.reshape(x.shape[0], self.minres, self.minres, -1)
        for layer in self._convtr_layers:
            x = layer(x)
        x += 0.5
        return x  # remove applying dist on here due to it is not possible to apply vmap on here


class MLP(eqx.Module):
    layers: list
    dist: eqx.Module
    out_shape: bool
    pdtype: str
    cdtype: str

    def __init__(
        self,
        key,
        num_layers,
        in_features,
        num_units,
        act,
        norm,
        out_shape=None,
        dist="mse",
        use_bias=True,
        outscale=1.0,
        winit="normal",
        binit=False,
        fan="in",
        fanin=0,
        pdtype="float32",
        cdtype="float32",
    ):
        self.pdtype = pdtype
        self.cdtype = cdtype

        main_key = key
        self.layers = []
        for i in range(num_layers):
            main_key, param_key = random.split(main_key, num=2)
            if i == 0:
                self.layers.append(
                    Linear(
                        param_key,
                        in_features=in_features,
                        out_features=num_units,
                        act=act,
                        norm=norm,
                        use_bias=use_bias,
                        outscale=outscale,
                        winit=winit,
                        binit=binit,
                        fan=fan,
                        fanin=fanin,
                        pdtype=self.pdtype,
                        cdtype=self.cdtype,
                    )
                )
            else:
                self.layers.append(
                    Linear(
                        param_key,
                        in_features=num_units,
                        out_features=num_units,
                        act=act,
                        norm=norm,
                        use_bias=use_bias,
                        outscale=outscale,
                        pdtype=self.pdtype,
                        cdtype=self.cdtype,
                    )
                )
        if out_shape is not None:
            self.dist = Dist(
                key=main_key,
                in_features=num_units,
                out_shape=out_shape,
                dist=dist,
                minstd=1.0,
                maxstd=1.0,
                unimix=0.0,
                bins=255,
                outscale=0.1,
                use_bias=True,
                winit="normal",
                fan="in",
                fanin=0,
                pdtype=self.pdtype,
                cdtype=self.cdtype,
            )
            self.out_shape = True
        else:
            self.dist = eqx.nn.Identity()
            self.out_shape = False

    def __call__(self, x):
        x = cast_to_compute(x, self.cdtype)
        input_shape = x.shape
        x = x.reshape([-1, x.shape[-1]])
        for layer in self.layers:
            x = layer(x)
        x = x.reshape((*input_shape[:2], -1))
        if self.out_shape:
            x = self.dist(x)
            return x
        else:
            return x


class Dist(eqx.Module):
    _mean: eqx.Module
    _std: eqx.Module
    _dist: str
    shape: tuple
    padding: int
    num_unit: int
    out_shape: tuple
    minstd: float
    maxstd: float
    unimix: float
    bins: int
    pdtype: str = "float32"
    cdtype: str = "float32"

    def __init__(
        self,
        key,
        in_features,
        out_shape,
        dist="mse",
        minstd=1.0,
        maxstd=1.0,
        unimix=0.0,
        bins=255,
        outscale=0.1,
        use_bias=True,
        winit: str = "normal",
        fan: str = "in",
        fanin=0,
        pdtype="float32",
        cdtype="float32",
    ):
        self.minstd = minstd
        self.maxstd = maxstd
        self.unimix = unimix
        self.bins = bins
        self._dist = dist
        self.shape = ()
        self.out_shape = out_shape if isinstance(out_shape, tuple) else tuple(out_shape)
        self.num_unit = int(np.prod(self.out_shape))
        self.pdtype = pdtype
        self.cdtype = cdtype

        if "twohot" in self._dist or self._dist == "softmax":
            self.padding = int(self.bins % 2)
            self.shape = (*self.out_shape, self.bins + self.padding)
            self.num_unit = int(np.prod(self.shape))
        else:
            self.padding = 0

        if "normal" in dist:
            mean_key, std_key = random.split(key, num=2)
            self._mean = Linear(
                mean_key,
                in_features=in_features,
                out_features=(
                    int(np.prod(self.shape))
                    if len(self.shape)
                    else int(np.prod(self.out_shape))
                ),
                use_bias=use_bias,
                outscale=outscale,
                winit=winit,
                fan=fan,
                fanin=fanin,
                pdtype=self.pdtype,
                cdtype=self.cdtype,
            )
            self._std = Linear(
                std_key,
                in_features=in_features,
                out_features=int(np.prod(self.out_shape)),
                use_bias=use_bias,
                outscale=outscale,
                winit=winit,
                fan=fan,
                fanin=fanin,
                pdtype=self.pdtype,
                cdtype=self.cdtype,
            )
        else:
            self._mean = Linear(
                key,
                in_features=in_features,
                out_features=(
                    int(np.prod(self.shape))
                    if len(self.shape)
                    else int(np.prod(self.out_shape))
                ),
                use_bias=use_bias,
                outscale=outscale,
                winit=winit,
                fan=fan,
                fanin=fanin,
                pdtype=self.pdtype,
                cdtype=self.cdtype,
            )
            self._std = eqx.nn.Identity()

    def __call__(self, inputs):
        dist = self.inner(inputs)
        return dist

    def inner(self, inputs):
        out = self._mean(inputs)
        out = out.reshape(inputs.shape[:-1] + self.shape).astype("float32")
        out = out[..., : -self.padding] if self.padding else out

        if "normal" in self._dist:
            std = self._std(inputs)
            std = std.reshape(inputs.shape[:-1] + self.out_shape).astype("float32")

        if self._dist == "symlog_mse":
            fwd, bwd = symlog, symexp
            return TransformedMseDist(out, len(self.out_shape), fwd, bwd)

        if self._dist == "hyperbolic_mse":
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
            return TransformedMseDist(out, len(self.out_shape), fwd, bwd)

        if self._dist == "symlog_and_twohot":
            bins = np.linspace(-20, 20, out.shape[-1])
            return TwoHotDist(out, bins, len(self.out_shape), symlog, symexp)

        if self._dist == "symexp_twohot":
            if out.shape[-1] % 2 == 1:
                half = jnp.linspace(
                    -20, 0, (out.shape[-1] - 1) // 2 + 1, dtype="float32"
                )
                half = symexp(half)
                bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
            else:
                half = jnp.linspace(-20, 0, out.shape[-1] // 2, dtype="float32")
                half = symexp(half)
                bins = jnp.concatenate([half, -half[::-1]], 0)
            return TwoHotDist(out, bins, len(self.out_shape))

        if self._dist == "hyperbolic_twohot":
            eps = 0.001
            f = lambda x: np.sign(x) * (
                np.square(
                    np.sqrt(1 + 4 * eps * (eps + 1 + np.abs(x))) / 2 / eps - 1 / 2 / eps
                )
                - 1
            )
            bins = f(np.linspace(-300, 300, out.shape[-1]))
            return TwoHotDist(out, bins, len(self.out_shape))

        if self._dist == "mse":
            return MSEDist(out, len(self.out_shape), "sum")

        if self._dist == "huber":
            return HuberDist(out, len(self.out_shape), "sum")

        if self._dist == "normal":
            lo, hi = self.minstd, self.maxstd
            std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
            dist = tfd.Normal(jnp.tanh(out), std)
            dist = tfd.Independent(dist, len(self.out_shape))
            dist.minent = self.num_unit * tfd.Normal(0.0, lo).entropy()
            dist.maxent = self.num_unit * tfd.Normal(0.0, hi).entropy()
            return dist

        if self._dist == "trunc_normal":
            lo, hi = self.minstd, self.maxstd
            std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
            dist = tfd.TruncatedNormal(jnp.tanh(out), std, -1, 1)
            dist = tfd.Independent(dist, len(self.out_shape))
            dist.minent = self.num_unit * (
                tfd.TruncatedNormal(1.0, lo, -1, 1).entropy()
            )
            dist.maxent = self.num_unit * (
                tfd.TruncatedNormal(0.0, hi, -1, 1).entropy()
            )
            return dist

        if self._dist == "binary":
            dist = tfd.Bernoulli(out)
            if self.out_shape:
                dist = tfd.Independent(dist, len(self.out_shape))
            return dist

        if self._dist == "softmax":
            dist = tfd.Categorical(out)
            if len(self.out_shape) > 1:
                dist = tfd.Independent(dist, len(self.out_shape) - 1)
            return dist

        if self._dist == "onehot":
            if self.unimix:
                probs = jax.nn.softmax(out, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self.unimix) * probs + self.unimix * uniform
                out = jnp.log(probs)
            dist = OneHotDist(out)
            if len(self.out_shape) > 1:
                dist = tfd.Independent(dist, len(self.out_shape) - 1)
            dist.minent = 0.0
            dist.maxent = np.prod(self.out_shape[:-1]) * np.log(self.out_shape[-1])
            return dist

        raise NotImplementedError(self.dist)


class Conv2D(eqx.Module):
    weight: jax.Array
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
        self.weight = Initializer(dist=winit, scale=outscale, mode=fan)(
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
                self.weight.astype(self.cdtype),
                (self.stride, self.stride),
                self.pad.upper(),
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
        else:
            x = jax.lax.conv_general_dilated(
                x,
                self.weight.astype(self.cdtype),
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
