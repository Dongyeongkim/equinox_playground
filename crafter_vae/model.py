import jax
import equinox as eqx
from jax import random
import jax.numpy as jnp
from utils import cast_to_compute
from networks import Conv2D, Linear


class VAE(eqx.Module):
    enc: eqx.Module
    dec: eqx.Module
    dist: eqx.Module
    latent_dim: int
    pdtype: str = "float32"
    cdtype: str = "float32"

    def __init__(
        self,
        key,
        latent_dim,
        debug_outer,
        channel_depth,
        channel_multipliers,
        kernel_size,
        stride,
        norm="rms",
        act="silu",
        minres=4,
        use_rgb=True,
        pdtype="float32",
        cdtype="float32",
    ):
        enc_param_key, dec_param_key, dist_param_key = random.split(key, num=3)
        self.enc = ImageEncoder(
            enc_param_key,
            debug_outer,
            channel_depth,
            channel_multipliers,
            kernel_size,
            stride,
            norm,
            act,
            minres,
            use_rgb,
            pdtype,
            cdtype,
        )
        self.dec = ImageDecoder(
            dec_param_key,
            latent_dim,
            debug_outer,
            channel_depth,
            channel_multipliers,
            kernel_size,
            stride,
            norm,
            act,
            minres,
            use_rgb,
            pdtype,
            cdtype,
        )
        self.dist = Dist(
            dist_param_key,
            in_features=(minres**2) * channel_depth * channel_multipliers[-1],
            out_features=latent_dim,
            pdtype=pdtype,
            cdtype=cdtype,
        )
        self.latent_dim = latent_dim
        self.pdtype = pdtype
        self.cdtype = cdtype

    @eqx.filter_jit
    def __call__(self, x, key):
        x = cast_to_compute(x, self.cdtype)
        enc_x = self.enc(x)
        stoch, distinfo = self.dist(enc_x, key)
        recon_x = self.dec(stoch)
        return recon_x, distinfo
    
    @eqx.filter_jit
    def generate(self, z):
        return self.dec(z)


class Dist(eqx.Module):
    _mean: eqx.Module
    _logvar: eqx.Module
    pdtype: str = "float32"
    cdtype: str = "float32"

    def __init__(
        self,
        key,
        in_features: int,
        out_features: int,
        pdtype: str = "float32",
        cdtype: str = "float32",
    ):
        meankey, stdkey = random.split(key, num=2)

        self._mean = Linear(
            meankey, in_features, out_features, pdtype=pdtype, cdtype=cdtype
        )
        self._logvar = Linear(
            stdkey, in_features, out_features, pdtype=pdtype, cdtype=cdtype
        )

        self.pdtype = pdtype
        self.cdtype = cdtype

    def __call__(self, x, key):
        x = cast_to_compute(x, compute_dtype=self.cdtype)
        mean = self._mean(x)
        logvar = self._logvar(x)
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(key, logvar.shape)
        stoch = mean + jax.lax.stop_gradient(eps) * std
        return stoch, {"mean": mean, "logvar": logvar}


class ImageEncoder(eqx.Module):
    _conv_layers: list
    pdtype: str = "float32"
    cdtype: str = "float32"

    def __init__(
        self,
        key,
        debug_outer,
        channel_depth,
        channel_multipliers,
        kernel_size,
        stride,
        norm="rms",
        act="silu",
        minres=4,
        use_rgb=True,
        pdtype="float32",
        cdtype="float32",
    ):

        channels = (3 if use_rgb else 1,) + tuple(
            [channel_depth * mult for mult in channel_multipliers]
        )

        self._conv_layers = []
        for i in range(len(channel_multipliers)):
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
                    pdtype=pdtype,
                    cdtype=cdtype,
                )
            )
        self.pdtype = pdtype
        self.cdtype = cdtype

    def __call__(self, x):
        x = cast_to_compute(x, self.cdtype)
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
        latent_dim,
        debug_outer,
        channel_depth,
        channel_multipliers,
        kernel_size,
        stride,
        norm="rms",
        act="silu",
        minres=4,
        use_rgb=True,
        pdtype="float32",
        cdtype="float32",
    ):
        channels = (3 if use_rgb else 1,) + tuple(
            [channel_depth * mult for mult in channel_multipliers]
        )

        key, param_key = random.split(key, num=2)
        self._linear_proj = Linear(
            param_key,
            in_features=latent_dim,
            out_features=(minres**2)*channels[-1],
            act=act,
            norm=norm,
            pdtype=pdtype,
            cdtype=cdtype,
        )
        self._convtr_layers = []
        for i in reversed(range(1, len(channels))):
            stride_ = 1 if (debug_outer and (i == 1)) else stride
            key, param_key = random.split(key, num=2)
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
        return x
