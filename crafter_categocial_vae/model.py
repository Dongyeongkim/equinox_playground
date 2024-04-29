import jax
import equinox as eqx
from jax import random
import jax.numpy as jnp
from utils import cast_to_compute
from networks import Conv2D, Linear
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class CategoricalVAE(eqx.Module):
    enc: eqx.Module
    dec: eqx.Module
    dist: eqx.Module
    latent_dim: int
    latent_cls: int
    pdtype: str = "float32"
    cdtype: str = "float32"

    def __init__(
        self,
        key,
        latent_dim,
        latent_cls,
        unimix,
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
            latent_cls,
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
            latent_dim,
            latent_cls,
            unimix,
            in_features=(minres**2) * channel_depth * channel_multipliers[-1],
            pdtype=pdtype,
            cdtype=cdtype,
        )
        self.latent_dim = latent_dim
        self.latent_cls = latent_cls
        self.pdtype = pdtype
        self.cdtype = cdtype

    @eqx.filter_jit
    def __call__(self, x, key):
        x = cast_to_compute(x, self.cdtype)
        enc_x = self.enc(x)
        stoch = self.dist(enc_x, key)
        recon_x = self.dec(stoch)
        return recon_x
    
    @eqx.filter_jit
    def generate(self, z):
        return self.dec(z)


class OneHotDist(tfd.OneHotCategorical):
    def __init__(self, logits=None, probs=None, dtype="float32"):
        super().__init__(logits, probs, dtype)
    
    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return super()._parameter_properties(dtype)
    
    def sample(self, sample_shape=(), seed=None):
        sample = jax.lax.stop_gradient(super().sample(sample_shape, seed))
        probs = self._pad(super().probs_parameter(), sample.shape)
        sample = jax.lax.stop_gradient(sample) + (probs - jax.lax.stop_gradient(probs)).astype(sample.dtype)
        return sample
    
    def _pad(self, tensor, shape):
        while len(tensor.shape) < len(shape):
            tensor = tensor[None]
        return tensor



class Dist(eqx.Module):
    _logit: eqx.Module
    latent_dim: int
    latent_cls: int
    unimix: float
    pdtype: str = "float32"
    cdtype: str = "float32"

    def __init__(
        self,
        key,
        latent_dim,
        latent_cls,
        unimix, 
        in_features: int,
        pdtype: str = "float32",
        cdtype: str = "float32",
    ):
        self._logit = Linear(
            key, in_features, latent_dim*latent_cls, pdtype=pdtype, cdtype=cdtype
        )

        self.latent_dim = latent_dim
        self.latent_cls = latent_cls
        self.unimix = unimix
        self.pdtype = pdtype
        self.cdtype = cdtype

    def __call__(self, x, key):
        x = cast_to_compute(x, compute_dtype=self.cdtype)
        x = self._logit(x)
        logit = x.reshape(x.shape[0], self.latent_dim, self.latent_cls)
        if self.unimix:
            probs = jax.nn.softmax(logit, -1)
            uniform = jnp.ones_like(probs) / probs.shape[-1]
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logit = jnp.log(probs)
        else:
            logit = jax.nn.log_softmax(logit, -1)
        grad_applied_sample = OneHotDist(logit, dtype=self.cdtype).sample(seed=key)
        return grad_applied_sample.reshape(x.shape[0], -1)
        


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
        latent_dim,
        latent_cls,
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
            in_features=latent_dim*latent_cls,
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
        x += 0.5
        return x
