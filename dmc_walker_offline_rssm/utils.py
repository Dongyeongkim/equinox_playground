import collections
import re

import jax
import optax
import equinox as eqx
import jax.numpy as jnp
from optax._src import base
from optax._src.clipping import unitwise_norm, unitwise_clip
from jax.tree_util import tree_map

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
mean_ = lambda v: jnp.float32(jnp.mean(v))
min_ = lambda v: jnp.float32(jnp.min(v))
max_ = lambda v: jnp.float32(jnp.max(v))
per_ = lambda u, v: jnp.float32(jnp.percentile(u, v))
minimum_ = lambda u, v: jnp.float32(jnp.minimum(u, v))
maximum_ = lambda u, v: jnp.float32(jnp.maximum(u, v))


# will add optimizer


class Optimizer(eqx.Module):
    lr: float
    scaler: str
    eps: float
    beta1: float
    beta2: float

    # Learning rate
    warmup: int
    anneal: int

    # Regularization
    wd: float
    wd_pattern: str

    # Clipping
    pmin: float
    globclip: float
    agc: float

    # Smoothing
    momentum: bool
    nesterov: bool

    # chain(optimiser)
    chain: optax.chain

    def __init__(
            self,
            lr,
            scaler="adam",
            eps=1e-7,
            beta1=0.9,
            beta2=0.999,
            warmup=1000,
            anneal=0,
            wd=0.0,
            wd_pattern=r"/weight$",
            pmin=1e-3,
            globclip=0.0,
            agc=0.0,
            momentum=False,
            nesterov=False,
    ):
        self.lr = lr
        self.scaler = scaler
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.warmup = warmup
        self.anneal = anneal
        self.wd = wd
        self.wd_pattern = wd_pattern
        self.pmin = pmin
        self.globclip = globclip
        self.agc = agc
        self.momentum = momentum
        self.nesterov = nesterov

        chain = []
        if self.globclip:
            chain.append(optax.clip_by_global_norm(self.globclip))
        if self.agc:
            chain.append(eqx_adaptive_grad_clip(self.agc, self.pmin))

        if self.scaler == "adam":
            chain.append(optax.scale_by_adam(self.beta1, self.beta2, self.eps))

        elif self.scaler == "rms":
            chain.append(scale_by_rms(self.beta2, self.eps))

        else:
            raise NotImplementedError(self.scaler)

        if self.momentum:
            chain.append(scale_by_momentum(self.beta1, self.nesterov))

        if self.wd:
            assert not self.wd_pattern[0].isnumeric(), self.wd_pattern
            pattern = re.compile(self.wd_pattern)
            wdmaskfn = lambda params: {k: bool(pattern.search(k)) for k in params}
            chain.append(optax.add_decayed_weights(self.wd, wdmaskfn))

        if isinstance(self.lr, dict):
            chain.append(scale_by_groups({pfx: -lr for pfx, lr in self.lr.items()}))
        else:
            chain.append(optax.scale(-self.lr))

        self.chain = optax.chain(*chain)

        ## float16 is not allowed, only bfloat16 and float32 or above precision are allowed.

    def init(self, modules):
        return self.chain.init(eqx.filter(modules, eqx.is_array))

    @eqx.filter_jit
    def update(self, opt_state, key, lossfn, modules, data, state):
        (total_loss, loss_and_info), grads = eqx.filter_value_and_grad(lossfn, has_aux=True)(modules, key, data, state)
        updates, opt_state = self.chain.update(grads, opt_state, modules)
        modules = eqx.apply_updates(modules, updates)
        return modules, opt_state, total_loss, loss_and_info


# video grid


def video_grid(video):
    B, T, H, W, C = video.shape
    return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


# normalising function


def symlog(x):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


# reset


def traj_reset(xs, reset):
    def fn(x):
        mask = reset
        while len(mask.shape) < len(x.shape):
            mask = mask[..., None]
        return x * (1 - mask.astype(x.dtype))

    return jax.tree_util.tree_map(fn, xs)


# computing function


def cast_to_compute(values, compute_dtype):
    return jax.tree_util.tree_map(
        lambda x: x if x.dtype == compute_dtype else x.astype(compute_dtype), values
    )


# tensor stats


def tensorstats(key, tensor, prefix=None):
    assert tensor.size > 0, tensor.shape
    assert jnp.issubdtype(tensor.dtype, jnp.floating), tensor.dtype
    tensor = tensor.astype("float32")  # To avoid overflows.
    metrics = {
        "mean": tensor.mean(),
        "std": tensor.std(),
        "mag": jnp.abs(tensor).mean(),
        "min": tensor.min(),
        "max": tensor.max(),
        "dist": subsample(key, tensor),
    }
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    return metrics


def subsample(key, values, amount=1024):
    values = values.flatten()
    if len(values) > amount:
        values = jax.random.permutation(key, values)[:amount]
    return values


# adaptive_gradient_clip for equinox

AdaptiveGradClipState = base.EmptyState


def eqx_adaptive_grad_clip(clipping: float, eps: float = 1e-3):
    def init_fn(params):
        del params
        return AdaptiveGradClipState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)
        params = eqx.filter(params, eqx.is_array)  # parameter filtering for eqx module
        g_norm, p_norm = jax.tree_util.tree_map(unitwise_norm, (updates, params))
        # Maximum allowable norm.
        max_norm = jax.tree_util.tree_map(
            lambda x: clipping * jnp.maximum(x, eps), p_norm
        )
        # If grad norm > clipping * param_norm, rescale.
        updates = jax.tree_util.tree_map(unitwise_clip, g_norm, max_norm, updates)
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)


# scale-by-rms


def scale_by_rms(beta=0.999, eps=1e-8):
    def init_fn(params):
        nu = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t, "float32"), params)
        step = jnp.zeros((), "int32")
        return (step, nu)

    def update_fn(updates, state, params=None):
        step, nu = state
        step = optax.safe_int32_increment(step)
        nu = jax.tree_util.tree_map(
            lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates
        )
        nu_hat = optax.bias_correction(nu, beta, step)
        updates = jax.tree_util.tree_map(
            lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat
        )
        return updates, (step, nu)

    return optax.GradientTransformation(init_fn, update_fn)


# scale-by-momentum


def scale_by_momentum(beta=0.9, nesterov=False):
    def init_fn(params):
        mu = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t, "float32"), params)
        step = jnp.zeros((), "int32")
        return (step, mu)

    def update_fn(updates, state, params=None):
        step, mu = state
        step = optax.safe_int32_increment(step)
        mu = optax.update_moment(updates, mu, beta, 1)
        if nesterov:
            mu_nesterov = optax.update_moment(updates, mu, beta, 1)
            mu_hat = optax.bias_correction(mu_nesterov, beta, step)
        else:
            mu_hat = optax.bias_correction(mu, beta, step)
        return mu_hat, (step, mu)

    return optax.GradientTransformation(init_fn, update_fn)


# scale-by-groups


def expand_groups(groups, keys):
    if isinstance(groups, (float, int)):
        return {key: groups for key in keys}
    groups = {
        group if group.endswith("/") else f"{group}/": value
        for group, value in groups.items()
    }
    assignment = {}
    groupcount = collections.defaultdict(int)
    for key in keys:
        matches = [prefix for prefix in groups if key.startswith(prefix)]
        if not matches:
            raise ValueError(
                f"Parameter {key} not fall into any of the groups:\n"
                + "".join(f"- {group}\n" for group in groups.keys())
            )
        if len(matches) > 1:
            raise ValueError(
                f"Parameter {key} fall into more than one of the groups:\n"
                + "".join(f"- {group}\n" for group in groups.keys())
            )
        assignment[key] = matches[0]
        groupcount[matches[0]] += 1
    for group in groups.keys():
        if not groupcount[group]:
            raise ValueError(
                f"Group {group} did not match any of the {len(keys)} keys."
            )
    expanded = {key: groups[assignment[key]] for key in keys}
    return expanded


def scale_by_groups(groups):
    def init_fn(params):
        return ()

    def update_fn(updates, state, params=None):
        scales = expand_groups(groups, updates.keys())
        updates = jax.tree_util.treemap(lambda u, s: u * s, updates, scales)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


# Distributions


class OneHotDist(tfd.OneHotCategorical):
    def __init__(self, logits=None, probs=None, dtype="float32"):
        super().__init__(logits, probs, dtype)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return super()._parameter_properties(dtype)

    def sample(self, sample_shape=(), seed=None):
        sample = sg(super().sample(sample_shape, seed))
        probs = self._pad(super().probs_parameter(), sample.shape)
        sample = sg(sample) + (probs - sg(probs)).astype(sample.dtype)
        return sample

    def _pad(self, tensor, shape):
        while len(tensor.shape) < len(shape):
            tensor = tensor[None]
        return tensor


class MSEDist:

    def __init__(self, mode, dims, agg="sum"):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        self.batch_shape = mode.shape[: len(mode.shape) - dims]
        self.event_shape = mode.shape[len(mode.shape) - dims:]

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class HuberDist:

    def __init__(self, mode, dims, agg="sum"):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        self.batch_shape = mode.shape[: len(mode.shape) - dims]
        self.event_shape = mode.shape[len(mode.shape) - dims:]

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        distance = jnp.sqrt(1 + distance) - 1
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class TransformedMseDist:

    def __init__(self, mode, dims, fwd, bwd, agg="sum", tol=1e-8):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._fwd = fwd
        self._bwd = bwd
        self._agg = agg
        self._tol = tol
        self.batch_shape = mode.shape[: len(mode.shape) - dims]
        self.event_shape = mode.shape[len(mode.shape) - dims:]

    def mode(self):
        return self._bwd(self._mode)

    def mean(self):
        return self._bwd(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - self._fwd(value)) ** 2
        distance = jnp.where(distance < self._tol, 0, distance)
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class TwoHotDist:

    def __init__(self, logits, bins, dims=0, transfwd=None, transbwd=None):
        assert logits.shape[-1] == len(bins), (logits.shape, len(bins))
        assert logits.dtype == "float32", logits.dtype
        assert bins.dtype == "float32", bins.dtype
        self.logits = logits
        self.probs = jax.nn.softmax(logits)
        self.dims = tuple([-x for x in range(1, dims + 1)])
        self.bins = jnp.array(bins)
        self.transfwd = transfwd or (lambda x: x)
        self.transbwd = transbwd or (lambda x: x)
        self.batch_shape = logits.shape[: len(logits.shape) - dims - 1]
        self.event_shape = logits.shape[len(logits.shape) - dims: -1]

    def mean(self):
        # The naive implementation results in a non-zero result even if the bins
        # are symmetric and the probabilities uniform, because the sum operation
        # goes left to right, accumulating numerical errors. Instead, we use a
        # symmetric sum to ensure that the predicted rewards and values are
        # actually zero at initialization.
        # return self.transbwd((self.probs * self.bins).sum(-1))
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = self.probs[..., :m]
            p2 = self.probs[..., m: m + 1]
            p3 = self.probs[..., m + 1:]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m: m + 1]
            b3 = self.bins[..., m + 1:]
            wavg = (p2 * b2).sum(-1) + ((p1 * b1)[..., ::-1] + (p3 * b3)).sum(-1)
            return self.transbwd(wavg)
        else:
            p1 = self.probs[..., : n // 2]
            p2 = self.probs[..., n // 2:]
            b1 = self.bins[..., : n // 2]
            b2 = self.bins[..., n // 2:]
            wavg = ((p1 * b1)[..., ::-1] + (p2 * b2)).sum(-1)
            return self.transbwd(wavg)

    def mode(self):
        return self.transbwd((self.probs * self.bins).sum(-1))

    def log_prob(self, x):
        assert x.dtype == "float32", x.dtype
        x = self.transfwd(x)
        below = (self.bins <= x[..., None]).astype("int32").sum(-1) - 1
        above = len(self.bins) - (self.bins > x[..., None]).astype("int32").sum(-1)
        below = jnp.clip(below, 0, len(self.bins) - 1)
        above = jnp.clip(above, 0, len(self.bins) - 1)
        equal = below == above
        dist_to_below = jnp.where(equal, 1, jnp.abs(self.bins[below] - x))
        dist_to_above = jnp.where(equal, 1, jnp.abs(self.bins[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
                jax.nn.one_hot(below, len(self.bins)) * weight_below[..., None]
                + jax.nn.one_hot(above, len(self.bins)) * weight_above[..., None]
        )
        log_pred = self.logits - jax.scipy.special.logsumexp(
            self.logits, -1, keepdims=True
        )
        return (target * log_pred).sum(-1).sum(self.dims)


class SlowUpdater(eqx.Module):
    updates: int = eqx.field()

    def __init__(self, src, dst, fraction=1.0, period=1):
        self.src = src
        self.dst = dst
        self.fraction = fraction
        self.period = period
        self.updates = 0

    @eqx.filter_jit
    def __call__(self):
        assert self.src.find()
        updates = self.updates
        need_init = (updates == 0)
        need_update = (updates % self.period == 0)
        mix = jnp.clip(1.0 * need_init + self.fraction * need_update, 0, 1)
        params = {
            k.replace(f'/{self.src.name}/', f'/{self.dst.name}/'): v
            for k, v in self.src.find().items()}
        ema = tree_map(
            lambda s, d: mix * s + (1 - mix) * d,
            params, self.dst.find())
        for name, param in ema.items():
            assert param.dtype == jnp.float32, (
                f'EMA of {name} should be float32 not {param.dtype}')
        self.dst.put(ema)
        self.updates = updates + 1


class Moments(eqx.Module):
    rate: float = 0.01
    limit: float = 1e-8
    perclo: float = 5.0
    perchi: float = 95.0

    mean: jnp.float32 = eqx.field()
    sqrs: jnp.float32 = eqx.field()
    corr: jnp.float32 = eqx.field()
    low: jnp.float32 = eqx.field()
    high: jnp.float32 = eqx.field()

    impl: str = eqx.static_field()

    def __init__(self, impl='mean_std'):
        self.impl = impl

        self.low = jnp.float32(0.)
        self.high = jnp.float32(0.)
        self.corr = jnp.float32(0.)
        self.sqrs = jnp.float32(0.)
        self.mean = jnp.float32(0.)

        if self.impl == 'off': pass
        elif self.impl == 'mean_std': pass
        elif self.impl == 'min_max': pass
        elif self.impl == 'perc': pass
        elif self.impl == 'perc_corr': pass
        else:
            raise NotImplementedError(self.impl)

    @eqx.filter_jit
    def __call__(self, x, update=True):
        update and self.update(x)
        return self.stats()

    @eqx.filter_jit
    def update(self, _x):
        x = sg(_x.astype('float32'))
        m = self.rate
        if self.impl == 'off':
            return self
        elif self.impl == 'mean_std':
            new_mean = (1 - m) * self.mean + m * mean_(x)
            new_sqrs = (1 - m) * self.sqrs + m * mean_(x * x)
            new_corr = (1 - m) * self.corr + m * 1.0
            return eqx.tree_at(lambda mod: (mod.mean, mod.sqrs, mod.corr), self, (new_mean, new_sqrs, new_corr))
        elif self.impl == 'min_max':
            low, high = min_(x), max_(x)
            new_low = (1 - m) * minimum_(self.low, low) + m * low
            new_high = (1 - m) * maximum_(self.high, high) + m * high
            return eqx.tree_at(lambda mod: (mod.low, mod.high), self, (new_low, new_high))
        elif self.impl == 'perc':
            low, high = per_(x, self.perclo), per_(x, self.perchi)
            new_low = (1 - m) * self.low + m * low
            new_high = (1 - m) * self.high + m * high
            return eqx.tree_at(lambda mod: (mod.low, mod.high), self, (new_low, new_high))
        elif self.impl == 'perc_corr':
            low, high = per_(x, self.perclo), per_(x, self.perchi)
            new_low = (1 - m) * self.low + m * low
            new_high = (1 - m) * self.high + m * high
            new_corr = (1 - m) * self.corr + m * 1.0
            return eqx.tree_at(lambda mod: (mod.low, mod.high, mod.corr), self, (new_low, new_high, new_corr))
        else:
            raise NotImplementedError(self.impl)

    @eqx.filter_jit
    def stats(self):
        if self.impl == 'off':
            return 0.0, 1.0
        elif self.impl == 'mean_std':
            corr = jnp.maximum(self.rate, self.corr)
            mean = self.mean / corr
            std = jnp.sqrt(jax.nn.relu(self.sqrs / corr - mean ** 2))
            std = jnp.maximum(self.limit, std)
            return sg(mean), sg(std)
        elif self.impl == 'min_max':
            offset = self.low
            span = self.high - self.low
            span = jnp.maximum(self.limit, span)
            return sg(offset), sg(span)
        elif self.impl == 'perc':
            offset = self.low
            span = self.high - self.low
            span = jnp.maximum(self.limit, span)
            return sg(offset), sg(span)
        elif self.impl == 'perc_corr':
            corr = jnp.maximum(self.rate, self.corr)
            lo = self.low / corr
            hi = self.high / corr
            span = hi - lo
            span = jnp.maximum(self.limit, span)
            return sg(lo), sg(span)
        else:
            raise NotImplementedError(self.impl)


if __name__ == '__main__':
    m = Moments()
    x = jnp.float32(5)
    print(m(x))
    print(m.stats())
    print(m(x))
    print(m.stats())
