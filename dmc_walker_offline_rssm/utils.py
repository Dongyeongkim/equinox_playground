import jax
import equinox as eqx
import jax.numpy as jnp
from optax._src import base
from optax._src.clipping import unitwise_norm, unitwise_clip

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)


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
        self.event_shape = mode.shape[len(mode.shape) - dims :]

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
        self.event_shape = mode.shape[len(mode.shape) - dims :]

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
        self.event_shape = mode.shape[len(mode.shape) - dims :]

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
        self.event_shape = logits.shape[len(logits.shape) - dims : -1]

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
            p2 = self.probs[..., m : m + 1]
            p3 = self.probs[..., m + 1 :]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m : m + 1]
            b3 = self.bins[..., m + 1 :]
            wavg = (p2 * b2).sum(-1) + ((p1 * b1)[..., ::-1] + (p3 * b3)).sum(-1)
            return self.transbwd(wavg)
        else:
            p1 = self.probs[..., : n // 2]
            p2 = self.probs[..., n // 2 :]
            b1 = self.bins[..., : n // 2]
            b2 = self.bins[..., n // 2 :]
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
