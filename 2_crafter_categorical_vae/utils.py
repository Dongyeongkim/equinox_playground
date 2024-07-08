import jax
import math
import numpy as np
import equinox as eqx
from PIL import Image
import jax.numpy as jnp
from optax._src import base
from optax._src.clipping import unitwise_norm, unitwise_clip

# input normalisation

def symlog(x):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)

def symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

# loss functions

@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.vmap
def mse(recon, obs):
    return jnp.sum((recon-obs)**2)


# computing util(precision casting)

def cast_to_compute(values, compute_dtype):
    return jax.tree_util.tree_map(lambda x: x if x.dtype == compute_dtype else x.astype(compute_dtype), values)


# visualising util

def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.0, format_img=None):
    """Make a grid of images and Save it into an image file.

    Args:
      ndarray (array_like): 4D mini-batch images of shape (B x H x W x C)
      fp:  A filename(string) or file object
      nrow (int, optional): Number of images displayed in each row of the grid.
        The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
      padding (int, optional): amount of padding. Default: ``2``.
      pad_value (float, optional): Value for the padded pixels. Default: ``0``.
      format_img(Optional):  If omitted, the format to use is determined from the
        filename extension. If a file object was used instead of a filename,
        this parameter should always be used.
    """

    if not (
        isinstance(ndarray, jnp.ndarray)
        or (
            isinstance(ndarray, list)
            and all(isinstance(t, jnp.ndarray) for t in ndarray)
        )
    ):
        raise TypeError(f'array_like of tensors expected, got {type(ndarray)}')

    ndarray = jnp.asarray(ndarray)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = (
        int(ndarray.shape[1] + padding),
        int(ndarray.shape[2] + padding),
    )
    num_channels = ndarray.shape[3]
    grid = jnp.full(
        (height * ymaps + padding, width * xmaps + padding, num_channels),
        pad_value,
    ).astype(jnp.float32)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid = grid.at[
                y * height + padding : (y + 1) * height,
                x * width + padding : (x + 1) * width,
            ].set(ndarray[k])
            k = k + 1

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = np.array(jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8))
    im = Image.fromarray(ndarr.copy())
    im.save(fp, format=format_img)


# adaptive_gradient_clip for equinox

AdaptiveGradClipState = base.EmptyState

def eqx_adaptive_grad_clip(clipping: float, eps: float = 1e-3):
    def init_fn(params):
        del params
        return AdaptiveGradClipState()
    
    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)
        params = eqx.filter(params, eqx.is_array)
        g_norm, p_norm = jax.tree_util.tree_map(unitwise_norm, (updates, params))
        # Maximum allowable norm.
        max_norm = jax.tree_util.tree_map(lambda x: clipping * jnp.maximum(x, eps), p_norm)
        # If grad norm > clipping * param_norm, rescale.
        updates = jax.tree_util.tree_map(unitwise_clip, g_norm, max_norm, updates)
        return updates, state
    
    return base.GradientTransformation(init_fn, update_fn)

