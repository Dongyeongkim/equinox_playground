import jax
from jax import random
import jax.numpy as jnp
import ml_collections
from dataset import get_dataset
from worldmodel import WorldModel

config = ml_collections.ConfigDict(
    {
        "encoder": {
            "channel_depth": 16,
            "channel_mults": (1, 2, 3, 4, 4),
            "act": "silu",
            "norm": "rms",
            "winit": "normal",
            "debug_outer": True,
            "kernel_size": 5,
            "stride": 2,
            "minres": 4,
            "cdtype": "bfloat16",
        },
        "rssm": {
            "deter": 1024,
            "hidden": 256,
            "latent_dim": 32,
            "latent_cls": 16,
            "act": "silu",
            "norm": "rms",
            "unimix": 0.01,
            "outscale": 1.0,
            "winit": "normal",
            "num_imglayer": 2,
            "num_obslayer": 1,
            "num_dynlayer": 1,
            "blocks": 8,
            "block_fans": False,
            "block_norm": False,
            "cdtype": "bfloat16",
        },
        "decoder": {
            "channel_depth": 16,
            "channel_mults": (1, 2, 3, 4, 4),
            "act": "silu",
            "norm": "rms",
            "winit": "normal",
            "debug_outer": True,
            "kernel_size": 5,
            "stride": 2,
            "minres": 4,
            "cdtype": "bfloat16",
        },
        "reward_head": {
            "num_layers": 1,
            "in_features": 1536,
            "num_units": 256,
            "act": "silu",
            "norm": "rms",
            "out_shape": (),
            "dist": "symexp_twohot",
            "outscale": 0.0,
            "winit": "normal",
            "cdtype": "bfloat16",
        },
        "cont_head": {
            "num_layers": 1,
            "in_features": 1536,
            "num_units": 256,
            "act": "silu",
            "norm": "rms",
            "out_shape": (),
            "dist": "binary",
            "outscale": 0.0,
            "winit": "normal",
            "cdtype": "bfloat16",
        },
        "seed": 0,
        "lr": 5e-4,
        "batch_size": 16,
        "traj_length": 64,
        "precision": 16,
    }
)



param_key, training_key = random.split(random.key(config.seed), num=2)

wm = WorldModel(param_key, (64, 64, 3), 6, config)

ds = get_dataset(
    "buffer/",
    config.batch_size,
    config.traj_length,
    "bfloat16" if config.precision == 16 else "float32",
)

data = {}
state = wm.initial(config.batch_size)

for image, action, reward, is_first, cont in zip(ds["image"], ds["action"], ds["reward"], ds["is_first"], ds["cont"]):
    data.update({"image": jnp.array(image / 255.0)})
    data.update({"action": jnp.array(action)})
    data.update({"reward": jnp.array(reward)})
    data.update({"is_first": jnp.array(is_first)})
    data.update({"cont": jnp.array(cont)})
    training_key, partial_key = random.split(training_key, num=2)
    loss, metrics = wm.loss(partial_key, data, state)
    for k, v in loss.items():
        print(f'{k}: {v.shape}')
    for k, v in metrics.items():
        print(f'{k}: {v.shape}')

def lossfn(loss: dict):
    pass
