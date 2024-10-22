import os
from jax import random
import jax.numpy as jnp


def get_crafter_dataset(rng, path, cdtype="float32"):
    ds = {}
    data = jnp.concatenate([jnp.array(jnp.load(path+elem)['image'], dtype=cdtype) / 255 for elem in os.listdir(path)])
    data = random.permutation(rng, data)
    ds['train'], ds['test'] = data[:int(0.8*len(data))], data[int(0.8*len(data)):]
    return ds['train'], ds['test']
    
