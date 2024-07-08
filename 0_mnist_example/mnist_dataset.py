from datasets import load_dataset

import numpy as np
import jax.numpy as jnp


def get_dataset_hf(dtype="float32"):
    mnist = load_dataset("mnist")

    ds = {}

    for split in ["train", "test"]:
        ds[split] = {
            "image": np.array([np.array(im) for im in mnist[split]["image"]]),
            "label": np.array(mnist[split]["label"]),
        }

        # cast to jnp and rescale pixel values
        ds[split]["image"] = (ds[split]["image"]).astype(dtype) / 255
        ds[split]["label"] = (ds[split]["label"]).astype(dtype)

        # append trailing channel dimension
        ds[split]["image"] = jnp.expand_dims(ds[split]["image"], 3)

    return ds["train"], ds["test"]
