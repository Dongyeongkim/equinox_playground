import jax
import optax
import ml_collections
import equinox as eqx
from model import CategoricalVAE
from jax import random
from jax import numpy as jnp
from crafter_dataset import get_crafter_dataset
from utils import mse, symlog, symexp, save_image, eqx_adaptive_grad_clip


@eqx.filter_jit
def loss(model, x, key):
    recon_x = model(x, key)
    rec_loss = mse(recon_x, symlog(x)).mean()
    return rec_loss


@eqx.filter_jit
def train_step(model, optim, opt_state, x, key):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, key)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


@eqx.filter_jit
def eval_f(model, x, key, z_key, number_of_samples=64):
    z_key1, z_key2 = random.split(z_key, num=2)
    recon_x = model(x, key)
    rec_loss = mse(recon_x, symlog(x)).mean()
    comparison = jnp.concatenate([x[:8], symexp(recon_x[:8])])
    sampled_from_normal_prior = symexp(model.generate(
        jax.nn.one_hot(
            random.categorical(
                z_key2,
                jax.nn.log_softmax(
                    random.normal(
                        z_key1,
                        (number_of_samples, model.latent_dim, model.latent_cls),
                        dtype=model.cdtype,
                    ),
                    -1,
                ),
            ),
            model.latent_cls,
        ).reshape(number_of_samples, -1)
    ))
    return (
        {"loss": rec_loss, "rec_loss": rec_loss},
        comparison,
        sampled_from_normal_prior,
    )


def train_and_evaluate(config: ml_collections.ConfigDict):
    main_key, param_key = random.split(random.key(config.seed), num=2)
    model = CategoricalVAE(
        param_key,
        config.latent_dim,
        config.latent_cls,
        config.unimix,
        config.debug_outer,
        config.channel_depth,
        config.channel_multipliers,
        config.kernel_size,
        config.stride,
        cdtype=config.cdtype,
    )
    optim = optax.chain(eqx_adaptive_grad_clip(0.3), optax.rmsprop(learning_rate=config.learning_rate, eps=1e-20, momentum=True))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    main_key, dataset_key = random.split(main_key, num=2)
    train_ds, test_ds = get_crafter_dataset(
        dataset_key, "crafter_dataset/", cdtype="bfloat16"
    )
    steps_per_epoch = len(train_ds) // config.batch_size

    for epoch in range(config.num_epochs):
        for i in range(steps_per_epoch):
            batch = train_ds[config.batch_size * i : config.batch_size * (i + 1)]
            main_key, sampling_key = random.split(main_key, num=2)
            model, opt_state, loss_value = train_step(
                model, optim, opt_state, batch, sampling_key
            )

        main_key, skey1, skey2 = random.split(main_key, num=3)
        metrics, comparison, sample = eval_f(
            model, test_ds[64 * epoch : 64 * (epoch + 1)], skey1, skey2
        )
        save_image(comparison, f"results/reconstruction_{epoch}.png", nrow=8)
        save_image(sample, f"results/sample_{epoch}.png", nrow=8)

        print(
            "eval epoch: {}, loss: {:.4f}, RECON_LOSS: {:.4f}".format(
                epoch + 1, float(metrics["loss"]), float(metrics["rec_loss"])
            )
        )


if __name__ == "__main__":
    import os
    import ml_collections

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.num_epochs = 100
    config.batch_size = 1024
    config.learning_rate = 4e-5
    config.latent_dim = 32
    config.latent_dim = 32
    config.latent_cls = 32
    config.unimix = 0.01
    config.debug_outer = True
    config.channel_depth = 32
    config.channel_multipliers = (1, 2, 3, 4, 4)
    config.kernel_size = 5
    config.stride = 2
    config.cdtype = "bfloat16"

    train_and_evaluate(config)
