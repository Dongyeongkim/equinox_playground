import optax
import ml_collections
import equinox as eqx
from model import VAE
from jax import random
from jax import numpy as jnp
from crafter_dataset import get_crafter_dataset
from utils import mse, kl_divergence, save_image


@eqx.filter_jit
def loss(model, x, key):
    recon_x, distinfo = model(x, key)
    rec_loss = mse(recon_x, x).mean()
    kl_loss = kl_divergence(distinfo["mean"], distinfo["logvar"]).mean()
    return rec_loss + kl_loss


@eqx.filter_jit
def train_step(model, optim, opt_state, x, key):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, key)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


@eqx.filter_jit
def eval_f(model, x, key, z_key, number_of_samples=64):
    recon_x, distinfo = model(x, key)
    rec_loss = mse(recon_x, x).mean()
    kl_loss = kl_divergence(distinfo["mean"], distinfo["logvar"]).mean()
    comparison = jnp.concatenate([x[:8], recon_x[:8]])
    sampled_from_normal_prior = model.generate(
        random.normal(z_key, (number_of_samples, model.latent_dim), dtype=model.cdtype)
    )
    return (
        {"loss": rec_loss + kl_loss, "rec_loss": rec_loss, "kl_loss": kl_loss},
        comparison,
        sampled_from_normal_prior,
    )


def train_and_evaluate(config: ml_collections.ConfigDict):
    main_key, param_key = random.split(random.key(config.seed), num=2)
    model = VAE(param_key, config.latent_dim, config.debug_outer, config.channel_depth, config.channel_multipliers, config.kernel_size, config.stride, cdtype=config.cdtype)
    optim = optax.adam(learning_rate=config.learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    main_key, dataset_key = random.split(main_key, num=2)
    train_ds, test_ds = get_crafter_dataset(dataset_key, 'crafter_dataset/', cdtype="bfloat16")
    steps_per_epoch = len(train_ds) // config.batch_size

    for epoch in range(config.num_epochs):
        for i in range(steps_per_epoch):
            batch = train_ds[config.batch_size*i:config.batch_size*(i+1)]
            main_key, sampling_key = random.split(main_key, num=2)
            model, opt_state, loss_value = train_step(model, optim, opt_state, batch, sampling_key)
        
        main_key, skey1, skey2 = random.split(main_key, num=3)
        metrics, comparison, sample = eval_f(model, test_ds[64*epoch:64*(epoch+1)], skey1, skey2)
        save_image(comparison, f'results/reconstruction_{epoch}.png', nrow=8)
        save_image(sample, f'results/sample_{epoch}.png', nrow=8)
        
        print('eval epoch: {}, loss: {:.4f}, RECON_LOSS: {:.4f}, KL_LOSS: {:.4f}'.format(epoch + 1, float(metrics['loss']), float(metrics['rec_loss']), float(metrics['kl_loss'])))



if __name__ == '__main__':
    import os
    import ml_collections

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.num_epochs = 100
    config.batch_size = 1024
    config.learning_rate = 4e-5
    config.latent_dim = 50
    config.debug_outer = True
    config.channel_depth = 32
    config.channel_multipliers = (1, 2, 3, 4, 4)
    config.kernel_size = 5
    config.stride = 2
    config.cdtype = "bfloat16"

    train_and_evaluate(config)