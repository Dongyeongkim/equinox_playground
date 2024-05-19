import jax
import optax
import numpy as np
import ml_collections
import equinox as eqx
from jax import random
import jax.numpy as jnp
from dataset import get_dataset
from worldmodel import WorldModel
import tensorboardX as tensorboard
from utils import eqx_adaptive_grad_clip, video_grid


@eqx.filter_jit
def loss(worldmodel, key, data, state):
    real_loss = {}
    losses, metrics = worldmodel.loss(key, data, state)
    loss = 0
    for k, v in losses.items():
        partial_loss = v.sum(axis=1).mean()
        if k == "rep":
            partial_loss *= 0.1
        real_loss.update({k: partial_loss})
        loss += partial_loss

    return loss, (real_loss, metrics)


@eqx.filter_jit
def train_step(model, optim, opt_state, key, data, state):
    (total_loss, loss_and_info), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
        model, key, data, state
    )
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, total_loss, loss_and_info


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    summary_writer = tensorboard.SummaryWriter(workdir)
    param_key, training_key = random.split(random.key(config.seed), num=2)

    wm = WorldModel(param_key, (64, 64, 3), 6, config)
    optim = optax.chain(
        optax.clip_by_global_norm(1000.0),
        optax.adamaxw(learning_rate=config.lr),
    )

    ds = get_dataset(
        "buffer/",
        config.batch_size,
        config.traj_length,
        "float32",
    )
    data = {}
    state = wm.initial(config.batch_size)
    opt_state = optim.init(eqx.filter(wm, eqx.is_array))
    for epoch in range(config.num_epoch):
        for step, (image, action, reward, is_first, cont) in enumerate(
            zip(ds["image"], ds["action"], ds["reward"], ds["is_first"], ds["cont"])
        ):
            data.update({"image": jnp.array(image / 255.0)})
            data.update({"action": jnp.array(action)})
            data.update({"reward": jnp.array(reward)})
            data.update({"is_first": jnp.array(is_first)})
            data.update({"cont": jnp.array(cont)})
            training_key, partial_key = random.split(training_key, num=2)
            wm, opt_state, total_loss, loss_and_info = train_step(
                wm, optim, opt_state, partial_key, data, state
            )
            if (epoch * len(ds["is_first"]) + step) % 1000 == 0:
                print(f"epoch: {epoch} step: {step}, curent loss is: {total_loss}")
            loss_info, info = loss_and_info
            current_step = len(ds["is_first"]) * epoch + step
            summary_writer.add_scalar(
                f"train/loss/total_loss", total_loss, global_step=current_step
            )
            for k, v in loss_info.items():
                summary_writer.add_scalar(
                    f"train/loss/{k}_loss", v, global_step=current_step
                )
            
            if epoch * len(ds["is_first"]) + step % 1000 == 0:
                training_key, report_key = random.split(training_key)
                report = wm.report(report_key, data)
                for key, value in report.items():
                    if len(value.shape) == 0:
                        summary_writer.add_scalar(f"train/log/{key}", value, global_step=current_step)
                    elif len(value.shape) == 1:
                        if len(value) > 1024:
                            value = value.copy()
                            np.random.shuffle(value)
                            value = value[:1024]
                        summary_writer.add_histogram(f"train/log/{key}", value, global_step=current_step)
                    elif len(value.shape) == 2:
                        summary_writer.add_image(f"train/log/{key}", value[..., None], global_step=current_step)
                    elif len(value.shape) == 3:
                        summary_writer.add_image(f"train/log/{key}", value, global_step=current_step)
                    elif len(value.shape) == 4:
                        summary_writer.add_image(f"train/log/{key}", value, global_step=current_step)
                    elif len(value.shape) == 5:
                        summary_writer.add_video(f"train/log/{key}", value, global_step=current_step)
                    else:
                        raise NotImplementedError



if __name__ == "__main__":
    config = ml_collections.ConfigDict(
        {
            "encoder": {
                "channel_depth": 32,
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
                "deter": 4096,
                "hidden": 512,
                "latent_dim": 32,
                "latent_cls": 32,
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
                "channel_depth": 32,
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
                "in_features": 5120,
                "num_units": 512,
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
                "in_features": 5120,
                "num_units": 512,
                "act": "silu",
                "norm": "rms",
                "out_shape": (),
                "dist": "binary",
                "outscale": 0.0,
                "winit": "normal",
                "cdtype": "bfloat16",
            },
            "seed": 0,
            "lr": 1e-4,
            "batch_size": 16,
            "traj_length": 64,
            "precision": 16,
            "num_epoch": 100,
        }
    )

    train_and_evaluate(config, "exp_local/")
