import os
import hydra
import numpy as np
import ml_collections
import equinox as eqx
from jax import random
import jax.numpy as jnp
from dataset import get_dataset
from worldmodel import WorldModel
import tensorboardX as tensorboard
from utils import Optimizer


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
    optim = Optimizer(lr=4e-5, scaler="rms", eps=1e-20, agc=0.3, momentum=True)

    ds = get_dataset(
        "buffer/",
        config.batch_size,
        config.traj_length,
        "float32",
    )
    data = {}
    state = wm.initial(config.batch_size)
    opt_state = optim.init(wm)
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
            wm, opt_state, total_loss, loss_and_info = optim.update(opt_state, partial_key, loss, wm, data, state)
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
            
            if current_step % 1000 == 0:
                training_key, report_key = random.split(training_key)
                report = wm.report(report_key, data)
                for key, value in report.items():
                    if len(value.shape) == 0:
                        summary_writer.add_scalar(f"report/log/{key}", value, global_step=current_step)
                    elif len(value.shape) == 1:
                        if len(value) > 1024:
                            value = value.copy()
                            np.random.shuffle(value)
                            value = value[:1024]
                        summary_writer.add_histogram(f"report/log/{key}", value, global_step=current_step)
                    elif len(value.shape) == 2:
                        summary_writer.add_image(f"report/log/{key}", value[..., None], global_step=current_step)
                    elif len(value.shape) == 3:
                        summary_writer.add_image(f"report/log/{key}", value, global_step=current_step, dataformats="HWC")
                    elif len(value.shape) == 4:
                        summary_writer.add_images(f"report/log/{key}", value, global_step=current_step, dataformats="NHWC")
                    elif len(value.shape) == 5:
                        summary_writer.add_video(f"report/log/{key}", value, global_step=current_step, dataformats="NTHWC")
                    else:
                        raise NotImplementedError

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    config = ml_collections.ConfigDict(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    path = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    train_and_evaluate(config, path)


if __name__ == "__main__":
    main()
