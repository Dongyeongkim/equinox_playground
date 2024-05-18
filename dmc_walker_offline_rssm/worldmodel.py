import jax
import optax
import equinox as eqx
from jax import random
import jax.numpy as jnp
from utils import eqx_adaptive_grad_clip, MSEDist
from networks import RSSM, ImageEncoder, ImageDecoder, MLP


class WorldModel(eqx.Module):
    rssm: eqx.Module
    encoder: eqx.Module
    heads: dict

    obs_space: tuple
    act_space: int
    config: dict

    def __init__(self, key, obs_space, act_space, config):
        encoder_param_key, rssm_param_key, heads_param_key = random.split(key, num=3)
        self.obs_space = obs_space
        self.act_space = act_space
        config.rssm.action_dim = self.act_space
        config.rssm.channel_depth = config.encoder.channel_depth
        config.rssm.channel_mults = config.encoder.channel_mults
        config.decoder.deter = config.rssm.deter
        config.decoder.latent_dim = config.rssm.latent_dim
        config.decoder.latent_cls = config.rssm.latent_cls
        self.encoder = ImageEncoder(encoder_param_key, **config.encoder)
        self.rssm = RSSM(rssm_param_key, **config.rssm)
        dec_param_key, rew_param_key, cont_param_key = random.split(
            heads_param_key, num=3
        )
        self.heads = {
            "decoder": eqx.filter_vmap(
                ImageDecoder(
                    dec_param_key,
                    **config.decoder,
                ),
                in_axes=1,
                out_axes=1,
            ),
            "reward": MLP(
                rew_param_key,
                **config.reward_head,
            ),
            "cont": MLP(
                cont_param_key,
                **config.cont_head,
            ),
        }
        self.config = config.to_dict()

    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros(
            (batch_size, self.act_space)
        )  # act_space should be integer
        return prev_latent, prev_action

    def loss(self, key, data, state):
        step_key, loss_key = random.split(key, num=2)
        embeds = eqx.filter_vmap(self.encoder, in_axes=1, out_axes=1)(data["image"])
        prev_latent, prev_action = state
        prev_actions = jnp.concatenate(
            [prev_action[:, None, ...], data["action"][:, :-1, ...]], 1
        )
        outs = self.rssm.observe(
            step_key, prev_latent, prev_actions, embeds, data["is_first"]
        )
        loss, metrics = self.rssm.loss(loss_key, outs)
        feat = jnp.concatenate(
            [outs["stoch"].reshape(*outs["stoch"].shape[:-2], -1), outs["deter"]], -1
        )
        for name, head in self.heads.items():
            log_name = name
            data_name = name
            dist = head(feat)
            if data_name == "decoder":
                log_name = "recon"
                data_name = "image"
                dist = MSEDist(dist.astype("float32"), 3, "sum")
            metrics.update({log_name: dist.mean()})
            loss.update({log_name: -dist.log_prob(data[data_name].astype("float32"))})

        return loss, metrics

    def report(self, key, data):
        pass


class ImagActorCritic(eqx.Module):
    pass


class VFunction(eqx.Module):
    pass


if __name__ == "__main__":
    import ml_collections

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
            },
            "reward_head": {
                "num_layers": 1,
                "in_features": 1280,
                "num_units": 256,
                "act": "silu",
                "norm": "rms",
                "out_shape": (),
                "dist": "symexp_twohot",
                "outscale": 0.0,
                "winit": "normal",
            },
            "cont_head": {
                "num_layers": 1,
                "in_features": 1280,
                "num_units": 256,
                "act": "silu",
                "norm": "rms",
                "out_shape": (),
                "dist": "binary",
                "outscale": 0.0,
                "winit": "normal",
            },
            "lr": 5e-4,
        }
    )

    wm = WorldModel(jax.random.key(0), (64, 64, 3), 6, config)
