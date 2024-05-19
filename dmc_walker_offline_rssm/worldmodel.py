import jax
import optax
import equinox as eqx
from jax import random
import jax.numpy as jnp
from utils import eqx_adaptive_grad_clip, TransformedMseDist, symlog, symexp, video_grid
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
        _, outs = self.rssm.observe(
            step_key, prev_latent, prev_actions, embeds, data["is_first"]
        )
        loss, metrics = self.rssm.loss(loss_key, outs)
        feat = self.rssm.get_feat(outs)
        for name, head in self.heads.items():
            log_name = name
            data_name = name
            dist = head(feat)
            if data_name == "decoder":
                log_name = "recon"
                data_name = "image"
                dist = TransformedMseDist(
                    dist.astype("float32"), 3, symlog, symexp, "sum"
                )
            loss.update({log_name: -dist.log_prob(data[data_name].astype("float32"))})

        return loss, metrics

    def imagine(self, key, policy, start, horizon):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        start["key"] = key
        start["action"] = policy(start)

        def step(prev, _):
            prev = prev.copy()
            carry, _ = self.rssm.img_step(prev, prev.pop("action"))
            return {**carry, "action": policy(carry)}

        traj = jax.lax.scan(
            f=lambda *a, **kw: self.obs_step(*a, **kw),
            init=jnp.arange(horizon),
            xs=start,
            unroll=self.config.imag_unroll,
        )

        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def report(self, key, data):
        loss_key, obs_key, img_key = random.split(key, num=3)
        state = self.initial(len(data["is_first"]))
        report = {}
        losses, metrics = self.loss(loss_key, data, state)
        report.update({f"{k}_loss": v.sum(axis=-1).mean() for k, v in losses.items()})
        report.update(metrics)
        carry, outs = self.rssm.observe(
            obs_key,
            self.rssm.initial(8),
            data["action"][:8, :5, ...],
            eqx.filter_vmap(self.encoder, in_axes=1, out_axes=1)(data["image"])[:8, :5, ...],
            data["is_first"][:8, :5, ...],
        )
        
        feat = self.rssm.get_feat(outs)
        recon = symexp(self.heads["decoder"](feat))
        _, states = self.rssm.imagine(img_key, carry, data["action"][:8, 5:, ...])
        feat = self.rssm.get_feat(states)
        openl = symexp(self.heads["decoder"](feat))
        truth = data["image"][:8].astype(jnp.float32)
        model = jnp.concatenate([recon[:, :5], openl], 1)
        error = (model - truth + 1) / 2
        video = jnp.concatenate([truth, model, error], 2)
        report[f"openl_video"] = video
        report[f"openl_image"] = video_grid(video)
        
        return report


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
