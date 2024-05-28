import jax
import optax
import numpy as np
import equinox as eqx
from jax import random
import jax.numpy as jnp
from utils import MSEDist, image_grid, tensorstats, subsample
from networks import RSSM, ImageEncoder, ImageDecoder, MLP
from ml_collections import FrozenConfigDict
from typing import Callable


sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)


class WorldModel(eqx.Module):
    rssm: eqx.Module
    encoder: eqx.Module
    heads: dict
    obs_space: tuple
    act_space: int
    config: FrozenConfigDict

    def __init__(self, key, obs_space, act_space, config):
        encoder_param_key, rssm_param_key, heads_param_key = random.split(key, num=3)
        self.obs_space = {"0": obs_space}
        self.act_space = act_space
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
                out_shape=(),
                **config.reward_head,
            ),
            "cont": MLP(
                cont_param_key,
                out_shape=(),
                **config.cont_head,
            ),
        }
        self.config = FrozenConfigDict(config)

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
        for name, head in self.heads.items():
            log_name = name
            data_name = name
            if data_name == "decoder":
                log_name = "recon"
                data_name = "image"
                dist = head(outs)
                dist = MSEDist(dist.astype("float32"), 3, "sum")
            else:
                feat = self.rssm.get_feat(outs)
                dist = head(feat)
            loss.update({log_name: -dist.log_prob(data[data_name].astype("float32"))})

        return loss, metrics

    def imagine(self, key, policy, start, horizon):
        first_cont = (1.0 - start["is_terminal"]).astype("float32")
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
        loss_key, obs_key, oloop_key, img_key = random.split(key, num=4)
        state = self.initial(len(data["is_first"]))
        report = {}
        losses, metrics = self.loss(loss_key, data, state)
        report.update({f"{k}_loss": v.sum(axis=-1).mean() for k, v in losses.items()})
        report.update(metrics)
        carry, outs = self.rssm.observe(
            obs_key,
            self.rssm.initial(8),
            data["action"][:8, ...],
            eqx.filter_vmap(self.encoder, in_axes=1, out_axes=1)(data["image"])[:8, ...],
            data["is_first"][:8, ...],
        )
        full_recon = np.float32(self.heads["decoder"](outs).astype("float32"))
        truth = np.float32(data["image"][:8])
        error = (full_recon - truth + 1) / 2
        recon_video = np.concatenate([truth, full_recon, error], 2)
        report[f"recon_video"] = recon_video
        carry, outs = self.rssm.observe(
            oloop_key,
            self.rssm.initial(8),
            data["action"][:8, :5, ...],
            eqx.filter_vmap(self.encoder, in_axes=1, out_axes=1)(data["image"])[
                :8, :5, ...
            ],
            data["is_first"][:8, :5, ...],
        )
        recon = np.float32(self.heads["decoder"](outs).astype("float32"))
        _, states = self.rssm.imagine(img_key, carry, data["action"][:8, 5:, ...])
        openl = np.float32(self.heads["decoder"](states).astype("float32"))
        truth = np.float32(data["image"][:8])
        model = np.concatenate([recon[:, :5], openl], 1)
        error = (model - truth + 1) / 2
        video = np.concatenate([truth, model, error], 2)
        report[f"openl_video"] = video
        report[f"openl_image"] = image_grid(video[:6, :16, ...])

        return report


class ImagActorCritic(eqx.Module):
    actor: eqx.Module
    critics: dict
    scales: dict
    act_space: dict
    grad: str

    config: FrozenConfigDict

    def __init__(self, key, critics, scales, act_space, config):
        critics = {k: v for k, v in critics.items() if scales[k]}

        for k, scale in scales.items():
            assert not scale or k in critics, k
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        self.scales = scales
        self.act_space = act_space

        self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
        disc = act_space.discrete

        self.actor = MLP(
            key,
            out_shape=act_space.shape,
            **config.actor,
            dist=config.actor_dist_disc if disc else config.actor_dist_cont,
        )
        self.retnorms = {k: 0 for k in critics}

        self.config = FrozenConfigDict(config)

    def initial(self, batch_size):
        return {}

    def policy(self, carry, state):
        return carry, {"action": self.actor(state)}

    def loss(self, key, traj):
        rew_key, ret_key, normed_ret_key = random.split(key, num=3)
        metrics = {}
        advs = []
        total = sum(self.scales[k] for k in self.critics)
        for key, critic in self.critics.items():
            rew, ret, base = critic.score(traj, self.actor)
            offset, invscale = self.retnorms[key](ret)
            normed_ret = (ret - offset) / invscale
            normed_base = (base - offset) / invscale
            advs.append((normed_ret - normed_base) * self.scales[key] / total)
            metrics.update(tensorstats(rew_key, rew, f"{key}_reward"))
            metrics.update(tensorstats(ret_key, ret, f"{key}_return_raw"))
            metrics.update(
                tensorstats(normed_ret_key, normed_ret, f"{key}_return_normed")
            )
            metrics[f"{key}_return_rate"] = (jnp.abs(ret) >= 0.5).mean()
        adv = jnp.stack(advs).sum(0)
        policy = self.actor(sg(traj))
        logpi = policy.log_prob(sg(traj["action"]))[:-1]
        loss = {"backprop": -adv, "reinforce": -logpi * sg(adv)}[self.grad]
        ent = policy.entropy()[:-1]
        loss -= self.config.actent * ent
        loss *= sg(traj["weight"])[:-1]
        loss *= self.config.loss_scales.actor
        metrics.update(self._metrics(traj, policy, logpi, ent, adv))
        return loss.mean(), metrics

    def _metrics(self, key, traj, policy, logpi, ent, adv):
        metrics = {}
        act_key, rand_key, ent_key, logpi_key, adv_key, subsample_key = random.split(
            key, num=6
        )
        ent = policy.entropy()[:-1]
        rand = (ent - policy.minent) / (policy.maxent - policy.minent)
        rand = rand.mean(range(2, len(rand.shape)))
        act = traj["action"]
        act = jnp.argmax(act, -1) if self.act_space.discrete else act
        metrics.update(tensorstats(act_key, act, "action"))
        metrics.update(tensorstats(rand_key, rand, "policy_randomness"))
        metrics.update(tensorstats(ent_key, ent, "policy_entropy"))
        metrics.update(tensorstats(logpi_key, logpi, "policy_logprob"))
        metrics.update(tensorstats(adv_key, adv, "adv"))
        metrics["imag_weight_dist"] = subsample(subsample_key, traj["weight"])
        return metrics


class VFunction(eqx.Module):
    net: eqx.Module
    slow: eqx.Module
    updater: eqx.Module
    rewfn: Callable
    config: FrozenConfigDict

    def __init__(self, key, rewfn, config):
        net_key, slow_key = random.split(key, num=2)
        self.net = MLP(net_key, out_shape=(), **config.critic)
        self.slow = MLP(slow_key, out_shape=(), **config.critic)
        self.rewfn = rewfn
        self.updater = eqx.nn.Identity()
        self.config = FrozenConfigDict(config)

    def loss(self, key, traj, target):
        metrics = {}
        traj = {k: v[:-1] for k, v in traj.items()}
        dist = self.net(traj)
        loss = -dist.log_prob(sg(target))
        if self.config.critic_slowreg == "logprob":
            reg = -dist.log_prob(sg(self.slow(traj).mean()))
        elif self.config.critic_slowreg == "xent":
            reg = -jnp.einsum(
                "...i,...i->...", sg(self.slow(traj).probs), jnp.log(dist.probs)
            )
        else:
            raise NotImplementedError(self.config.critic_slowreg)
        loss += self.config.loss_scales.slowreg * reg
        loss = (loss * sg(traj["weight"])).mean()
        loss *= self.config.loss_scales.critic
        metrics = tensorstats(key, dist.mean())
        return loss, metrics

    def score(self, traj, actor=None):
        rew = self.rewfn(traj)
        assert (
            len(rew) == len(traj["action"]) - 1
        ), "should provide rewards for all but last action"
        discount = 1 - 1 / self.config.horizon
        disc = traj["cont"][1:] * discount
        value = self.net(traj).mean()
        vals = [value[-1]]
        interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
        for t in reversed(range(len(disc))):
            vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
        ret = jnp.stack(list(reversed(vals))[:-1])
        return rew, ret, value[:-1]


if __name__ == "__main__":
    import ml_collections

    config = ml_collections.ConfigDict(
        {
            "critic": {
                "num_layers": 3,
                "in_features": 512,
                "num_units": 512,
                "act": "silu",
                "norm": "rms",
            }
        }
    )
    vfunction = VFunction(jax.random.key(0), lambda x: x**2, config)
