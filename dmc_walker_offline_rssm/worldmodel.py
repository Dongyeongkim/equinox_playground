import jax
import equinox as eqx
from jax import random
import jax.numpy as jnp
from networks import RSSM, ImageEncoder, ImageDecoder, MLP


class WorldModel(eqx.Module):
    rssm: eqx.Module
    encoder: eqx.Module
    heads: dict

    obs_space: tuple
    act_space: tuple
    config: dict
    

    def __init__(self, key, obs_space, act_space, config):
        encoder_param_key, rssm_param_key, heads_param_key = random.split(key, num=3)
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.config = config

        self.encoder = ImageEncoder(encoder_param_key, **config.encoder)
        self.rssm = RSSM(rssm_param_key, **config.rssm)
        dec_param_key, rew_param_key, cont_param_key = random.split(
            heads_param_key, num=3
        )
        self.heads = {
            "decoder": ImageDecoder(
                dec_param_key,
                **config.decoder,
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
    
    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        return prev_latent, prev_action
    
    def loss(self, key, data, state):
        step_key, loss_key = random.split(key, num=2)
        embeds = eqx.filter_vmap(self.encoder, in_axes=1)(data['obs'])
        prev_latent, prev_action = state
        prev_actions = jnp.concatenate([prev_action[:, None, ...], data['action'][:, :-1, ...]], 1)
        outs = self.rssm.observe(step_key, prev_latent, prev_actions, embeds, data['is_first'])
        loss, metrics = self.rssm.loss(loss_key, outs)
        feat = jnp.concatenate([outs["stoch"], outs["deter"]], -1)
        for name, head in self.heads.items():
            dist = head(feat)
            loss.update({name: -dist.log_prob(jnp.squeeze(data[name]).astype("float32")).sum()})
        
        return loss, metrics
    


class ImagActorCritic(eqx.Module):
    pass


class VFunction(eqx.Module):
    pass
