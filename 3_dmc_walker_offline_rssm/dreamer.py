import jax
import jax.numpy as jnp
import equinox as eqx



import jaxagent
import behaviour
from worldmodel import WorldModel
from utils import cast_to_compute


import yaml


@jaxagent.Wrapper
class DreamerV3(eqx.Module):
    config: dict
    obs_space: dict
    act_space: dict
    step: int
    wm: eqx.Module
    task_behavior: eqx.Module
    expl_behavior: eqx.Module

    configs = yaml.YAML(typ='safe').load(
        (embodied.Path(__file__).parent / 'configs.yaml').read())

    def __init__(self, obs_space, act_space, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.step = step
        self.wm = WorldModel(obs_space, act_space, config, name='wm')
        self.task_behavior = getattr(behaviour, config.task_behavior)(
            self.wm, self.act_space, self.config, name='task_behavior')
        if config.expl_behavior == 'None':
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviour, config.expl_behavior)(
                self.wm, self.act_space, self.config, name='expl_behavior')

    def policy_initial(self, batch_size):
        return (
            self.wm.initial(batch_size),
            self.task_behavior.initial(batch_size),
            self.expl_behavior.initial(batch_size))

    def train_initial(self, batch_size):
        return self.wm.initial(batch_size)

    def policy(self, obs, state, mode='train'):
        obs = self.preprocess(obs)
        (prev_latent, prev_action), task_state, expl_state = state
        embed = self.wm.encoder(obs)
        latent, _ = self.wm.rssm.obs_step(
            prev_latent, prev_action, embed, obs['is_first'])
        self.expl_behavior.policy(latent, expl_state)
        task_outs, task_state = self.task_behavior.policy(latent, task_state)
        expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
        if mode == 'eval':
            outs = task_outs
            outs['action'] = outs['action'].sample(seed=nj.rng())
            outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
        elif mode == 'explore':
            outs = expl_outs
            outs['log_entropy'] = outs['action'].entropy()
            outs['action'] = outs['action'].sample(seed=nj.rng())
        elif mode == 'train':
            outs = task_outs
            outs['log_entropy'] = outs['action'].entropy()
            outs['action'] = outs['action'].sample(seed=nj.rng())
        state = ((latent, outs['action']), task_state, expl_state)
        return outs, state

    def train(self, data, state):
        # self.config.jax.jit and print('Tracing train function.')
        # metrics = {}
        # data = self.preprocess(data)
        # state, wm_outs, mets = self.wm.train(data, state)
        # metrics.update(mets)
        # context = {**data, **wm_outs['post']}
        # start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
        # _, mets = self.task_behavior.train(self.wm.imagine, start, context)
        # metrics.update(mets)
        # if self.config.expl_behavior != 'None':
        #     _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
        #     metrics.update({'expl_' + key: value for key, value in mets.items()})
        # outs = {}
        # return outs, state, metrics
        pass

    def report(self, data):
        data = self.preprocess(data)
        report = {}
        report.update(self.wm.report(data))
        mets = self.task_behavior.report(data)
        report.update({f'task_{k}': v for k, v in mets.items()})
        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)
            report.update({f'expl_{k}': v for k, v in mets.items()})
        return report

    def preprocess(self, obs):
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith('log_') or key in ('key',):
                continue
            if len(value.shape) > 3 and value.dtype == jnp.uint8:
                value = cast_to_compute(value) / 255.0
            else:
                value = value.astype(jnp.float32)
            obs[key] = value
        obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
        return obs
