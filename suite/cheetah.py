import jax
from jax import numpy as jnp
import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.io import html, mjcf, model

from etils import epath
from flax import struct
from ml_collections import config_dict
import mujoco
from mujoco import mjx
class Cheetah(PipelineEnv):
  def __init__(
      self,
      forward_reward_weight=1.0,
      ctrl_cost_weight=0.1,
      reset_noise_scale=0.1,
      exclude_current_positions_from_observation=True,
      **kwargs
  ):
    #path = epath.resource_path('brax') / 'envs/assets/half_cheetah.xml'
    #sys = mjcf.load(path)
    path=epath.Path('cheetah.xml')
    mj_model = mujoco.MjModel.from_xml_path((path).as_posix())
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6
    sys = mjcf.load_model(mj_model)
    n_frames = 5

#    if backend in ['spring', 'positional']:
#      sys = sys.replace(dt=0.003125)
#      n_frames = 16
#      gear = jp.array([120, 90, 60, 120, 100, 100])
#      sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
    kwargs['backend']='mjx'
    super().__init__(sys=sys, **kwargs)

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )

  def reset(self, rng: jnp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=low, maxval=hi
    )
    qvel = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

    pipeline_state = self.pipeline_init(qpos, qvel)

    obs = self._get_obs(pipeline_state)
    reward, done, zero = jnp.zeros(3)
    metrics = {
        'x_position': zero,
        'x_velocity': zero,
        'reward_ctrl': zero,
        'reward_run': zero,
    }
    return State(pipeline_state, obs, reward, done, metrics)

  def step(self, state: State, action: jnp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    pipeline_state0 = state.pipeline_state
    #assert pipeline_state0  is not None
    pipeline_state = self.pipeline_step(pipeline_state0, action)

    x_velocity = (
        pipeline_state.x.pos[0, 0] - pipeline_state0.x.pos[0, 0]
    ) / self.dt
    forward_reward = self._forward_reward_weight * x_velocity
    ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))

    obs = self._get_obs(pipeline_state)
    reward = forward_reward - ctrl_cost
    state.metrics.update(
        x_position=pipeline_state.x.pos[0, 0],
        x_velocity=x_velocity,
        reward_run=forward_reward,
        reward_ctrl=-ctrl_cost,
    )

    return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

  def _get_obs(self, pipeline_state: base.State) -> jnp.ndarray:
    """Returns the environment observations."""
    position = pipeline_state.q
    velocity = pipeline_state.qd

    if self._exclude_current_positions_from_observation:
      position = position[1:]

    return jnp.concatenate((position, velocity))
#example format
#envs.register_environment('cheetah', Cheetah)