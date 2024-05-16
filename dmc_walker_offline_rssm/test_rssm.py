import jax
import jax.numpy as jnp
from networks import RSSM

batch_size = 32
traj_length = 64
action_dim = 4


actions = jnp.ones((batch_size, traj_length, action_dim))


rssm = RSSM(jax.random.key(0), 1024, 256, action_dim, 32, 16, 16, (1, 2, 3, 4, 4))
carry = rssm.initial(32)
carry, out = rssm.imagine(jax.random.key(0), carry, actions)
