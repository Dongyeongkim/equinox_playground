import jax
import jax.numpy as jnp
from networks import RSSM


deter = 1024
hidden = 256

latent_dim = 32
latent_cls = 16

minres = 4
channel_depth = 16
channel_mults = (1, 2, 3, 4, 4)
batch_size = 32
traj_length = 64
action_dim = 4
embed_dim = (minres**2) * channel_mults[-1] * channel_depth


actions = jnp.ones((batch_size, traj_length, action_dim))

rssm = RSSM(
    jax.random.key(0),
    deter,
    hidden,
    action_dim,
    latent_dim,
    latent_cls,
    channel_depth,
    channel_mults,
)
carry = rssm.initial(32)
carry, out = rssm.imagine(jax.random.key(0), carry, actions)

print(carry.keys())
print("_________")
print(out.keys())

carry = rssm.initial(32)
embeds = jnp.ones((batch_size, traj_length, embed_dim))
resets = jnp.zeros((batch_size, traj_length))
out = rssm.observe(jax.random.key(0), carry, actions, embeds, resets)
print("_________")
print(out.keys())
