import jax
import equinox as eqx



class WorldModel(eqx.Module):
    rssm: eqx.Module
    enc: eqx.Module
    heads: dict

    def __init__(self, kw):
        pass


