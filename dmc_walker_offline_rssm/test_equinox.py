import jax
import jax.numpy as jnp
import equinox as eqx

class test_dictionary_input(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self):
        self.weight = jnp.ones((10))
        self.bias = jnp.ones((10))

    def __call__(self, x):
        return {'weight': x['weight']*self.weight, 'bias': x['bias']*self.bias}
    



if __name__ == '__main__':
    testmodule = test_dictionary_input()
    x = {'weight': jnp.ones((1,)), 'bias': jnp.zeros((1,))}
    print(testmodule(x))