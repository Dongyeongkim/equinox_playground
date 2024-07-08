import jax
import optax
import equinox as eqx
from jax import vmap
from jax import random
import jax.numpy as jnp
from networks import Conv2D
from networks import Linear
from mnist_dataset import get_dataset_hf


class CNN(eqx.Module):
    conv_layers: list
    linear_layers: list

    def __init__(self, key, compute_dtype="float32"):
        param_key1, param_key2, param_key3, param_key4 = random.split(key, num=4)
        self.conv_layers = [
            Conv2D(
                param_key1,
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                act="relu",
                norm="none",
                cdtype=compute_dtype,
            ),
            vmap(
                eqx.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
                in_axes=-1,
                out_axes=-1,
            ),
            Conv2D(
                param_key2,
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                act="relu",
                norm="none",
                cdtype=compute_dtype,
            ),
            vmap(eqx.nn.AvgPool2d(kernel_size=2, stride=2), in_axes=-1, out_axes=-1),
        ]
        self.linear_layers = [
            Linear(
                param_key3, in_features=3136, out_features=256, act="relu", norm="rms"
            ),
            Linear(param_key4, in_features=256, out_features=10),
        ]

    def __call__(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.reshape(x.shape[0], -1)
        for layer in self.linear_layers:
            x = layer(x)
        return x


def loss(model, x, y):
    return cross_entropy(y, model(x))

loss = eqx.filter_jit(loss)


def cross_entropy(labels, logits):
    logits = jax.nn.log_softmax(logits)
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))


@eqx.filter_jit
def compute_accuracy(model: CNN, x, y):
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    pred_y = model(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean((y == pred_y))

@eqx.filter_jit
def evaluate(rng, model: CNN, test_ds, batch_size):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    avg_loss = 0
    avg_acc = 0

    test_ds_size = len(test_ds["image"])
    steps_per_epoch = test_ds_size // batch_size
    perms = random.permutation(rng, len(test_ds["image"]))
    perms = perms[: steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        avg_loss += loss(
            model, test_ds["image"][perm, ...], test_ds["label"][perm, ...]
        )
        avg_acc += compute_accuracy(
            model, test_ds["image"][perm, ...], test_ds["label"][perm, ...]
        )
    return avg_loss / len(perms) , avg_acc / len(perms)


optimiser = optax.adamw(learning_rate=3e-4, b1=0.1)


def train(
    model: CNN,
    train_ds,
    test_ds,
    optim: optax.GradientTransformation,
    num_epoch: int,
    batch_size: int,
    print_every: int,
) -> CNN:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(model: CNN, opt_state, x, y):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    train_ds_size = len(train_ds["image"])
    steps_per_epoch = train_ds_size // batch_size
    for epoch in range(num_epoch):
        perms = random.permutation(random.key(epoch), train_ds_size)
        perms = perms[: steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))
        for step, perm in enumerate(perms):
            model, opt_state, train_loss = make_step(
                model,
                opt_state,
                train_ds["image"][perm, ...],
                train_ds["label"][perm, ...],
            )
            if (step % print_every) == 0:
                test_loss, test_accuracy = evaluate(
                    random.key(step), model, test_ds, batch_size
                )
                print(
                    f"epoch={epoch}, step={step}, train_loss={train_loss.item()}, "
                    f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
                )
    return model


if __name__ == "__main__":
    train_ds, test_ds = get_dataset_hf(dtype="bfloat16")
    model = CNN(random.key(0), compute_dtype="bfloat16")
    trained_model = train(
        model,
        train_ds,
        test_ds,
        optimiser,
        num_epoch=100,
        batch_size=1024,
        print_every=20,
    )
