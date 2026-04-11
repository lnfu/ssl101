import jax
import jax.numpy as jnp


def init_params(
    key: jax.Array,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
) -> dict:
    """Initialize MLP parameters with small random weights.

    Args:
        key: JAX PRNG key.
        input_dim: Flattened input dimension (e.g. 784 for 28×28 images).
        hidden_dim: Hidden layer width.
        num_classes: Number of output classes.

    Returns:
        Dict with keys W1, b1, W2, b2.
    """
    k1, k2 = jax.random.split(key)
    scale = 0.01
    return {
        "W1": jax.random.normal(k1, (input_dim, hidden_dim)) * scale,
        "b1": jnp.zeros(hidden_dim),
        "W2": jax.random.normal(k2, (hidden_dim, num_classes)) * scale,
        "b2": jnp.zeros(num_classes),
    }


def forward(params: dict, X: jax.Array) -> jax.Array:
    """MLP forward pass: Linear → ReLU → Linear → Softmax.

    Args:
        params: Dict with W1, b1, W2, b2.
        X: float32 array of shape (N, input_dim).

    Returns:
        Probability array of shape (N, num_classes).
    """
    z1 = X @ params["W1"] + params["b1"]
    a1 = jnp.maximum(0.0, z1)  # ReLU
    z2 = a1 @ params["W2"] + params["b2"]

    # Numerically stable softmax: subtract row-wise max before exp
    z2 = z2 - jnp.max(z2, axis=-1, keepdims=True)
    exp_z2 = jnp.exp(z2)
    return exp_z2 / exp_z2.sum(axis=-1, keepdims=True)


def cross_entropy_loss(params: dict, X: jax.Array, y: jax.Array) -> jax.Array:
    """Mean cross-entropy loss.

    Args:
        params: Model parameters.
        X: float32 (N, input_dim).
        y: int32 (N,) class indices.

    Returns:
        Scalar loss value.
    """
    probs = forward(params, X)
    log_probs = jnp.log(probs + 1e-9)
    one_hot = jnp.eye(probs.shape[-1])[y]
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def accuracy(params: dict, X: jax.Array, y: jax.Array) -> jax.Array:
    """Fraction of correct predictions.

    Args:
        params: Model parameters.
        X: float32 (N, input_dim).
        y: int32 (N,) ground-truth class indices.

    Returns:
        Scalar accuracy in [0, 1].
    """
    preds = jnp.argmax(forward(params, X), axis=-1)
    return jnp.mean(preds == y)
