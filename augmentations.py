import augmax
import jax
import jax.numpy as jnp


def make_augmentation(image_size: int = 32) -> augmax.Chain:
    return augmax.Chain(
        augmax.RandomCrop(image_size, image_size),
        augmax.HorizontalFlip(),
    )


def augment_batch(
    chain: augmax.Chain, rng: jax.Array, images: jax.Array, padding: int = 4
) -> jax.Array:
    # Pad before crop: (N, H, W, C) → (N, H+2p, W+2p, C)
    images = jnp.pad(images, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
    rngs = jax.random.split(rng, images.shape[0])
    return jax.vmap(chain)(rngs, images)
