import augmax
import jax
import jax.numpy as jnp

_STL10_MEAN = jnp.array([0.4467, 0.4398, 0.4066])
_STL10_STD = jnp.array([0.2603, 0.2566, 0.2713])


def make_augmentation(image_size: int = 96) -> augmax.Chain:
    return augmax.Chain(
        augmax.RandomSizedCrop(image_size, image_size),
        augmax.HorizontalFlip(),
        augmax.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
        ),
        augmax.RandomGrayscale(p=0.2),
        augmax.GaussianBlur(p=0.5),
        augmax.Normalize(mean=_STL10_MEAN, std=_STL10_STD),
    )


def augment_batch(
    chain: augmax.Chain,
    rng: jax.Array,
    images: jax.Array,
) -> jax.Array:
    rngs = jax.random.split(rng, images.shape[0])
    return jax.vmap(chain)(rngs, images)
