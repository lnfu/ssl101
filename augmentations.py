import augmax
import jax
import jax.numpy as jnp

_CIFAR10_MEAN = jnp.array([0.4914, 0.4822, 0.4465])
_CIFAR10_STD = jnp.array([0.2470, 0.2435, 0.2616])


def make_augmentation(image_size: int = 32) -> augmax.Chain:
    return augmax.Chain(
        augmax.RandomSizedCrop(image_size, image_size),
        augmax.HorizontalFlip(),
        augmax.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
        ),
        augmax.RandomGrayscale(p=0.2),
        augmax.GaussianBlur(p=0.5),
        augmax.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
    )


def augment_batch(
    chain: augmax.Chain,
    rng: jax.Array,
    images: jax.Array,
) -> jax.Array:
    rngs = jax.random.split(rng, images.shape[0])
    return jax.vmap(chain)(rngs, images)
