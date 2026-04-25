import augmax
import jax
import jax.numpy as jnp

# STL-10 image statistics (used for Normalize in every augmentation chain).
_STL10_IMAGE_SIZE = 96
_STL10_MEAN = jnp.array([0.4467, 0.4398, 0.4066])
_STL10_STD = jnp.array([0.2603, 0.2566, 0.2713])


def make_pretrain_augmentation(
    image_size: int = _STL10_IMAGE_SIZE,
) -> augmax.Chain:
    """SimCLR pretraining view: crop, flip, color jitter, grayscale, blur, normalize."""
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


def make_probe_train_augmentation(
    image_size: int = _STL10_IMAGE_SIZE,
) -> augmax.Chain:
    """Linear probe training view: crop, flip, normalize."""
    return augmax.Chain(
        augmax.RandomSizedCrop(image_size, image_size),
        augmax.HorizontalFlip(),
        augmax.Normalize(mean=_STL10_MEAN, std=_STL10_STD),
    )


def make_normalize_only() -> augmax.Chain:
    """Normalize-only chain, used for eval and for the no-random-augment path."""
    return augmax.Chain(
        augmax.Normalize(mean=_STL10_MEAN, std=_STL10_STD),
    )


def augment_batch(
    chain: augmax.Chain,
    rng: jax.Array,
    images: jax.Array,
) -> jax.Array:
    rngs = jax.random.split(rng, images.shape[0])
    return jax.vmap(chain)(rngs, images)
