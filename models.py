import jax.numpy as jnp
from jax.nn import initializers as jax_initializers
from flax import nnx

# kaiming_normal(fan_out, nonlinearity='relu') — matches torchvision ResNet18
_kaiming_normal = jax_initializers.variance_scaling(2.0, "fan_out", "normal")


class BasicBlock(nnx.Module):
    def __init__(
        self, in_features: int, out_features: int, strides: int, rngs: nnx.Rngs
    ) -> None:
        if strides != 1 or in_features != out_features:
            self.downsample_conv = nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(1, 1),
                strides=strides,
                use_bias=False,
                kernel_init=_kaiming_normal,
                rngs=rngs,
            )
            self.downsample_bn = nnx.BatchNorm(
                num_features=out_features,
                rngs=rngs,
            )
        else:
            self.downsample_conv = None
            self.downsample_bn = None

        self.conv1 = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(3, 3),
            padding=1,
            strides=strides,
            use_bias=False,
            kernel_init=_kaiming_normal,
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(num_features=out_features, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(3, 3),
            padding=1,
            use_bias=False,
            kernel_init=_kaiming_normal,
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(num_features=out_features, rngs=rngs)

    def __call__(
        self, x: jnp.ndarray, use_running_average: bool = False
    ) -> jnp.ndarray:
        if self.downsample_conv is None:
            identity = x
        else:
            identity = self.downsample_conv(x)
            identity = self.downsample_bn(
                identity, use_running_average=use_running_average
            )

        x = self.conv1(x)
        x = self.bn1(x, use_running_average=use_running_average)
        x = nnx.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, use_running_average=use_running_average)

        x += identity
        x = nnx.relu(x)

        return x


class Layer1(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.block1 = BasicBlock(64, 64, 1, rngs)
        self.block2 = BasicBlock(64, 64, 1, rngs)

    def __call__(
        self, x: jnp.ndarray, use_running_average: bool = False
    ) -> jnp.ndarray:
        x = self.block1(x, use_running_average)
        x = self.block2(x, use_running_average)
        return x


class Layer2(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.block1 = BasicBlock(64, 128, 2, rngs)
        self.block2 = BasicBlock(128, 128, 1, rngs)

    def __call__(
        self, x: jnp.ndarray, use_running_average: bool = False
    ) -> jnp.ndarray:
        x = self.block1(x, use_running_average)
        x = self.block2(x, use_running_average)
        return x


class Layer3(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.block1 = BasicBlock(128, 256, 2, rngs)
        self.block2 = BasicBlock(256, 256, 1, rngs)

    def __call__(
        self, x: jnp.ndarray, use_running_average: bool = False
    ) -> jnp.ndarray:
        x = self.block1(x, use_running_average)
        x = self.block2(x, use_running_average)
        return x


class Layer4(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.block1 = BasicBlock(256, 512, 2, rngs)
        self.block2 = BasicBlock(512, 512, 1, rngs)

    def __call__(
        self, x: jnp.ndarray, use_running_average: bool = False
    ) -> jnp.ndarray:
        x = self.block1(x, use_running_average)
        x = self.block2(x, use_running_average)
        return x


class ResNet18(nnx.Module):
    def __init__(self, num_classes: int, rngs: nnx.Rngs) -> None:
        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=3,
            use_bias=False,
            kernel_init=_kaiming_normal,
            rngs=rngs,
        )
        self.bn = nnx.BatchNorm(num_features=64, rngs=rngs)

        self.layer1 = Layer1(rngs)
        self.layer2 = Layer2(rngs)
        self.layer3 = Layer3(rngs)
        self.layer4 = Layer4(rngs)

        self.fc = nnx.Linear(
            in_features=512,
            out_features=num_classes,
            rngs=rngs,
        )

    def __call__(
        self, x: jnp.ndarray, use_running_average: bool = False
    ) -> jnp.ndarray:
        x = self.conv1(x)
        x = self.bn(x, use_running_average=use_running_average)
        x = nnx.relu(x)
        x = nnx.max_pool(
            x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
        )

        x = self.layer1(x, use_running_average)
        x = self.layer2(x, use_running_average)
        x = self.layer3(x, use_running_average)
        x = self.layer4(x, use_running_average)

        x = x.mean(axis=(1, 2))  # (B, H, W, C) -> (B, C)
        x = self.fc(x)

        return x
