
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class InstanceNormalization3D(layers.Layer):
    """Normalize each channel across the 3D spatial dimensions."""

    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gamma = self.add_weight(
            name="gamma",
            shape=(channels,),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(channels,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        mean, variance = tf.nn.moments(
            x, axes=[1, 2, 3], keepdims=True
        )
        x = (x - mean) * tf.math.rsqrt(
            variance + self.epsilon
        )
        return (
            x * self.gamma[None, None, None, None, :]
            + self.beta[None, None, None, None, :]
        )

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


def depthwise_separable_conv3d(
    x, channels, kernel_size, dilation_rate, name
):
    """Grouped 3D convolution followed by pointwise convolution."""

    # Depthwise operation: each channel is convolved separately.
    x = layers.Conv3D(
        filters=channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        groups=channels,
        use_bias=False,
        name=f"{name}_depthwise",
    )(x)

    # Pointwise operation: mix information across channels.
    x = layers.Conv3D(
        filters=channels,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=f"{name}_pointwise",
    )(x)

    return x


def directional_branch(
    x, channels, kernel_size, dilation_rate, name, dropout_rate
):
    x = depthwise_separable_conv3d(
        x, channels, kernel_size, dilation_rate, name
    )
    x = InstanceNormalization3D(name=f"{name}_norm")(x)
    x = layers.LeakyReLU(
        negative_slope=0.01, name=f"{name}_activation"
    )(x)
    x = layers.Dropout(
        dropout_rate, name=f"{name}_dropout"
    )(x)
    return x


def tce_block(x, channels=None, dropout_rate=0.1, name="tce"):
    """
    Triaxial Context Encoding (TCE).

    Input and output:
        (batch, depth, height, width, channels)
    """

    if channels is None:
        channels = x.shape[-1]

    if channels is None:
        raise ValueError("The input channel count must be known.")

    channels = int(channels)

    # Three directional branches from the paper.
    axial = directional_branch(
        x, channels,
        kernel_size=(1, 3, 3),
        dilation_rate=(1, 2, 2),
        name=f"{name}_axial",
        dropout_rate=dropout_rate,
    )

    coronal = directional_branch(
        x, channels,
        kernel_size=(3, 1, 3),
        dilation_rate=(2, 1, 2),
        name=f"{name}_coronal",
        dropout_rate=dropout_rate,
    )

    sagittal = directional_branch(
        x, channels,
        kernel_size=(3, 3, 1),
        dilation_rate=(2, 2, 1),
        name=f"{name}_sagittal",
        dropout_rate=dropout_rate,
    )

    # Average the three branches to create a global descriptor.
    average_features = layers.Average(name=f"{name}_branch_average")(
        [axial, coronal, sagittal]
    )
    descriptor = layers.GlobalAveragePooling3D(
        name=f"{name}_global_pool"
    )(average_features)

    # Learn three attention scores and normalize them with softmax.
    hidden_units = max(channels // 4, 1)
    scores = layers.Dense(
        hidden_units, activation="relu",
        name=f"{name}_attention_fc1"
    )(descriptor)
    scores = layers.Dense(
        3, name=f"{name}_attention_fc2"
    )(scores)
    weights = layers.Softmax(
        name=f"{name}_branch_weights"
    )(scores)

    # Apply one learned weight to each spatial branch.
    weights = layers.Reshape(
        (1, 1, 1, 1, 3), name=f"{name}_weight_reshape"
    )(weights)

    branches = layers.Lambda(
        lambda tensors: tf.stack(tensors, axis=-1),
        output_shape= x.shape[1:] + (3,),
        name=f"{name}_stack_branches",
    )([axial, coronal, sagittal])

    weighted = layers.Multiply(name=f"{name}_weighted_branches")(
        [branches, weights]
    )
    fused = layers.Lambda(
        lambda tensor: tf.reduce_sum(tensor, axis=-1),
        output_shape=x.shape[1:],
        name=f"{name}_dynamic_fusion",
    )(weighted)

    # Aggregate the fused features.
    output = layers.Conv3D(
        channels,
        kernel_size=1,
        padding="same",
        name=f"{name}_output_projection",
    )(fused)

    return output



if __name__ == "__main__":
    inputs = layers.Input(shape=(32, 32, 32, 8))
    outputs = tce_block(inputs, name="tce_test")

    model = tf.keras.Model(inputs, outputs)

    sample = tf.random.normal((1, 32, 32, 32, 8))
    result = model(sample, training=False)

    # Create a model that outputs the attention weights.
    weight_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(
            "tce_test_branch_weights"
        ).output,
    )

    branch_weights = weight_model(
        sample, training=False
    ).numpy()

    print("Input shape: ", sample.shape)
    print("Output shape:", result.shape)
    print("Branch weights:", branch_weights)
    print("Weight sum:", branch_weights.sum(axis=-1))
