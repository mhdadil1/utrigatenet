
import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(x, filters):
    """Two 3D convolutions in one encoder/decoder block."""
    x = layers.Conv3D(
        filters, kernel_size=3, padding="same",
        activation="relu"
    )(x)

    x = layers.Conv3D(
        filters, kernel_size=3, padding="same",
        activation="relu"
    )(x)

    return x


def build_baseline_unet(
    input_shape=(32, 32, 32, 4),
    num_classes=4,
    base_filters=8
):
    """
    Small 3D U-Net for initial development tests.

    Input:  3D volume with four MRI channels.
    Output: voxel-wise class probabilities.

    This is a CPU-friendly scaffold, not the full
    UTriGate-Net architecture from the paper.
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    e1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling3D(pool_size=2)(e1)

    e2 = conv_block(p1, base_filters * 2)
    p2 = layers.MaxPooling3D(pool_size=2)(e2)

    e3 = conv_block(p2, base_filters * 4)
    p3 = layers.MaxPooling3D(pool_size=2)(e3)

    # Bottleneck
    b = conv_block(p3, base_filters * 8)

    # Decoder
    u3 = layers.Conv3DTranspose(
        base_filters * 4, kernel_size=2, strides=2
    )(b)
    u3 = layers.Concatenate()([u3, e3])
    d3 = conv_block(u3, base_filters * 4)

    u2 = layers.Conv3DTranspose(
        base_filters * 2, kernel_size=2, strides=2
    )(d3)
    u2 = layers.Concatenate()([u2, e2])
    d2 = conv_block(u2, base_filters * 2)

    u1 = layers.Conv3DTranspose(
        base_filters, kernel_size=2, strides=2
    )(d2)
    u1 = layers.Concatenate()([u1, e1])
    d1 = conv_block(u1, base_filters)

    outputs = layers.Conv3D(
        num_classes, kernel_size=1, activation="softmax"
    )(d1)

    return Model(inputs, outputs, name="Baseline_3D_UNet")


if __name__ == "__main__":
    model = build_baseline_unet()
    model.summary()

    # One synthetic volume; batch size = 1
    sample = tf.random.normal((1, 32, 32, 32, 4))
    prediction = model(sample, training=False)

    print("Input shape:     ", sample.shape)
    print("Prediction shape:", prediction.shape)
