import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tce import tce_block
from models.gmf import gmf_block


def build_utrigate_net(input_shape=(32, 32, 32, 4), num_classes=4):
    inputs = layers.Input(shape=input_shape, name="mri_input")

    # Split input channels into FLAIR, T1, T1CE and T2.
    modality_features = []

    for index, modality in enumerate(["flair", "t1", "t1ce", "t2"]):
        x = layers.Lambda(
            lambda tensor, i=index: tensor[..., i:i + 1],
            output_shape=input_shape[:-1] + (1,),
            name=f"extract_{modality}"
        )(inputs)

        # Create modality-specific feature maps.
        x = layers.Conv3D(
            8, 3, padding="same", activation="relu",
            name=f"{modality}_stem"
        )(x)

        x = tce_block(x, name=f"{modality}_tce")
        modality_features.append(x)

    # Fuse modality-specific representations.
    fused = gmf_block(modality_features, name="gmf")

    # Compact encoder.
    skip = layers.Conv3D(
        16, 3, padding="same", activation="relu",
        name="encoder_conv"
    )(fused)

    down = layers.MaxPool3D(
        pool_size=2, name="encoder_pool"
    )(skip)

    down = layers.Conv3D(
        24, 3, padding="same", activation="relu",
        name="bottleneck_conv1"
    )(down)

    down = layers.Conv3D(
        24, 3, padding="same", activation="relu",
        name="bottleneck_conv2"
    )(down)

    # Decoder.
    up = layers.UpSampling3D(
        size=2, name="decoder_upsample"
    )(down)

    up = layers.Concatenate(name="decoder_skip")([up, skip])

    up = layers.Conv3D(
        16, 3, padding="same", activation="relu",
        name="decoder_conv"
    )(up)

    # Four-class segmentation probabilities.
    outputs = layers.Conv3D(
        num_classes, 1, padding="same",
        activation="softmax", name="segmentation_output"
    )(up)

    return Model(inputs, outputs, name="UTriGateNet_Prototype")


if __name__ == "__main__":
    model = build_utrigate_net()
    model.summary()

    test_input = tf.random.normal((1, 32, 32, 32, 4))
    prediction = model(test_input, training=False)

    print("Input shape:", test_input.shape)
    print("Prediction shape:", prediction.shape)
    print("Probability sum at one voxel:",
          float(tf.reduce_sum(prediction[0, 0, 0, 0, :])))

    save_path = PROJECT_ROOT / "models" / "utrigate_prototype.keras"
    model.save(save_path)
    print("Prototype model saved to:", save_path)
