
import tensorflow as tf
from tensorflow.keras import layers


def gmf_block(modality_features, name="gmf"):
    """
    Gated Modality Fusion.
    Each input tensor must have the same shape.
    """
    gated_features = []

    for index, feature in enumerate(modality_features):
        channels = feature.shape[-1]

        transformed = layers.Conv3D(
            channels, 3, padding="same",
            name=f"{name}_transform_{index}"
        )(feature)

        transformed = layers.LayerNormalization(
            axis=-1, name=f"{name}_norm_{index}"
        )(transformed)

        gate = layers.Conv3D(
            channels, 1, padding="same",
            activation="sigmoid",
            name=f"{name}_gate_{index}"
        )(transformed)

        gated = layers.Multiply(
            name=f"{name}_multiply_{index}"
        )([feature, gate])

        gated_features.append(gated)

    fused = layers.Concatenate(
        name=f"{name}_concat"
    )(gated_features)

    fused = layers.Conv3D(
        16, 1, padding="same",
        activation="relu", name=f"{name}_projection"
    )(fused)

    return fused
