
import tensorflow as tf


def urwl_loss(y_true, y_pred, num_classes=4, lam=0.1, epsilon=1e-7):
    """
    Compact prototype of uncertainty-regularized weighted loss.

    y_true: (batch, depth, height, width), integer labels 0-3
    y_pred: (batch, depth, height, width, classes), softmax probabilities
    """
    y_true = tf.cast(y_true, tf.int32)

    # One-hot encode the four segmentation classes.
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)

    # Estimate class weights from voxel frequencies in this batch.
    axes = [0, 1, 2, 3]
    class_counts = tf.reduce_sum(y_true_one_hot, axis=axes)
    total_voxels = tf.reduce_sum(class_counts)

    class_weights = total_voxels / (
        num_classes * (class_counts + epsilon)
    )
    class_weights = tf.stop_gradient(class_weights)

    # Weighted soft Dice loss.
    intersection = tf.reduce_sum(
        y_true_one_hot * y_pred, axis=axes
    )
    denominator = tf.reduce_sum(
        y_true_one_hot + y_pred, axis=axes
    )

    dice_per_class = (
        2.0 * intersection + epsilon
    ) / (denominator + epsilon)

    dice_loss = tf.reduce_sum(
        class_weights * (1.0 - dice_per_class)
    ) / (tf.reduce_sum(class_weights) + epsilon)

    # Entropy regularization: encourages confident predictions.
    entropy = -tf.reduce_sum(
        y_pred * tf.math.log(y_pred), axis=-1
    )
    entropy_loss = tf.reduce_mean(entropy)

    return dice_loss + lam * entropy_loss
