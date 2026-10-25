
import tensorflow as tf
from losses.urwl import urwl_loss

# One batch of 32 x 32 x 32 voxels with integer labels.
y_true = tf.zeros((1, 32, 32, 32), dtype=tf.int32)

# Simulated four-class softmax predictions.
logits = tf.random.normal((1, 32, 32, 32, 4))
y_pred = tf.nn.softmax(logits, axis=-1)

loss = urwl_loss(y_true, y_pred)

print("Ground-truth shape:", y_true.shape)
print("Prediction shape:", y_pred.shape)
print("URWL prototype loss:", float(loss))
print("Loss is finite:", bool(tf.math.is_finite(loss)))
