import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.utrigatenet import build_utrigate_net
from losses.urwl import urwl_loss


# 1. Load the prepared MRI sample
data_path = PROJECT_ROOT / "data" / "processed" / "patient_sample.npz"
data = np.load(data_path)

image = data["image"].astype(np.float32)
label = data["label"].astype(np.int32)

# Add batch dimension
image = np.expand_dims(image, axis=0)

print("MRI input shape:", image.shape)
print("Ground-truth shape:", label.shape)

# 2. Rebuild the same model architecture
model = build_utrigate_net(
    input_shape=(32, 32, 32, 4),
    num_classes=4
)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss=urwl_loss
)

# 3. Load the trained weights
model_path = PROJECT_ROOT / "models" / "utrigate_trained_prototype.keras"
model.load_weights(model_path)
trained_model = model

print("Trained weights loaded successfully!")

# 4. Predict segmentation probabilities
probabilities = trained_model.predict(image, verbose=0)

# Convert probabilities into class labels
prediction = np.argmax(probabilities[0], axis=-1)

# 5. Display prediction information
print("\n--- Prediction Results ---")
print("Ground-truth shape:", label.shape)
print("Prediction shape:", prediction.shape)
print("Ground-truth classes:", np.unique(label))
print("Predicted classes:", np.unique(prediction))

# 6. Calculate Dice score for each tumor class
print("\n--- Dice Scores ---")
for class_id in [1, 2, 3]:
    true_mask = label == class_id
    pred_mask = prediction == class_id
    denominator = true_mask.sum() + pred_mask.sum()

    if denominator == 0:
        dice = float("nan")
    else:
        intersection = np.logical_and(true_mask, pred_mask).sum()
        dice = 2.0 * intersection / denominator

    print(f"Class {class_id} Dice score: {dice:.4f}")

print("\nPrediction test completed.")
