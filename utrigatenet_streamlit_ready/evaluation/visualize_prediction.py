import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.utrigatenet import build_utrigate_net
from losses.urwl import urwl_loss

# Load MRI sample and labels
data_path = PROJECT_ROOT / "data" / "processed" / "patient_sample.npz"
data = np.load(data_path)

image = data["image"].astype(np.float32)
label = data["label"].astype(np.int32)

# Rebuild model and load trained weights
model = build_utrigate_net(
    input_shape=(32, 32, 32, 4),
    num_classes=4
)

model.compile(optimizer="adam", loss=urwl_loss)

model_path = PROJECT_ROOT / "models" / "utrigate_trained_prototype.keras"
model.load_weights(model_path)

# Predict segmentation
probabilities = model.predict(image[np.newaxis, ...], verbose=0)
prediction = np.argmax(probabilities[0], axis=-1)

# Choose the central axial slice
slice_index = image.shape[2] // 2

# Display FLAIR, actual mask and predicted mask
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image[:, :, slice_index, 0], cmap="gray")
axes[0].set_title("FLAIR MRI")

axes[1].imshow(label[:, :, slice_index], vmin=0, vmax=3)
axes[1].set_title("Ground Truth")

axes[2].imshow(prediction[:, :, slice_index], vmin=0, vmax=3)
axes[2].set_title("Predicted Segmentation")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "prediction_comparison.png"

plt.savefig(output_path, dpi=150)
print("Comparison saved to:", output_path)
