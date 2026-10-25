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

print("Available data keys:", data.files)

image = data["image"].astype(np.float32)
label = data["label"].astype(np.int32)

# 2. Add batch dimensions
image = np.expand_dims(image, axis=0)
label = np.expand_dims(label, axis=0)

print("MRI input shape:", image.shape)
print("Segmentation label shape:", label.shape)
print("Labels present:", np.unique(label))

# 3. Build the model
model = build_utrigate_net(
    input_shape=(32, 32, 32, 4),
    num_classes=4
)

# 4. Compile with our prototype loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss=urwl_loss
)

# 5. Run a very small training test
model.fit(
    image,
    label,
    epochs=1,
    batch_size=1
)

# 6. Save the trained prototype
save_path = PROJECT_ROOT / "models" / "utrigate_trained_prototype.keras"
model.save(save_path)

print("Training smoke test completed.")
print("Saved model to:", save_path)
