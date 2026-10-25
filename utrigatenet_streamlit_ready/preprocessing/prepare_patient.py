
from pathlib import Path

import nibabel as nib
import numpy as np

# 1. Patient location
patient_id = "BraTS19_2013_10_1"

patient_path = (
    Path(r"C:\Users\GOD\Brats 2019\MICCAI_BraTS_2019_Data_Training")
    / "HGG"
    / patient_id
)

modalities = ["flair", "t1", "t1ce", "t2"]

# 2. Load and normalize each MRI modality
images = []

for modality in modalities:
    file_path = patient_path / f"{patient_id}_{modality}.nii"
    volume = nib.load(str(file_path)).get_fdata(
        dtype=np.float32
    )

    # Normalize only non-background voxels
    brain_mask = volume != 0
    normalized = np.zeros_like(volume)

    if np.any(brain_mask):
        values = volume[brain_mask]
        mean = values.mean()
        std = values.std()

        if std > 0:
            normalized[brain_mask] = (
                values - mean
            ) / std

    images.append(normalized)

# Shape: (240, 240, 155, 4)
image_4ch = np.stack(images, axis=-1)

# 3. Load segmentation mask
seg_path = patient_path / f"{patient_id}_seg.nii"
seg = nib.load(str(seg_path)).get_fdata().astype(np.uint8)

# Map original labels 0, 1, 2, 4 to classes 0, 1, 2, 3
# Class 0 = background; classes 1–3 = tumor subregions
labels = np.zeros_like(seg, dtype=np.uint8)
labels[seg == 1] = 1
labels[seg == 2] = 2
labels[seg == 4] = 3

# 4. Find the center of the annotated tumor
tumor_coords = np.argwhere(seg > 0)

if len(tumor_coords) == 0:
    raise ValueError("No tumor labels found in this patient.")

center = tumor_coords.mean(axis=0).astype(int)

# 5. Extract a 32 x 32 x 32 patch
patch_size = 32
half = patch_size // 2

starts = [
    max(0, min(int(c) - half, dim - patch_size))
    for c, dim in zip(center, seg.shape)
]

slices = tuple(
    slice(start, start + patch_size)
    for start in starts
)

image_patch = image_4ch[slices + (slice(None),)]
label_patch = labels[slices]

# 6. Save prepared sample
output_dir = Path(str(Path(__file__).resolve().parent.parent / "data" / "processed"))
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "patient_sample.npz"

np.savez_compressed(
    output_file,
    image=image_patch.astype(np.float32),
    label=label_patch.astype(np.uint8),
)

print("Preprocessing completed!")
print("Image patch shape:", image_patch.shape)
print("Label patch shape:", label_patch.shape)
print("MRI channels:", modalities)
print("Patch labels:", np.unique(label_patch))
print("Saved to:", output_file)
