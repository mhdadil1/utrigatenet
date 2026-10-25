
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Dataset and patient paths
patient_path = Path(
    r"C:\Users\GOD\Brats 2019\MICCAI_BraTS_2019_Data_Training"
    r"\HGG\BraTS19_2013_10_1"
)

patient_id = "BraTS19_2013_10_1"
modalities = ["flair", "t1", "t1ce", "t2"]

# Load the four MRI scans
volumes = {}

for modality in modalities:
    path = patient_path / f"{patient_id}_{modality}.nii"
    volumes[modality] = nib.load(str(path)).get_fdata(
        dtype=np.float32
    )

# Load the segmentation mask
seg_path = patient_path / f"{patient_id}_seg.nii"
segmentation = nib.load(str(seg_path)).get_fdata()

# Select a central slice
slice_index = segmentation.shape[2] // 2

# Create a figure
fig, axes = plt.subplots(1, 5, figsize=(18, 4))

for ax, modality in zip(axes[:4], modalities):
    volume = volumes[modality]
    image_slice = volume[:, :, slice_index]

    # Normalize nonzero pixels for display
    nonzero = image_slice[image_slice > 0]

    if nonzero.size:
        low, high = np.percentile(nonzero, [1, 99])
        image_slice = np.clip(
            (image_slice - low) / (high - low + 1e-8),
            0, 1
        )

    ax.imshow(image_slice.T, cmap="gray", origin="lower")
    ax.set_title(modality.upper())
    ax.axis("off")

# Display tumor labels
mask_slice = segmentation[:, :, slice_index].T
axes[4].imshow(volumes["flair"][:, :, slice_index].T,
               cmap="gray", origin="lower")
axes[4].imshow(
    np.ma.masked_where(mask_slice == 0, mask_slice),
    cmap="jet", alpha=0.5, origin="lower",
    vmin=1, vmax=4
)
axes[4].set_title("FLAIR + Tumor Mask")
axes[4].axis("off")

plt.tight_layout()

# Save the figure
output_dir = Path(str(Path(__file__).resolve().parent.parent / "outputs"))
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "patient_visualization.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"Visualization saved to: {output_file}")

plt.show()
