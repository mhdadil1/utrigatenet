
from pathlib import Path
import nibabel as nib

# Dataset location
dataset_path = Path(
    r"C:\Users\GOD\Brats 2019\MICCAI_BraTS_2019_Data_Training"
)

# Select one HGG patient
patient_path = (
    dataset_path / "HGG" / "BraTS19_2013_10_1"
)

modalities = ["flair", "t1", "t1ce", "t2", "seg"]

for modality in modalities:
    file_path = patient_path / (
        f"BraTS19_2013_10_1_{modality}.nii"
    )

    image = nib.load(str(file_path))
    volume = image.get_fdata()

    print(f"\n{modality.upper()}")
    print("Shape:", volume.shape)
    print("Data type:", volume.dtype)
    print("Minimum:", volume.min())
    print("Maximum:", volume.max())


import numpy as np

# Inspect the segmentation labels
seg_path = patient_path / "BraTS19_2013_10_1_seg.nii"
seg_image = nib.load(str(seg_path))
segmentation = seg_image.get_fdata()

labels, counts = np.unique(
    segmentation, return_counts=True
)

print("\nSEGMENTATION LABELS")
for label, count in zip(labels, counts):
    print(f"Label {int(label)}: {count:,} voxels")

print("\nMRI DATA CHECK")
print("All four modalities have the same shape:", all(
    nib.load(
        str(patient_path / f"BraTS19_2013_10_1_{m}.nii")
    ).shape == segmentation.shape
    for m in ["flair", "t1", "t1ce", "t2"]
))
