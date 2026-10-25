import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
import streamlit as st
import tensorflow as tf

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.urwl import urwl_loss
from models.utrigatenet import build_utrigate_net

# Page configuration
st.set_page_config(
    page_title="UTriGate-Net | 3D Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = {
    0: "Background (Class 0)",
    1: "Necrotic / Non-Enhancing Core — NCR/NET (Class 1)",
    2: "Peritumoral Edema — ED (Class 2)",
    3: "GD-Enhancing Tumor — ET (Class 3)",
}

CLASS_COLORS = ["#111827", "#3b82f6", "#10b981", "#f59e0b"]
SEG_CMAP = ListedColormap(CLASS_COLORS)
MODALITIES = ["FLAIR", "T1", "T1CE", "T2"]


@st.cache_resource(show_spinner="Loading UTriGate-Net 3D model and weights...")
def load_segmentation_model(weights_filename: str):
    """Build UTriGate-Net architecture, load trained weights, and build TCE weight extractor."""
    model = build_utrigate_net(input_shape=(32, 32, 32, 4), num_classes=4)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=urwl_loss,
    )
    weights_path = PROJECT_ROOT / "models" / weights_filename
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")
    model.load_weights(str(weights_path))

    # Build auxiliary model to inspect TCE directional branch attention weights
    branch_outputs = [
        model.get_layer(f"{mod}_tce_branch_weights").output
        for mod in ["flair", "t1", "t1ce", "t2"]
    ]
    attention_model = tf.keras.Model(inputs=model.input, outputs=branch_outputs)
    return model, attention_model


@st.cache_data(show_spinner="Loading MRI sample...")
def load_default_sample():
    """Load the pre-packaged 32x32x32x4 BraTS patient sample."""
    sample_path = PROJECT_ROOT / "data" / "processed" / "patient_sample.npz"
    data = np.load(sample_path)
    image = data["image"].astype(np.float32)
    label = data["label"].astype(np.int32) if "label" in data.files else None
    return image, label


def preprocess_nifti_uploads(modality_files: dict, seg_file=None):
    """Normalize 4 NIfTI volumes and extract a 32x32x32 tumor-centered (or volume-centered) patch."""
    images = []
    for mod in ["flair", "t1", "t1ce", "t2"]:
        uploaded = modality_files[mod]
        suffix = ".nii.gz" if uploaded.name.endswith(".nii.gz") else ".nii"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        volume = nib.load(tmp_path).get_fdata(dtype=np.float32)
        Path(tmp_path).unlink(missing_ok=True)

        brain_mask = volume != 0
        normalized = np.zeros_like(volume, dtype=np.float32)
        if np.any(brain_mask):
            values = volume[brain_mask]
            mean = values.mean()
            std = values.std()
            if std > 0:
                normalized[brain_mask] = (values - mean) / std
        images.append(normalized)

    image_4ch = np.stack(images, axis=-1)
    vol_shape = image_4ch.shape[:3]

    label_patch = None
    if seg_file is not None:
        suffix = ".nii.gz" if seg_file.name.endswith(".nii.gz") else ".nii"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(seg_file.getbuffer())
            tmp_path = tmp.name
        seg = nib.load(tmp_path).get_fdata().astype(np.uint8)
        Path(tmp_path).unlink(missing_ok=True)

        labels = np.zeros_like(seg, dtype=np.int32)
        labels[seg == 1] = 1
        labels[seg == 2] = 2
        labels[seg == 4] = 3

        tumor_coords = np.argwhere(seg > 0)
        if len(tumor_coords) > 0:
            center = tumor_coords.mean(axis=0).astype(int)
        else:
            center = np.array(vol_shape) // 2
    else:
        labels = None
        center = np.array(vol_shape) // 2

    patch_size = 32
    half = patch_size // 2
    starts = [
        max(0, min(int(c) - half, dim - patch_size))
        for c, dim in zip(center, vol_shape)
    ]
    slices = tuple(slice(start, start + patch_size) for start in starts)
    image_patch = image_4ch[slices + (slice(None),)]
    if labels is not None:
        label_patch = labels[slices]

    return image_patch.astype(np.float32), label_patch


def compute_dice_scores(label: np.ndarray, prediction: np.ndarray):
    """Calculate per-class Dice similarity score and voxel counts."""
    metrics = []
    for class_id in [1, 2, 3]:
        true_mask = label == class_id
        pred_mask = prediction == class_id
        true_count = int(true_mask.sum())
        pred_count = int(pred_mask.sum())
        denominator = true_count + pred_count

        if denominator == 0:
            dice = float("nan")
        else:
            intersection = int(np.logical_and(true_mask, pred_mask).sum())
            dice = (2.0 * intersection) / denominator

        metrics.append(
            {
                "Class ID": class_id,
                "Subregion": CLASS_NAMES[class_id],
                "Dice Score": dice,
                "Ground Truth Voxels": true_count,
                "Predicted Voxels": pred_count,
            }
        )
    return metrics


def get_plane_slice(volume: np.ndarray, plane: str, idx: int):
    """Extract a 2D slice along Axial, Coronal, or Sagittal plane."""
    if plane == "Axial (Z)":
        return volume[:, :, idx]
    if plane == "Coronal (Y)":
        return volume[:, idx, :]
    return volume[idx, :, :]


def render_modality_strip(image: np.ndarray, plane: str, slice_idx: int):
    """Plot all 4 MRI modalities side-by-side at the chosen slice."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), facecolor="#0e1117")
    for idx, mod_name in enumerate(MODALITIES):
        ax = axes[idx]
        slc = get_plane_slice(image[..., idx], plane, slice_idx)
        ax.imshow(slc.T, cmap="gray", origin="lower")
        ax.set_title(f"{mod_name}", color="white", fontsize=11, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    return fig


def render_segmentation_comparison(
    image: np.ndarray,
    prediction: np.ndarray,
    probabilities: np.ndarray,
    label: np.ndarray | None,
    plane: str,
    slice_idx: int,
    modality_idx: int,
):
    """Plot MRI modality, Ground Truth (if available), Predicted Mask, and Confidence Map."""
    ncols = 4 if label is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(3.6 * ncols, 3.6), facecolor="#0e1117")

    mri_slice = get_plane_slice(image[..., modality_idx], plane, slice_idx)
    pred_slice = get_plane_slice(prediction, plane, slice_idx)
    conf_slice = get_plane_slice(np.max(probabilities, axis=-1), plane, slice_idx)

    # 1. Selected MRI Modality
    axes[0].imshow(mri_slice.T, cmap="gray", origin="lower")
    axes[0].set_title(
        f"{MODALITIES[modality_idx]} MRI", color="white", fontsize=11, fontweight="bold"
    )
    axes[0].axis("off")

    col_ptr = 1
    # 2. Ground Truth (optional)
    if label is not None:
        gt_slice = get_plane_slice(label, plane, slice_idx)
        axes[col_ptr].imshow(mri_slice.T, cmap="gray", origin="lower")
        axes[col_ptr].imshow(
            np.ma.masked_where(gt_slice.T == 0, gt_slice.T),
            cmap=SEG_CMAP,
            vmin=0,
            vmax=3,
            alpha=0.75,
            origin="lower",
        )
        axes[col_ptr].set_title(
            "Ground Truth Overlay", color="white", fontsize=11, fontweight="bold"
        )
        axes[col_ptr].axis("off")
        col_ptr += 1

    # 3. Predicted Segmentation Overlay
    axes[col_ptr].imshow(mri_slice.T, cmap="gray", origin="lower")
    axes[col_ptr].imshow(
        np.ma.masked_where(pred_slice.T == 0, pred_slice.T),
        cmap=SEG_CMAP,
        vmin=0,
        vmax=3,
        alpha=0.75,
        origin="lower",
    )
    axes[col_ptr].set_title(
        "Predicted Overlay", color="white", fontsize=11, fontweight="bold"
    )
    axes[col_ptr].axis("off")
    col_ptr += 1

    # 4. Raw Predicted Class Map
    axes[col_ptr].imshow(pred_slice.T, cmap=SEG_CMAP, vmin=0, vmax=3, origin="lower")
    axes[col_ptr].set_title(
        "Predicted Classes (0–3)", color="white", fontsize=11, fontweight="bold"
    )
    axes[col_ptr].axis("off")

    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(4)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=2,
        frameon=False,
        labelcolor="white",
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.08),
    )
    plt.tight_layout()
    return fig


def main():
    st.title("🧠 UTriGate-Net: 3D Brain Tumor MRI Segmentation")
    st.markdown(
        """
        Interactive 3D multi-modal brain tumor segmentation built with **Triaxial Context Encoding (TCE)**,
        **Gated Modality Fusion (GMF)**, and **Uncertainty-Regularized Weighted Loss (URWL)** on 4-channel
        BraTS MRI volumes (`FLAIR`, `T1`, `T1CE`, `T2`).
        """
    )

    # Sidebar controls
    st.sidebar.header("⚙️ Model & Data Configuration")
    weights_choice = st.sidebar.selectbox(
        "Model Checkpoint",
        options=[
            "utrigate_trained_prototype.keras",
            "utrigate_prototype.keras",
        ],
        index=0,
        help="Select the trained prototype checkpoint or the initial prototype weights.",
    )

    data_source = st.sidebar.radio(
        "Input MRI Volume Source",
        options=[
            "Built-in BraTS Sample (patient_sample.npz)",
            "Upload Preprocessed Patch (.npz)",
            "Upload Raw NIfTI Volumes (.nii / .nii.gz)",
        ],
        index=0,
    )

    image, label = None, None

    if data_source == "Built-in BraTS Sample (patient_sample.npz)":
        try:
            image, label = load_default_sample()
            st.sidebar.success("Loaded `data/processed/patient_sample.npz` (32×32×32×4)")
        except Exception as exc:
            st.error(f"Could not load default sample: {exc}")
            st.stop()

    elif data_source == "Upload Preprocessed Patch (.npz)":
        uploaded_npz = st.sidebar.file_uploader(
            "Upload `.npz` file (`image` key with shape 32×32×32×4)", type=["npz"]
        )
        if uploaded_npz is not None:
            npz_data = np.load(uploaded_npz)
            if "image" not in npz_data.files:
                st.error("Uploaded `.npz` must contain an `'image'` array.")
                st.stop()
            image = npz_data["image"].astype(np.float32)
            label = (
                npz_data["label"].astype(np.int32)
                if "label" in npz_data.files
                else None
            )
            if image.shape != (32, 32, 32, 4):
                st.error(
                    f"Expected `image` shape `(32, 32, 32, 4)`, got `{image.shape}`."
                )
                st.stop()
        else:
            st.info("Please upload a `.npz` file in the sidebar to run inference.")
            st.stop()

    else:
        st.sidebar.markdown("Upload all 4 MRI modalities (`.nii` or `.nii.gz`):")
        mod_uploads = {}
        for mod in ["flair", "t1", "t1ce", "t2"]:
            mod_uploads[mod] = st.sidebar.file_uploader(
                f"{mod.upper()} NIfTI", type=["nii", "gz"], key=f"nii_{mod}"
            )
        seg_upload = st.sidebar.file_uploader(
            "Ground Truth Segmentation (`seg.nii`, optional)",
            type=["nii", "gz"],
            key="nii_seg",
        )
        if all(mod_uploads.values()):
            with st.spinner("Normalizing modalities & extracting 32×32×32 patch..."):
                image, label = preprocess_nifti_uploads(mod_uploads, seg_upload)
        else:
            st.info(
                "Upload all 4 MRI NIfTI files (`FLAIR`, `T1`, `T1CE`, `T2`) in the sidebar to extract a 32×32×32 patch and run segmentation."
            )
            st.stop()

    # Load model and run inference
    try:
        model, attention_model = load_segmentation_model(weights_choice)
    except Exception as exc:
        st.error(f"Failed to load model `{weights_choice}`: {exc}")
        st.stop()

    batch_input = np.expand_dims(image, axis=0)
    probabilities = model.predict(batch_input, verbose=0)[0]
    prediction = np.argmax(probabilities, axis=-1).astype(np.int32)
    branch_weights_list = attention_model.predict(batch_input, verbose=0)

    # Calculate URWL loss & prediction confidence
    mean_confidence = float(np.mean(np.max(probabilities, axis=-1)))
    loss_val = None
    if label is not None:
        loss_tensor = urwl_loss(
            tf.constant(label[np.newaxis, ...], dtype=tf.int32),
            tf.constant(probabilities[np.newaxis, ...], dtype=tf.float32),
        )
        loss_val = float(loss_tensor.numpy())

    # Top KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Input Patch Shape", f"{image.shape[0]}×{image.shape[1]}×{image.shape[2]}×{image.shape[3]}")
    col2.metric("Predicted Classes Present", str(np.unique(prediction).tolist()))
    col3.metric("Mean Voxel Confidence", f"{mean_confidence:.2%}")
    col4.metric(
        "URWL Loss",
        f"{loss_val:.4f}" if loss_val is not None else "N/A (No GT)",
    )

    tab_explore, tab_metrics, tab_architecture, tab_saved = st.tabs(
        [
            "🔍 Interactive 3D Slice Viewer",
            "📊 Dice Scores & Voxel Metrics",
            "🧭 TCE Triaxial Attention & Architecture",
            "🖼️ Saved Reference Outputs",
        ]
    )

    with tab_explore:
        ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.2, 2.0])
        with ctrl1:
            plane = st.selectbox(
                "Anatomical Viewing Plane",
                options=["Axial (Z)", "Coronal (Y)", "Sagittal (X)"],
                index=0,
            )
        with ctrl2:
            modality_name = st.selectbox(
                "Reference MRI Modality", options=MODALITIES, index=0
            )
            modality_idx = MODALITIES.index(modality_name)
        with ctrl3:
            slice_idx = st.slider(
                "Slice Index", min_value=0, max_value=31, value=16, step=1
            )

        st.subheader(f"Segmentation Comparison — {plane} Slice {slice_idx}")
        comp_fig = render_segmentation_comparison(
            image=image,
            prediction=prediction,
            probabilities=probabilities,
            label=label,
            plane=plane,
            slice_idx=slice_idx,
            modality_idx=modality_idx,
        )
        st.pyplot(comp_fig, use_container_width=True)
        plt.close(comp_fig)

        st.subheader(f"All 4 MRI Modalities — {plane} Slice {slice_idx}")
        mod_fig = render_modality_strip(image=image, plane=plane, slice_idx=slice_idx)
        st.pyplot(mod_fig, use_container_width=True)
        plt.close(mod_fig)

    with tab_metrics:
        if label is not None:
            st.subheader("Tumor Subregion Dice Similarity Scores")
            dice_rows = compute_dice_scores(label, prediction)
            valid_dices = [
                r["Dice Score"]
                for r in dice_rows
                if not np.isnan(r["Dice Score"])
            ]
            mean_dice = float(np.mean(valid_dices)) if valid_dices else 0.0

            m_cols = st.columns(4)
            for idx, row in enumerate(dice_rows):
                score_str = (
                    f"{row['Dice Score']:.4f}"
                    if not np.isnan(row["Dice Score"])
                    else "N/A"
                )
                m_cols[idx].metric(
                    f"Class {row['Class ID']} Dice",
                    score_str,
                    help=row["Subregion"],
                )
            m_cols[3].metric("Mean Tumor Dice", f"{mean_dice:.4f}")

            st.table(
                [
                    {
                        "Class ID": r["Class ID"],
                        "Tumor Subregion": r["Subregion"],
                        "Dice Score": (
                            f"{r['Dice Score']:.4f}"
                            if not np.isnan(r["Dice Score"])
                            else "N/A"
                        ),
                        "Ground Truth Voxels": f"{r['Ground Truth Voxels']:,}",
                        "Predicted Voxels": f"{r['Predicted Voxels']:,}",
                    }
                    for r in dice_rows
                ]
            )
        else:
            st.info(
                "Ground-truth segmentation mask not provided for this volume. Showing predicted voxel distribution only."
            )

        st.subheader("Predicted Voxel Distribution Across All 4 Classes")
        total_voxels = int(prediction.size)
        dist_rows = []
        for cid in range(4):
            count = int((prediction == cid).sum())
            dist_rows.append(
                {
                    "Class ID": cid,
                    "Class Name": CLASS_NAMES[cid],
                    "Predicted Voxels": f"{count:,}",
                    "Volume Share": f"{(count / total_voxels):.2%}",
                }
            )
        st.table(dist_rows)

    with tab_architecture:
        st.subheader("Triaxial Context Encoding (TCE) Dynamic Plane Attention")
        st.markdown(
            """
            Each MRI modality passes through a dedicated **TCE block** (`models/tce.py`) with three anisotropic
            depthwise-separable 3D dilated convolution branches (**Axial `(1×3×3)`**, **Coronal `(3×1×3)`**, and
            **Sagittal `(3×3×1)`**). Below are the learned softmax attention weights for the current 3D volume:
            """
        )
        att_rows = []
        for mod_name, w_arr in zip(MODALITIES, branch_weights_list):
            w = w_arr[0]
            att_rows.append(
                {
                    "MRI Modality": mod_name,
                    "Axial Weight": f"{float(w[0]):.4f}",
                    "Coronal Weight": f"{float(w[1]):.4f}",
                    "Sagittal Weight": f"{float(w[2]):.4f}",
                }
            )
        st.table(att_rows)

        st.subheader("UTriGate-Net Pipeline Summary")
        st.markdown(
            """
            1. **Modality Split & Stem (`models/utrigatenet.py`)**: Splits the `(32, 32, 32, 4)` input into `FLAIR`, `T1`, `T1CE`, and `T2` streams and applies 3D stem convolutions (`8` filters).
            2. **Triaxial Context Encoding (`models/tce.py`)**: Extracts directional features along Axial, Coronal, and Sagittal axes with `InstanceNormalization3D` and dynamically weights them via global pooling + 2-layer MLP + Softmax.
            3. **Gated Modality Fusion (`models/gmf.py`)**: Applies learned sigmoid gating masks (`Conv3D` + `LayerNormalization`) to each modality stream before concatenating and projecting to `16` fused channels.
            4. **Compact 3D Encoder–Decoder (`models/utrigatenet.py`)**: Encodes via `Conv3D(16)` + `MaxPool3D(2)` + bottleneck `Conv3D(24)×2`, upsamples with skip connection, and outputs 4-class voxel-wise softmax probabilities.
            5. **Uncertainty-Regularized Weighted Loss (`losses/urwl.py`)**: Combines inverse-frequency class-weighted soft 3D Dice loss with voxel-wise Shannon entropy regularization (`λ = 0.1`).
            """
        )

    with tab_saved:
        st.subheader("Pre-Generated Evaluation Figures (`outputs/`)")
        out_col1, out_col2 = st.columns(2)
        pat_viz = PROJECT_ROOT / "outputs" / "patient_visualization.png"
        pred_viz = PROJECT_ROOT / "outputs" / "prediction_comparison.png"

        with out_col1:
            if pat_viz.exists():
                st.image(
                    str(pat_viz),
                    caption="BraTS 2019 Patient Multi-Modal Visualization (outputs/patient_visualization.png)",
                    use_container_width=True,
                )
            else:
                st.warning("`outputs/patient_visualization.png` not found.")

        with out_col2:
            if pred_viz.exists():
                st.image(
                    str(pred_viz),
                    caption="Central Slice Prediction Comparison (outputs/prediction_comparison.png)",
                    use_container_width=True,
                )
            else:
                st.warning("`outputs/prediction_comparison.png` not found.")


if __name__ == "__main__":
    main()
