# UTriGate-Net: 3D Brain Tumor MRI Segmentation

Interactive 3D multi-modal MRI brain tumor segmentation built with **Triaxial Context Encoding (TCE)**, **Gated Modality Fusion (GMF)**, and **Uncertainty-Regularized Weighted Loss (URWL)**.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud
1. Push the contents of this folder to the root of a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app**, select your repository and branch, and set **Main file path** to `streamlit_app.py`.
3. Click **Deploy!**.
