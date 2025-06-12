import streamlit as st
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import io
import math

st.set_page_config(page_title="PCA Image Compressor", layout="wide")
st.title("📸 PCA Image Compression (RGB)")
st.markdown("Upload an image and select the number of components to compress it using PCA.")


def calculate_mse(original, compressed):
    return np.mean((original.astype("float32") - compressed.astype("float32")) ** 2)

def calculate_psnr(mse, max_pixel=255.0):
    return 20 * math.log10(max_pixel / math.sqrt(mse)) if mse != 0 else float("inf")


def apply_pca_rgb(img_array, n_components):
    compressed_channels = []
    for i in range(3):  # R, G, B
        pca = PCA(n_components=n_components)
        channel = img_array[:, :, i]
        transformed = pca.fit_transform(channel)
        reconstructed = pca.inverse_transform(transformed)
        compressed_channels.append(reconstructed)
    reconstructed_img = np.stack(compressed_channels, axis=2)
    return np.clip(reconstructed_img, 0, 255).astype(np.uint8)


uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((300, 300))  
    img_array = np.array(img)

    orig_buf = io.BytesIO()
    img.save(orig_buf, format="PNG")
    orig_bytes = orig_buf.getvalue()
    original_size_kb = len(orig_bytes) / 1024


    st.subheader("🔧 Select Number of PCA Components")
    n_components = st.slider("Components per channel", min_value=1, max_value=150, value=50)

    with st.spinner("Compressing image using PCA..."):
        compressed_img = apply_pca_rgb(img_array, n_components)


        comp_buf = io.BytesIO()
        Image.fromarray(compressed_img).save(comp_buf, format="PNG")
        comp_bytes = comp_buf.getvalue()
        compressed_size_kb = len(comp_bytes) / 1024


        mse = calculate_mse(img_array, compressed_img)
        psnr = calculate_psnr(mse)

    display_size = (250, 250)
    img_display = Image.fromarray(img_array).resize(display_size)
    comp_display = Image.fromarray(compressed_img).resize(display_size)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_display, caption=f"🖼️ Original Image ({original_size_kb:.2f} KB)")
    with col2:
        st.image(comp_display, caption=f"🗜️ Compressed (PCA={n_components}, {compressed_size_kb:.2f} KB)")

    st.markdown("---")
    st.subheader("📊 Compression Metrics")
    st.markdown(f"""
    - 🔢 **Original Size:** {original_size_kb:.2f} KB  
    - 🗜️ **Compressed Size:** {compressed_size_kb:.2f} KB  
    - 📉 **MSE:** {mse:.2f}  
    - 🔍 **PSNR:** {psnr:.2f} dB  
    """)
    st.download_button("⬇️ Download Compressed Image", data=comp_bytes, file_name="compressed_pca.png", mime="image/png")
