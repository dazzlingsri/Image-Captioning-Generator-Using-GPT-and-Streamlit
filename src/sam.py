import streamlit as st
import os
import random
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from model import Net
from utils import ConfigS, ConfigL, download_weights

# Function to generate caption from an image
def generate_caption(img, model, temperature):
    with torch.no_grad():
        caption, _ = model(img, temperature)
    return caption

# Streamlit app code
def main():
    st.title("Image Caption Generator")
    st.sidebar.header("Settings")

    # Model size selector
    size_option = st.sidebar.selectbox("Select Model Size", ['S', 'L'])
    config = ConfigL() if size_option == 'L' else ConfigS()

    # Temperature slider
    temperature = st.sidebar.slider("Temperature for Sampling", min_value=0.1, max_value=2.0, value=1.0)

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True, width=300)

        # Load the model
        ckp_path = os.path.join(config.weights_dir, 'model.pt')
        model = Net(
            clip_model=config.clip_model,
            text_model=config.text_model,
            ep_len=config.ep_len,
            num_layers=config.num_layers,
            n_heads=config.n_heads,
            forward_expansion=config.forward_expansion,
            dropout=config.dropout,
            max_len=config.max_len,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        if not os.path.exists(config.weights_dir):
            os.makedirs(config.weights_dir)

        if not os.path.isfile(ckp_path):
            download_weights(ckp_path, size_option)

        checkpoint = torch.load(ckp_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()

        if st.button("Generate Caption"):
            caption = generate_caption(image, model, temperature)
            st.write("Generated Caption:", caption)

if __name__ == "__main__":
    main()
