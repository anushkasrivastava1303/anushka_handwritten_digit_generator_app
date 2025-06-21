import streamlit as st
import torch
from vae_mnist import VAE
import matplotlib.pyplot as plt

device = torch.device("cpu")
model = VAE()
model.load_state_dict(torch.load("vae_mnist.pth", map_location=device))
model.eval()

st.title("Handwritten Digit Generator (VAE)")
st.markdown("Note: VAE is unsupervised. Digit selection is for labeling only.")

digit = st.selectbox("Choose digit (0–9):", list(range(10)))

if st.button("Generate 5 Images"):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        z = torch.randn(1, 20)
        with torch.no_grad():
            sample = model.decode(z).reshape(28, 28).numpy()
        axs[i].imshow(sample, cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
