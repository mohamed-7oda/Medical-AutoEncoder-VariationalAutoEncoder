import os
import numpy as np
from models.vae import build_vae
from utils.data_loader import load_dataset
from utils.visualization import plot_loss
import matplotlib.pyplot as plt

# CONFIG
DATA_PATH = "/content/medical_mnist"
REGION = "AbdomenCT"

def generate_samples(decoder, latent_dim=2, n=10):
    z = np.random.normal(size=(n, latent_dim))
    images = decoder.predict(z)

    plt.figure(figsize=(10, 2))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
    plt.show()

def main():
    print(f"Training VAE on {REGION}")

    data_dir = os.path.join(DATA_PATH, REGION)
    dataset = load_dataset(data_dir)

    vae, encoder, decoder = build_vae()

    history = vae.fit(dataset, epochs=10)

    # Loss
    plot_loss(history, "VAE Loss")

    # Generate samples
    generate_samples(decoder)

    # Save model
    vae.save_weights(f"../results/vae_{REGION}.weights.h5")

if __name__ == "__main__":
    main()
