import matplotlib.pyplot as plt
import numpy as np

def plot_loss(history, title):
    plt.plot(history.history['loss'])
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def show_reconstruction(model, dataset):
    for batch in dataset.take(1):
        x, _ = batch
        recon = model.predict(x)

        plt.figure(figsize=(10, 4))
        for i in range(5):
            plt.subplot(2, 5, i+1)
            plt.imshow(x[i], cmap='gray')
            plt.axis('off')

            plt.subplot(2, 5, i+6)
            plt.imshow(recon[i], cmap='gray')
            plt.axis('off')
        plt.show()
