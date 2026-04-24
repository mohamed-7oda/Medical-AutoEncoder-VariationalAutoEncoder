import os
from models.ae import build_ae
from utils.data_loader import load_dataset
from utils.visualization import show_reconstruction, plot_loss

# CONFIG
DATA_PATH = "/content/medical_mnist"
REGION = "AbdomenCT"   # change easily

def main():
    print(f"Training AE on {REGION}")

    data_dir = os.path.join(DATA_PATH, REGION)
    dataset = load_dataset(data_dir)

    model = build_ae()

    history = model.fit(dataset, epochs=10)

    # Visualization
    plot_loss(history, "AE Loss")
    show_reconstruction(model, dataset)

    # Save model
    model.save_weights(f"../results/ae_{REGION}.weights.h5")

if __name__ == "__main__":
    main()
