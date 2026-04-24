import os
from models.ae import build_ae
from models.vae import build_vae
from utils.data_loader import load_dataset

extract_path = "/content/medical_mnist"
regions = os.listdir(extract_path)

results = {}

for region in regions:
    print(f"\nTraining {region}")

    data_dir = os.path.join(extract_path, region)
    dataset = load_dataset(data_dir)

    ae = build_ae()
    ae.fit(dataset, epochs=3)

    vae, _, _ = build_vae()
    vae.fit(dataset, epochs=3)

    results[region] = {"AE": ae, "VAE": vae}
