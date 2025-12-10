import torch
import load_model
from umap import UMAP
from sklearn.decomposition import PCA
import json
import pickle

models = [
    "gmm_cnn_mnist_0.pth",
    "gmm_cnn_mnist_1.pth",
    "gmm_cnn_mnist_2.pth",
    "gmm_cnn_mnist_3.pth",
    "gmm_cnn_mnist_4.pth",
    "gmm_cnn_mnist_5.pth",
    "gmm_cnn_mnist_6.pth",
]

distance_types = ["Euclidean", "Cosine"]

class ModelData:
    def __init__(self):
        self.pca_umaps = None
        self.vae_umaps = None
        self.img_pca = None
        self.img_vae = None
        self.labels = None


def run_one_model(model_name):
    umap = UMAP()
    print(f'Processing model: {model_name}')
    model_path = f"TDA-Proj/ckpts/{model_name}"
    config_path = model_path[:-4] + ".json"

    model, train_loader, _, _ = load_model.load_model_from_path(model_path, 2000)
    device = model.parameters().__next__().device

    embed_dim = json.load(open(config_path))['embedding_dim']

    pca = PCA(n_components=embed_dim)

    pca_umaps = []
    vae_umaps = []
    img_pca = []
    img_vae = []

    labels = []

    for i, batch in enumerate(train_loader):
        print(f'  Processing batch {i+1}...')
        images, batch_labels = batch
        images = images.to(device)

        labels.append(batch_labels)

        mu, _ = model.encode(images)

        temp_img_pca = pca.fit_transform(
                            images.cpu().numpy().reshape(images.shape[0], -1)
                        )
        pca_umaps.append(
            torch.tensor(umap.fit_transform(temp_img_pca))
        )
        vae_umaps.append(
            torch.tensor(umap.fit_transform(mu.cpu().numpy()))
        )
        img_pca.append(
            torch.tensor(
                temp_img_pca
            )
        )
        img_vae.append(mu.cpu())

        if i > 20:
            break

    model_data = ModelData()
    model_data.pca_umaps = torch.stack(pca_umaps, dim=0)
    model_data.vae_umaps = torch.stack(vae_umaps, dim=0)
    model_data.img_pca = torch.stack(img_pca, dim=0)
    model_data.img_vae = torch.stack(img_vae, dim=0)
    model_data.labels = torch.stack(labels, dim=0)

    with open(f'model_data/model_{model_name}.pkl', 'wb') as f:
        pickle.dump(model_data, f)


def run_distance_calculations():
    for model_name in models:
        run_one_model(model_name)


if __name__ == "__main__":
    with torch.inference_mode():
        #run_distance_calculations()
        run_one_model(models[-1])
