import pickle
import torch
import matplotlib.pyplot as plt

from distances import ModelData
from distances import models

def plot_two_umaps(X1, labels1, X2, labels2, title1='UMap 1', title2='UMap 2', modelname='', extra_filepath=''):
    X1 = X1.cpu().numpy() if hasattr(X1, 'cpu') else X1
    labels1 = labels1.cpu().numpy() if hasattr(labels1, 'cpu') else labels1
    X2 = X2.cpu().numpy() if hasattr(X2, 'cpu') else X2
    labels2 = labels2.cpu().numpy() if hasattr(labels2, 'cpu') else labels2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc1 = axes[0].scatter(X1[:, 0], X1[:, 1], c=labels1, cmap='tab10', s=10)
    axes[0].set_title(title1 + f' - {modelname}')
    plt.colorbar(sc1, ax=axes[0])

    sc2 = axes[1].scatter(X2[:, 0], X2[:, 1], c=labels2, cmap='tab10', s=10)
    axes[1].set_title(title2 + f' - {modelname}')
    plt.colorbar(sc2, ax=axes[1])

    plt.tight_layout()
    filename = f'images/umaps/' + title1 + title2 + f'_{modelname}' + f'_{extra_filepath}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def one_model_umap_plots(model_index):
    with open('model_data/model_' + models[model_index] + '.pkl', 'rb') as f:
        model_data = pickle.load(f)

    for j in range(model_data.pca_umaps.shape[0]):
        plot_two_umaps(
            model_data.pca_umaps[j,...], 
            model_data.labels[j,...],
            model_data.vae_umaps[j,...], 
            model_data.labels[j,...],
            title1=f'PCA UMap',
            title2=f'VAE UMap',
            modelname=models[model_index],
            extra_filepath=f'batch{j+1}'
        )

    print(f"Model {models[model_index]}:")

for i in range(len(models)):
    one_model_umap_plots(i)

print('Hello')