import pickle
import torch
import matplotlib.pyplot as plt

import numpy as np
from ripser import ripser
from persim import plot_diagrams

from distances import ModelData
from distances import models

def plot_barcodes_unified(dgms, ax=None, max_death=None, colors=None, linewidth=2):
    """
    Unified barcode plot for multiple homology dimensions.
    
    Parameters:
        dgms: list of ndarray diagrams, as returned by ripser()['dgms']
        ax: optional matplotlib axis; if None, creates a new figure
        max_death: optional float to cap infinite intervals
        colors: list of colors per dimension
        linewidth: thickness of bars
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,4))

    if colors is None:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    # Determine a maximum value for infinite deaths
    if max_death is None:
        finite_deaths = []
        for dgm in dgms:
            finite = dgm[np.isfinite(dgm[:,1]), 1]
            if len(finite) > 0:
                finite_deaths.extend(finite)
        max_death = max(finite_deaths) * 1.1 if finite_deaths else 1.0

    y_offset = 0
    for dim, dgm in enumerate(dgms):
        color = colors[dim % len(colors)]
        for birth, death in dgm:
            if not np.isfinite(death):
                death = max_death
            ax.hlines(y_offset, birth, death, colors=color, linewidth=linewidth)
            y_offset += 1

        # Add annotation label for the dimension
        ax.text(max_death * 1.01,
                y_offset - len(dgm) / 2 if len(dgm) > 0 else y_offset,
                f"H{dim}",
                color=color,
                va="center")

        y_offset += 1  # spacing between dimensions

    ax.set_xlabel("Scale parameter")
    ax.set_ylabel("Feature index")
    ax.invert_yaxis()
    ax.set_ylim(y_offset, -1)


def one_model_persistence_diagrams(model_index):
    with open('model_data/model_' + models[model_index] + '.pkl', 'rb') as f:
        model_data = pickle.load(f)

    for j in range(model_data.img_pca.shape[0]):
        if j > 4:
            break
        print(f"    Processing batch {j}...")
        img_pca = model_data.img_pca[j]
        img_vae = model_data.img_vae[j]
        labels = model_data.labels[j]

        diagrams_pca = ripser(img_pca[:500,...].numpy(), maxdim=2)['dgms']
        diagrams_vae = ripser(img_vae[:500,...].numpy(), maxdim=2)['dgms']
        
        if True:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            plot_barcodes_unified(diagrams_pca, ax=axes[0])
            axes[0].set_title(f'PCA')
            plot_barcodes_unified(diagrams_vae, ax=axes[1])
            axes[1].set_title(f'VAE')
            fig.suptitle(f'Persistence Barcodes - {models[model_index]} - Batch {j}')
            plt.savefig(f'images/barcodes/bc_{models[model_index]}_batch{j}.png')
            plt.close()

        if False:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            plot_diagrams(diagrams_pca, show=False, ax=axes[0])
            axes[0].set_title(f'PCA')
            plot_diagrams(diagrams_vae, show=False, ax=axes[1])
            axes[1].set_title(f'VAE')
            fig.suptitle(f'Persistence Diagrams - {models[i]} - Batch {j}')
            plt.savefig(f'images/pers_diags/pd_{models[i]}_batch{j}.png')
            plt.close()



for i in range(len(models)):
    one_model_persistence_diagrams(i)

print('Hello')