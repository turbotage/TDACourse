import sys
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import argparse
import load_model
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize 3D AE latent space')
    parser.add_argument('model_path', type=str, help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size for visualization (default: 2000)')
    args = parser.parse_args()

    # Load model and train_loader using the same logic as visualize_reconstruction
    model, train_loader, _, _ = load_model.load_model_from_path(args.model_path, args.batch_size)
    device = next(model.parameters()).device

    # Get a batch of images and labels from the train_loader
    images, labels = next(iter(train_loader))
    images = images.to(device)

    # Encode images to get 3D latent representations
    with torch.no_grad():
        mu, _ = model.encode(images)
    mu = mu.cpu().numpy()
    labels = labels.cpu().numpy()

    assert mu.shape[1] == 3, f"Expected latent dimension 3, got {mu.shape[1]}"

    # Define axis permutations and filenames
    permutations = [
        ((0, 1, 2), 'xyz'),
        ((1, 2, 0), 'yzx'),
        ((2, 0, 1), 'zxy'),
    ]

    for axes, name in permutations:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(mu[:, axes[0]], mu[:, axes[1]], mu[:, axes[2]], c=labels, cmap='tab10', s=10)
        ax.set_xlabel(f'z{name[0]}')
        ax.set_ylabel(f'z{name[1]}')
        ax.set_zlabel(f'z{name[2]}')
        fig.colorbar(scatter, ax=ax, label='Label')
        plt.title(f'3D Latent Space Projection ({name})')
        plt.tight_layout()
        filename = f'images/3d_latent/3d_latent_scatter_{name}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved 3D latent scatter plot to {filename}')
