import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os

import load_model

# Add TDA-Proj to path
sys.path.append('./TDA-Proj')

from Models import GMMVAE_CNN, GMMVAE
from TrainingDatasets import get_concentric_triangles_dataloader

def visualize_concentric_triangles(model_path, num_samples=500, config_path=None):
    """
    Visualize concentric triangles reconstruction by plotting original vs reconstructed 2D points.
    
    Args:
        model_path: Path to the saved model checkpoint (.pth file)
        num_samples: Number of samples to visualize (default: 500)
        config_path: Path to config JSON (optional, will try to infer from model_path)
    """
    
    model, train_loader, test_loader, _ = load_model.load_model(
        model_path, 
        batch_size=num_samples, 
        dataset='concentrictriangles', 
        config_path=config_path
    )

    device = next(model.parameters()).device

    # Get a batch of test data
    data_batch, labels = next(iter(train_loader))
    actual_samples = min(num_samples, len(labels))
    data_batch = data_batch[:actual_samples].to(device)
    labels = labels[:actual_samples]

    # The input has shape (batch, 2, 2) where data is embedded
    # From the loader: dataset_expanded[:, :, 0] = train_dataset.X[:, 0][:,None]
    # This takes the x-coordinate and broadcasts it, so [:, :, 0] contains [x, x]
    # But the original 2D data should have both x and y coordinates
    # The embedding should be: first row is [x, 0], second row is [y, 0]
    # So to extract the original 2D points: [data[0,0], data[1,0]]
    original_embedded = data_batch.cpu().numpy()  # Shape: (batch, 2, 2)
    
    # Extract 2D points - row 0 has x in column 0, row 1 has y in column 0
    original_points = original_embedded[:, :, 0]  # Shape: (batch, 2)

    # Reconstruct
    with torch.no_grad():
        outputs = model(data_batch)
        reconstructed = outputs['recon']  # Shape: (batch, 4) - flattened

    # Reshape reconstructed to (batch, 2, 2) and extract the same way
    reconstructed_points = reconstructed.reshape(num_samples, 2, 2)[:, :, 0].cpu().numpy()  # Shape: (batch, 2, 2)

    # Create color map based on labels
    colors = ['red', 'blue', 'green', 'cyan', 'magenta']
    label_colors = [colors[label % len(colors)] for label in labels]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot original points
    axes[0].scatter(original_points[:, 0], original_points[:, 1], 
                    c=label_colors, alpha=0.6, s=20)
    axes[0].set_title('Original Concentric Triangles')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)

    # Plot reconstructed points
    axes[1].scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], 
                    c=label_colors, alpha=0.6, s=20)
    axes[1].set_title('Reconstructed Points')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('concentric_triangles_reconstruction.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: concentric_triangles_reconstruction.png")
    plt.show()

    # Calculate reconstruction MSE
    mse = np.mean((original_points - reconstructed_points) ** 2)
    print(f"\nReconstruction MSE: {mse:.6f}")
    
    # Per-coordinate MSE
    mse_x = np.mean((original_points[:, 0] - reconstructed_points[:, 0]) ** 2)
    mse_y = np.mean((original_points[:, 1] - reconstructed_points[:, 1]) ** 2)
    print(f"MSE (X-coordinate): {mse_x:.6f}")
    print(f"MSE (Y-coordinate): {mse_y:.6f}")

    # Plot reconstruction error as arrows (overlay on original)
    fig2, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(original_points[:, 0], original_points[:, 1], 
               c=label_colors, alpha=0.6, s=20, label='Original')
    
    # Draw arrows from original to reconstructed (limit for clarity)
    num_arrows = min(100, actual_samples)
    for i in range(num_arrows):
        ax.arrow(original_points[i, 0], original_points[i, 1],
                reconstructed_points[i, 0] - original_points[i, 0],
                reconstructed_points[i, 1] - original_points[i, 1],
                head_width=0.02, head_length=0.03, fc='gray', ec='gray', alpha=0.3)
    
    ax.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], 
               c='black', marker='x', s=20, alpha=0.4, label='Reconstructed')
    
    ax.set_title('Reconstruction Error (arrows from original to reconstructed)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('concentric_triangles_error.png', dpi=150, bbox_inches='tight')
    print("Saved error visualization to: concentric_triangles_error.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize concentric triangles reconstructions')
    parser.add_argument('--model_path', type=str, default='TDA-Proj/ckpts/gmm_cnn_concentric_triangles_0.pth',
                        help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--num_samples', type=int, default=5000, 
                        help='Number of samples to visualize (default: 500)')
    
    args = parser.parse_args()

    # Automatically use config file with same name as model_path, replacing .pth with .json
    if args.model_path.endswith('.pth'):
        config_path = args.model_path[:-4] + '.json'
    else:
        config_path = None

    visualize_concentric_triangles(args.model_path, args.num_samples, config_path)
