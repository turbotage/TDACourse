import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import argparse
import json
import os

import load_model

# Add TDA-Proj to path
sys.path.append('./TDA-Proj')

from Models import GMMVAE_CNN, GMMVAE
from TrainingDatasets import get_cifar10_dataloader, get_mnist_dataloader

def denormalize_cifar10(tensor):
    """Denormalize CIFAR10 images to [0, 1] range for display."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    return tensor * std + mean

def denormalize_mnist(tensor):
    """Denormalize MNIST images to [0, 1] range for display."""
    mean = torch.tensor([0.1307]).view(1, 1, 1)
    std = torch.tensor([0.3081]).view(1, 1, 1)
    return tensor * std + mean

def visualize_reconstruction(model_path, num_images=5, dataset='cifar10', config_path=None):

    model, train_loader, test_loader, is_color = load_model.load_model(
                            model_path, 
                            batch_size=num_images, 
                            dataset=dataset, 
                            config_path=config_path
                            )

    device = model.parameters().__next__().device

    # Always get a batch of test images after test_loader is created
    images, labels = next(iter(test_loader))
    images = images[:num_images]
    labels = labels[:num_images]
    images = images.to(device)

    # Reconstruct images (call model ONCE)
    with torch.no_grad():
        outputs = model(images)
        reconstructed = outputs['recon']

    # Move to CPU and clamp
    images = torch.clamp(images.cpu(), 0, 1)
    reconstructed = torch.clamp(reconstructed.cpu(), 0, 1)

    # Plot the results
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 3, 6))
    C, H, W = images.shape[1:]
    for i in range(num_images):
        # Original image
        if is_color:
            img = images[i].permute(1, 2, 0).numpy()
        else:
            img = images[i].squeeze().numpy()
        axes[0, i].imshow(img, cmap='gray' if not is_color else None)
        axes[0, i].set_title(f'Original (Label: {labels[i].item()})')
        axes[0, i].axis('off')
        # Reconstructed image
        recon_img = reconstructed[i].reshape(C, H, W)
        if is_color:
            recon = recon_img.permute(1, 2, 0).numpy()
        else:
            recon = recon_img.squeeze().numpy()
        axes[1, i].imshow(recon, cmap='gray' if not is_color else None)
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig('images/reconstruction/reconstruction_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: images/reconstruction/reconstruction_comparison.png")

    # Reshape reconstructed to (B, C, H, W) for MSE calculation
    reconstructed_img = reconstructed.reshape(images.shape)
    mse_per_image = torch.mean((images - reconstructed_img) ** 2, dim=[1, 2, 3])
    print("\nMSE per image:")
    for i, mse in enumerate(mse_per_image):
        print(f"  Image {i+1}: {mse.item():.4f}")
    print(f"Average MSE: {mse_per_image.mean().item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize model reconstructions')
    parser.add_argument('model_path', type=str, help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to visualize (default: 5)')
    parser.add_argument('--dataset', type=str, default=None, choices=['cifar10', 'mnist'],
                        help='Dataset to use (if not provided, inferred from model_path)')

    args = parser.parse_args()

    # Infer dataset from model_path if not provided
    dataset = args.dataset
    if dataset is None:
        model_path_lower = args.model_path.lower()
        if 'mnist' in model_path_lower:
            dataset = 'mnist'
        elif 'cifar' in model_path_lower:
            dataset = 'cifar10'
        else:
            raise ValueError("Could not infer dataset from model_path. Please specify --dataset explicitly.")


    # Automatically use config file with same name as model_path, replacing .pth with .json
    if args.model_path.endswith('.pth'):
        config_path = args.model_path[:-4] + '.json'
    else:
        config_path = None

    visualize_reconstruction(args.model_path, args.num_images, dataset, config_path)

    