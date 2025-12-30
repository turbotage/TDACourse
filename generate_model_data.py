"""
General script to generate ModelData for all datasets (MNIST, CIFAR, ConcentricTriangles).
Generates embeddings for: raw data, VAE embeddings, PCA embeddings, and UMAP projections.
Uses config files from TDA-Proj/cfgs/ to determine dataset and model settings.
"""
import torch
import sys
import os
import json
import pickle
import numpy as np
from pathlib import Path
from umap import UMAP
from sklearn.decomposition import PCA

# Add TDA-Proj to path
sys.path.append('./TDA-Proj')

import load_model


class ModelData:
    """Container for all embedding types and labels."""
    def __init__(self):
        self.raw_data = None  # Raw flattened data
        self.vae_embeddings = None  # VAE latent embeddings
        self.pca_embeddings = None  # PCA embeddings
        self.umap_raw = None  # UMAP of raw data
        self.umap_pca = None  # UMAP of PCA embeddings
        self.umap_vae = None  # UMAP of VAE embeddings
        self.labels = None
        self.dataset_name = None
        self.model_name = None


def get_dataset_name_from_config(config):
    """Extract dataset name from config, handling different naming conventions."""
    dataset = config.get('dataset', '').upper()
    if 'MNIST' in dataset:
        return 'MNIST'
    elif 'CIFAR' in dataset:
        return 'CIFAR10'
    elif 'TRIANGLE' in dataset or 'CONCENTRIC' in dataset:
        return 'ConcentricTriangles'
    else:
        raise ValueError(f"Unknown dataset in config: {dataset}")


def get_model_name_from_path(model_path):
    """Extract model name from checkpoint path."""
    return Path(model_path).stem


def get_diverse_batch_from_dataset(dataloader, num_samples, random_seed=42):
    """
    Sample one batch of diverse data from the dataset, balanced across labels.
    
    Args:
        dataloader: PyTorch DataLoader to sample from
        num_samples: Number of diverse samples to return
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (data, labels) where:
            - data: numpy array of shape (num_samples, ...) containing the diverse samples
            - labels: numpy array of shape (num_samples,) containing the labels
    """
    np.random.seed(random_seed)
    
    # Collect data and labels from dataloader
    all_data_list = []
    all_labels_list = []
    
    print(f'  Collecting data from dataset for diverse sampling...')
    for batch in dataloader:
        # Handle different batch formats
        if len(batch) == 2:
            images, batch_labels = batch
        elif len(batch) == 3:
            images, batch_labels, _ = batch  # Skip embeddings if present
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
        
        # Convert to numpy
        images_np = images.cpu().numpy()
        labels_np = batch_labels.cpu().numpy()
        
        all_data_list.append(images_np)
        all_labels_list.append(labels_np)
        
        # Collect enough data to ensure we can get diverse samples
        # Stop when we have at least num_samples * 2 samples to ensure diversity
        total_collected = sum(len(labels) for labels in all_labels_list)
        if total_collected >= num_samples * 2:
            break
    
    # Stack all collected data
    # Use np.concatenate for multi-dimensional arrays (like images)
    all_data = np.concatenate(all_data_list, axis=0)
    all_labels = np.concatenate(all_labels_list)
    
    n_total = all_data.shape[0]
    print(f'    Collected {n_total} samples from dataset')
    
    # Select diverse samples (balanced across labels)
    print(f'    Selecting diverse samples (max {num_samples} samples, balanced labels)...')
    unique_labels = np.unique(all_labels)
    n_classes = len(unique_labels)
    samples_per_class = max(1, num_samples // n_classes)
    max_samples = min(num_samples, n_total)
    
    selected_indices = []
    for label in unique_labels:
        label_indices = np.where(all_labels == label)[0]
        # Select samples_per_class samples from this class
        n_select = min(samples_per_class, len(label_indices))
        if n_select > 0:
            selected = np.random.choice(label_indices, size=n_select, replace=False)
            selected_indices.extend(selected)
    
    # If we haven't reached num_samples, add more samples randomly
    if len(selected_indices) < max_samples:
        remaining_indices = [i for i in range(n_total) if i not in selected_indices]
        n_needed = max_samples - len(selected_indices)
        if len(remaining_indices) > 0:
            additional = np.random.choice(remaining_indices, size=min(n_needed, len(remaining_indices)), replace=False)
            selected_indices.extend(additional)
    
    # Limit to max_samples
    selected_indices = selected_indices[:max_samples]
    selected_indices = np.array(selected_indices)
    
    # Select the diverse samples
    diverse_data = all_data[selected_indices]
    diverse_labels = all_labels[selected_indices]
    
    # Print label distribution
    unique_selected, counts = np.unique(diverse_labels, return_counts=True)
    print(f'    Selected {len(selected_indices)} samples with label distribution:')
    for label, count in zip(unique_selected, counts):
        print(f'      Class {label}: {count} samples')
    
    return diverse_data, diverse_labels


def generate_model_data(config_path, model_path=None, num_samples=2000, use_reparameterized=False):
    """
    Generate ModelData for a given config and model using a diverse batch from the dataset.
    
    Args:
        config_path: Path to config JSON file
        model_path: Path to model checkpoint (.pth file). If None, uses save_path from config.
        num_samples: Number of diverse samples to select from the dataset
        use_reparameterized: If True, use reparameterized samples (z = mu + eps*std) instead of just mu.
                            If False, use only the mean (mu) of the latent distribution.
    
    Returns:
        ModelData object
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dataset_name = get_dataset_name_from_config(config)
    
    # Get model path
    if model_path is None:
        model_path = os.path.join('TDA-Proj', config.get('save_path', ''))
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Cannot generate ModelData.")
    
    print(f'Processing dataset: {dataset_name}')
    print(f'Using model: {model_path}')
    
    # Map dataset name to load_model format
    dataset_map = {
        'MNIST': 'mnist',
        'CIFAR10': 'cifar10',
        'ConcentricTriangles': 'concentrictriangles'
    }
    load_dataset_name = dataset_map[dataset_name]
    
    # Get embedding dimension from config
    embed_dim = config.get('embedding_dim', 32)
    
    # Load model and dataloader
    # Use a larger batch size for the dataloader to ensure we have enough data for diverse sampling
    dataloader_batch_size = max(num_samples, 1000)
    model, train_loader, _, _ = load_model.load_model(
        model_path=model_path,
        batch_size=dataloader_batch_size,
        dataset=load_dataset_name,
        config_path=config_path
    )
    
    device = next(model.parameters()).device
    model.eval()
    
    # Get diverse batch from dataset
    print('  Sampling diverse batch from dataset...')
    diverse_data, diverse_labels = get_diverse_batch_from_dataset(
        dataloader=train_loader,
        num_samples=num_samples,
        random_seed=42
    )
    
    # Convert diverse data to torch tensors
    diverse_data_tensor = torch.tensor(diverse_data, dtype=torch.float32).to(device)
    diverse_labels_tensor = torch.tensor(diverse_labels, dtype=torch.long)
    
    # Generate embeddings from diverse batch
    embedding_method = "reparameterized samples (z = mu + eps*std)" if use_reparameterized else "mean (mu) only"
    print(f'  Generating embeddings from diverse batch (using {embedding_method})...')
    
    with torch.inference_mode():
        # Get VAE embeddings
        mu, logvar = model.encode(diverse_data_tensor)
        if use_reparameterized:
            # Use reparameterized sample: z = mu + eps * std
            z = model.reparameterize(mu, logvar)
            vae_embeddings = z.cpu().numpy()
        else:
            # Use only the mean (mu) of the latent distribution
            vae_embeddings = mu.cpu().numpy()
        
        # Flatten images for raw data
        all_raw_data = diverse_data.reshape(diverse_data.shape[0], -1)
        all_vae_embeddings = vae_embeddings
        all_labels = diverse_labels_tensor
    
    print(f'  Collected {all_raw_data.shape[0]} samples')
    print(f'  VAE embedding shape: {all_vae_embeddings.shape}')
    print(f'  VAE embedding dimension: {embed_dim}')
    
    # Check if VAE embeddings are collapsed (all points very close together)
    vae_std = np.std(all_vae_embeddings, axis=0)
    vae_mean_std = np.mean(vae_std)
    print(f'  VAE embedding std per dimension (mean): {vae_mean_std:.4f}')
    if vae_mean_std < 0.1:
        print(f'  WARNING: VAE embeddings appear collapsed (very low variance)!')
        print(f'  This suggests the model may not have learned meaningful representations.')
        print(f'  Consider checking model training or increasing beta value.')
    
    print('  Second pass: computing PCA and UMAP...')
    
    # Fit PCA on full dataset
    # Use min of embed_dim and number of features (PCA can't extract more components than features)
    n_features = all_raw_data.shape[1]
    pca_n_components = min(embed_dim, n_features)
    pca = PCA(n_components=pca_n_components)
    all_pca_embeddings = pca.fit_transform(all_raw_data)
    
    # Fit UMAP on all selected data
    print(f'    Fitting UMAP on raw data (all {all_raw_data.shape[0]} samples)...')
    umap_raw_transformer = UMAP(n_components=2, random_state=42)
    umap_raw_transformer.fit(all_raw_data)
    umap_raw = umap_raw_transformer.transform(all_raw_data)
    
    # UMAP on PCA embeddings
    print(f'    Computing UMAP on PCA embeddings (all {all_pca_embeddings.shape[0]} samples)...')
    umap_pca_transformer = UMAP(n_components=2, random_state=42)
    umap_pca_transformer.fit(all_pca_embeddings)
    umap_pca_proj = umap_pca_transformer.transform(all_pca_embeddings)
    
    # Handle VAE embeddings based on dimensionality
    if embed_dim == 2:
        # VAE embeddings are already 2D - use directly
        print('    VAE embeddings are 2D - using directly (no UMAP needed)')
        umap_vae_proj = all_vae_embeddings
    elif embed_dim == 3:
        # VAE embeddings are 3D - use first 2 dimensions
        print('    VAE embeddings are 3D - using first 2 dimensions (no UMAP needed)')
        umap_vae_proj = all_vae_embeddings[:, :2]
    else:
        # VAE embeddings are high-dimensional - apply UMAP
        print(f'    Computing UMAP on VAE embeddings (dim={embed_dim}, all {all_vae_embeddings.shape[0]} samples)...')
        umap_vae_transformer = UMAP(n_components=2, random_state=42)
        umap_vae_transformer.fit(all_vae_embeddings)
        umap_vae_proj = umap_vae_transformer.transform(all_vae_embeddings)
    
    # Create ModelData object
    model_data = ModelData()
    model_data.raw_data = torch.tensor(all_raw_data)
    model_data.vae_embeddings = torch.tensor(all_vae_embeddings)
    model_data.pca_embeddings = torch.tensor(all_pca_embeddings)
    model_data.umap_raw = torch.tensor(umap_raw)
    model_data.umap_pca = torch.tensor(umap_pca_proj)
    model_data.umap_vae = torch.tensor(umap_vae_proj)
    model_data.labels = all_labels
    model_data.dataset_name = dataset_name
    model_data.model_name = get_model_name_from_path(model_path)
    
    return model_data


def save_model_data(model_data, output_dir='model_data'):
    """Save ModelData to pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'model_{model_data.model_name}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f'Saved model data to {output_path}')
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ModelData for TDA analysis')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=2000, help='Number of diverse samples to select from dataset')
    parser.add_argument('--use_reparameterized', action='store_true', help='Use reparameterized samples (z = mu + eps*std) instead of just mu')
    parser.add_argument('--output_dir', type=str, default='model_data', help='Output directory for saved data')
    
    args = parser.parse_args()
    
    model_data = generate_model_data(
        config_path=args.config,
        model_path=args.model_path,
        num_samples=args.num_samples,
        use_reparameterized=args.use_reparameterized
    )
    save_model_data(model_data, output_dir=args.output_dir)

