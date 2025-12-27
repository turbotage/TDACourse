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
from ProjectDatasets import DATASET_LOADERS
from TrainingDatasets import DATASET_LOADERS as TRAINING_DATASET_LOADERS


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


def load_embeddings_from_npz(embeds_path, max_samples=None):
    """
    Load VAE embeddings from .npz file.
    
    Args:
        embeds_path: Path to .npz file (relative to TDA-Proj/ or absolute)
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        numpy array of embeddings (train set)
    """
    # Try relative to TDA-Proj first, then absolute
    full_path = os.path.join('TDA-Proj', embeds_path)
    if not os.path.exists(full_path):
        full_path = embeds_path
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Embeddings file not found: {embeds_path}")
    
    print(f'    Loading embeddings from: {full_path}')
    embeddings = np.load(full_path)
    
    if 'train' in embeddings:
        train_embeddings = embeddings['train']
    else:
        # If no 'train' key, assume the file contains a single array
        train_embeddings = embeddings[list(embeddings.keys())[0]]
    
    if max_samples is not None and train_embeddings.shape[0] > max_samples:
        train_embeddings = train_embeddings[:max_samples]
    
    return train_embeddings


def generate_model_data(config_path, model_path=None, embeds_path=None, batch_size=2000, max_batches=None, use_reparameterized=False):
    """
    Generate ModelData for a given config and model.
    
    Args:
        config_path: Path to config JSON file
        model_path: Path to model checkpoint (.pth file). If None, uses save_path from config.
        embeds_path: Path to embeddings .npz file. If provided, uses embeddings from file.
                     If None, uses model_path to load model and generate embeddings.
        batch_size: Batch size for data loading
        max_batches: Maximum number of batches to process (None for all)
        use_reparameterized: If True, use reparameterized samples (z = mu + eps*std) instead of just mu.
                            If False, use only the mean (mu) of the latent distribution.
    
    Returns:
        ModelData object
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dataset_name = get_dataset_name_from_config(config)
    
    # Determine whether to use model or embeddings
    # Priority: if embeds_path is explicitly provided, use it; otherwise use model
    model_available = False
    
    if embeds_path is not None:
        # Use embeddings file if explicitly provided
        print(f'Processing dataset: {dataset_name}')
        print(f'Using embeddings from file: {embeds_path}')
        model_available = False
    else:
        # Use model path (load model and generate embeddings)
        if model_path is None:
            model_path = os.path.join('TDA-Proj', config.get('save_path', ''))
        
        if not os.path.exists(model_path):
            # Model not found, try to fall back to embeddings from config
            embeds_path = config.get('embeds_path', '')
            if embeds_path:
                print(f'Processing dataset: {dataset_name}')
                print(f'Model path: {model_path} (not found, using embeddings from config)')
                print(f'Embeddings path: {embeds_path}')
                model_available = False
            else:
                raise FileNotFoundError(f"Model not found at {model_path} and no embeds_path provided or in config. Cannot generate ModelData.")
        else:
            # Model exists, use it
            print(f'Processing dataset: {dataset_name}')
            print(f'Using model: {model_path}')
            model_available = True
    
    # Map dataset name to load_model format
    dataset_map = {
        'MNIST': 'mnist',
        'CIFAR10': 'cifar10',
        'ConcentricTriangles': 'concentrictriangles'
    }
    load_dataset_name = dataset_map[dataset_name]
    
    # Get embedding dimension from config
    embed_dim = config.get('embedding_dim', 32)
    
    # Storage for all embeddings
    raw_data_list = []
    vae_embeddings_list = []
    labels_list = []
    
    if model_available:
        # Load model and generate embeddings
        model, train_loader, _, _ = load_model.load_model(
            model_path=model_path,
            batch_size=batch_size,
            dataset=load_dataset_name,
            config_path=config_path
        )
        
        device = next(model.parameters()).device
        model.eval()
        
        # First pass: collect all data
        embedding_method = "reparameterized samples (z = mu + eps*std)" if use_reparameterized else "mean (mu) only"
        print(f'  First pass: collecting data (using {embedding_method})...')
        with torch.inference_mode():
            for i, batch in enumerate(train_loader):
                if max_batches is not None and i >= max_batches:
                    break
                
                print(f'    Processing batch {i+1}...')
                
                # Handle different batch formats
                if len(batch) == 2:
                    images, batch_labels = batch
                elif len(batch) == 3:
                    images, batch_labels, _ = batch  # Skip embeddings if present
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")
                
                images = images.to(device)
                batch_labels = batch_labels.cpu()
                
                labels_list.append(batch_labels)
                
                # Get VAE embeddings
                mu, logvar = model.encode(images)
                if use_reparameterized:
                    # Use reparameterized sample: z = mu + eps * std
                    z = model.reparameterize(mu, logvar)
                    vae_emb = z.cpu()
                else:
                    # Use only the mean (mu) of the latent distribution
                    vae_emb = mu.cpu()
                vae_embeddings_list.append(vae_emb)
                
                # Flatten images for raw data
                images_flat = images.cpu().numpy().reshape(images.shape[0], -1)
                raw_data_list.append(torch.tensor(images_flat))
        
        # Stack all data
        all_raw_data = torch.cat(raw_data_list, dim=0).numpy()
        all_vae_embeddings = torch.cat(vae_embeddings_list, dim=0).numpy()
        all_labels = torch.cat(labels_list, dim=0)
        
    else:
        # Load embeddings from file and get raw data from dataloader
        print('  Loading embeddings from file...')
        all_vae_embeddings_full = load_embeddings_from_npz(embeds_path)
        
        # Get dataloader for raw data and labels (use TrainingDatasets, not ProjectDatasets)
        data_getter, _ = TRAINING_DATASET_LOADERS[dataset_name]
        train_loader, _ = data_getter(batch_size=batch_size, shuffle_train=False)
        
        # Collect raw data and labels from first batch only
        print('  Collecting raw data and labels (one batch, diverse labels)...')
        all_raw_data_full = []
        all_labels_full = []
        
        # Collect first batch to get labels
        first_batch = next(iter(train_loader))
        images, batch_labels = first_batch
        batch_labels = batch_labels.cpu()
        images_flat = images.cpu().numpy().reshape(images.shape[0], -1)
        all_raw_data_full.append(images_flat)
        all_labels_full.append(batch_labels.numpy())
        
        # Stack first batch
        all_raw_data_full = np.vstack(all_raw_data_full)
        all_labels_full = np.concatenate(all_labels_full)
        
        # Ensure embeddings and raw data have matching number of samples
        n_total = min(all_raw_data_full.shape[0], all_vae_embeddings_full.shape[0])
        if all_raw_data_full.shape[0] != all_vae_embeddings_full.shape[0]:
            print(f'    Warning: Mismatch in sample counts. Using {n_total} samples.')
            all_raw_data_full = all_raw_data_full[:n_total]
            all_vae_embeddings_full = all_vae_embeddings_full[:n_total]
            all_labels_full = all_labels_full[:n_total]
        
        # Select diverse samples (at most one batch, balanced across labels)
        print(f'    Selecting diverse samples (max {batch_size} samples, balanced labels)...')
        np.random.seed(42)  # For reproducibility
        unique_labels = np.unique(all_labels_full)
        n_classes = len(unique_labels)
        samples_per_class = max(1, batch_size // n_classes)
        max_samples = min(batch_size, n_total)
        
        selected_indices = []
        for label in unique_labels:
            label_indices = np.where(all_labels_full == label)[0]
            # Select samples_per_class samples from this class
            n_select = min(samples_per_class, len(label_indices))
            if n_select > 0:
                selected = np.random.choice(label_indices, size=n_select, replace=False)
                selected_indices.extend(selected)
        
        # If we haven't reached batch_size, add more samples randomly
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
        all_raw_data = all_raw_data_full[selected_indices]
        all_vae_embeddings = all_vae_embeddings_full[selected_indices]
        all_labels = torch.tensor(all_labels_full[selected_indices])
        
        # Print label distribution
        unique_selected, counts = np.unique(all_labels.numpy(), return_counts=True)
        print(f'    Selected {len(selected_indices)} samples with label distribution:')
        for label, count in zip(unique_selected, counts):
            print(f'      Class {label}: {count} samples')
    
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
    
    # Fit UMAP on full dataset (or sample if too large)
    umap_sample_size = min(5000, all_raw_data.shape[0])
    np.random.seed(42)  # For reproducibility
    
    # UMAP on raw data
    umap_raw_transformer = UMAP(n_components=2, random_state=42)
    if all_raw_data.shape[0] > umap_sample_size:
        print(f'    Fitting UMAP on raw data (sample of {umap_sample_size} samples)...')
        sample_indices_raw = np.random.choice(all_raw_data.shape[0], umap_sample_size, replace=False)
        umap_raw_transformer.fit(all_raw_data[sample_indices_raw])
    else:
        umap_raw_transformer.fit(all_raw_data)
    umap_raw = umap_raw_transformer.transform(all_raw_data)
    
    # UMAP on PCA embeddings
    print('    Computing UMAP on PCA embeddings...')
    umap_pca_transformer = UMAP(n_components=2, random_state=42)
    if all_pca_embeddings.shape[0] > umap_sample_size:
        sample_indices_pca = np.random.choice(all_pca_embeddings.shape[0], umap_sample_size, replace=False)
        umap_pca_transformer.fit(all_pca_embeddings[sample_indices_pca])
    else:
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
        print(f'    Computing UMAP on VAE embeddings (dim={embed_dim})...')
        umap_vae_transformer = UMAP(n_components=2, random_state=42)
        if all_vae_embeddings.shape[0] > umap_sample_size:
            sample_indices_vae = np.random.choice(all_vae_embeddings.shape[0], umap_sample_size, replace=False)
            umap_vae_transformer.fit(all_vae_embeddings[sample_indices_vae])
        else:
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
    if model_available:
        model_data.model_name = get_model_name_from_path(model_path)
    else:
        # Use embeddings path for model name, or fall back to model path if available
        if embeds_path:
            model_data.model_name = Path(embeds_path).stem
        elif model_path:
            model_data.model_name = get_model_name_from_path(model_path)
        else:
            model_data.model_name = "unknown"
    
    return model_data


def save_model_data(model_data, output_dir='model_data'):
    """Save ModelData to pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'model_{model_data.model_name}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f'Saved model data to {output_path}')
    return output_path


def process_all_configs(cfgs_dir='TDA-Proj/cfgs', batch_size=8000, max_batches=None, use_reparameterized=False):
    """
    Process all config files in the cfgs directory.
    
    Args:
        cfgs_dir: Directory containing config files
        batch_size: Batch size for data loading
        max_batches: Maximum number of batches to process per config
    """
    cfgs_path = Path(cfgs_dir)
    config_files = list(cfgs_path.glob('*.json'))
    
    # Filter to relevant configs (GMM models)
    relevant_configs = [f for f in config_files if 'GMM' in f.name]
    
    print(f'Found {len(relevant_configs)} config files to process')
    
    for config_file in relevant_configs:
        try:
            print(f'\n{"="*60}')
            print(f'Processing config: {config_file.name}')
            print(f'{"="*60}')
            
            model_data = generate_model_data(
                config_path=str(config_file),
                batch_size=batch_size,
                max_batches=max_batches,
                use_reparameterized=use_reparameterized
            )
            
            save_model_data(model_data)
            
        except Exception as e:
            print(f'Error processing {config_file.name}: {e}')
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ModelData for TDA analysis')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--embeds_path', type=str, default=None, help='Path to embeddings .npz file (if provided, uses embeddings instead of model)')
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size for data loading')
    parser.add_argument('--max_batches', type=int, default=None, help='Maximum number of batches to process')
    parser.add_argument('--use_reparameterized', action='store_true', help='Use reparameterized samples (z = mu + eps*std) instead of just mu')
    parser.add_argument('--all', action='store_true', help='Process all configs in TDA-Proj/cfgs')
    parser.add_argument('--output_dir', type=str, default='model_data', help='Output directory for saved data')
    
    args = parser.parse_args()
    
    if args.all:
        process_all_configs(batch_size=args.batch_size, max_batches=args.max_batches, use_reparameterized=args.use_reparameterized)
    elif args.config:
        model_data = generate_model_data(
            config_path=args.config,
            model_path=args.model_path,
            embeds_path=args.embeds_path,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            use_reparameterized=args.use_reparameterized
        )
        save_model_data(model_data, output_dir=args.output_dir)
    else:
        parser.print_help()
        print('\nError: Must specify either --config or --all')

