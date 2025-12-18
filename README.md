# TDA Course Project: Topological Data Analysis of GMM-VAE Latent Spaces

## Overview

This project implements and analyzes **Gaussian Mixture Model Variational Autoencoders (GMM-VAE)** with a focus on applying **Topological Data Analysis (TDA)** to understand the structure of learned latent representations. The project includes:

- **Model Training**: GMM-VAE models with CNN and MLP architectures
- **Dataset Support**: MNIST, CIFAR-10, and synthetic concentric triangles
- **TDA Analysis**: Persistence diagrams, barcodes, and topological features
- **Visualization**: UMAP projections, 3D latent space plots, and reconstruction comparisons

## Project Structure

```
TDACourse/
├── TDA-Proj/                    # Core project code
│   ├── Models.py                # GMM-VAE model implementations
│   ├── Training.py              # Training script
│   ├── TrainingDatasets.py      # Dataset loaders
│   ├── ProjectDatasets.py       # Dataset wrappers with embeddings
│   ├── GenerateEmbeddings.py    # Embedding generation utilities
│   ├── base_dataset.py          # Base dataset class
│   ├── concentric_triangles_dataset.py  # Synthetic dataset
│   ├── preamble.py             # Utility functions (optimizers, loss, seeds)
│   ├── count_params.py          # Model parameter counting
│   ├── cfgs/                    # Model configuration files
│   │   ├── GMMCNNMNIST.json
│   │   ├── GMMCNNCifar.json
│   │   ├── GMMCifar.json
│   │   └── GMMConcentricTriangles.json
│   ├── embeds/                  # Precomputed embeddings
│   │   ├── gmm_cnn_cifar.npz
│   │   └── gmm_cnn_mnist.npz
│   └── ckpts/                   # Model checkpoints (not in repo)
│
├── model_data/                  # Processed model data (pickled)
│   └── model_gmm_cnn_mnist_*.pth.pkl
│
├── images/                      # Generated visualizations
│   ├── 3d_latent/              # 3D latent space scatter plots
│   ├── barcodes/               # Persistence barcode diagrams
│   ├── pers_diags/             # Persistence diagrams
│   ├── reconstruction/         # Reconstruction comparisons
│   ├── triangles/              # Concentric triangles visualizations
│   └── umaps/                  # UMAP projection plots
│
├── load_model.py               # Model loading utilities
├── distances.py                # Distance computation and model data processing
├── tda_analysis_psdiag.py      # Persistence diagram analysis
├── tda_analysis_umap_plots.py  # UMAP visualization generation
├── visualize_reconstruction.py # Reconstruction visualization
├── visualize_3d_ae.py          # 3D latent space visualization
├── visualize_concentric_triangles.py  # Triangle dataset visualization
├── dash_plotting.py            # Interactive Dash visualizations
└── small_tests.py              # Testing utilities
```

## Installation

### Requirements

```bash
pip install torch torchvision
pip install numpy matplotlib
pip install scikit-learn umap-learn
pip install ripser persim
pip install tqdm
pip install dash  # For interactive visualizations (optional)
```

### Setup

1. Clone or navigate to the project directory
2. Ensure the `TDA-Proj` directory is in your Python path (scripts handle this automatically)
3. Download datasets (MNIST and CIFAR-10 will be downloaded automatically on first use)

## Usage

### Training Models

Train a GMM-VAE model using a configuration file:

```bash
cd TDA-Proj
python Training.py --config cfgs/GMMCNNMNIST.json
```

Configuration files specify:
- Model architecture (CNN or MLP)
- Dataset (MNIST, CIFAR-10, or concentric triangles)
- Hyperparameters (embedding dimension, number of classes, etc.)
- Training parameters (epochs, batch size, optimizer, etc.)

### Generating Embeddings

Generate embeddings from trained models:

```bash
cd TDA-Proj
python GenerateEmbeddings.py --config cfgs/GMMCNNMNIST.json --model_path ckpts/model.pth
```

### Visualizing Reconstructions

Compare original and reconstructed images:

```bash
python visualize_reconstruction.py model_data/model_gmm_cnn_mnist_0.pth --num_images 8 --dataset mnist
```

### 3D Latent Space Visualization

Visualize latent representations in 3D:

```bash
python visualize_3d_ae.py
```

### TDA Analysis

#### Persistence Diagrams and Barcodes

```bash
python tda_analysis_psdiag.py
```

This generates persistence diagrams and barcode plots for different homology dimensions.

#### UMAP Visualizations

```bash
python tda_analysis_umap_plots.py
```

Generates UMAP projections comparing PCA and VAE latent spaces.

### Loading Models Programmatically

```python
import load_model

model, train_loader, test_loader, is_color = load_model.load_model(
    model_path='model_data/model_gmm_cnn_mnist_0.pth',
    batch_size=64,
    dataset='mnist'
)
```

## Key Components

### Models

**GMMVAE_CNN**: CNN-based GMM-VAE for image datasets (MNIST, CIFAR-10)
- Encoder: Convolutional layers with residual blocks
- Decoder: Transposed convolutions
- Prior: Learnable Gaussian Mixture Model in latent space

**GMMVAE**: MLP-based GMM-VAE for simpler datasets
- Encoder/Decoder: Fully connected layers
- Suitable for low-dimensional synthetic data

### Datasets

- **MNIST**: 28×28 grayscale digit images (10 classes)
- **CIFAR-10**: 32×32 color images (10 classes)
- **Concentric Triangles**: Synthetic 2D dataset with topological structure

### TDA Tools

- **Ripser**: Computes persistence diagrams from point clouds
- **Persistence Diagrams**: Visualize topological features (holes, voids, etc.)
- **Barcodes**: Alternative visualization of persistence intervals
- **UMAP**: Dimensionality reduction for visualization

## Configuration Files

Configuration files in `TDA-Proj/cfgs/` specify:

```json
{
  "model": "GMMVAE_CNN",
  "dataset": "mnist",
  "embedding_dim": 32,
  "num_classes": 10,
  "conv_channels": [32, 64, 128],
  "conv_strides": [2, 2, 2],
  "epochs": 50,
  "batch_size": 128,
  "optimizer": {"name": "Adam", "lr": 0.001},
  "criterion": "mse"
}
```

## Output Files

### Model Checkpoints
- Saved as `.pth` files in `TDA-Proj/ckpts/`
- Associated JSON config files with same name (`.json`)

### Processed Data
- Pickled model data in `model_data/` containing:
  - PCA and VAE embeddings
  - UMAP projections
  - Labels and metadata

### Visualizations
- **Reconstructions**: `images/reconstruction/reconstruction_comparison_*.png`
- **3D Latent Spaces**: `images/3d_latent/3d_latent_scatter_*.png`
- **Barcodes**: `images/barcodes/bc_*.png`
- **Persistence Diagrams**: `images/pers_diags/*.png`
- **UMAP Plots**: `images/umaps/*.png`

## Key Scripts

- **`Training.py`**: Main training script with epoch management, loss computation, and checkpointing
- **`load_model.py`**: Utilities for loading trained models and setting up data loaders
- **`distances.py`**: Computes distances and processes model data for TDA analysis
- **`tda_analysis_psdiag.py`**: Generates persistence diagrams and barcodes
- **`tda_analysis_umap_plots.py`**: Creates UMAP visualizations comparing PCA vs VAE spaces

## Notes

- To get embeddings, use the dataset getters in `ProjectDatasets.py`
- Models are run for MNIST and CIFAR-10; concentric triangles models are experimental
- Random seeds are fixed (seed=69) for reproducibility
- The project uses beta-VAE formulation with configurable KL weight

## License

[Add license information if applicable]

## Authors

[Add author information]

