import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from concentric_triangles_dataset import ConcentricTrianglesDatasetTorch
import numpy as np

class DatasetPlusEmbeddings(Dataset):
    """
    A dataset wrapper that adds precomputed embeddings to each data sample.

    Args:
        base_dataset (torch.utils.data.Dataset): The original dataset.
        embeddings (torch.Tensor): Precomputed embeddings corresponding to each data sample.
    """
    def __init__(self, base_dataset, embeddings):
        self.base_dataset = base_dataset
        self.embeddings = embeddings

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data, label = self.base_dataset[idx]
        embedding = self.embeddings[idx]
        return data, label, embedding

def get_cifar10_dataloader(batch_size=64, shuffle_train = True, embedding_path = 'embeds/gmm_cnn_cifar.npz'):
    """
    Returns a DataLoader for the CIFAR-10 dataset.

    Args:
        batch_size (int): Number of samples per batch.
        train (bool): If True, creates dataset from training set, otherwise from test set.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: DataLoader for the CIFAR-10 dataset.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    embeddings = np.load(embedding_path)
    train_embeds = torch.from_numpy(embeddings['train']).float()
    test_embeds = torch.from_numpy(embeddings['test']).float()
    train_dataset = DatasetPlusEmbeddings(datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform), train_embeds)
    test_dataset = DatasetPlusEmbeddings(datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform), test_embeds)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def get_mnist_dataloader(batch_size=64, shuffle_train = True, embedding_path = 'embeds/gmm_cnn_mnist.npz'):
    """
    Returns a DataLoader for the MNIST dataset.

    Args:
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: DataLoader for the MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    embeddings = np.load(embedding_path)
    train_embeds = torch.from_numpy(embeddings['train']).float()
    test_embeds = torch.from_numpy(embeddings['test']).float()
    train_dataset = DatasetPlusEmbeddings(datasets.MNIST(root='data', train=True, download=True, transform=transform), train_embeds)
    test_dataset = DatasetPlusEmbeddings(datasets.MNIST(root='data', train=False, download=True, transform=transform), test_embeds)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def get_concentric_triangles_dataloader(batch_size=64, num_samples=1000, shuffle_train = True):
    """
    Returns a DataLoader for the Concentric Triangles dataset.

    Args:
        batch_size (int): Number of samples per batch.
        num_samples (int): Total number of samples in the dataset.

    Returns:
        DataLoader: DataLoader for the Concentric Triangles dataset.
    """
    dataset = ConcentricTrianglesDatasetTorch(num_samples=num_samples)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8 * num_samples), num_samples - int(0.8 * num_samples)])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

DATASET_LOADERS = {
    'CIFAR10': (get_cifar10_dataloader,(3,32,32)),
    'MNIST': (get_mnist_dataloader,(1,28,28)),
    'ConcentricTriangles': (get_concentric_triangles_dataloader, (1,2,1))
}