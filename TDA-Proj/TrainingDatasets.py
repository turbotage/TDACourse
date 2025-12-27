import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from concentric_triangles_dataset import ConcentricTrianglesDatasetTorch
import numpy as np

def get_cifar10_dataloader(batch_size=64, shuffle_train = True):
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
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def get_mnist_dataloader(batch_size=64, shuffle_train = True):
    """
    Returns a DataLoader for the MNIST dataset.

    Args:
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: DataLoader for the MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def get_concentric_triangles_dataloader(batch_size=64, num_samples=10000, shuffle_train = True):
    """
    Returns a DataLoader for the Concentric Triangles dataset.

    Args:
        batch_size (int): Number of samples per batch.
        num_samples (int): Total number of samples in the dataset.

    Returns:
        DataLoader: DataLoader for the Concentric Triangles dataset.
    """
    # Generate full dataset
    train_dataset = ConcentricTrianglesDatasetTorch(number_of_samples=int(num_samples*0.8))
    test_dataset = ConcentricTrianglesDatasetTorch(number_of_samples=int(num_samples*0.2))

    # dataset_expanded = np.zeros((train_dataset.X.shape[0], 2, 2), dtype=np.float32)
    # dataset_expanded[:, :, 0] = train_dataset.X
    # train_dataset.X = dataset_expanded

    # dataset_expanded = np.zeros((test_dataset.X.shape[0], 2, 2), dtype=np.float32)
    # dataset_expanded[:, :, 0] = test_dataset.X
    # test_dataset.X = dataset_expanded
    
    #convert to float32
    train_dataset.X = train_dataset.X.astype(np.float32)
    test_dataset.X = test_dataset.X.astype(np.float32)
    #print shape
    print(f"Train dataset shape: {train_dataset.X.shape}")
    print(f"Test dataset shape: {test_dataset.X.shape}")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

DATASET_LOADERS = {
    'CIFAR10': (get_cifar10_dataloader,(3,32,32)),
    'MNIST': (get_mnist_dataloader,(1,28,28)),
    'ConcentricTriangles': (get_concentric_triangles_dataloader, (1,2))
}