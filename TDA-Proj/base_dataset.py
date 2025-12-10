"""
Abstract base class for datasets with raw data only (no embedding).
"""
from abc import ABC, abstractmethod
from sklearn.utils import shuffle  # type: ignore[import]
from sklearn.metrics import pairwise_distances  # type: ignore[import]
RANDOM_STATE = 42


class BaseDataset(ABC):
    """
    Abstract base class for datasets with raw data and distance matrices (no embedding).
    Subclasses must implement `_load_data` and `plot_image`.
    """

    def __init__(self, number_of_samples=100):
        """
        Initialize the dataset and load raw data.
        """
        self.number_of_samples = number_of_samples

        # Load the actual data (implemented by subclasses)
        self._load_data()
        # Shuffle the data
        # self.X, self.y = shuffle(self.X, self.y, random_state=RANDOM_STATE)

    @abstractmethod
    def _load_data(self):
        """
        Load the dataset and set self.X (features) and self.y (labels).
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def plot_image(self, index):
        """
        Visualize a single data point at the given index.
        Must be implemented by subclasses.
        """
        pass

    @property
    def shape(self):
        """Return the shape of the feature matrix."""
        return self.X.shape

    @property
    def labels(self):
        """Return the labels array."""
        return self.y

    @property
    def data(self):
        """Return features and labels as a tuple."""
        return self.X, self.y

    def print_first_10_labels(self):
        """Print the first 10 labels."""
        print(self.y[:10])

    def print_first_10_images(self):
        """Print the first 10 feature vectors."""
        print(self.X[:10])

    def print_first_10_images_and_labels(self):
        """Print the first 10 feature vectors and their labels."""
        print(self.X[:10], self.y[:10])

    def get_distance_features(self):
        """
        Return the feature matrix used for distance computations.
        Raw datasets return `self.X`. Embedded datasets can override this.
        """
        return self.X[:self.number_of_samples]

    def get_distance_matrix(self, distance_metric='euclidean'):
        """
        Compute pairwise distances on the dataset's distance features.
        Raw datasets use X; embedded datasets may override the distance features.
        """
        features = self.get_distance_features()
        return pairwise_distances(features, metric=distance_metric)

    # Raw datasets do not provide an embedding plot

