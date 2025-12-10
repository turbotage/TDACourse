import numpy as np
import pandas as pd  # type: ignore[import]
import matplotlib.pyplot as plt
from base_dataset import BaseDataset  # type: ignore[import]
from torch.utils.data import Dataset
RANDOM_STATE = 42

class ConcentricTrianglesDataset(BaseDataset):

    def __init__(self, number_of_samples=100,
                 n_triangles=5, points_per_triangle=200, sizes=None, noise=0.0, rotation=0.0, random_state=None):
        """
        Initialize the Concentric Triangles dataset.

        Parameters:
        - number_of_samples: Number of samples to use for embeddings
        - n_triangles: Number of concentric triangles to generate
        - points_per_triangle: Number of points per triangle
        - sizes: List/array of sizes (side lengths) for triangles
        - noise: Standard deviation of Gaussian noise
        - rotation: Rotation angle in radians (default 0.0)
        - random_state: Random state for reproducibility (defaults to RANDOM_STATE from config)
        """
        # Store generation parameters
        self.n_triangles = n_triangles
        self.points_per_triangle = points_per_triangle
        self.sizes = sizes
        self.noise = noise
        self.rotation = rotation
        self.random_state = random_state if random_state is not None else RANDOM_STATE

        # Call parent constructor which will call _load_data
        super().__init__(number_of_samples)

    def _load_data(self):
        """Generate fresh concentric triangles data."""
        self.df = self.make_concentric_triangles(
            n_triangles=self.n_triangles,
            points_per_triangle=self.points_per_triangle,
            sizes=self.sizes,
            noise=self.noise,
            rotation=self.rotation,
            random_state=self.random_state
        )

        # Extract X (features) and y (labels)
        self.X = self.df[['x', 'y']].values
        self.y = self.df['label'].values

    def make_concentric_triangles(self, n_triangles=3, points_per_triangle=200, sizes=None, noise=0.05, rotation=0.0, random_state=None):
        """
        Generate 2D points lying on concentric equilateral triangles with optional Gaussian noise.
        Returns a pandas DataFrame with columns: x, y, label, size, position.

        Parameters:
        - n_triangles: number of concentric triangles
        - points_per_triangle: number of points per triangle (int or list of ints for per-triangle counts)
        - sizes: list/array of sizes (side lengths). If None, sizes will be [1,2,...,n_triangles]
        - noise: standard deviation of Gaussian noise added to (x,y)
        - rotation: rotation angle in radians for all triangles
        - random_state: int or None for reproducibility
        """
        rng = np.random.default_rng(random_state)

        # allow points_per_triangle to be either int or list
        if isinstance(points_per_triangle, int):
            counts = [points_per_triangle] * n_triangles
        else:
            counts = list(points_per_triangle)
            if len(counts) != n_triangles:
                raise ValueError("points_per_triangle list must match n_triangles length")

        if sizes is None:
            sizes = np.arange(1, n_triangles+1)
        else:
            sizes = np.array(sizes)
            if sizes.shape[0] != n_triangles:
                raise ValueError("sizes length must equal n_triangles")

        rows = []
        for i, (size, cnt) in enumerate(zip(sizes, counts)):
            # Generate points along the triangle perimeter
            points = self._generate_triangle_points(size, cnt, rotation, rng)

            # Add noise
            x = points[:, 0] + rng.normal(0, noise, size=cnt)
            y = points[:, 1] + rng.normal(0, noise, size=cnt)

            label = np.full(cnt, i, dtype=int)
            size_col = np.full(cnt, size, dtype=float)
            positions = np.linspace(0, 1, cnt)  # Position along perimeter (0 to 1)

            for xi, yi, lab, sz, pos in zip(x, y, label, size_col, positions):
                rows.append((xi, yi, int(lab), float(sz), float(pos)))

        df = pd.DataFrame(rows, columns=["x", "y", "label", "size", "position"])

        return df

    def _generate_triangle_points(self, size, n_points, rotation, rng):
        """
        Generate n_points uniformly distributed along the perimeter of an equilateral triangle.

        Parameters:
        - size: side length of the triangle
        - n_points: number of points to generate
        - rotation: rotation angle in radians
        - rng: numpy random generator

        Returns:
        - Array of shape (n_points, 2) with (x, y) coordinates
        """
        # Vertices of an equilateral triangle centered at origin
        # The triangle is oriented with one vertex pointing up
        height = size * np.sqrt(3) / 2
        vertices = np.array([
            [0, 2*height/3],                    # Top vertex
            [-size/2, -height/3],               # Bottom left vertex
            [size/2, -height/3]                 # Bottom right vertex
        ])

        # Apply rotation
        if rotation != 0:
            cos_rot = np.cos(rotation)
            sin_rot = np.sin(rotation)
            rotation_matrix = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])
            vertices = vertices @ rotation_matrix.T

        # Calculate perimeter length
        perimeter = 3 * size

        # Generate random positions along the perimeter
        positions = np.sort(rng.uniform(0, perimeter, size=n_points))

        points = []
        for pos in positions:
            if pos < size:
                # First edge: from vertex 0 to vertex 1
                t = pos / size
                point = vertices[0] * (1 - t) + vertices[1] * t
            elif pos < 2 * size:
                # Second edge: from vertex 1 to vertex 2
                t = (pos - size) / size
                point = vertices[1] * (1 - t) + vertices[2] * t
            else:
                # Third edge: from vertex 2 to vertex 0
                t = (pos - 2 * size) / size
                point = vertices[2] * (1 - t) + vertices[0] * t

            points.append(point)

        return np.array(points)

    def display_dataframe(self, df):
        print(df.head(12).to_string(index=False))

    def plot_concentric(self, figsize=(6,6), marker='o', s=20, alpha=0.8):
        """
        Scatter-plot the concentric triangles dataset. Each label plotted with a separate point series.
        """
        labels = sorted(self.df['label'].unique())
        plt.figure(figsize=figsize)
        for lab in labels:
            sub = self.df[self.df['label'] == lab]
            plt.scatter(sub['x'], sub['y'], label=f"label={lab}, size={sub['size'].iloc[0]:.2f}", marker=marker, s=s, alpha=alpha)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Concentric triangles (synthetic)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_image(self, index):
        """
        Plot a single point from the dataset, highlighting it within the context of all triangles.
        Similar to MNIST's plot_image method.
        """
        print(f"Label: {self.y[index]}")
        print(f"Coordinates: x={self.X[index][0]:.4f}, y={self.X[index][1]:.4f}")

        # Plot all triangles with low opacity
        labels = sorted(self.df['label'].unique())
        plt.figure(figsize=(6,6))
        for lab in labels:
            sub = self.df[self.df['label'] == lab]
            plt.scatter(sub['x'], sub['y'], label=f"label={lab}", marker='o', s=20, alpha=0.3)

        # Highlight the specific point
        # plt.scatter(self.X[index][0], self.X[index][1],
        #            color='red', s=200, marker='*',
        #            edgecolors='black', linewidths=2,
        #            label=f'Selected point (index={index})', zorder=5)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.title(f'Point at index {index} (Label: {self.y[index]})')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Raw dataset: no embedding plot here


class ConcentricTrianglesDatasetTorch(Dataset):

    def __init__(self, number_of_samples=100,
                 n_triangles=5, points_per_triangle=200, sizes=None, noise=0.0, rotation=0.0, random_state=None):
        self.ctds = ConcentricTrianglesDataset(
            number_of_samples=number_of_samples,
            n_triangles=n_triangles,
            points_per_triangle=points_per_triangle,
            sizes=sizes,
            noise=noise,
            rotation=rotation,
            random_state=random_state
        )
        self.X = self.ctds.X
        self.y = self.ctds.y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Example usage
if __name__ == "__main__":
    dataset = ConcentricTrianglesDataset(number_of_samples=1000)
    dataset.print_first_10_labels()
    dataset.print_first_10_images_and_labels()
    dataset.plot_image(0)

