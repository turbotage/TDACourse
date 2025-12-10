import numpy as np
import sys
sys.path.append('TDA-Proj')
import matplotlib.pyplot as plt

import ProjectDatasets as pd

#gmm_data = np.load('TDA-Proj/embeds/gmm_cnn_cifar.npz')

train_dataloader, test_dataloader = pd.get_cifar10_dataloader(batch_size=1000, embedding_path='TDA-Proj/embeds/gmm_cnn_cifar.npz')

batch = next(iter(train_dataloader))

# Get the first image from the batch (shape: (3, 32, 32))
image_tensor = batch[0][200]

# Convert to numpy and transpose to (32, 32, 3) for matplotlib
image_np = image_tensor.numpy().transpose(1, 2, 0)

# Display the image
plt.figure()
plt.imshow(image_np)
plt.show()

print('Hello')
