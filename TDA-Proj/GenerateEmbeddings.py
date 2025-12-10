import torch
import json
import argparse
from TrainingDatasets import DATASET_LOADERS
from tqdm import tqdm
import os
import random
import numpy as np
from Models import MODELS



def epoch(model, dataloader, device='cpu'):
    """
    Perform one epoch of training or evaluation.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): Device to run the computations on ('cpu' or 'cuda').
    """
    model.to(device)
    model.eval()
    embeddings = []

    for inputs, _ in dataloader:
        inputs = inputs.to(device)

        # Forward pass
        outputs_mu, outputs_logvar = model.encoder(inputs)
        embeddings.append(outputs_mu.cpu().detach().numpy())
    embeddings = np.vstack(embeddings)

    return embeddings

def fix_random_seeds(seed=69):
    """
    Fix random seeds.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(config):
    model = MODELS[config['model']]
    dataset = config['dataset']
    fix_random_seeds(config.get('random_seed', 69))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_getter, input_shape = DATASET_LOADERS[dataset]
    model = model(input_shape = input_shape, embedding_dim=config['embedding_dim'], num_classes=config['num_classes']).to(device)
    model.load_state_dict(torch.load(config['save_path'], map_location=device))
    train_data, test_data = data_getter(batch_size=config['batch_size'], shuffle_train = False)

    train_embeddings = epoch(model, train_data, device)
    test_embeddings = epoch(model, test_data, device)

    np.savez_compressed(config.get('embeds_path', 'embeddings.npz'), train=train_embeddings, test=test_embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)