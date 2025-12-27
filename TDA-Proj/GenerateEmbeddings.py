import torch
import json
import argparse
from TrainingDatasets import DATASET_LOADERS
import numpy as np
from Models import load_model_from_config
from preamble import *

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

        # Forward pass - use model.encode() which handles input shape properly
        outputs_mu, outputs_logvar = model.encode(inputs)
        embeddings.append(outputs_mu.cpu().detach().numpy())
    embeddings = np.vstack(embeddings)

    return embeddings



def main(config):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = config['dataset']
    data_getter, input_shape = DATASET_LOADERS[dataset]
    
    model = load_model_from_config(config, input_shape).to(device)
    
    model.load_state_dict(torch.load(config['save_path'], map_location=device))
    
    fix_random_seeds(config.get('random_seed', 69))
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