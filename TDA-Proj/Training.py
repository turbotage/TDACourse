import json
import argparse
import os
import glob
from TrainingDatasets import DATASET_LOADERS
from tqdm import tqdm
from Models import MODELS, load_model_from_config
from preamble import *

def epoch(model, dataloader, criterion, optimizer=None, device='cpu', verbose=True, beta=1.0):
    """
    Perform one epoch of training or evaluation.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training. If None, evaluation mode is assumed.
        device (str): Device to run the computations on ('cpu' or 'cuda').
        beta (float): KL weight for VAE models (beta-VAE parameter).
    """
    model.to(device)
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    running_loss = 0.0
    running_recon_loss = 0.0
    running_kl = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader) if verbose else dataloader
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Determine loss depending on model output type
        if isinstance(outputs, dict):
            # VAE-style model: expect model.compute_loss to return {'loss': tensor, ...}
            if hasattr(model, 'compute_loss'):
                loss_info = model.compute_loss(inputs, outputs, beta=beta)
                loss = loss_info['loss']
            else:
                # Fallback: try to compute reconstruction loss with provided criterion
                recon = outputs.get('recon')
                if recon is None:
                    raise ValueError("Model returned dict but no 'recon' key and no compute_loss method available")
                inputs_flat = inputs.view(inputs.size(0), -1) if inputs.dim() > 2 else inputs
                if callable(criterion):
                    loss = criterion(recon, inputs_flat)
                    loss_info = {'loss': loss, 'recon_loss': loss.item(), 'kl': 0.0}
                else:
                    raise ValueError("No valid criterion provided to compute reconstruction loss")

            # Backprop / optimization step
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_recon_loss += loss_info.get('recon_loss', 0.0) * batch_size
            running_kl += loss_info.get('kl', 0.0) * batch_size
            total += batch_size

            # For VAE training we don't compute classification accuracy; keep correct=0
        else:
            # Standard supervised model path: outputs is logits/predictions
            # Assume criterion knows how to compare outputs with labels
            loss = criterion(outputs, labels)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            # Try to compute accuracy when outputs are class logits
            try:
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
            except Exception:
                # If outputs are not class logits, skip accuracy
                pass

            total += batch_size
        pbar.set_postfix({'loss': running_loss / max(total, 1)})
    epoch_loss = running_loss / max(total, 1)
    epoch_recon = running_recon_loss / max(total, 1)
    epoch_kl = running_kl / max(total, 1)
    epoch_acc = (correct / total) if total > 0 else 0.0

    return epoch_loss, epoch_acc, epoch_recon, epoch_kl

def main(config):
    fix_random_seeds(config.get('random_seed', 69))
    
    dataset = config['dataset']
    num_epochs = config['num_epochs']
    verbose = config.get('verbose', True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_getter, input_shape = DATASET_LOADERS[dataset]
    
    model = load_model_from_config(config, input_shape).to(device)

    # instantiate criterion and optimizer from config if provided
    criterion = create_criterion(config.get('criterion'))
    optimizer = create_optimizer(model.parameters(), config.get('optimizer'))

    train_data, test_data = data_getter(batch_size=config['batch_size'], shuffle_train = True)
    #train_data.dataset.dataset.X = train_data.dataset.dataset.X.astype(np.float32)
    #test_data.dataset.dataset.X = test_data.dataset.dataset.X.astype(np.float32)
    
    best_val_loss = float('inf')
    best_state_dict = None
    
    # Beta annealing: gradually increase KL weight for better clustering
    beta_start = config.get('beta_start', 0.0)
    beta_end = config.get('beta_end', config.get('beta', 1.0))
    warmup_epochs = config.get('beta_warmup_epochs', 0)
    annealing_epochs = config.get('beta_annealing_epochs', 10)  # How many epochs to anneal over
    
    for epoch_idx in range(num_epochs):
        # Compute current beta
        if epoch_idx < warmup_epochs:
            # Stay at beta_start during warmup
            beta = beta_start
        elif epoch_idx < warmup_epochs + annealing_epochs:
            # Linear annealing from beta_start to beta_end
            progress = (epoch_idx - warmup_epochs) / annealing_epochs
            beta = beta_start + (beta_end - beta_start) * progress
        else:
            # After annealing, stay at beta_end
            beta = beta_end
        
        train_loss, train_acc, train_recon, train_kl = epoch(model, train_data, criterion, optimizer, device, verbose, beta=beta)
        test_loss, test_acc, test_recon, test_kl = epoch(model, test_data, criterion, None, device, verbose, beta=beta)

        if verbose:
            print(f"Epoch {epoch_idx+1}/{num_epochs} (β={beta:.3f}) - "
                  f"Train: Loss={train_loss:.4f}, Recon={train_recon:.4f}, "
                  f"KL={train_kl:.4f} (β*KL={beta*train_kl:.4f}) - "
                  f"Test: Loss={test_loss:.4f}, Recon={test_recon:.4f}, "
                  f"KL={test_kl:.4f} (β*KL={beta*test_kl:.4f})")
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_state_dict = model.state_dict()
    
    # Generate unique checkpoint name with index
    save_path = config.get('save_path', 'ckpts/best_model.pth')
    ckpt_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    
    # Find next available index
    os.makedirs(ckpt_dir, exist_ok=True)
    existing_ckpts = glob.glob(os.path.join(ckpt_dir, f"{base_name}_*.pth"))
    existing_indices = []
    for ckpt in existing_ckpts:
        try:
            idx = int(os.path.basename(ckpt).split('_')[-1].split('.')[0])
            existing_indices.append(idx)
        except ValueError:
            pass
    
    next_idx = max(existing_indices) + 1 if existing_indices else 0
    
    # Save checkpoint and config with same index
    ckpt_path = os.path.join(ckpt_dir, f"{base_name}_{next_idx}.pth")
    config_path = os.path.join(ckpt_dir, f"{base_name}_{next_idx}.json")
    
    torch.save(best_state_dict, ckpt_path)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    if verbose:
        print(f"\nSaved checkpoint: {ckpt_path}")
        print(f"Saved config: {config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network model.")
    parser.add_argument('--config', type=str, required=False, help='Path to the configuration JSON file.', default='cfgs/GMMConcentricTriangles.json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)