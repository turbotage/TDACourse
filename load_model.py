import torch
import json
import os
import sys

# Add TDA-Proj to path
sys.path.append('./TDA-Proj')

from Models import GMMVAE_CNN, GMMVAE
from TrainingDatasets import get_cifar10_dataloader, get_mnist_dataloader

def load_model_from_path(path, batchsize):
	if path.endswith('.pth'):
		config_path = path[:-4] + '.json'
	else:
		raise ValueError("Model path must end with .pth")

	if 'mnist' in path.lower():
		dataset = 'mnist'
	elif 'cifar' in path.lower():
		dataset = 'cifar10'
	else:
		raise ValueError("Cannot infer dataset from model path. Please specify.")

	return load_model(path, batchsize, dataset=dataset, config_path=config_path)


def load_model(model_path, batch_size, dataset='cifar10', config_path=None):
	"""
	Load a trained model and visualize input vs reconstructed images.
	Args:
		model_path: Path to the saved model checkpoint (.pth file)
		num_images: Number of images to visualize (default: 5)
		dataset: 'cifar10' or 'mnist'
		config_path: Path to config JSON (optional, will try to infer from model_path)
	"""
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	# Load the model checkpoint
	print(f"Loading model from: {model_path}")
	checkpoint = torch.load(model_path, map_location=device)

	# Try to load matching config file (same name as checkpoint but .json)
	if config_path is None:
		config_path = model_path.replace('.pth', '.json')
	if not os.path.exists(config_path):
		raise FileNotFoundError(f"Config file not found: {config_path}")
	with open(config_path, 'r') as f:
		model_config = json.load(f)

	# Determine model type
	model_type = model_config.get('model_type', 'cnn')

	# Set up model and dataloader
	# Build model_kwargs from config, matching training logic
	model_kwargs = {}
	if dataset == 'cifar10':
		model_kwargs['input_shape'] = (3, 32, 32)
	elif dataset == 'mnist':
		model_kwargs['input_shape'] = (1, 28, 28)
	else:
		raise ValueError(f"Unknown dataset: {dataset}")

	# Common model params
	if 'conv_channels' in model_config:
		model_kwargs['conv_channels'] = model_config['conv_channels']
	if 'conv_strides' in model_config:
		model_kwargs['conv_strides'] = model_config['conv_strides']
	if 'embedding_dim' in model_config:
		model_kwargs['embedding_dim'] = model_config['embedding_dim']
	if 'num_classes' in model_config:
		model_kwargs['num_classes'] = model_config['num_classes']
	if 'use_residual' in model_config:
		model_kwargs['use_residual'] = model_config['use_residual']
	if 'num_residual_blocks' in model_config:
		model_kwargs['num_residual_blocks'] = model_config['num_residual_blocks']

	if dataset == 'cifar10' or (dataset == 'mnist' and model_type == 'cnn'):
		model = GMMVAE_CNN(**model_kwargs)
		is_color = (dataset == 'cifar10')
	elif dataset == 'mnist':
		model = GMMVAE(
			input_dim=28*28,
			hidden_dim=512,
			embedding_dim=model_config.get('embedding_dim', 32),
			num_classes=model_config.get('num_classes', 10)
		)
		is_color = False

	if dataset == 'cifar10':
		train_loader, test_loader = get_cifar10_dataloader(batch_size=batch_size)
	else:
		train_loader, test_loader = get_mnist_dataloader(batch_size=batch_size)

	# Load model weights
	model.load_state_dict(checkpoint)
	model = model.to(device)
	model.eval()

	return model, train_loader, test_loader, is_color
