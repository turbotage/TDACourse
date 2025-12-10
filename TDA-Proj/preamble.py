import torch.optim as optim
import torch
import os
import random
import numpy as np

def create_optimizer(params, opt_config):
    """Create a torch optimizer from a config.

    opt_config can be:
      - None -> returns None
      - a string name of optimizer (e.g. 'Adam') -> uses default lr 1e-3
      - a dict: { 'name': 'Adam', 'lr': 0.001, ... }
    """
    if opt_config is None:
        return None

    # If user passed an already-constructed optimizer object, return it
    if isinstance(opt_config, optim.Optimizer):
        return opt_config

    name = None
    kwargs = {}
    if isinstance(opt_config, str):
        name = opt_config
        kwargs = {'lr': 1e-3}
    elif isinstance(opt_config, dict):
        name = opt_config.get('name')
        # copy all other keys as kwargs except name
        kwargs = {k: v for k, v in opt_config.items() if k != 'name'}
    else:
        raise ValueError('optimizer config must be None, a string, a dict, or an optimizer instance')

    if name is None:
        raise ValueError('optimizer config missing name')

    # map common lowercase names
    name_map = {
        'adam': 'Adam',
        'adamw': 'AdamW',
        'sgd': 'SGD',
        'rmsprop': 'RMSprop',
    }
    opt_name = name_map.get(name.lower(), name)

    if not hasattr(optim, opt_name):
        raise ValueError(f'Unknown optimizer: {opt_name}')

    OptimClass = getattr(optim, opt_name)
    return OptimClass(params, **kwargs)


def create_criterion(crit_config):
    """Create a loss/criterion from config.

    crit_config can be None, string, dict, or an already-constructed loss instance.
    """
    import torch.nn as nn

    if crit_config is None:
        return None
    if isinstance(crit_config, nn.modules.loss._Loss):
        return crit_config
    if isinstance(crit_config, str):
        name = crit_config.lower()
        if name in ('crossentropy', 'cross_entropy', 'crossent', 'ce'):
            return nn.CrossEntropyLoss()
        if name in ('mse', 'mse_loss', 'mse-loss'):
            return nn.MSELoss()
        if name in ('bce', 'bcewithlogits', 'bce_with_logits'):
            return nn.BCEWithLogitsLoss()
        raise ValueError(f'Unknown criterion string: {crit_config}')
    if isinstance(crit_config, dict):
        name = crit_config.get('name')
        params = {k: v for k, v in crit_config.items() if k != 'name'}
        name = name.lower()
        if name in ('crossentropy', 'cross_entropy', 'crossent', 'ce'):
            return nn.CrossEntropyLoss(**params)
        if name in ('mse', 'mse_loss', 'mse-loss'):
            return nn.MSELoss(**params)
        if name in ('bce', 'bcewithlogits', 'bce_with_logits'):
            return nn.BCEWithLogitsLoss(**params)
        raise ValueError(f'Unknown criterion name in dict: {name}')

    raise ValueError('criterion config must be None, a string, a dict, or a loss instance')


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
