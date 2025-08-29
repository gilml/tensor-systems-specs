import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility across random, numpy, and torch.
    If deterministic=True, configures torch/cuDNN for reproducible behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
