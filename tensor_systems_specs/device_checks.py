import torch


def ensure_device(tensor, device='cuda'):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if device == 'cuda' and torch.cuda.is_available():
        return tensor.to('cuda')
    return tensor.to('cpu')
