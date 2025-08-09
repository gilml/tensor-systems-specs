import torch


def trace_shape(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        print(f"[TSS] {name} → Not a tensor")
        return
    print(
        f"[TSS] {name} → Shape: {tuple(tensor.shape)}, Device: {tensor.device}, Dtype: {tensor.dtype}")
