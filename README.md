# Tensor Systems — Specs

Reference package for tensor performance diagnostics, shape contracts, and device management in PyTorch model pipelines.

## Install (development mode)

```bash
pip install -e .
```

## Usage

```python
import tensor_systems_specs as tss
import torch

x = torch.rand((64, 3, 224, 224))
tss.trace_shape("Image batch", x)
```

## Reproducibility

```python
import tensor_systems_specs as tss

tss.set_seed(42)
```

## Current Modules

1. Execution timing utilities — measure execution time for tensor operations or functions.
2. Shape tracing — log shape, device, and dtype information.
3. Device checks — ensure tensors are on the correct device (CPU/GPU) for computation.
4. Reproducibility — set seeds across random, numpy, and torch for repeatable experiments.
