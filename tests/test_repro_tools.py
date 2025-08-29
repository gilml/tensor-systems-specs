import random, numpy as np, torch
import tensor_systems_specs as tss


def test_set_seed_reproducibility():
    tss.set_seed(100)
    vals1 = (random.random(), np.random.rand(), torch.rand(1).item())

    tss.set_seed(100)
    vals2 = (random.random(), np.random.rand(), torch.rand(1).item())

    assert vals1 == vals2
