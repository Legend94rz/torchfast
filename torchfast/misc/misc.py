from asyncio import gather
import os
from threading import local
from typing import Optional, Any, Type
import numpy as np
import random
import torch
from enum import Flag
from torch import distributed


class Stage(Flag):
    TRAIN = 1
    VALIDATION = 2
    INFERENCE = 3


def all_reduce_mean(tensor) -> None:
    # inplace operation!
    world_size = distributed.get_world_size()
    distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
    tensor /= world_size


def all_gather_same_ndim(tensor):
    # tensor must has same ndim on each node
    world_size = distributed.get_world_size()
    local_shape = torch.tensor(tensor.shape, device='cuda')
    shape_list = [torch.empty_like(local_shape, device='cuda') for _ in range(world_size)]
    distributed.all_gather(shape_list, local_shape)
    max_size = max([shape.prod() for shape in shape_list]).item()
    padded = torch.empty(max_size, device='cuda')
    padded[:local_shape.prod()] = tensor.reshape(-1)

    tensor_list = [torch.empty(max_size, device='cuda') for i in range(world_size)]
    distributed.all_gather(tensor_list, padded)
    return [x[:shape_list[i].prod()].reshape(*shape_list[i]) for i, x in enumerate(tensor_list)]


def seed_everything(seed: Optional[int] = None) -> int:
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    In addition, sets the env variable `PL_GLOBAL_SEED` which will be passed to
    spawned subprocesses (e.g. ddp_spawn backend).
    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED")
        seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    if not (min_seed_value <= seed <= max_seed_value):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    return random.randint(min_seed_value, max_seed_value)


def model_summary(module: torch.nn.Module):
    pass


def safe_type_cast(t: type, v: Any, default: Optional[Any] = None):
    try:
        return t(v)
    except (ValueError, TypeError):
        return default
