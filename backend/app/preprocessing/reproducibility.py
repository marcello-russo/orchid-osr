import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    print(f"Setting seed: {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.manual_seed(seed)  # MPS typically uses the main manual_seed
        # MPS doesn't have a specific deterministic flag like CUDNN yet, but manual_seed covers randomness
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
