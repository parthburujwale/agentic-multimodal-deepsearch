import os
import random

import numpy as np
import torch


def configure_reproducibility(
    seed: int = 42,
):
    """
    Configure deterministic inference for the DeepSearch pipeline.
    """

    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        ":4096:8",
    )

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False

    if (
        torch.cuda.is_available()
        and hasattr(torch.backends.cuda, "matmul")
    ):
        torch.backends.cuda.matmul.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    except Exception:
        pass

    print(
        "\nReproducibility configured:"
    )

    print(
        f"  Seed: {seed}"
    )

    print(
        "  cuDNN benchmark: False"
    )

    print(
        "  cuDNN deterministic: True"
    )

    print(
        "  TF32: disabled"
    )

    print(
        "  Deterministic algorithms: enabled"
    )


def reset_random_state(
    seed: int = 42,
):
    """
    Reset Python, NumPy, and PyTorch random states before
    individual planner, verifier, or refiner operations.
    """

    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)