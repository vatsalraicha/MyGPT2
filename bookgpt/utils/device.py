"""MPS/CPU device selection and utilities for Apple Silicon."""

import logging
import torch

logger = logging.getLogger(__name__)


def get_device(force_cpu: bool = False) -> torch.device:
    """Get the best available device (MPS preferred, CPU fallback).

    Args:
        force_cpu: If True, always return CPU device.

    Returns:
        torch.device for MPS or CPU.
    """
    if force_cpu:
        logger.info("Forced CPU mode")
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            logger.info("Using MPS (Metal Performance Shaders) device")
            return torch.device("mps")
        else:
            logger.warning("MPS not built in this PyTorch installation, falling back to CPU")
    else:
        logger.warning("MPS not available on this system, falling back to CPU")

    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """Get the recommended dtype for the given device.

    MPS supports float16 but some ops may fail; float32 is safer.
    """
    if device.type == "mps":
        return torch.float32  # float16 has op gaps on MPS; use float32 for reliability
    return torch.float32


def mps_memory_info() -> dict:
    """Return current MPS memory allocation info.

    Returns:
        Dict with allocated and driver-allocated bytes, or empty if not MPS.
    """
    if not torch.backends.mps.is_available():
        return {}
    return {
        "allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
    }


def mps_synchronize():
    """Synchronize MPS device (wait for all kernels to complete)."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def mps_empty_cache():
    """Release MPS cached memory back to the system."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all backends."""
    import numpy as np
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        # MPS uses the CPU generator for random ops
        torch.manual_seed(seed)
    logger.info(f"Random seed set to {seed}")
