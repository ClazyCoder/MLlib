"""
Device management utilities for PyTorch models and tensors.

This module provides utilities for automatic device detection and moving
objects (models, tensors, batches) to the appropriate device (CPU/GPU/MPS).
"""

import torch
from typing import Any, Optional


def check_device() -> str:
    """
    Check the current device and return its type.
    """
    if torch.backends.mps.is_available():
        return 'MPS'
    elif torch.cuda.is_available():
        return 'GPU'
    else:
        return 'CPU'


def get_device(device_name: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    If device_name is specified, returns that device. Otherwise, automatically
    detects the best available device (MPS > CUDA > CPU).
    
    Args:
        device_name: Optional device name ('cuda', 'cpu', 'mps', 'cuda:0', etc.)
                    If None, automatically selects the best available device.
    
    Returns:
        torch.device: The selected device.
    
    Examples:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device('cuda:1')  # Specific GPU
        >>> device = get_device('cpu')  # Force CPU
    """
    if device_name:
        return torch.device(device_name)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')  # Apple Silicon
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def move_to_device(
    obj: Any,
    device: torch.device,
    non_blocking: bool = False
) -> Any:
    """
    Move an object (tensor, model, list, dict, etc.) to the specified device.
    
    This function recursively handles:
    - torch.Tensor
    - torch.nn.Module
    - Lists and tuples (returns same type)
    - Dictionaries
    - Other types (returned unchanged)
    
    Args:
        obj: Object to move to device.
        device: Target device.
        non_blocking: If True, tries to convert asynchronously (for CUDA).
    
    Returns:
        The object moved to the specified device.
    
    Examples:
        >>> model = move_to_device(model, device)
        >>> batch = move_to_device({'images': images, 'labels': labels}, device)
        >>> tensors = move_to_device([tensor1, tensor2], device)
    """
    if isinstance(obj, (list, tuple)):
        moved = [move_to_device(item, device, non_blocking) for item in obj]
        return type(obj)(moved)  # Preserve list/tuple type
    
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in obj.items()}
    
    elif hasattr(obj, 'to'):
        # torch.Tensor or torch.nn.Module
        return obj.to(device, non_blocking=non_blocking)
    
    else:
        # Other types (int, str, None, etc.) - return as is
        return obj
