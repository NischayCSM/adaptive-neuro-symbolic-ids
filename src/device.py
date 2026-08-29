"""
Shared device configuration for PyTorch.

Every script that creates a model or tensor should import from here
instead of duplicating ``torch.device(...)`` logic.

Usage
-----
    from src.device import DEVICE, print_device_info

    print_device_info()
    model = MyModel().to(DEVICE)
    x = torch.tensor(data).to(DEVICE)
"""

import torch

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device() -> torch.device:
    """Return the project-wide default device."""
    return DEVICE


def print_device_info() -> None:
    """Log device details at startup (call once per script entry-point)."""
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
