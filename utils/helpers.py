"""
Utility functions for Stable Diffusion Model Comparison Project
"""

import json
import time
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Union
from functools import wraps

import numpy as np
import torch
from PIL import Image


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_image(image: Image.Image, path: Union[str, Path], create_dirs: bool = True):
    """
    Save a PIL Image to disk.

    Args:
        image: PIL Image object
        path: Path to save the image
        create_dirs: If True, create parent directories if they don't exist
    """
    path = Path(path)
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)

    image.save(path)


def load_image(path: Union[str, Path]) -> Image.Image:
    """
    Load an image from disk.

    Args:
        path: Path to the image file

    Returns:
        PIL Image object
    """
    return Image.open(path).convert("RGB")


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2):
    """
    Save data to a JSON file.

    Args:
        data: Dictionary to save
        path: Path to save the JSON file
        indent: Indentation level for pretty printing
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        path: Path to the JSON file

    Returns:
        Dictionary containing the loaded data
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure execution time of a function.

    Args:
        func: Function to measure

    Returns:
        Wrapped function that prints execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"\n[TIMING] {func.__name__} took {elapsed_time:.2f} seconds "
              f"({elapsed_time / 60:.2f} minutes)")

        return result
    return wrapper


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "Code block", verbose: bool = True):
        """
        Initialize timer.

        Args:
            name: Name of the code block being timed
            verbose: If True, print timing information
        """
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed_time = None

    def __enter__(self):
        """Start the timer."""
        self.start_time = time.time()
        if self.verbose:
            print(f"[TIMER] Starting: {self.name}")
        return self

    def __exit__(self, *args):
        """Stop the timer and print elapsed time."""
        self.elapsed_time = time.time() - self.start_time
        if self.verbose:
            print(f"[TIMER] Finished: {self.name} in {self.elapsed_time:.2f}s "
                  f"({self.elapsed_time / 60:.2f}m)")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get GPU memory usage information.

    Returns:
        Dictionary with memory info (allocated, reserved, total)
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0}

    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
    }


def print_gpu_memory():
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        print("[GPU] No GPU available")
        return

    info = get_gpu_memory_info()
    print(f"[GPU] Memory - Allocated: {info['allocated_gb']:.2f}GB, "
          f"Reserved: {info['reserved_gb']:.2f}GB, "
          f"Total: {info['total_gb']:.2f}GB")


def clear_gpu_memory():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[GPU] Cache cleared")


def batch_list(items: List, batch_size: int) -> List[List]:
    """
    Split a list into batches.

    Args:
        items: List to split
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_comparison_grid(
    images: List[Image.Image],
    labels: List[str],
    grid_size: tuple = (2, 2),
    label_height: int = 30
) -> Image.Image:
    """
    Create a grid of images with labels.

    Args:
        images: List of PIL Images
        labels: List of labels for each image
        grid_size: (rows, cols) for the grid
        label_height: Height of label area in pixels

    Returns:
        Combined grid image
    """
    rows, cols = grid_size
    if len(images) != rows * cols:
        raise ValueError(f"Number of images ({len(images)}) must match "
                        f"grid size ({rows}x{cols}={rows*cols})")

    # Get image dimensions (assume all images are same size)
    img_width, img_height = images[0].size

    # Create new image with space for labels
    total_width = img_width * cols
    total_height = (img_height + label_height) * rows
    grid_image = Image.new('RGB', (total_width, total_height), color='white')

    # Paste images into grid
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols

        x = col * img_width
        y = row * (img_height + label_height) + label_height

        grid_image.paste(img, (x, y))

        # Add label (simple text - can be enhanced with PIL.ImageDraw)
        # For now, we'll skip text rendering as it requires font handling

    return grid_image


def ensure_dir_exists(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_exists(path: Union[str, Path]) -> bool:
    """
    Check if a file exists.

    Args:
        path: File path

    Returns:
        True if file exists, False otherwise
    """
    return Path(path).exists()


def get_file_size(path: Union[str, Path]) -> float:
    """
    Get file size in MB.

    Args:
        path: File path

    Returns:
        File size in MB
    """
    return Path(path).stat().st_size / (1024 * 1024)


def format_number(num: float, decimals: int = 2) -> str:
    """
    Format a number for display.

    Args:
        num: Number to format
        decimals: Number of decimal places

    Returns:
        Formatted string
    """
    return f"{num:.{decimals}f}"


class ProgressTracker:
    """Track progress of long-running operations."""

    def __init__(self, total: int, description: str = "Processing",
                 print_every: int = 100):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items to process
            description: Description of the operation
            print_every: Print progress every N items
        """
        self.total = total
        self.description = description
        self.print_every = print_every
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        """
        Update progress.

        Args:
            n: Number of items processed
        """
        self.current += n

        if self.current % self.print_every == 0 or self.current == self.total:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.current) / rate if rate > 0 else 0

            print(f"[PROGRESS] {self.description}: {self.current}/{self.total} "
                  f"({100 * self.current / self.total:.1f}%) - "
                  f"{format_time(elapsed)} elapsed, "
                  f"{format_time(remaining)} remaining, "
                  f"{rate:.2f} it/s")

    def finish(self):
        """Mark operation as finished."""
        elapsed = time.time() - self.start_time
        rate = self.total / elapsed if elapsed > 0 else 0
        print(f"[PROGRESS] {self.description}: Complete! "
              f"{self.total} items in {format_time(elapsed)} "
              f"({rate:.2f} it/s)")


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")

    # Test timer
    with Timer("Test operation"):
        time.sleep(1)

    # Test GPU info
    print_gpu_memory()

    # Test seeds
    set_random_seeds(42)
    print(f"Random number: {random.random()}")

    print("\nUtilities test complete!")
