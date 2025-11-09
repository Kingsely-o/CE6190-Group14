"""
Configuration file for Stable Diffusion Model Comparison Project
Contains all constants, paths, and default parameters
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
COCO_DIR = DATA_DIR / "coco_5k"
COCO_IMAGES_DIR = COCO_DIR / "images"
COCO_ANNOTATIONS_FILE = COCO_DIR / "annotations.json"
PROMPTS_FILE = DATA_DIR / "prompts.txt"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
EXP1_DIR = RESULTS_DIR / "exp1"
EXP2_DIR = RESULTS_DIR / "exp2"
EXP3_DIR = RESULTS_DIR / "exp3"
FIGURES_DIR = RESULTS_DIR / "figures"

# Model output directories
SD_V15_DIR = EXP1_DIR / "sd_v15"
SD_V21_DIR = EXP1_DIR / "sd_v21"

# Results files
EXP1_RESULTS_FILE = EXP1_DIR / "exp1_results.json"
EXP2_RESULTS_FILE = EXP2_DIR / "exp2_results.json"
EXP3_RESULTS_FILE = EXP3_DIR / "exp3_results.json"

# Create directories if they don't exist
for directory in [DATA_DIR, COCO_DIR, COCO_IMAGES_DIR,
                  RESULTS_DIR, EXP1_DIR, EXP2_DIR, EXP3_DIR, FIGURES_DIR,
                  SD_V15_DIR, SD_V21_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# Model identifiers on HuggingFace
SD_V15_MODEL_ID = "runwayml/stable-diffusion-v1-5"
SD_V21_MODEL_ID = "stabilityai/stable-diffusion-2-1"

# Model names for saving/loading
MODEL_NAMES = {
    "sd_v15": "Stable Diffusion v1.5",
    "sd_v21": "Stable Diffusion v2.1"
}

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================

# Default generation parameters
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SEED = 42
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512

# Experiment 3 hyperparameter ranges
GUIDANCE_SCALES = [3.0, 5.0, 7.5, 10.0, 15.0]
INFERENCE_STEPS = [20, 50, 100]

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

# COCO dataset settings
# Using yerevann/coco-karpathy which is the standard COCO caption dataset
# This dataset contains ~5000 images in test split with 5 captions per image
COCO_DATASET_NAME = "yerevann/coco-karpathy"
COCO_SPLIT = "test"
COCO_NUM_SAMPLES = 5000  # Can be reduced to 1000 for testing

# Custom prompts categories
PROMPT_CATEGORIES = [
    "simple",
    "scenes",
    "multi_object",
    "detailed",
    "hard"
]

NUM_PROMPTS_PER_CATEGORY = 20
TOTAL_CUSTOM_PROMPTS = 100

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# FID calculation settings
FID_BATCH_SIZE = 50
FID_DIMS = 2048  # InceptionV3 feature dimension

# Inception Score settings
IS_SPLITS = 10

# CLIP Score settings
CLIP_MODEL_NAME = "ViT-B/32"

# Device settings
DEVICE = "cuda"  # Change to "cpu" if no GPU available

# ============================================================================
# EXECUTION SETTINGS
# ============================================================================

# Memory optimization
USE_XFORMERS = False  # Disabled due to compatibility issues with flash_attn
ENABLE_ATTENTION_SLICING = True
ENABLE_VAE_SLICING = True

# Progress tracking
PRINT_EVERY = 100  # Print progress every N images

# Multiprocessing
NUM_WORKERS = 4

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Plot settings
FIGURE_DPI = 300
FIGURE_SIZE = (10, 6)
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# Image grid settings
GRID_COLS = 4
GRID_ROWS = 4
COMPARISON_GRID_SIZE = (512, 512)

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

# Random seeds for reproducibility
RANDOM_SEED = 42
TORCH_SEED = 42
NUMPY_SEED = 42

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_output_dir(model_name: str, experiment: str = "exp1") -> Path:
    """
    Get the output directory for a specific model and experiment.

    Args:
        model_name: Name of the model (e.g., 'sd_v15', 'sd_v21')
        experiment: Experiment name (e.g., 'exp1', 'exp2')

    Returns:
        Path to model output directory
    """
    exp_dir = RESULTS_DIR / experiment
    model_dir = exp_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def print_config():
    """Print current configuration settings."""
    print("=" * 80)
    print("STABLE DIFFUSION COMPARISON - CONFIGURATION")
    print("=" * 80)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"\nModels:")
    print(f"  - SD v1.5: {SD_V15_MODEL_ID}")
    print(f"  - SD v2.1: {SD_V21_MODEL_ID}")
    print(f"\nGeneration Settings:")
    print(f"  - Steps: {DEFAULT_NUM_INFERENCE_STEPS}")
    print(f"  - Guidance Scale: {DEFAULT_GUIDANCE_SCALE}")
    print(f"  - Seed: {DEFAULT_SEED}")
    print(f"  - Image Size: {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}")
    print(f"\nDataset:")
    print(f"  - COCO Samples: {COCO_NUM_SAMPLES}")
    print(f"  - Custom Prompts: {TOTAL_CUSTOM_PROMPTS}")
    print(f"\nDevice: {DEVICE}")
    print(f"Memory Optimization: xformers={USE_XFORMERS}, "
          f"attention_slicing={ENABLE_ATTENTION_SLICING}")
    print("=" * 80)


if __name__ == "__main__":
    print_config()
