"""
Data loading utilities for COCO-5K dataset and custom prompts
"""

import json
import os
import requests
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from io import BytesIO

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    COCO_DIR,
    COCO_IMAGES_DIR,
    COCO_ANNOTATIONS_FILE,
    PROMPTS_FILE,
    COCO_DATASET_NAME,
    COCO_SPLIT,
    COCO_NUM_SAMPLES,
    PROMPT_CATEGORIES
)
from utils.helpers import save_json, load_json, Timer, ProgressTracker


def download_image_from_url(url: str, max_retries: int = 3, timeout: int = 10) -> Optional[Image.Image]:
    """
    Download image from URL with retry logic.

    Args:
        url: Image URL
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds for each request

    Returns:
        PIL Image object or None if download fails
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return image
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
                continue
            return None
    return None


def download_coco_5k(num_samples: int = COCO_NUM_SAMPLES, force_download: bool = False):
    """
    Download COCO-5K dataset from HuggingFace yerevann/coco-karpathy.

    This dataset contains:
    - 'url': Image URL to download from
    - 'sentences': List of dictionaries with caption information

    This function will:
    1. Load the dataset from HuggingFace
    2. Download images from URLs
    3. Extract captions from sentences field
    4. Save everything to local disk

    Args:
        num_samples: Number of samples to download (default 5000)
        force_download: If True, re-download even if data exists

    Raises:
        RuntimeError: If dataset cannot be loaded
        ValueError: If critical errors occur during processing
    """
    print(f"\n{'='*80}")
    print(f"DOWNLOADING COCO DATASET FROM {COCO_DATASET_NAME}")
    print(f"{'='*80}")
    print(f"Target samples: {num_samples}")
    print(f"Output directory: {COCO_DIR}")

    # Check if already downloaded
    if COCO_ANNOTATIONS_FILE.exists() and not force_download:
        print(f"\n[DATA] ✓ COCO data already exists at {COCO_DIR}")
        print("[DATA] Use force_download=True to re-download")
        return

    # Create directories
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    COCO_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[DATA] Created directories")

    with Timer("COCO dataset download"):
        # Load dataset from HuggingFace
        print(f"\n[DATA] Loading dataset from HuggingFace...")
        print(f"  Dataset: {COCO_DATASET_NAME}")
        print(f"  Split: {COCO_SPLIT}")

        try:
            dataset = load_dataset(COCO_DATASET_NAME, split=COCO_SPLIT)
            print(f"[DATA] ✓ Dataset loaded: {len(dataset)} total samples available")
        except Exception as e:
            raise RuntimeError(
                f"FATAL: Cannot load dataset '{COCO_DATASET_NAME}' split '{COCO_SPLIT}'.\n"
                f"Error: {e}\n"
                "Please verify:\n"
                "  1. Dataset name is correct\n"
                "  2. You have internet access\n"
                "  3. HuggingFace Hub is accessible"
            ) from e

        # Verify dataset is not empty
        if len(dataset) == 0:
            raise ValueError("FATAL: Dataset is empty! Cannot proceed without data.")

        # Select subset if needed
        total_available = len(dataset)
        if total_available < num_samples:
            print(f"[WARNING] Requested {num_samples} samples but only {total_available} available")
            print(f"[DATA] Will download all {total_available} samples")
            num_samples = total_available
        else:
            dataset = dataset.select(range(num_samples))
            print(f"[DATA] Selected first {num_samples} samples")

        # Inspect first sample to understand structure
        print(f"\n[DATA] Inspecting dataset structure...")
        first_sample = dataset[0]
        print(f"  Available fields: {list(first_sample.keys())}")

        # Process samples with detailed progress tracking
        print(f"\n[DATA] Processing {len(dataset)} samples...")
        print(f"[DATA] Will download images from URLs and extract captions")
        print(f"[DATA] Images will be saved to: {COCO_IMAGES_DIR}")

        annotations = []
        failed_downloads = 0

        # Use tqdm for real progress tracking
        with tqdm(total=len(dataset), desc="Downloading images", unit="img") as pbar:
            for idx, sample in enumerate(dataset):
                # Extract image URL
                image_url = sample.get('url', None)
                if not image_url:
                    print(f"\n[WARNING] Sample {idx}: No 'url' field found, keys: {list(sample.keys())}")
                    failed_downloads += 1
                    pbar.update(1)
                    continue

                # Extract caption from sentences
                caption = None
                sentences = sample.get('sentences', None)

                if sentences:
                    # sentences is typically a list of dicts with 'raw' or 'sentence' field
                    if isinstance(sentences, list) and len(sentences) > 0:
                        first_sentence = sentences[0]
                        if isinstance(first_sentence, dict):
                            # Try common caption fields
                            caption = first_sentence.get('raw') or first_sentence.get('sentence') or first_sentence.get('caption')
                        elif isinstance(first_sentence, str):
                            caption = first_sentence
                    elif isinstance(sentences, dict):
                        caption = sentences.get('raw') or sentences.get('sentence')

                # Fallback to other caption fields
                if not caption:
                    caption = sample.get('caption') or sample.get('text')

                if not caption or not isinstance(caption, str) or len(caption.strip()) == 0:
                    print(f"\n[WARNING] Sample {idx}: No valid caption found")
                    failed_downloads += 1
                    pbar.update(1)
                    continue

                # Download image from URL
                image = download_image_from_url(image_url)
                if image is None:
                    print(f"\n[WARNING] Sample {idx}: Failed to download image from {image_url}")
                    failed_downloads += 1
                    pbar.update(1)
                    continue

                # Save image to local disk
                image_filename = f"coco_{idx:05d}.jpg"
                image_path = COCO_IMAGES_DIR / image_filename

                try:
                    image.save(image_path, "JPEG", quality=95)
                except Exception as e:
                    print(f"\n[WARNING] Sample {idx}: Failed to save image: {e}")
                    failed_downloads += 1
                    pbar.update(1)
                    continue

                # Store annotation
                annotations.append({
                    "image_id": idx,
                    "image_filename": image_filename,
                    "image_path": str(image_path),
                    "caption": caption.strip(),
                    "original_url": image_url
                })

                pbar.update(1)

                # Periodic status update
                if (idx + 1) % 100 == 0:
                    pbar.set_postfix({
                        'saved': len(annotations),
                        'failed': failed_downloads
                    })

        # Verify we got some data
        if len(annotations) == 0:
            raise ValueError(
                f"FATAL: No valid samples were processed!\n"
                f"Total attempted: {len(dataset)}\n"
                f"Failed downloads: {failed_downloads}\n"
                "Please check your internet connection and the dataset URLs."
            )

        # Save annotations to JSON
        print(f"\n[DATA] Saving annotations...")
        save_json({
            "annotations": annotations,
            "num_samples": len(annotations),
            "failed_downloads": failed_downloads,
            "dataset_name": COCO_DATASET_NAME,
            "split": COCO_SPLIT
        }, COCO_ANNOTATIONS_FILE)

        print(f"\n{'='*80}")
        print(f"DOWNLOAD COMPLETE")
        print(f"{'='*80}")
        print(f"✓ Successfully processed: {len(annotations)} samples")
        print(f"✗ Failed downloads: {failed_downloads}")
        print(f"✓ Images saved to: {COCO_IMAGES_DIR}")
        print(f"✓ Annotations saved to: {COCO_ANNOTATIONS_FILE}")
        print(f"{'='*80}")


def load_coco_5k() -> List[Dict[str, str]]:
    """
    Load COCO-5K dataset from disk.

    Returns:
        List of dictionaries containing image paths and captions
    """
    if not COCO_ANNOTATIONS_FILE.exists():
        print("[ERROR] COCO data not found. Run download_coco_5k() first.")
        print("[INFO] Attempting to download now...")
        download_coco_5k()

    print(f"[DATA] Loading COCO annotations from {COCO_ANNOTATIONS_FILE}")
    data = load_json(COCO_ANNOTATIONS_FILE)
    annotations = data["annotations"]

    print(f"[DATA] Loaded {len(annotations)} COCO samples")
    return annotations


def load_custom_prompts() -> Dict[str, List[str]]:
    """
    Load custom prompts from prompts.txt file.

    Returns:
        Dictionary mapping category names to lists of prompts
    """
    print(f"[DATA] Loading custom prompts from {PROMPTS_FILE}")

    if not PROMPTS_FILE.exists():
        raise FileNotFoundError(f"Prompts file not found: {PROMPTS_FILE}")

    prompts_by_category = {category: [] for category in PROMPT_CATEGORIES}
    current_category = None

    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Check for category header
            if line.startswith('[CATEGORY:'):
                # Extract category name
                current_category = line.split(':')[1].strip().rstrip(']')
                if current_category not in prompts_by_category:
                    prompts_by_category[current_category] = []
            elif current_category:
                # Add prompt to current category
                prompts_by_category[current_category].append(line)

    # Print statistics
    total_prompts = sum(len(prompts) for prompts in prompts_by_category.values())
    print(f"[DATA] Loaded {total_prompts} prompts across {len(prompts_by_category)} categories:")
    for category, prompts in prompts_by_category.items():
        print(f"  - {category}: {len(prompts)} prompts")

    return prompts_by_category


def get_coco_images_and_captions(num_samples: Optional[int] = None) -> Tuple[List[Image.Image], List[str]]:
    """
    Load COCO images and captions as PIL Images and text.

    Args:
        num_samples: Number of samples to load (None for all)

    Returns:
        Tuple of (images, captions)
    """
    annotations = load_coco_5k()

    if num_samples:
        annotations = annotations[:num_samples]

    images = []
    captions = []

    print(f"[DATA] Loading {len(annotations)} COCO images...")
    tracker = ProgressTracker(len(annotations), "Loading images")

    for ann in annotations:
        try:
            image = Image.open(ann["image_path"]).convert("RGB")
            images.append(image)
            captions.append(ann["caption"])
            tracker.update()
        except Exception as e:
            print(f"[ERROR] Failed to load image {ann['image_path']}: {e}")

    tracker.finish()

    print(f"[DATA] Loaded {len(images)} images and captions")
    return images, captions


def get_all_custom_prompts() -> List[str]:
    """
    Get all custom prompts as a flat list.

    Returns:
        List of all prompts
    """
    prompts_by_category = load_custom_prompts()
    all_prompts = []
    for prompts in prompts_by_category.values():
        all_prompts.extend(prompts)
    return all_prompts


def save_dataset_info():
    """
    Save information about the datasets.
    """
    info = {
        "coco": {
            "num_samples": COCO_NUM_SAMPLES,
            "dataset_name": COCO_DATASET_NAME,
            "split": COCO_SPLIT,
            "data_dir": str(COCO_DIR)
        },
        "custom_prompts": {
            "categories": PROMPT_CATEGORIES,
            "num_categories": len(PROMPT_CATEGORIES),
            "prompts_per_category": 20,
            "total_prompts": 100,
            "prompts_file": str(PROMPTS_FILE)
        }
    }

    info_file = Path(__file__).parent / "dataset_info.json"
    save_json(info, info_file)
    print(f"[DATA] Saved dataset info to {info_file}")


def test_data_loading():
    """
    Test data loading functions.
    """
    print("\n" + "=" * 80)
    print("TESTING DATA LOADING")
    print("=" * 80)

    # Test custom prompts
    print("\n1. Testing custom prompts loading...")
    prompts_by_category = load_custom_prompts()

    print("\nSample prompts from each category:")
    for category, prompts in prompts_by_category.items():
        print(f"\n{category.upper()}:")
        for i, prompt in enumerate(prompts[:3], 1):
            print(f"  {i}. {prompt}")

    # Test COCO loading (without downloading)
    print("\n2. Testing COCO dataset...")
    if COCO_ANNOTATIONS_FILE.exists():
        annotations = load_coco_5k()
        print(f"\nFirst 3 COCO annotations:")
        for i, ann in enumerate(annotations[:3], 1):
            print(f"  {i}. Image: {ann['image_filename']}")
            print(f"     Caption: {ann['caption']}")
    else:
        print("[INFO] COCO dataset not downloaded yet.")
        print("[INFO] Run download_coco_5k() to download the dataset.")

    print("\n" + "=" * 80)
    print("DATA LOADING TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Run tests
    test_data_loading()

    # Optionally download COCO (commented out to avoid automatic download)
    # print("\nDownloading COCO dataset...")
    # download_coco_5k(num_samples=100)  # Start with 100 for testing
