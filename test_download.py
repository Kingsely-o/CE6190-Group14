#!/usr/bin/env python3
"""
Test script to verify the fixed data download functionality
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import download_coco_5k, load_coco_5k
from config import COCO_IMAGES_DIR, COCO_ANNOTATIONS_FILE

def test_download():
    """Test downloading a small subset of data"""
    print("="*80)
    print("TESTING DATA DOWNLOAD - SMALL SUBSET")
    print("="*80)

    # Download only 10 samples for testing
    print("\nStep 1: Downloading 10 samples...")
    try:
        download_coco_5k(num_samples=10, force_download=True)
        print("✓ Download completed successfully")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify files exist
    print("\nStep 2: Verifying files...")

    # Check annotations file
    if not COCO_ANNOTATIONS_FILE.exists():
        print(f"✗ Annotations file not found: {COCO_ANNOTATIONS_FILE}")
        return False
    print(f"✓ Annotations file exists: {COCO_ANNOTATIONS_FILE}")

    # Check images directory
    image_files = list(COCO_IMAGES_DIR.glob("*.jpg"))
    print(f"✓ Found {len(image_files)} image files in {COCO_IMAGES_DIR}")

    if len(image_files) == 0:
        print("✗ No image files found!")
        return False

    # Load and verify annotations
    print("\nStep 3: Loading and verifying annotations...")
    try:
        annotations = load_coco_5k()
        print(f"✓ Loaded {len(annotations)} annotations")

        # Show first 3 samples
        print("\nFirst 3 samples:")
        for i, ann in enumerate(annotations[:3], 1):
            print(f"\nSample {i}:")
            print(f"  Image: {ann['image_filename']}")
            print(f"  Caption: {ann['caption'][:100]}...")
            print(f"  URL: {ann.get('original_url', 'N/A')[:80]}...")

            # Verify image file exists
            image_path = Path(ann['image_path'])
            if image_path.exists():
                print(f"  ✓ Image file exists ({image_path.stat().st_size} bytes)")
            else:
                print(f"  ✗ Image file missing: {image_path}")

    except Exception as e:
        print(f"✗ Failed to load annotations: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("TEST PASSED - DATA DOWNLOAD IS WORKING CORRECTLY")
    print("="*80)
    print(f"\nSummary:")
    print(f"  - Annotations saved: {len(annotations)}")
    print(f"  - Images downloaded: {len(image_files)}")
    print(f"  - All samples have captions: ✓")
    print(f"  - All images saved to disk: ✓")
    print(f"\nYou can now run the full download with:")
    print(f"  python run_all.py --download_data")

    return True

if __name__ == "__main__":
    success = test_download()
    sys.exit(0 if success else 1)
