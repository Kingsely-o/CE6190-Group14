"""
Experiment 1: Baseline Performance Comparison
Compare SD v1.5 vs v2.1 on COCO-5K dataset
Metrics: FID, CLIP Score, Inception Score, Generation Time
"""

import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    COCO_NUM_SAMPLES,
    EXP1_DIR,
    EXP1_RESULTS_FILE,
    SD_V15_DIR,
    SD_V21_DIR,
    COCO_IMAGES_DIR,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_SEED,
    DEVICE
)
from models.model_loader import load_sd_v15, load_sd_v21, generate_image, unload_model
from data.data_loader import load_coco_5k, download_coco_5k
from evaluation.metrics import (
    calculate_fid,
    calculate_clip_score,
    calculate_inception_score,
    load_images_from_dir
)
from utils.helpers import (
    save_json,
    save_image,
    Timer,
    ProgressTracker,
    set_random_seeds,
    print_gpu_memory
)


def generate_images_for_model(
    model_pipeline,
    model_name: str,
    annotations: list,
    output_dir: Path,
    num_samples: int = None
):
    """
    Generate images for all COCO captions using a single model.

    Args:
        model_pipeline: Loaded StableDiffusion pipeline
        model_name: Name of the model (for logging)
        annotations: List of COCO annotations
        output_dir: Directory to save generated images
        num_samples: Number of samples to generate (None for all)

    Returns:
        Dictionary with generation metadata
    """
    if num_samples:
        annotations = annotations[:num_samples]

    print(f"\n{'=' * 80}")
    print(f"GENERATING IMAGES WITH {model_name.upper()}")
    print(f"{'=' * 80}")
    print(f"Number of samples: {len(annotations)}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track timing
    generation_times = []
    tracker = ProgressTracker(len(annotations), f"Generating {model_name}")

    # Set seed for reproducibility
    set_random_seeds(DEFAULT_SEED)

    for idx, ann in enumerate(annotations):
        caption = ann["caption"]

        # Generate image
        start_time = time.time()

        try:
            image = generate_image(
                model_pipeline,
                prompt=caption,
                num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
                guidance_scale=DEFAULT_GUIDANCE_SCALE,
                seed=DEFAULT_SEED + idx  # Different seed per image
            )

            # Save image
            image_filename = f"gen_{idx:05d}.png"
            image_path = output_dir / image_filename
            save_image(image, image_path)

            # Track time
            elapsed = time.time() - start_time
            generation_times.append(elapsed)

        except Exception as e:
            print(f"\n[ERROR] Failed to generate image {idx}: {e}")
            generation_times.append(0)

        tracker.update()

    tracker.finish()

    # Calculate statistics
    avg_time = sum(generation_times) / len(generation_times) if generation_times else 0
    total_time = sum(generation_times)

    print(f"\n[{model_name}] Generation complete!")
    print(f"  - Total images: {len(annotations)}")
    print(f"  - Total time: {total_time / 60:.2f} minutes")
    print(f"  - Average time per image: {avg_time:.2f} seconds")

    return {
        "model_name": model_name,
        "num_images": len(annotations),
        "total_time_seconds": total_time,
        "average_time_seconds": avg_time,
        "generation_times": generation_times
    }


def evaluate_model(
    model_name: str,
    generated_images_dir: Path,
    real_images_dir: Path,
    captions: list
):
    """
    Evaluate generated images using multiple metrics.

    Args:
        model_name: Name of the model
        generated_images_dir: Directory with generated images
        real_images_dir: Directory with real COCO images
        captions: List of captions for CLIP score

    Returns:
        Dictionary with all evaluation metrics
    """
    print(f"\n{'=' * 80}")
    print(f"EVALUATING {model_name.upper()}")
    print(f"{'=' * 80}")

    results = {"model_name": model_name}

    # Load generated images
    print(f"\n[EVAL] Loading generated images from {generated_images_dir}...")
    generated_images = load_images_from_dir(generated_images_dir)
    print(f"[EVAL] Loaded {len(generated_images)} generated images")

    # 1. Calculate FID
    try:
        print("\n[EVAL] Calculating FID...")
        fid_score = calculate_fid(
            generated_images_dir,
            real_images_dir,
            device=DEVICE
        )
        results["fid"] = fid_score
    except Exception as e:
        print(f"[ERROR] FID calculation failed: {e}")
        results["fid"] = None

    # 2. Calculate CLIP Score
    try:
        print("\n[EVAL] Calculating CLIP Score...")
        clip_score = calculate_clip_score(
            generated_images[:len(captions)],
            captions,
            device=DEVICE
        )
        results["clip_score"] = clip_score
    except Exception as e:
        print(f"[ERROR] CLIP Score calculation failed: {e}")
        results["clip_score"] = None

    # 3. Calculate Inception Score
    try:
        print("\n[EVAL] Calculating Inception Score...")
        is_mean, is_std = calculate_inception_score(
            generated_images,
            device=DEVICE
        )
        results["inception_score_mean"] = is_mean
        results["inception_score_std"] = is_std
    except Exception as e:
        print(f"[ERROR] Inception Score calculation failed: {e}")
        results["inception_score_mean"] = None
        results["inception_score_std"] = None

    print(f"\n[EVAL] {model_name} evaluation complete!")
    print("\nResults:")
    print(f"  - FID: {results.get('fid', 'N/A')}")
    print(f"  - CLIP Score: {results.get('clip_score', 'N/A')}")
    print(f"  - Inception Score: {results.get('inception_score_mean', 'N/A')} ± {results.get('inception_score_std', 'N/A')}")

    return results


def run_experiment_1(num_samples: int = None, skip_generation: bool = False):
    """
    Run Experiment 1: Baseline comparison on COCO-5K.

    Args:
        num_samples: Number of samples to use (None for all configured)
        skip_generation: If True, skip generation and only run evaluation
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: BASELINE PERFORMANCE COMPARISON")
    print("=" * 80)

    # Use configured number or override
    if num_samples is None:
        num_samples = COCO_NUM_SAMPLES

    with Timer("Experiment 1"):
        # Step 1: Load COCO dataset
        print("\n[EXP1] Step 1: Loading COCO dataset...")
        try:
            annotations = load_coco_5k()
        except Exception as e:
            print(f"[ERROR] Failed to load COCO: {e}")
            print("[INFO] Downloading COCO dataset...")
            download_coco_5k(num_samples=num_samples)
            annotations = load_coco_5k()

        annotations = annotations[:num_samples]
        captions = [ann["caption"] for ann in annotations]

        print(f"[EXP1] Loaded {len(annotations)} annotations")

        # Initialize results dictionary
        results = {
            "experiment": "exp1_baseline",
            "num_samples": len(annotations),
            "models": {}
        }

        # Step 2: Generate images with SD v1.5
        if not skip_generation:
            print("\n[EXP1] Step 2: Generating images with SD v1.5...")
            pipeline_v15 = load_sd_v15(device=DEVICE)
            print_gpu_memory()

            generation_results_v15 = generate_images_for_model(
                pipeline_v15,
                "SD v1.5",
                annotations,
                SD_V15_DIR,
                num_samples=num_samples
            )

            results["models"]["sd_v15"] = generation_results_v15

            # Unload model to free memory
            unload_model(pipeline_v15)
            print_gpu_memory()

            # Step 3: Generate images with SD v2.1
            print("\n[EXP1] Step 3: Generating images with SD v2.1...")
            pipeline_v21 = load_sd_v21(device=DEVICE)
            print_gpu_memory()

            generation_results_v21 = generate_images_for_model(
                pipeline_v21,
                "SD v2.1",
                annotations,
                SD_V21_DIR,
                num_samples=num_samples
            )

            results["models"]["sd_v21"] = generation_results_v21

            # Unload model
            unload_model(pipeline_v21)
            print_gpu_memory()
        else:
            print("\n[EXP1] Skipping generation (skip_generation=True)")

        # Step 4: Evaluate SD v1.5
        print("\n[EXP1] Step 4: Evaluating SD v1.5...")
        eval_results_v15 = evaluate_model(
            "SD v1.5",
            SD_V15_DIR,
            COCO_IMAGES_DIR,
            captions
        )
        results["models"]["sd_v15"] = {
            **results["models"].get("sd_v15", {}),
            **eval_results_v15
        }

        # Step 5: Evaluate SD v2.1
        print("\n[EXP1] Step 5: Evaluating SD v2.1...")
        eval_results_v21 = evaluate_model(
            "SD v2.1",
            SD_V21_DIR,
            COCO_IMAGES_DIR,
            captions
        )
        results["models"]["sd_v21"] = {
            **results["models"].get("sd_v21", {}),
            **eval_results_v21
        }

        # Step 6: Save results
        print("\n[EXP1] Step 6: Saving results...")
        save_json(results, EXP1_RESULTS_FILE)
        print(f"[EXP1] Results saved to {EXP1_RESULTS_FILE}")

        # Step 7: Print comparison table
        print_comparison_table(results)

    print("\n" + "=" * 80)
    print("EXPERIMENT 1 COMPLETE!")
    print("=" * 80)

    return results


def print_comparison_table(results: dict):
    """
    Print a comparison table of results.

    Args:
        results: Results dictionary from experiment
    """
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON TABLE")
    print("=" * 80)

    # Extract metrics
    models = results.get("models", {})

    print(f"\n{'Metric':<25} {'SD v1.5':<20} {'SD v2.1':<20}")
    print("-" * 80)

    # FID
    fid_v15 = models.get("sd_v15", {}).get("fid", "N/A")
    fid_v21 = models.get("sd_v21", {}).get("fid", "N/A")
    fid_v15_str = f"{fid_v15:.2f}" if isinstance(fid_v15, (int, float)) else str(fid_v15)
    fid_v21_str = f"{fid_v21:.2f}" if isinstance(fid_v21, (int, float)) else str(fid_v21)
    print(f"{'FID (lower=better)':<25} {fid_v15_str:<20} {fid_v21_str:<20}")

    # CLIP Score
    clip_v15 = models.get("sd_v15", {}).get("clip_score", "N/A")
    clip_v21 = models.get("sd_v21", {}).get("clip_score", "N/A")
    clip_v15_str = f"{clip_v15:.4f}" if isinstance(clip_v15, (int, float)) else str(clip_v15)
    clip_v21_str = f"{clip_v21:.4f}" if isinstance(clip_v21, (int, float)) else str(clip_v21)
    print(f"{'CLIP Score (higher=better)':<25} {clip_v15_str:<20} {clip_v21_str:<20}")

    # Inception Score
    is_v15 = models.get("sd_v15", {}).get("inception_score_mean", "N/A")
    is_v21 = models.get("sd_v21", {}).get("inception_score_mean", "N/A")
    is_v15_str = f"{is_v15:.2f}" if isinstance(is_v15, (int, float)) else str(is_v15)
    is_v21_str = f"{is_v21:.2f}" if isinstance(is_v21, (int, float)) else str(is_v21)
    print(f"{'Inception Score':<25} {is_v15_str:<20} {is_v21_str:<20}")

    # Generation Time
    time_v15 = models.get("sd_v15", {}).get("average_time_seconds", "N/A")
    time_v21 = models.get("sd_v21", {}).get("average_time_seconds", "N/A")
    time_v15_str = f"{time_v15:.2f}s" if isinstance(time_v15, (int, float)) else str(time_v15)
    time_v21_str = f"{time_v21:.2f}s" if isinstance(time_v21, (int, float)) else str(time_v21)
    print(f"{'Avg Time per Image':<25} {time_v15_str:<20} {time_v21_str:<20}")

    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment 1: Baseline Comparison")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to use (default: from config)")
    parser.add_argument("--skip_generation", action="store_true",
                       help="Skip generation and only evaluate existing images")

    args = parser.parse_args()

    # Run experiment
    run_experiment_1(
        num_samples=args.num_samples,
        skip_generation=args.skip_generation
    )
