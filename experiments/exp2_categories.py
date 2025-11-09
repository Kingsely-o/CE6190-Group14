"""
Experiment 2: Category-wise Performance Analysis
Compare SD v1.5 vs v2.1 on custom prompts across different categories
Metrics: CLIP Score per category
"""

import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    EXP2_DIR,
    EXP2_RESULTS_FILE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_SEED,
    DEVICE,
    PROMPT_CATEGORIES
)
from models.model_loader import load_sd_v15, load_sd_v21, generate_image, unload_model
from data.data_loader import load_custom_prompts
from evaluation.metrics import CLIPScorer
from utils.helpers import (
    save_json,
    save_image,
    Timer,
    ProgressTracker,
    set_random_seeds,
    print_gpu_memory
)


def generate_images_for_prompts(
    model_pipeline,
    model_name: str,
    prompts_by_category: dict,
    output_dir: Path
):
    """
    Generate images for all custom prompts.

    Args:
        model_pipeline: Loaded StableDiffusion pipeline
        model_name: Name of the model
        prompts_by_category: Dictionary of prompts by category
        output_dir: Directory to save generated images

    Returns:
        Dictionary mapping categories to lists of generated image paths
    """
    print(f"\n{'=' * 80}")
    print(f"GENERATING IMAGES WITH {model_name.upper()}")
    print(f"{'=' * 80}")

    output_dir.mkdir(parents=True, exist_ok=True)

    generated_images = {category: [] for category in prompts_by_category.keys()}
    generation_times = []

    # Set seed for reproducibility
    set_random_seeds(DEFAULT_SEED)

    total_prompts = sum(len(prompts) for prompts in prompts_by_category.values())
    tracker = ProgressTracker(total_prompts, f"Generating {model_name}")

    prompt_idx = 0

    for category, prompts in prompts_by_category.items():
        print(f"\n[{model_name}] Processing category: {category} ({len(prompts)} prompts)")

        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        for idx, prompt in enumerate(prompts):
            start_time = time.time()

            try:
                # Generate image
                image = generate_image(
                    model_pipeline,
                    prompt=prompt,
                    num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
                    guidance_scale=DEFAULT_GUIDANCE_SCALE,
                    seed=DEFAULT_SEED + prompt_idx
                )

                # Save image
                image_filename = f"{category}_{idx:03d}.png"
                image_path = category_dir / image_filename
                save_image(image, image_path)

                generated_images[category].append({
                    "prompt": prompt,
                    "image_path": str(image_path),
                    "prompt_idx": prompt_idx
                })

                elapsed = time.time() - start_time
                generation_times.append(elapsed)

            except Exception as e:
                print(f"\n[ERROR] Failed to generate image for prompt '{prompt}': {e}")

            prompt_idx += 1
            tracker.update()

    tracker.finish()

    # Calculate statistics
    avg_time = sum(generation_times) / len(generation_times) if generation_times else 0
    total_time = sum(generation_times)

    print(f"\n[{model_name}] Generation complete!")
    print(f"  - Total prompts: {total_prompts}")
    print(f"  - Total time: {total_time / 60:.2f} minutes")
    print(f"  - Average time per image: {avg_time:.2f} seconds")

    return generated_images, {
        "total_time_seconds": total_time,
        "average_time_seconds": avg_time
    }


def evaluate_category_performance(
    model_name: str,
    generated_images_by_category: dict,
    device: str = DEVICE
):
    """
    Evaluate CLIP scores for each category.

    Args:
        model_name: Name of the model
        generated_images_by_category: Dictionary of generated images by category
        device: Device for computation

    Returns:
        Dictionary with CLIP scores per category
    """
    print(f"\n{'=' * 80}")
    print(f"EVALUATING {model_name.upper()} - CATEGORY-WISE CLIP SCORES")
    print(f"{'=' * 80}")

    # Initialize CLIP scorer
    clip_scorer = CLIPScorer(device=device)

    category_scores = {}

    for category, image_data in generated_images_by_category.items():
        print(f"\n[EVAL] Category: {category}")

        # Load images
        from PIL import Image
        images = []
        prompts = []

        for data in image_data:
            try:
                img = Image.open(data["image_path"]).convert("RGB")
                images.append(img)
                prompts.append(data["prompt"])
            except Exception as e:
                print(f"[ERROR] Failed to load image {data['image_path']}: {e}")

        if not images:
            print(f"[WARNING] No images loaded for category {category}")
            continue

        # Calculate CLIP score
        clip_scores = clip_scorer.calculate_clip_scores_list(images, prompts)
        avg_clip_score = sum(clip_scores) / len(clip_scores) if clip_scores else 0

        category_scores[category] = {
            "num_images": len(images),
            "clip_scores": clip_scores,
            "average_clip_score": avg_clip_score
        }

        print(f"[EVAL] {category}: CLIP Score = {avg_clip_score:.4f}")

    # Calculate overall average
    all_scores = []
    for scores in category_scores.values():
        all_scores.extend(scores["clip_scores"])

    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0

    print(f"\n[EVAL] {model_name} Overall Average CLIP Score: {overall_avg:.4f}")

    return {
        "category_scores": category_scores,
        "overall_average": overall_avg
    }


def run_experiment_2(skip_generation: bool = False):
    """
    Run Experiment 2: Category-wise performance analysis.

    Args:
        skip_generation: If True, skip generation and only evaluate
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: CATEGORY-WISE PERFORMANCE ANALYSIS")
    print("=" * 80)

    with Timer("Experiment 2"):
        # Step 1: Load custom prompts
        print("\n[EXP2] Step 1: Loading custom prompts...")
        prompts_by_category = load_custom_prompts()

        # Initialize results
        results = {
            "experiment": "exp2_categories",
            "categories": list(prompts_by_category.keys()),
            "models": {}
        }

        # Directories for each model
        exp2_v15_dir = EXP2_DIR / "sd_v15"
        exp2_v21_dir = EXP2_DIR / "sd_v21"

        # Step 2: Generate images with SD v1.5
        if not skip_generation:
            print("\n[EXP2] Step 2: Generating images with SD v1.5...")
            pipeline_v15 = load_sd_v15(device=DEVICE)
            print_gpu_memory()

            generated_v15, gen_stats_v15 = generate_images_for_prompts(
                pipeline_v15,
                "SD v1.5",
                prompts_by_category,
                exp2_v15_dir
            )

            results["models"]["sd_v15"] = {
                "model_name": "SD v1.5",
                "generation_stats": gen_stats_v15
            }

            # Unload model
            unload_model(pipeline_v15)
            print_gpu_memory()

            # Step 3: Generate images with SD v2.1
            print("\n[EXP2] Step 3: Generating images with SD v2.1...")
            pipeline_v21 = load_sd_v21(device=DEVICE)
            print_gpu_memory()

            generated_v21, gen_stats_v21 = generate_images_for_prompts(
                pipeline_v21,
                "SD v2.1",
                prompts_by_category,
                exp2_v21_dir
            )

            results["models"]["sd_v21"] = {
                "model_name": "SD v2.1",
                "generation_stats": gen_stats_v21
            }

            # Unload model
            unload_model(pipeline_v21)
            print_gpu_memory()
        else:
            print("\n[EXP2] Skipping generation (skip_generation=True)")
            # Load existing image paths
            generated_v15 = load_generated_images(exp2_v15_dir, prompts_by_category)
            generated_v21 = load_generated_images(exp2_v21_dir, prompts_by_category)

        # Step 4: Evaluate SD v1.5
        print("\n[EXP2] Step 4: Evaluating SD v1.5...")
        eval_results_v15 = evaluate_category_performance(
            "SD v1.5",
            generated_v15,
            device=DEVICE
        )
        results["models"]["sd_v15"]["evaluation"] = eval_results_v15

        # Step 5: Evaluate SD v2.1
        print("\n[EXP2] Step 5: Evaluating SD v2.1...")
        eval_results_v21 = evaluate_category_performance(
            "SD v2.1",
            generated_v21,
            device=DEVICE
        )
        results["models"]["sd_v21"]["evaluation"] = eval_results_v21

        # Step 6: Save results
        print("\n[EXP2] Step 6: Saving results...")
        save_json(results, EXP2_RESULTS_FILE)
        print(f"[EXP2] Results saved to {EXP2_RESULTS_FILE}")

        # Step 7: Print comparison table
        print_category_comparison_table(results)

    print("\n" + "=" * 80)
    print("EXPERIMENT 2 COMPLETE!")
    print("=" * 80)

    return results


def load_generated_images(output_dir: Path, prompts_by_category: dict):
    """
    Load existing generated images from disk.

    Args:
        output_dir: Directory containing generated images
        prompts_by_category: Dictionary of prompts by category

    Returns:
        Dictionary of generated images by category
    """
    generated_images = {category: [] for category in prompts_by_category.keys()}

    for category, prompts in prompts_by_category.items():
        category_dir = output_dir / category

        if not category_dir.exists():
            print(f"[WARNING] Category directory not found: {category_dir}")
            continue

        for idx, prompt in enumerate(prompts):
            image_filename = f"{category}_{idx:03d}.png"
            image_path = category_dir / image_filename

            if image_path.exists():
                generated_images[category].append({
                    "prompt": prompt,
                    "image_path": str(image_path),
                    "prompt_idx": idx
                })

    return generated_images


def print_category_comparison_table(results: dict):
    """
    Print a comparison table of category-wise CLIP scores.

    Args:
        results: Results dictionary from experiment
    """
    print("\n" + "=" * 80)
    print("CATEGORY-WISE COMPARISON TABLE")
    print("=" * 80)

    models = results.get("models", {})

    # Extract category scores
    v15_scores = models.get("sd_v15", {}).get("evaluation", {}).get("category_scores", {})
    v21_scores = models.get("sd_v21", {}).get("evaluation", {}).get("category_scores", {})

    print(f"\n{'Category':<20} {'SD v1.5 CLIP':<20} {'SD v2.1 CLIP':<20} {'Difference':<15}")
    print("-" * 80)

    for category in results.get("categories", []):
        v15_score = v15_scores.get(category, {}).get("average_clip_score", 0)
        v21_score = v21_scores.get(category, {}).get("average_clip_score", 0)
        diff = v21_score - v15_score

        v15_str = f"{v15_score:.4f}" if v15_score else "N/A"
        v21_str = f"{v21_score:.4f}" if v21_score else "N/A"
        diff_str = f"{diff:+.4f}" if v15_score and v21_score else "N/A"

        print(f"{category:<20} {v15_str:<20} {v21_str:<20} {diff_str:<15}")

    # Overall averages
    v15_overall = models.get("sd_v15", {}).get("evaluation", {}).get("overall_average", 0)
    v21_overall = models.get("sd_v21", {}).get("evaluation", {}).get("overall_average", 0)
    overall_diff = v21_overall - v15_overall

    # Format values
    v15_str = f"{v15_overall:.4f}" if v15_overall else "N/A"
    v21_str = f"{v21_overall:.4f}" if v21_overall else "N/A"
    diff_str = f"{overall_diff:+.4f}" if (v15_overall and v21_overall) else "N/A"

    print("-" * 80)
    print(f"{'OVERALL':<20} {v15_str:<20} {v21_str:<20} {diff_str:<15}")

    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment 2: Category-wise Analysis")
    parser.add_argument("--skip_generation", action="store_true",
                       help="Skip generation and only evaluate existing images")

    args = parser.parse_args()

    # Run experiment
    run_experiment_2(skip_generation=args.skip_generation)
