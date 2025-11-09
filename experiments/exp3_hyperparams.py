"""
Experiment 3 (Optional): Hyperparameter Sensitivity Analysis
Test different guidance scales and inference steps
Metrics: CLIP Score and generation time for each setting
"""

import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    EXP3_DIR,
    EXP3_RESULTS_FILE,
    GUIDANCE_SCALES,
    INFERENCE_STEPS,
    DEFAULT_SEED,
    DEVICE
)
from models.model_loader import load_sd_v15, load_sd_v21, generate_image, unload_model
from data.data_loader import load_custom_prompts
from evaluation.metrics import CLIPScorer
from utils.helpers import (
    save_json,
    save_image,
    Timer,
    set_random_seeds,
    print_gpu_memory
)


def test_hyperparameter(
    model_pipeline,
    model_name: str,
    prompts: list,
    param_name: str,
    param_value: float,
    output_dir: Path,
    fixed_steps: int = 50,
    fixed_guidance: float = 7.5
):
    """
    Test a single hyperparameter setting.

    Args:
        model_pipeline: Loaded StableDiffusion pipeline
        model_name: Name of the model
        prompts: List of test prompts
        param_name: Name of parameter being tested ('guidance' or 'steps')
        param_value: Value of the parameter
        output_dir: Directory to save images
        fixed_steps: Fixed number of steps (when testing guidance)
        fixed_guidance: Fixed guidance scale (when testing steps)

    Returns:
        Dictionary with results
    """
    print(f"\n[{model_name}] Testing {param_name}={param_value}")

    # Determine parameters
    if param_name == "guidance":
        num_steps = fixed_steps
        guidance = param_value
    else:  # param_name == "steps"
        num_steps = int(param_value)
        guidance = fixed_guidance

    # Create output directory
    output_subdir = output_dir / f"{param_name}_{param_value}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Generate images
    images = []
    generation_times = []

    set_random_seeds(DEFAULT_SEED)

    for idx, prompt in enumerate(prompts):
        start_time = time.time()

        try:
            image = generate_image(
                model_pipeline,
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                seed=DEFAULT_SEED + idx
            )

            # Save image
            image_path = output_subdir / f"img_{idx:03d}.png"
            save_image(image, image_path)

            images.append(image)
            elapsed = time.time() - start_time
            generation_times.append(elapsed)

        except Exception as e:
            print(f"[ERROR] Failed to generate image for prompt '{prompt}': {e}")
            continue

    # Calculate CLIP score
    clip_scorer = CLIPScorer(device=DEVICE)
    clip_scores = clip_scorer.calculate_clip_scores_list(images, prompts)
    avg_clip_score = sum(clip_scores) / len(clip_scores) if clip_scores else 0

    # Calculate time statistics
    avg_time = sum(generation_times) / len(generation_times) if generation_times else 0

    results = {
        param_name: param_value,
        "num_steps": num_steps,
        "guidance_scale": guidance,
        "clip_score": avg_clip_score,
        "avg_time_seconds": avg_time,
        "individual_clip_scores": clip_scores,
        "generation_times": generation_times
    }

    print(f"  - CLIP Score: {avg_clip_score:.4f}")
    print(f"  - Avg Time: {avg_time:.2f}s")

    return results


def run_experiment_3(
    model_choice: str = "sd_v15",
    test_guidance: bool = True,
    test_steps: bool = True,
    num_test_prompts: int = 20
):
    """
    Run Experiment 3: Hyperparameter sensitivity analysis.

    Args:
        model_choice: Which model to test ('sd_v15' or 'sd_v21')
        test_guidance: If True, test guidance scale values
        test_steps: If True, test inference steps values
        num_test_prompts: Number of prompts to use for testing

    Returns:
        Results dictionary
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)

    with Timer("Experiment 3"):
        # Step 1: Load test prompts
        print("\n[EXP3] Step 1: Loading test prompts...")
        prompts_by_category = load_custom_prompts()

        # Use a subset of prompts from first category for testing
        first_category = list(prompts_by_category.keys())[0]
        test_prompts = prompts_by_category[first_category][:num_test_prompts]

        print(f"[EXP3] Using {len(test_prompts)} prompts from '{first_category}' category")

        # Step 2: Load model
        print(f"\n[EXP3] Step 2: Loading model {model_choice}...")
        if model_choice == "sd_v15":
            pipeline = load_sd_v15(device=DEVICE)
            model_name = "SD v1.5"
        else:
            pipeline = load_sd_v21(device=DEVICE)
            model_name = "SD v2.1"

        print_gpu_memory()

        # Initialize results
        results = {
            "experiment": "exp3_hyperparams",
            "model": model_name,
            "num_test_prompts": len(test_prompts),
            "test_prompts": test_prompts
        }

        # Create output directory
        model_output_dir = EXP3_DIR / model_choice

        # Step 3: Test guidance scales
        if test_guidance:
            print(f"\n[EXP3] Step 3: Testing guidance scales...")
            print(f"Testing values: {GUIDANCE_SCALES}")

            guidance_results = []
            for guidance_value in GUIDANCE_SCALES:
                result = test_hyperparameter(
                    pipeline,
                    model_name,
                    test_prompts,
                    "guidance",
                    guidance_value,
                    model_output_dir,
                    fixed_steps=50
                )
                guidance_results.append(result)

            results["guidance_scale_tests"] = guidance_results

        # Step 4: Test inference steps
        if test_steps:
            print(f"\n[EXP3] Step 4: Testing inference steps...")
            print(f"Testing values: {INFERENCE_STEPS}")

            steps_results = []
            for steps_value in INFERENCE_STEPS:
                result = test_hyperparameter(
                    pipeline,
                    model_name,
                    test_prompts,
                    "steps",
                    steps_value,
                    model_output_dir,
                    fixed_guidance=7.5
                )
                steps_results.append(result)

            results["inference_steps_tests"] = steps_results

        # Step 5: Unload model
        print("\n[EXP3] Step 5: Cleaning up...")
        unload_model(pipeline)
        print_gpu_memory()

        # Step 6: Save results
        print("\n[EXP3] Step 6: Saving results...")
        save_json(results, EXP3_RESULTS_FILE)
        print(f"[EXP3] Results saved to {EXP3_RESULTS_FILE}")

        # Step 7: Print summary
        print_hyperparameter_summary(results)

    print("\n" + "=" * 80)
    print("EXPERIMENT 3 COMPLETE!")
    print("=" * 80)

    return results


def print_hyperparameter_summary(results: dict):
    """
    Print summary of hyperparameter tests.

    Args:
        results: Results dictionary from experiment
    """
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SENSITIVITY SUMMARY")
    print("=" * 80)

    model_name = results.get("model", "Unknown")
    print(f"\nModel: {model_name}")

    # Guidance scale results
    if "guidance_scale_tests" in results:
        print("\n" + "-" * 80)
        print("GUIDANCE SCALE SENSITIVITY")
        print("-" * 80)
        print(f"{'Guidance Scale':<20} {'CLIP Score':<20} {'Avg Time (s)':<20}")
        print("-" * 80)

        for result in results["guidance_scale_tests"]:
            guidance = result.get("guidance_scale", 0)
            clip_score = result.get("clip_score", 0)
            avg_time = result.get("avg_time_seconds", 0)

            print(f"{guidance:<20.1f} {clip_score:<20.4f} {avg_time:<20.2f}")

        print("-" * 80)

    # Inference steps results
    if "inference_steps_tests" in results:
        print("\n" + "-" * 80)
        print("INFERENCE STEPS SENSITIVITY")
        print("-" * 80)
        print(f"{'Num Steps':<20} {'CLIP Score':<20} {'Avg Time (s)':<20}")
        print("-" * 80)

        for result in results["inference_steps_tests"]:
            steps = result.get("num_steps", 0)
            clip_score = result.get("clip_score", 0)
            avg_time = result.get("avg_time_seconds", 0)

            print(f"{steps:<20} {clip_score:<20.4f} {avg_time:<20.2f}")

        print("-" * 80)

    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment 3: Hyperparameter Sensitivity")
    parser.add_argument("--model", type=str, default="sd_v15",
                       choices=["sd_v15", "sd_v21"],
                       help="Which model to test")
    parser.add_argument("--no_guidance", action="store_true",
                       help="Skip guidance scale tests")
    parser.add_argument("--no_steps", action="store_true",
                       help="Skip inference steps tests")
    parser.add_argument("--num_prompts", type=int, default=20,
                       help="Number of test prompts to use")

    args = parser.parse_args()

    # Run experiment
    run_experiment_3(
        model_choice=args.model,
        test_guidance=not args.no_guidance,
        test_steps=not args.no_steps,
        num_test_prompts=args.num_prompts
    )
