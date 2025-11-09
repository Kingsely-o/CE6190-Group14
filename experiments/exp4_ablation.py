"""
Experiment 4: Ablation Study of Key Components
Systematically remove or disable components to understand their contribution

This is a TRUE ablation study where we test "with vs without" each component.
"""

import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    SD_V15_MODEL_ID,
    RESULTS_DIR,
    DEVICE,
    DEFAULT_SEED,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE
)
from models.model_loader import load_sd_v15, load_sd_v21, generate_image, unload_model
from evaluation.metrics import CLIPScorer
from utils.helpers import save_json, save_image, Timer, set_random_seeds, print_gpu_memory


# ============================================================================
# SETUP
# ============================================================================

EXP4_DIR = RESULTS_DIR / "exp4_ablation"
EXP4_RESULTS_FILE = EXP4_DIR / "exp4_results.json"
EXP4_DIR.mkdir(parents=True, exist_ok=True)

# Test prompts for ablation study
TEST_PROMPTS = [
    "a professional photograph of a cat sitting on a wooden table",
    "a beautiful sunset over the ocean with orange and pink clouds",
    "a futuristic cityscape with flying cars and neon lights",
    "a close-up portrait of a person with curly hair",
    "a bowl of fresh fruit on a kitchen counter",
]


# ============================================================================
# ABLATION 1: TEXT CONDITIONING
# ============================================================================

def ablate_text_conditioning() -> dict:
    """
    Test the importance of text conditioning.

    Variants:
    1. Full Text (baseline) - with complete prompts
    2. Empty Text - unconditional generation (no text)
    3. Partial Text - only first 3 words

    This shows the contribution of text conditioning to quality.
    """
    print("\n" + "="*80)
    print("ABLATION 1: TEXT CONDITIONING")
    print("Testing: Full Text vs Empty Text vs Partial Text")
    print("="*80)

    results = {
        "ablation": "text_conditioning",
        "variants": {}
    }

    # Load model
    print("\n[SETUP] Loading model...")
    pipeline = load_sd_v15(device=DEVICE)
    print_gpu_memory()

    # Variant 1: FULL TEXT (Baseline)
    print("\n[VARIANT 1/3] Full Text Conditioning (Baseline)")
    full_results = generate_and_evaluate(
        pipeline, TEST_PROMPTS, TEST_PROMPTS,
        EXP4_DIR / "text_full", "Full Text"
    )
    results["variants"]["full_text"] = full_results

    # Variant 2: EMPTY TEXT (No conditioning)
    print("\n[VARIANT 2/3] Empty Text (Unconditional)")
    empty_prompts = [""] * len(TEST_PROMPTS)
    empty_results = generate_and_evaluate(
        pipeline, empty_prompts, TEST_PROMPTS,
        EXP4_DIR / "text_empty", "Empty Text"
    )
    results["variants"]["empty_text"] = empty_results

    # Variant 3: PARTIAL TEXT
    print("\n[VARIANT 3/3] Partial Text (First 3 words)")
    partial_prompts = [" ".join(p.split()[:3]) for p in TEST_PROMPTS]
    partial_results = generate_and_evaluate(
        pipeline, partial_prompts, TEST_PROMPTS,
        EXP4_DIR / "text_partial", "Partial Text"
    )
    results["variants"]["partial_text"] = partial_results

    # Calculate contributions
    baseline_score = full_results["avg_clip_score"]
    for variant_name, variant_data in results["variants"].items():
        if variant_name != "full_text":
            loss = baseline_score - variant_data["avg_clip_score"]
            loss_pct = (loss / baseline_score) * 100
            variant_data["quality_loss"] = loss
            variant_data["quality_loss_pct"] = loss_pct

    # Cleanup
    unload_model(pipeline)
    print_gpu_memory()

    # Print summary
    print("\n" + "-"*80)
    print("TEXT CONDITIONING SUMMARY:")
    print("-"*80)
    for name, data in results["variants"].items():
        score = data["avg_clip_score"]
        loss_pct = data.get("quality_loss_pct", 0)
        print(f"  {name:20s}: CLIP={score:.4f}  Loss: {loss_pct:+.1f}%")
    print("-"*80)

    return results


# ============================================================================
# ABLATION 2: CLASSIFIER-FREE GUIDANCE (CFG)
# ============================================================================

def ablate_cfg() -> dict:
    """
    Test the importance of Classifier-Free Guidance.

    Variants:
    1. With CFG (guidance=7.5) - baseline
    2. Without CFG (guidance=1.0) - no guidance amplification
    3. Unconditional (guidance=0.0) - ignore text completely

    This shows CFG's contribution to text-image alignment.
    """
    print("\n" + "="*80)
    print("ABLATION 2: CLASSIFIER-FREE GUIDANCE (CFG)")
    print("Testing: With CFG vs Without CFG vs Unconditional")
    print("="*80)

    results = {
        "ablation": "classifier_free_guidance",
        "variants": {}
    }

    # Load model
    print("\n[SETUP] Loading model...")
    pipeline = load_sd_v15(device=DEVICE)

    guidance_configs = [
        (7.5, "with_cfg", "With CFG"),
        (1.0, "without_cfg", "Without CFG"),
        (0.0, "unconditional", "Unconditional")
    ]

    for idx, (guidance, variant_id, variant_name) in enumerate(guidance_configs, 1):
        print(f"\n[VARIANT {idx}/3] {variant_name} (guidance={guidance})")
        variant_results = generate_with_guidance(
            pipeline, TEST_PROMPTS, guidance,
            EXP4_DIR / f"cfg_{variant_id}", variant_name
        )
        results["variants"][variant_id] = variant_results

    # Calculate CFG contribution
    baseline_score = results["variants"]["with_cfg"]["avg_clip_score"]
    without_score = results["variants"]["without_cfg"]["avg_clip_score"]
    cfg_contribution = baseline_score - without_score
    cfg_contribution_pct = (cfg_contribution / baseline_score) * 100

    results["cfg_contribution"] = cfg_contribution
    results["cfg_contribution_pct"] = cfg_contribution_pct

    # Cleanup
    unload_model(pipeline)

    # Print summary
    print("\n" + "-"*80)
    print("CFG SUMMARY:")
    print("-"*80)
    for name, data in results["variants"].items():
        score = data["avg_clip_score"]
        guidance = data["guidance_scale"]
        print(f"  {name:20s}: CLIP={score:.4f}  (guidance={guidance})")
    print(f"\n  CFG Contribution: {cfg_contribution_pct:.1f}%")
    print("-"*80)

    return results


# ============================================================================
# ABLATION 3: MODEL SIZE/ARCHITECTURE
# ============================================================================

def ablate_model_size() -> dict:
    """
    Compare different model architectures.

    Variants:
    1. SD v1.5 (baseline)
    2. SD v2.1 (different architecture)

    This shows the impact of model capacity and architecture.
    """
    print("\n" + "="*80)
    print("ABLATION 3: MODEL SIZE & ARCHITECTURE")
    print("Testing: SD v1.5 vs SD v2.1")
    print("="*80)

    results = {
        "ablation": "model_size",
        "variants": {}
    }

    # Variant 1: SD v1.5
    print("\n[VARIANT 1/2] Stable Diffusion v1.5")
    pipeline_v15 = load_sd_v15(device=DEVICE)
    v15_results = generate_and_evaluate_simple(
        pipeline_v15, TEST_PROMPTS,
        EXP4_DIR / "model_v15", "SD v1.5"
    )
    results["variants"]["sd_v15"] = v15_results
    unload_model(pipeline_v15)

    # Variant 2: SD v2.1
    print("\n[VARIANT 2/2] Stable Diffusion v2.1")
    pipeline_v21 = load_sd_v21(device=DEVICE)
    v21_results = generate_and_evaluate_simple(
        pipeline_v21, TEST_PROMPTS,
        EXP4_DIR / "model_v21", "SD v2.1"
    )
    results["variants"]["sd_v21"] = v21_results
    unload_model(pipeline_v21)

    # Compare
    v15_score = v15_results["avg_clip_score"]
    v21_score = v21_results["avg_clip_score"]
    difference = v21_score - v15_score
    difference_pct = (difference / v15_score) * 100

    results["architecture_difference"] = difference
    results["architecture_difference_pct"] = difference_pct

    # Print summary
    print("\n" + "-"*80)
    print("MODEL SIZE SUMMARY:")
    print("-"*80)
    print(f"  SD v1.5: CLIP={v15_score:.4f}  Time={v15_results['avg_time']:.2f}s")
    print(f"  SD v2.1: CLIP={v21_score:.4f}  Time={v21_results['avg_time']:.2f}s")
    print(f"  Difference: {difference_pct:+.1f}%")
    print("-"*80)

    return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_and_evaluate(
    pipeline,
    generation_prompts: list,
    evaluation_prompts: list,
    output_dir: Path,
    variant_name: str
) -> dict:
    """
    Generate images with generation_prompts and evaluate against evaluation_prompts.

    This allows testing unconditional/partial text while evaluating with full prompts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    gen_times = []

    set_random_seeds(DEFAULT_SEED)

    for idx, gen_prompt in enumerate(generation_prompts):
        start_time = time.time()

        image = generate_image(
            pipeline,
            prompt=gen_prompt,
            num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            seed=DEFAULT_SEED + idx
        )

        elapsed = time.time() - start_time

        img_path = output_dir / f"img_{idx:03d}.png"
        save_image(image, img_path)

        images.append(image)
        gen_times.append(elapsed)

        print(f"    Generated {idx+1}/{len(generation_prompts)}: {elapsed:.2f}s")

    # Evaluate with target prompts
    clip_scorer = CLIPScorer(device=DEVICE)
    clip_scores = clip_scorer.calculate_clip_scores_list(images, evaluation_prompts)

    results = {
        "variant_name": variant_name,
        "num_images": len(images),
        "clip_scores": clip_scores,
        "avg_clip_score": sum(clip_scores) / len(clip_scores),
        "avg_time": sum(gen_times) / len(gen_times)
    }

    print(f"  → {variant_name}: CLIP={results['avg_clip_score']:.4f}, Time={results['avg_time']:.2f}s")

    return results


def generate_with_guidance(
    pipeline,
    prompts: list,
    guidance_scale: float,
    output_dir: Path,
    variant_name: str
) -> dict:
    """Generate with specific guidance scale."""
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    gen_times = []

    set_random_seeds(DEFAULT_SEED)

    for idx, prompt in enumerate(prompts):
        start_time = time.time()

        image = generate_image(
            pipeline,
            prompt=prompt,
            num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
            guidance_scale=guidance_scale,
            seed=DEFAULT_SEED + idx
        )

        elapsed = time.time() - start_time

        img_path = output_dir / f"img_{idx:03d}.png"
        save_image(image, img_path)

        images.append(image)
        gen_times.append(elapsed)

        print(f"    Generated {idx+1}/{len(prompts)}: {elapsed:.2f}s")

    # Evaluate
    clip_scorer = CLIPScorer(device=DEVICE)
    clip_scores = clip_scorer.calculate_clip_scores_list(images, prompts)

    results = {
        "variant_name": variant_name,
        "guidance_scale": guidance_scale,
        "num_images": len(images),
        "clip_scores": clip_scores,
        "avg_clip_score": sum(clip_scores) / len(clip_scores),
        "avg_time": sum(gen_times) / len(gen_times)
    }

    print(f"  → {variant_name}: CLIP={results['avg_clip_score']:.4f}")

    return results


def generate_and_evaluate_simple(
    pipeline,
    prompts: list,
    output_dir: Path,
    variant_name: str
) -> dict:
    """Simple generation and evaluation."""
    return generate_and_evaluate(pipeline, prompts, prompts, output_dir, variant_name)


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_experiment_4(
    run_text: bool = True,
    run_cfg: bool = True,
    run_model: bool = True
):
    """
    Run Experiment 4: Ablation Study.

    Args:
        run_text: Run text conditioning ablation
        run_cfg: Run classifier-free guidance ablation
        run_model: Run model size ablation

    Returns:
        Results dictionary
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: ABLATION STUDY")
    print("Systematically testing component contributions")
    print("="*80)

    all_results = {
        "experiment": "exp4_ablation",
        "description": "True ablation study of key components",
        "test_prompts": TEST_PROMPTS,
        "ablations": {}
    }

    with Timer("Experiment 4"):
        if run_text:
            all_results["ablations"]["text_conditioning"] = ablate_text_conditioning()

        if run_cfg:
            all_results["ablations"]["cfg"] = ablate_cfg()

        if run_model:
            all_results["ablations"]["model_size"] = ablate_model_size()

    # Save results
    print("\n[SAVING] Saving results...")
    save_json(all_results, EXP4_RESULTS_FILE)
    print(f"  Saved to: {EXP4_RESULTS_FILE}")

    # Print comprehensive summary
    print_experiment_summary(all_results)

    print("\n" + "="*80)
    print("EXPERIMENT 4 COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {EXP4_DIR}")
    print(f"JSON results: {EXP4_RESULTS_FILE}")
    print("\nNext steps:")
    print("  1. Run visualization: python visualization/plot_results.py")
    print("  2. Check generated images in results/exp4_ablation/")
    print("  3. Use results for Section 3.2 (Ablation Study) of your report")

    return all_results


def print_experiment_summary(results: dict):
    """Print comprehensive summary of ablation study."""
    print("\n" + "="*80)
    print("EXPERIMENT 4: ABLATION STUDY SUMMARY")
    print("="*80)

    # Collect all contributions
    contributions = []

    # Text Conditioning
    if "text_conditioning" in results["ablations"]:
        text_data = results["ablations"]["text_conditioning"]["variants"]
        if "full_text" in text_data and "empty_text" in text_data:
            baseline = text_data["full_text"]["avg_clip_score"]
            ablated = text_data["empty_text"]["avg_clip_score"]
            contribution = ((baseline - ablated) / baseline) * 100
            contributions.append(("Text Conditioning", contribution))

            print("\n1. TEXT CONDITIONING ABLATION:")
            print("-" * 80)
            for name, data in text_data.items():
                score = data["avg_clip_score"]
                loss = data.get("quality_loss_pct", 0)
                if loss == 0:
                    print(f"  {name:25s}: {score:.4f} (baseline)")
                else:
                    print(f"  {name:25s}: {score:.4f} ({loss:+.1f}% loss)")
            print(f"\n  → Text Conditioning Contribution: {contribution:.1f}%")

    # CFG
    if "cfg" in results["ablations"]:
        cfg_contribution = results["ablations"]["cfg"].get("cfg_contribution_pct", 0)
        contributions.append(("Classifier-Free Guidance", cfg_contribution))

        print("\n2. CLASSIFIER-FREE GUIDANCE ABLATION:")
        print("-" * 80)
        cfg_data = results["ablations"]["cfg"]["variants"]
        for name, data in cfg_data.items():
            score = data["avg_clip_score"]
            guidance = data["guidance_scale"]
            print(f"  {name:25s}: {score:.4f} (guidance={guidance})")
        print(f"\n  → CFG Contribution: {cfg_contribution:.1f}%")

    # Model Size
    if "model_size" in results["ablations"]:
        model_data = results["ablations"]["model_size"]["variants"]
        diff_pct = results["ablations"]["model_size"].get("architecture_difference_pct", 0)
        contributions.append(("Model Architecture", abs(diff_pct)))

        print("\n3. MODEL SIZE/ARCHITECTURE ABLATION:")
        print("-" * 80)
        for name, data in model_data.items():
            score = data["avg_clip_score"]
            time_val = data["avg_time"]
            print(f"  {name:25s}: {score:.4f} (time: {time_val:.2f}s)")
        print(f"\n  → Architecture Difference: {diff_pct:+.1f}%")

    # Component Importance Ranking
    if contributions:
        contributions.sort(key=lambda x: x[1], reverse=True)

        print("\n" + "="*80)
        print("COMPONENT IMPORTANCE RANKING")
        print("="*80)

        for rank, (component, contribution) in enumerate(contributions, 1):
            stars = "⭐" * min(5, int(contribution / 10))
            print(f"  {rank}. {component:30s}: {contribution:6.1f}%  {stars}")

        print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment 4: Ablation Study")
    parser.add_argument("--text", action="store_true", help="Run text conditioning ablation only")
    parser.add_argument("--cfg", action="store_true", help="Run CFG ablation only")
    parser.add_argument("--model", action="store_true", help="Run model size ablation only")

    args = parser.parse_args()

    # If no specific ablation selected, run all
    run_all = not (args.text or args.cfg or args.model)

    run_experiment_4(
        run_text=run_all or args.text,
        run_cfg=run_all or args.cfg,
        run_model=run_all or args.model
    )
