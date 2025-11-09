"""
Visualization utilities for plotting experiment results
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from config import FIGURES_DIR, FIGURE_DPI, FIGURE_SIZE
from utils.helpers import load_json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = FIGURE_DPI
plt.rcParams['savefig.dpi'] = FIGURE_DPI


def plot_metrics_comparison(exp1_results: Dict, save_path: Optional[Path] = None):
    """
    Create a bar chart comparing metrics between models.

    Args:
        exp1_results: Results from Experiment 1
        save_path: Path to save the figure (default: FIGURES_DIR/metrics_comparison.png)
    """
    print("\n[PLOT] Creating metrics comparison chart...")

    if save_path is None:
        save_path = FIGURES_DIR / "metrics_comparison.png"

    models = exp1_results.get("models", {})
    sd_v15 = models.get("sd_v15", {})
    sd_v21 = models.get("sd_v21", {})

    # Extract metrics (note: FID is reversed for display - lower is better becomes higher is better)
    metrics = {
        "FID\n(lower=better)": [
            sd_v15.get("fid", 0),
            sd_v21.get("fid", 0)
        ],
        "CLIP Score\n(higher=better)": [
            sd_v15.get("clip_score", 0),
            sd_v21.get("clip_score", 0)
        ],
        "Inception Score\n(higher=better)": [
            sd_v15.get("inception_score_mean", 0),
            sd_v21.get("inception_score_mean", 0)
        ]
    }

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    model_names = ["SD v1.5", "SD v2.1"]
    colors = ['#3498db', '#e74c3c']

    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx]

        # Create bars
        x = np.arange(len(model_names))
        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black')

        # Customize
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}',
                       ha='center', va='bottom', fontsize=9)

    plt.suptitle("Stable Diffusion Model Comparison - Overall Metrics",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=FIGURE_DPI)
    plt.close()

    print(f"[PLOT] Saved metrics comparison to {save_path}")


def plot_category_comparison(exp2_results: Dict, save_path: Optional[Path] = None):
    """
    Create a grouped bar chart comparing CLIP scores across categories.

    Args:
        exp2_results: Results from Experiment 2
        save_path: Path to save the figure
    """
    print("\n[PLOT] Creating category comparison chart...")

    if save_path is None:
        save_path = FIGURES_DIR / "category_comparison.png"

    models = exp2_results.get("models", {})
    categories = exp2_results.get("categories", [])

    # Extract category scores
    sd_v15_scores = []
    sd_v21_scores = []

    v15_data = models.get("sd_v15", {}).get("evaluation", {}).get("category_scores", {})
    v21_data = models.get("sd_v21", {}).get("evaluation", {}).get("category_scores", {})

    for category in categories:
        v15_score = v15_data.get(category, {}).get("average_clip_score", 0)
        v21_score = v21_data.get(category, {}).get("average_clip_score", 0)
        sd_v15_scores.append(v15_score)
        sd_v21_scores.append(v21_score)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, sd_v15_scores, width, label='SD v1.5',
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, sd_v21_scores, width, label='SD v2.1',
                   color='#e74c3c', alpha=0.8, edgecolor='black')

    # Customize
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('CLIP Score', fontsize=12, fontweight='bold')
    ax.set_title('Category-wise CLIP Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories],
                       rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)

    add_value_labels(bars1)
    add_value_labels(bars2)

    plt.tight_layout()

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=FIGURE_DPI)
    plt.close()

    print(f"[PLOT] Saved category comparison to {save_path}")


def plot_hyperparameter_sensitivity(exp3_results: Dict, save_path: Optional[Path] = None):
    """
    Create charts showing hyperparameter sensitivity.

    Args:
        exp3_results: Results from Experiment 3
        save_path: Path to save the figure
    """
    print("\n[PLOT] Creating hyperparameter sensitivity charts...")

    if save_path is None:
        save_path = FIGURES_DIR / "hyperparameter_sensitivity.png"

    has_guidance = "guidance_scale_tests" in exp3_results
    has_steps = "inference_steps_tests" in exp3_results

    if not has_guidance and not has_steps:
        print("[PLOT] No hyperparameter test results found")
        return

    # Determine layout
    num_plots = sum([has_guidance, has_steps])
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))

    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot guidance scale sensitivity
    if has_guidance:
        guidance_tests = exp3_results["guidance_scale_tests"]

        guidance_values = [t["guidance_scale"] for t in guidance_tests]
        clip_scores = [t["clip_score"] for t in guidance_tests]
        times = [t["avg_time_seconds"] for t in guidance_tests]

        ax = axes[plot_idx]
        ax2 = ax.twinx()

        # Plot CLIP scores
        line1 = ax.plot(guidance_values, clip_scores, 'o-', color='#3498db',
                       linewidth=2, markersize=8, label='CLIP Score')
        ax.set_xlabel('Guidance Scale', fontsize=11, fontweight='bold')
        ax.set_ylabel('CLIP Score', fontsize=11, fontweight='bold', color='#3498db')
        ax.tick_params(axis='y', labelcolor='#3498db')
        ax.grid(True, alpha=0.3)

        # Plot times
        line2 = ax2.plot(guidance_values, times, 's--', color='#e74c3c',
                        linewidth=2, markersize=8, label='Gen Time')
        ax2.set_ylabel('Generation Time (s)', fontsize=11, fontweight='bold', color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')

        ax.set_title('Guidance Scale Sensitivity', fontsize=12, fontweight='bold')

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')

        plot_idx += 1

    # Plot inference steps sensitivity
    if has_steps:
        steps_tests = exp3_results["inference_steps_tests"]

        steps_values = [t["num_steps"] for t in steps_tests]
        clip_scores = [t["clip_score"] for t in steps_tests]
        times = [t["avg_time_seconds"] for t in steps_tests]

        ax = axes[plot_idx]
        ax2 = ax.twinx()

        # Plot CLIP scores
        line1 = ax.plot(steps_values, clip_scores, 'o-', color='#3498db',
                       linewidth=2, markersize=8, label='CLIP Score')
        ax.set_xlabel('Inference Steps', fontsize=11, fontweight='bold')
        ax.set_ylabel('CLIP Score', fontsize=11, fontweight='bold', color='#3498db')
        ax.tick_params(axis='y', labelcolor='#3498db')
        ax.grid(True, alpha=0.3)

        # Plot times
        line2 = ax2.plot(steps_values, times, 's--', color='#e74c3c',
                        linewidth=2, markersize=8, label='Gen Time')
        ax2.set_ylabel('Generation Time (s)', fontsize=11, fontweight='bold', color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')

        ax.set_title('Inference Steps Sensitivity', fontsize=12, fontweight='bold')

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')

    model_name = exp3_results.get("model", "Model")
    plt.suptitle(f'Hyperparameter Sensitivity Analysis - {model_name}',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=FIGURE_DPI)
    plt.close()

    print(f"[PLOT] Saved hyperparameter sensitivity to {save_path}")


def create_image_grid(
    image_paths: List[Path],
    labels: List[str],
    grid_size: tuple = (2, 2),
    save_path: Optional[Path] = None
):
    """
    Create a grid of images with labels.

    Args:
        image_paths: List of image file paths
        labels: List of labels for each image
        grid_size: (rows, cols)
        save_path: Path to save the grid
    """
    print("\n[PLOT] Creating image grid...")

    if save_path is None:
        save_path = FIGURES_DIR / "image_grid.png"

    rows, cols = grid_size
    num_images = rows * cols

    if len(image_paths) < num_images:
        print(f"[WARNING] Only {len(image_paths)} images provided, need {num_images}")
        num_images = len(image_paths)

    # Load images
    images = []
    for path in image_paths[:num_images]:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")

    if not images:
        print("[ERROR] No images loaded")
        return

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for idx, (img, label) in enumerate(zip(images, labels[:num_images])):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        ax.imshow(img)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.axis('off')

    # Hide empty subplots
    for idx in range(len(images), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.tight_layout()

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=FIGURE_DPI)
    plt.close()

    print(f"[PLOT] Saved image grid to {save_path}")


def create_side_by_side_comparison(
    v15_images: List[Path],
    v21_images: List[Path],
    prompts: List[str],
    num_examples: int = 4,
    save_path: Optional[Path] = None
):
    """
    Create side-by-side comparison of generated images.

    Args:
        v15_images: List of SD v1.5 image paths
        v21_images: List of SD v2.1 image paths
        prompts: List of prompts used
        num_examples: Number of examples to show
        save_path: Path to save the comparison
    """
    print("\n[PLOT] Creating side-by-side comparison...")

    if save_path is None:
        save_path = FIGURES_DIR / "side_by_side_comparison.png"

    num_examples = min(num_examples, len(v15_images), len(v21_images))

    # Create figure
    fig, axes = plt.subplots(num_examples, 2, figsize=(10, num_examples * 5))

    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_examples):
        # Load images
        try:
            img_v15 = Image.open(v15_images[idx]).convert("RGB")
            img_v21 = Image.open(v21_images[idx]).convert("RGB")

            # Display SD v1.5
            axes[idx, 0].imshow(img_v15)
            axes[idx, 0].set_title(f"SD v1.5", fontsize=11, fontweight='bold')
            axes[idx, 0].axis('off')

            # Display SD v2.1
            axes[idx, 1].imshow(img_v21)
            axes[idx, 1].set_title(f"SD v2.1", fontsize=11, fontweight='bold')
            axes[idx, 1].axis('off')

            # Add prompt as row label
            prompt = prompts[idx] if idx < len(prompts) else ""
            if len(prompt) > 60:
                prompt = prompt[:57] + "..."

            fig.text(0.5, 1 - (idx + 0.5) / num_examples, prompt,
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        except Exception as e:
            print(f"[ERROR] Failed to load images for example {idx}: {e}")

    plt.suptitle("Side-by-Side Model Comparison", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=FIGURE_DPI)
    plt.close()

    print(f"[PLOT] Saved side-by-side comparison to {save_path}")


def generate_all_plots():
    """Generate all plots from saved results."""
    print("\n" + "=" * 80)
    print("GENERATING ALL PLOTS")
    print("=" * 80)

    from config import EXP1_RESULTS_FILE, EXP2_RESULTS_FILE, EXP3_RESULTS_FILE, RESULTS_DIR

    # Plot Experiment 1 results
    if EXP1_RESULTS_FILE.exists():
        print("\n[PLOT] Processing Experiment 1 results...")
        exp1_results = load_json(EXP1_RESULTS_FILE)
        plot_metrics_comparison(exp1_results)
    else:
        print("[WARNING] Experiment 1 results not found")

    # Plot Experiment 2 results
    if EXP2_RESULTS_FILE.exists():
        print("\n[PLOT] Processing Experiment 2 results...")
        exp2_results = load_json(EXP2_RESULTS_FILE)
        plot_category_comparison(exp2_results)
    else:
        print("[WARNING] Experiment 2 results not found")

    # Plot Experiment 3 results
    if EXP3_RESULTS_FILE.exists():
        print("\n[PLOT] Processing Experiment 3 results...")
        exp3_results = load_json(EXP3_RESULTS_FILE)
        plot_hyperparameter_sensitivity(exp3_results)
    else:
        print("[INFO] Experiment 3 results not found (optional)")

    # Plot Experiment 4 results (Ablation Study)
    exp4_results_file = RESULTS_DIR / "exp4_ablation" / "exp4_results.json"
    if exp4_results_file.exists():
        print("\n[PLOT] Processing Experiment 4 results (Ablation Study)...")
        try:
            from visualization.plot_exp4 import generate_exp4_plots
            generate_exp4_plots(exp4_results_file)
        except Exception as e:
            print(f"[ERROR] Failed to generate exp4 plots: {e}")
    else:
        print("[INFO] Experiment 4 results not found (optional)")

    print("\n" + "=" * 80)
    print("ALL PLOTS GENERATED")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_plots()
