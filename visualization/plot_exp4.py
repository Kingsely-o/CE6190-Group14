"""
Visualization tools for Experiment 4 (Ablation Study)
Shows component contributions and performance degradation
"""

import json
from pathlib import Path
import sys
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from config import RESULTS_DIR, FIGURES_DIR


EXP4_DIR = RESULTS_DIR / "exp4_ablation"
EXP4_FIGURES_DIR = FIGURES_DIR / "exp4_ablation"
EXP4_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_results(results_file: Path = None) -> Dict:
    """Load true ablation study results from JSON file."""
    if results_file is None:
        results_file = EXP4_DIR / "exp4_ablation_results.json"

    with open(results_file, 'r') as f:
        return json.load(f)


# ============================================================================
# TEXT CONDITIONING ABLATION VISUALIZATION
# ============================================================================

def plot_text_conditioning_ablation(results: Dict):
    """
    Visualize impact of text conditioning.
    Shows the critical importance of text guidance.
    """
    text_data = results["ablations"].get("text_conditioning", {})
    if not text_data or "variants" not in text_data:
        print("No text conditioning data found")
        return

    variants = text_data["variants"]

    # Extract data
    variant_names = []
    clip_scores = []
    contribution_pcts = []

    baseline_score = variants["full_text"]["avg_clip_score"]

    for name, data in variants.items():
        if "avg_clip_score" not in data:
            continue

        # Clean up variant names for display
        display_name = name.replace("_", " ").title()
        variant_names.append(display_name)
        clip_scores.append(data["avg_clip_score"])

        # Calculate loss from baseline
        if name != "full_text":
            loss_pct = ((baseline_score - data["avg_clip_score"]) / baseline_score) * 100
            contribution_pcts.append(loss_pct)
        else:
            contribution_pcts.append(0)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Absolute CLIP Scores
    ax1 = axes[0]
    colors = ['green' if name == 'Full Text' else 'coral' for name in variant_names]
    bars1 = ax1.bar(variant_names, clip_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('CLIP Score', fontsize=13, fontweight='bold')
    ax1.set_title('Text Conditioning: Impact on Image-Text Alignment',
                 fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(baseline_score, color='green', linestyle='--', alpha=0.6, linewidth=2, label='Baseline')
    ax1.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: Quality Loss from Baseline
    ax2 = axes[1]
    colors2 = ['green' if pct == 0 else 'red' for pct in contribution_pcts]
    bars2 = ax2.bar(variant_names, contribution_pcts, color=colors2, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Quality Loss from Baseline (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Text Conditioning: Performance Degradation',
                 fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(0, color='green', linestyle='-', alpha=0.6, linewidth=2)

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = EXP4_FIGURES_DIR / "text_conditioning_ablation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved text conditioning ablation to {save_path}")
    plt.close()


# ============================================================================
# CFG ABLATION VISUALIZATION
# ============================================================================

def plot_cfg_ablation(results: Dict):
    """
    Visualize impact of Classifier-Free Guidance.
    Shows the importance of CFG for text-image alignment.
    """
    cfg_data = results["ablations"].get("cfg", {})
    if not cfg_data or "variants" not in cfg_data:
        print("No CFG data found")
        return

    variants = cfg_data["variants"]

    # Extract data
    variant_names = []
    guidance_scales = []
    clip_scores = []

    for name, data in variants.items():
        if "avg_clip_score" not in data:
            continue

        display_name = name.replace("_", " ").title()
        variant_names.append(display_name)
        guidance_scales.append(data.get("guidance_scale", 0))
        clip_scores.append(data["avg_clip_score"])

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: CLIP Score vs Guidance Scale
    ax1 = axes[0]
    colors = ['green', 'orange', 'red']
    bars1 = ax1.bar(variant_names, clip_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('CLIP Score', fontsize=13, fontweight='bold')
    ax1.set_title('Classifier-Free Guidance: Impact on Quality',
                 fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels with guidance scale
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}\n(g={guidance_scales[i]})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 2: Quality degradation
    baseline_score = clip_scores[0]  # "with_cfg"
    degradation = [(baseline_score - score) / baseline_score * 100 for score in clip_scores]

    ax2 = axes[1]
    colors2 = ['green' if d == 0 else 'red' for d in degradation]
    bars2 = ax2.bar(variant_names, degradation, color=colors2, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Quality Loss from Baseline (%)', fontsize=13, fontweight='bold')
    ax2.set_title('CFG Contribution to Image Quality',
                 fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(0, color='green', linestyle='-', alpha=0.6, linewidth=2)

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

    # Add CFG contribution text
    cfg_contribution = cfg_data.get("cfg_contribution_pct", 0)
    fig.text(0.5, 0.02, f'CFG Contribution: {cfg_contribution:.1f}%',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    save_path = EXP4_FIGURES_DIR / "cfg_ablation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved CFG ablation to {save_path}")
    plt.close()


# ============================================================================
# VAE ABLATION VISUALIZATION
# ============================================================================

def plot_vae_ablation(results: Dict):
    """
    Visualize impact of different VAE models.
    """
    vae_data = results["ablations"].get("vae", {})
    if not vae_data or "variants" not in vae_data:
        print("No VAE data found")
        return

    variants = vae_data["variants"]

    # Extract data
    variant_names = []
    clip_scores = []

    for name, data in variants.items():
        if "avg_clip_score" not in data:
            continue

        display_name = name.replace("_", " ").title()
        variant_names.append(display_name)
        clip_scores.append(data["avg_clip_score"])

    if not variant_names:
        print("No valid VAE data to plot")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['steelblue', 'coral'][:len(variant_names)]
    bars = ax.bar(variant_names, clip_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('CLIP Score', fontsize=13, fontweight='bold')
    ax.set_title('VAE Comparison: Impact on Image Quality',
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_path = EXP4_FIGURES_DIR / "vae_ablation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved VAE ablation to {save_path}")
    plt.close()


# ============================================================================
# MODEL SIZE ABLATION VISUALIZATION
# ============================================================================

def plot_model_size_ablation(results: Dict):
    """
    Visualize impact of model size/architecture.
    """
    model_data = results["ablations"].get("model_size", {})
    if not model_data or "variants" not in model_data:
        print("No model size data found")
        return

    variants = model_data["variants"]

    # Extract data
    model_names = []
    clip_scores = []
    gen_times = []

    for name, data in variants.items():
        if "avg_clip_score" not in data:
            continue

        display_name = name.replace("_", " ").upper()
        model_names.append(display_name)
        clip_scores.append(data["avg_clip_score"])
        gen_times.append(data.get("avg_time", 0))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: CLIP Score comparison
    ax1 = axes[0]
    bars1 = ax1.bar(model_names, clip_scores, color=['steelblue', 'coral'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('CLIP Score', fontsize=13, fontweight='bold')
    ax1.set_title('Model Comparison: Image Quality',
                 fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 2: Generation time comparison
    ax2 = axes[1]
    bars2 = ax2.bar(model_names, gen_times, color=['steelblue', 'coral'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Average Generation Time (seconds)', fontsize=13, fontweight='bold')
    ax2.set_title('Model Comparison: Inference Speed',
                 fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_path = EXP4_FIGURES_DIR / "model_size_ablation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved model size ablation to {save_path}")
    plt.close()


# ============================================================================
# COMPREHENSIVE COMPONENT CONTRIBUTION CHART
# ============================================================================

def plot_component_contributions(results: Dict):
    """
    Create a comprehensive bar chart showing all component contributions.
    This is the KEY FIGURE for ablation study.
    """
    print("\n[VIZ] Creating comprehensive component contributions chart...")

    contributions = {}

    # Text Conditioning
    if "text_conditioning" in results["ablations"]:
        text_data = results["ablations"]["text_conditioning"]["variants"]
        if "full_text" in text_data and "empty_text" in text_data:
            baseline = text_data["full_text"]["avg_clip_score"]
            ablated = text_data["empty_text"]["avg_clip_score"]
            contribution = ((baseline - ablated) / baseline) * 100
            contributions["Text\nConditioning"] = contribution

    # CFG
    if "cfg" in results["ablations"]:
        cfg_contribution = results["ablations"]["cfg"].get("cfg_contribution_pct", 0)
        contributions["Classifier-Free\nGuidance"] = cfg_contribution

    # Model Size (if v2.1 is better or worse)
    if "model_size" in results["ablations"]:
        model_data = results["ablations"]["model_size"]["variants"]
        if "sd_v15" in model_data and "sd_v21" in model_data:
            v15_score = model_data["sd_v15"]["avg_clip_score"]
            v21_score = model_data["sd_v21"]["avg_clip_score"]
            diff = ((v21_score - v15_score) / v15_score) * 100
            contributions["Model\nArchitecture"] = abs(diff)

    # VAE (if improved VAE exists)
    if "vae" in results["ablations"]:
        vae_data = results["ablations"]["vae"]["variants"]
        if "original_vae" in vae_data and "improved_vae" in vae_data:
            if "avg_clip_score" in vae_data["improved_vae"]:
                orig_score = vae_data["original_vae"]["avg_clip_score"]
                improved_score = vae_data["improved_vae"]["avg_clip_score"]
                diff = ((improved_score - orig_score) / orig_score) * 100
                contributions["VAE\nQuality"] = abs(diff)

    if not contributions:
        print("No contribution data available")
        return

    # Sort by contribution (descending)
    sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    component_names = [item[0] for item in sorted_contributions]
    contribution_values = [item[1] for item in sorted_contributions]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color gradient based on importance
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(component_names)))

    bars = ax.barh(component_names, contribution_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xlabel('Contribution to Image Quality (%)', fontsize=14, fontweight='bold')
    ax.set_title('Component Importance Ranking\n(Performance Degradation When Removed/Disabled)',
                fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f' {width:.1f}%',
               ha='left', va='center', fontsize=12, fontweight='bold')

    # Add importance stars
    for i, (name, value) in enumerate(sorted_contributions):
        if value > 50:
            stars = "⭐⭐⭐⭐⭐"
        elif value > 20:
            stars = "⭐⭐⭐⭐"
        elif value > 10:
            stars = "⭐⭐⭐"
        elif value > 5:
            stars = "⭐⭐"
        else:
            stars = "⭐"

        ax.text(-2, i, stars, ha='right', va='center', fontsize=10)

    plt.tight_layout()
    save_path = EXP4_FIGURES_DIR / "component_contributions.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved component contributions to {save_path}")
    print("\n  ⭐⭐⭐⭐⭐ This is the KEY FIGURE for your ablation study! ⭐⭐⭐⭐⭐\n")
    plt.close()


# ============================================================================
# COMPREHENSIVE SUMMARY DASHBOARD
# ============================================================================

def create_summary_dashboard(results: Dict):
    """
    Create a comprehensive dashboard showing all ablation results.
    """
    print("\n[VIZ] Creating summary dashboard...")

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Text Conditioning
    if "text_conditioning" in results["ablations"]:
        ax1 = fig.add_subplot(gs[0, 0])
        text_data = results["ablations"]["text_conditioning"]["variants"]
        names = [n.replace("_", "\n") for n in text_data.keys() if "avg_clip_score" in text_data[n]]
        scores = [d["avg_clip_score"] for d in text_data.values() if "avg_clip_score" in d]
        colors = ['green' if 'full' in n else 'coral' for n in names]
        ax1.bar(names, scores, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('CLIP Score', fontsize=10, fontweight='bold')
        ax1.set_title('Text Conditioning Ablation', fontsize=11, fontweight='bold')
        ax1.tick_params(axis='x', labelsize=8)
        ax1.grid(axis='y', alpha=0.3)

    # 2. CFG
    if "cfg" in results["ablations"]:
        ax2 = fig.add_subplot(gs[0, 1])
        cfg_data = results["ablations"]["cfg"]["variants"]
        names = [n.replace("_", "\n") for n in cfg_data.keys() if "avg_clip_score" in cfg_data[n]]
        scores = [d["avg_clip_score"] for d in cfg_data.values() if "avg_clip_score" in d]
        colors = ['green', 'orange', 'red'][:len(names)]
        ax2.bar(names, scores, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('CLIP Score', fontsize=10, fontweight='bold')
        ax2.set_title('Classifier-Free Guidance Ablation', fontsize=11, fontweight='bold')
        ax2.tick_params(axis='x', labelsize=8)
        ax2.grid(axis='y', alpha=0.3)

    # 3. Model Size
    if "model_size" in results["ablations"]:
        ax3 = fig.add_subplot(gs[1, 0])
        model_data = results["ablations"]["model_size"]["variants"]
        names = [n.upper() for n in model_data.keys() if "avg_clip_score" in model_data[n]]
        scores = [d["avg_clip_score"] for d in model_data.values() if "avg_clip_score" in d]
        ax3.bar(names, scores, color=['steelblue', 'coral'][:len(names)], alpha=0.7, edgecolor='black')
        ax3.set_ylabel('CLIP Score', fontsize=10, fontweight='bold')
        ax3.set_title('Model Size Comparison', fontsize=11, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)

    # 4. Latent Resolution
    if "latent_resolution" in results["ablations"]:
        ax4 = fig.add_subplot(gs[1, 1])
        res_data = results["ablations"]["latent_resolution"]["variants"]
        resolutions = []
        scores = []
        for name, data in res_data.items():
            if "avg_clip_score" in data:
                res = data.get("resolution", 0)
                resolutions.append(res)
                scores.append(data["avg_clip_score"])

        if resolutions:
            ax4.plot(resolutions, scores, 'o-', linewidth=2, markersize=8, color='steelblue')
            ax4.set_xlabel('Resolution (pixels)', fontsize=10, fontweight='bold')
            ax4.set_ylabel('CLIP Score', fontsize=10, fontweight='bold')
            ax4.set_title('Latent Resolution Impact', fontsize=11, fontweight='bold')
            ax4.grid(alpha=0.3)
            ax4.axvline(512, color='red', linestyle='--', alpha=0.5, label='Native (512)')
            ax4.legend()

    # 5 & 6: Component Contributions (large panel)
    ax5 = fig.add_subplot(gs[2, :])

    # Gather all contributions
    contributions = {}

    if "text_conditioning" in results["ablations"]:
        text_data = results["ablations"]["text_conditioning"]["variants"]
        if "full_text" in text_data and "empty_text" in text_data:
            baseline = text_data["full_text"]["avg_clip_score"]
            ablated = text_data["empty_text"]["avg_clip_score"]
            contribution = ((baseline - ablated) / baseline) * 100
            contributions["Text Conditioning"] = contribution

    if "cfg" in results["ablations"]:
        cfg_contribution = results["ablations"]["cfg"].get("cfg_contribution_pct", 0)
        contributions["CFG"] = cfg_contribution

    if contributions:
        sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        names = [item[0] for item in sorted_contributions]
        values = [item[1] for item in sorted_contributions]

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
        ax5.barh(names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax5.set_xlabel('Contribution (%)', fontsize=12, fontweight='bold')
        ax5.set_title('Component Importance Ranking', fontsize=13, fontweight='bold')
        ax5.grid(axis='x', alpha=0.3)

        for i, bar in enumerate(ax5.patches):
            width = bar.get_width()
            ax5.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {width:.1f}%',
                   ha='left', va='center', fontsize=11, fontweight='bold')

    plt.suptitle('True Ablation Study: Comprehensive Summary', fontsize=18, fontweight='bold', y=0.98)

    save_path = EXP4_FIGURES_DIR / "summary_dashboard.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary dashboard to {save_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_exp4_plots(results_file: Path = None):
    """
    Generate all visualizations for Experiment 4 (Ablation Study).
    """
    print("\n" + "="*80)
    print("GENERATING EXPERIMENT 4 VISUALIZATIONS")
    print("="*80)

    results = load_results(results_file)

    if "text_conditioning" in results["ablations"]:
        plot_text_conditioning_ablation(results)

    if "cfg" in results["ablations"]:
        plot_cfg_ablation(results)

    if "vae" in results["ablations"]:
        plot_vae_ablation(results)

    if "model_size" in results["ablations"]:
        plot_model_size_ablation(results)

    # KEY FIGURE: Component Contributions
    plot_component_contributions(results)

    # Summary Dashboard
    create_summary_dashboard(results)

    print("\n" + "="*80)
    print(f"ALL VISUALIZATIONS SAVED TO: {EXP4_FIGURES_DIR}")
    print("="*80)
    print("\nKEY FIGURES FOR YOUR REPORT:")
    print("  1. component_contributions.png  ⭐⭐⭐⭐⭐ (MOST IMPORTANT)")
    print("  2. text_conditioning_ablation.png")
    print("  3. cfg_ablation.png")
    print("  4. summary_dashboard.png")
    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Experiment 4 (Ablation Study) Results")
    parser.add_argument("--results", type=str, default=None,
                       help="Path to exp4 ablation results JSON file")

    args = parser.parse_args()

    results_file = Path(args.results) if args.results else None
    generate_exp4_plots(results_file)
