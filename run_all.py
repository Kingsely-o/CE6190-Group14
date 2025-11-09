"""
Main script to run all experiments and generate visualizations
"""

import argparse
from pathlib import Path
import sys

from config import print_config, COCO_NUM_SAMPLES
from utils.helpers import Timer

# Import experiment modules
from experiments.exp1_baseline import run_experiment_1
from experiments.exp2_categories import run_experiment_2
from experiments.exp3_hyperparams import run_experiment_3
from experiments.exp4_ablation import run_experiment_4
from visualization.plot_results import generate_all_plots


def run_all_experiments(
    run_exp1: bool = True,
    run_exp2: bool = True,
    run_exp3: bool = False,
    run_exp4: bool = False,
    generate_plots: bool = True,
    num_samples: int = None,
    skip_generation: bool = False
):
    """
    Run all experiments and generate visualizations.

    Args:
        run_exp1: Run Experiment 1 (baseline comparison)
        run_exp2: Run Experiment 2 (category analysis)
        run_exp3: Run Experiment 3 (hyperparameter sensitivity - optional)
        run_exp4: Run Experiment 4 (ablation study - optional)
        generate_plots: Generate all visualization plots
        num_samples: Number of COCO samples for Exp 1 (None for default)
        skip_generation: Skip image generation, only evaluate
    """
    print("\n" + "=" * 80)
    print("STABLE DIFFUSION MODEL COMPARISON PROJECT")
    print("Comparing SD v1.5 vs SD v2.1")
    print("=" * 80)

    # Print configuration
    print_config()

    with Timer("Complete pipeline"):
        # Run Experiment 1: Baseline Performance
        if run_exp1:
            print("\n" + "=" * 80)
            print("RUNNING EXPERIMENT 1: BASELINE PERFORMANCE")
            print("=" * 80)

            try:
                run_experiment_1(
                    num_samples=num_samples,
                    skip_generation=skip_generation
                )
                print("\n✓ Experiment 1 completed successfully")
            except Exception as e:
                print(f"\n✗ Experiment 1 failed: {e}")
                import traceback
                traceback.print_exc()

        # Run Experiment 2: Category Analysis
        if run_exp2:
            print("\n" + "=" * 80)
            print("RUNNING EXPERIMENT 2: CATEGORY-WISE ANALYSIS")
            print("=" * 80)

            try:
                run_experiment_2(skip_generation=skip_generation)
                print("\n✓ Experiment 2 completed successfully")
            except Exception as e:
                print(f"\n✗ Experiment 2 failed: {e}")
                import traceback
                traceback.print_exc()

        # Run Experiment 3: Hyperparameter Sensitivity (Optional)
        if run_exp3:
            print("\n" + "=" * 80)
            print("RUNNING EXPERIMENT 3: HYPERPARAMETER SENSITIVITY")
            print("=" * 80)

            try:
                run_experiment_3(
                    model_choice="sd_v15",
                    test_guidance=True,
                    test_steps=True
                )
                print("\n✓ Experiment 3 completed successfully")
            except Exception as e:
                print(f"\n✗ Experiment 3 failed: {e}")
                import traceback
                traceback.print_exc()

        # Run Experiment 4: Ablation Study (Optional)
        if run_exp4:
            print("\n" + "=" * 80)
            print("RUNNING EXPERIMENT 4: ABLATION STUDY")
            print("=" * 80)

            try:
                run_experiment_4(
                    run_text=True,
                    run_cfg=True,
                    run_model=True
                )
                print("\n✓ Experiment 4 completed successfully")
            except Exception as e:
                print(f"\n✗ Experiment 4 failed: {e}")
                import traceback
                traceback.print_exc()

        # Generate all plots
        if generate_plots:
            print("\n" + "=" * 80)
            print("GENERATING VISUALIZATIONS")
            print("=" * 80)

            try:
                generate_all_plots()
                print("\n✓ All plots generated successfully")
            except Exception as e:
                print(f"\n✗ Plot generation failed: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)

    print("\nResults saved to:")
    print(f"  - Experiment 1: results/exp1/ (Baseline Comparison)")
    print(f"  - Experiment 2: results/exp2/ (Category Analysis)")
    if run_exp3:
        print(f"  - Experiment 3: results/exp3/ (Hyperparameter Sensitivity)")
    if run_exp4:
        print(f"  - Experiment 4: results/exp4_ablation/ (Ablation Study)")
    if generate_plots:
        print(f"  - Visualizations: results/figures/")



def download_data_only():
    """Download COCO dataset only."""
    print("\n" + "=" * 80)
    print("DOWNLOADING COCO DATASET")
    print("=" * 80)

    from data.data_loader import download_coco_5k

    try:
        download_coco_5k(num_samples=COCO_NUM_SAMPLES)
        print("\n✓ COCO dataset downloaded successfully")
    except Exception as e:
        print(f"\n✗ COCO download failed: {e}")
        import traceback
        traceback.print_exc()


def test_setup():
    """Test the setup with a minimal run."""
    print("\n" + "=" * 80)
    print("TESTING SETUP")
    print("=" * 80)

    print("\n1. Testing configuration...")
    print_config()

    print("\n2. Testing data loading...")
    try:
        from data.data_loader import test_data_loading
        test_data_loading()
        print("✓ Data loading test passed")
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")

    print("\n3. Testing model loading...")
    try:
        from models.model_loader import load_sd_v15, unload_model
        from config import DEVICE

        pipeline = load_sd_v15(device=DEVICE)
        print("✓ Model loading successful")

        unload_model(pipeline)
        print("✓ Model unloading successful")
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("SETUP TEST COMPLETE")
    print("=" * 80)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Stable Diffusion Model Comparison Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all required experiments (1 and 2) with default settings
  python run_all.py

  # Run with fewer samples for testing
  python run_all.py --num_samples 100

  # Run only experiment 1
  python run_all.py --exp1_only

  # Run only experiment 2
  python run_all.py --exp2_only

  # Run all including optional experiments 3 and 4
  python run_all.py --include_exp3 --include_exp4

  # Run only ablation study (exp4)
  python run_all.py --exp4_only

  # Skip generation and only evaluate existing images
  python run_all.py --skip_generation

  # Download COCO dataset only
  python run_all.py --download_data

  # Test setup
  python run_all.py --test_setup

  # Generate plots only (from existing results)
  python run_all.py --plots_only
        """
    )

    parser.add_argument("--exp1_only", action="store_true",
                       help="Run only Experiment 1")
    parser.add_argument("--exp2_only", action="store_true",
                       help="Run only Experiment 2")
    parser.add_argument("--exp4_only", action="store_true",
                       help="Run only Experiment 4 (Ablation Study)")
    parser.add_argument("--include_exp3", action="store_true",
                       help="Include optional Experiment 3 (Hyperparameter Sensitivity)")
    parser.add_argument("--include_exp4", action="store_true",
                       help="Include optional Experiment 4 (Ablation Study)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of COCO samples for Exp 1 (default: from config)")
    parser.add_argument("--skip_generation", action="store_true",
                       help="Skip image generation, only evaluate existing images")
    parser.add_argument("--no_plots", action="store_true",
                       help="Don't generate plots")
    parser.add_argument("--plots_only", action="store_true",
                       help="Only generate plots from existing results")
    parser.add_argument("--download_data", action="store_true",
                       help="Download COCO dataset only")
    parser.add_argument("--test_setup", action="store_true",
                       help="Test the setup")

    args = parser.parse_args()

    # Handle special modes
    if args.download_data:
        download_data_only()
        return

    if args.test_setup:
        test_setup()
        return

    if args.plots_only:
        print("\nGenerating plots from existing results...")
        generate_all_plots()
        return

    # Determine which experiments to run
    if args.exp4_only:
        run_exp1 = False
        run_exp2 = False
        run_exp3 = False
        run_exp4 = True
    else:
        run_exp1 = not args.exp2_only
        run_exp2 = not args.exp1_only
        run_exp3 = args.include_exp3
        run_exp4 = args.include_exp4

    generate_plots = not args.no_plots

    # Run experiments
    run_all_experiments(
        run_exp1=run_exp1,
        run_exp2=run_exp2,
        run_exp3=run_exp3,
        run_exp4=run_exp4,
        generate_plots=generate_plots,
        num_samples=args.num_samples,
        skip_generation=args.skip_generation
    )


if __name__ == "__main__":
    main()
