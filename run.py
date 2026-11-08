"""
Unified CLI for Reproduction, Auditing, and Experiments.

Commands:
    python run.py audit
    python run.py check-masks
    python run.py run-experiments
    python run.py all

Can also run individual modules directly:
    python audit.py
    python check_masks.py
    python run_experiments.py
"""

import argparse
import sys
from audit import run_audit
from check_masks import run_check_masks
from run_experiments import run_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for Pneumonia Detection Reproducibility Audits & Experiments.",
        usage="%(prog)s [action] [options]"
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="all",
        choices=["audit", "check-masks", "check_masks", "run-experiments", "run_experiments", "all"],
        help="Command to run: 'audit', 'check-masks', 'run-experiments', or 'all' (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root path to dataset (default: data)",
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        default="data/processed/segmented_images",
        help="Path to segmented masks directory (default: data/processed/segmented_images)",
    )

    args = parser.parse_args()
    action = args.action.replace("_", "-")

    success = True
    if action == "audit":
        success = run_audit(args.data_dir)
    elif action == "check-masks":
        success = run_check_masks(args.masks_dir)
    elif action == "run-experiments":
        run_experiments()
    elif action == "all":
        print("\n>>> STEP 1/3: DATASET AUDIT")
        s1 = run_audit(args.data_dir)
        print("\n>>> STEP 2/3: LUNG SEGMENTATION MASK CHECK")
        s2 = run_check_masks(args.masks_dir)
        print("\n>>> STEP 3/3: RUNNING EXPERIMENTS & EXPORTING RESULTS TABLES")
        run_experiments()
        success = s1 and s2

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
