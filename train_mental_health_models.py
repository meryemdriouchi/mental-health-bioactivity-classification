"""
Train bioactivity classifiers for multiple mental health conditions.

For automatic updates with fresh ChEMBL data, prefer:
  python update_models.py --force
"""

from __future__ import annotations

import argparse
import sys

from model_updater import train_one_target, update_all_models
from targets_config import MODELS_DIR, all_targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mental health bioactivity models")
    parser.add_argument("--force", action="store_true", help="Force fresh ChEMBL download")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.force:
        results = update_all_models(force=True)
    else:
        results: dict = {}
        for target in all_targets():
            try:
                metrics = train_one_target(target, force_refresh=False)
                if metrics:
                    results[target.key] = metrics
            except Exception as exc:
                print(f"  ERROR training {target.key}: {exc}")

    if not results:
        print("\nERROR: No models were trained successfully.")
        sys.exit(1)

    print(f"\nDone. {len(results)}/{len(all_targets())} models trained.")
    print("Predict with: python predict_mental_health.py")
    print("Auto-update:  python update_models.py")


if __name__ == "__main__":
    main()
