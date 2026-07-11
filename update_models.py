#!/usr/bin/env python3
"""
Auto-update mental health bioactivity models with fresh ChEMBL data.

Usage:
  python update_models.py              # update only if data is stale (>7 days)
  python update_models.py --force      # force fresh download + retrain now
  python update_models.py --status     # show last update info without retraining
"""

from __future__ import annotations

import argparse
import json
import sys

from model_updater import update_all_models
from targets_config import MANIFEST_PATH, REFRESH_INTERVAL_DAYS, all_targets
from mental_health_common import cache_is_stale


def show_status() -> None:
    print(f"Auto-refresh interval: every {REFRESH_INTERVAL_DAYS} days\n")

    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        print(f"Last update: {manifest.get('last_update', 'unknown')}\n")
        for key, info in manifest.get("targets", {}).items():
            print(f"  {info.get('condition', key)}")
            print(f"    Compounds: {info.get('compounds')}  |  CV-AUC: {info.get('cv_roc_auc')}")
            print(f"    Source: {info.get('data_source')}  |  New last run: {info.get('new_compounds_added')}")
    else:
        print("No manifest yet. Run: python update_models.py --force")

    print("\nPer-target cache status:")
    for target in all_targets():
        stale = cache_is_stale(target, REFRESH_INTERVAL_DAYS)
        exists = target.model_path.exists()
        state = "needs update" if stale or not exists else "fresh"
        print(f"  {target.condition:30s}  model={'yes' if exists else 'no'}  cache={state}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-update mental health ML models")
    parser.add_argument("--force", action="store_true", help="Force ChEMBL download and retrain")
    parser.add_argument("--status", action="store_true", help="Show update status only")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    print("Mental Health Model Auto-Updater")
    print("=" * 60)
    if args.force:
        print("Mode: FORCE — downloading latest ChEMBL data for all targets")
    else:
        print(f"Mode: SMART — only updating targets older than {REFRESH_INTERVAL_DAYS} days")

    results = update_all_models(force=args.force)

    if not results and not args.force:
        print("\nAll models are up to date. Use --force to retrain anyway.")
    elif not results:
        print("\nWARNING: No models were updated. Check internet connection.")
        sys.exit(1)
    else:
        print(f"\nUpdated {len(results)} model(s).")
        print("Predict with: python predict_mental_health.py")


if __name__ == "__main__":
    main()
