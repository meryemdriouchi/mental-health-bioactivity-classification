"""
Train bioactivity classifiers for multiple mental health conditions.

One Random Forest model per protein target (SERT, D2, GABA-A, DAT, MAO-A, AChE).
Run: python train_mental_health_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from mental_health_common import (
    build_feature_matrix,
    clean_activity_data,
    load_activity_data,
    save_model_bundle,
    train_and_evaluate,
)
from targets_config import DATA_DIR, MODELS_DIR, all_targets


def train_one_target(target) -> dict | None:
    print(f"\n{'=' * 60}")
    print(f"Condition: {target.condition}")
    print(f"Target:    {target.target_name} ({target.chembl_id})")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_df, source = load_activity_data(target)
    clean_df = clean_activity_data(raw_df, source)

    if source == "chembl":
        clean_df.to_csv(target.cache_path, index=False)
        print(f"  Saved cache: {target.cache_path}")

    n_active = clean_df["label"].sum()
    n_inactive = (clean_df["label"] == 0).sum()
    print(f"  Compounds: {len(clean_df)}  |  Active: {n_active}  |  Inactive: {n_inactive}")

    if n_active < 5 or n_inactive < 5:
        print(f"  SKIPPED — need at least 5 active and 5 inactive (got {n_active}/{n_inactive})")
        return None

    X, y, valid_df = build_feature_matrix(clean_df)
    print(f"  Valid molecules: {len(valid_df)}")

    if y.sum() < 5 or (y == 0).sum() < 5:
        print("  SKIPPED — not enough valid compounds after fingerprinting")
        return None

    model, metrics = train_and_evaluate(X, y)
    save_model_bundle(target, model, metrics)
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  Model saved: {target.model_path}")
    return metrics


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    targets = all_targets()

    print(f"Training {len(targets)} mental health bioactivity models...")
    results: dict[str, dict] = {}

    for target in targets:
        try:
            metrics = train_one_target(target)
            if metrics:
                results[target.key] = metrics
        except Exception as exc:
            print(f"  ERROR training {target.key}: {exc}")

    # Summary report
    summary_path = MODELS_DIR / "training_summary.txt"
    lines = ["Mental Health Bioactivity Models — Training Summary", "=" * 60, ""]
    for target in targets:
        if target.key in results:
            m = results[target.key]
            lines.append(
                f"{target.condition:30s}  AUC={m['roc_auc']:.3f}  "
                f"compounds={m['n_compounds']}  (active={m['n_active']})"
            )
        else:
            lines.append(f"{target.condition:30s}  SKIPPED")
    lines.append("")
    lines.append(f"Models saved in: {MODELS_DIR}")
    summary_path.write_text("\n".join(lines))

    print(f"\n{'=' * 60}")
    print("\n".join(lines))
    print(f"\nSummary: {summary_path}")

    if not results:
        print("\nERROR: No models were trained successfully.")
        sys.exit(1)

    print(f"\nDone. {len(results)}/{len(targets)} models trained.")
    print("Predict with: python predict_mental_health.py")


if __name__ == "__main__":
    main()
