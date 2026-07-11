"""
Automatic model update pipeline.

Downloads fresh ChEMBL data, retrains models, and only keeps improvements.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mental_health_common import (
    append_training_history,
    build_feature_matrix,
    cache_is_stale,
    clean_activity_data,
    count_new_compounds,
    load_activity_data,
    load_previous_auc,
    save_model_bundle,
    should_deploy_model,
    train_and_evaluate,
    update_manifest,
    utc_now_iso,
)
from targets_config import DATA_DIR, MODELS_DIR, REFRESH_INTERVAL_DAYS, MentalHealthTarget, all_targets


def train_one_target(
    target: MentalHealthTarget,
    force_refresh: bool = False,
    min_active: int = 10,
    min_inactive: int = 10,
) -> dict | None:
    print(f"\n{'=' * 60}")
    print(f"Condition: {target.condition}")
    print(f"Target:    {target.target_name} ({target.chembl_id})")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    old_df = pd.read_csv(target.cache_path) if target.cache_path.exists() else None

    raw_df, source = load_activity_data(target, force_refresh=force_refresh)
    clean_df = clean_activity_data(raw_df, source)
    new_compounds = count_new_compounds(old_df, clean_df)

    if source == "chembl" or force_refresh:
        clean_df.to_csv(target.cache_path, index=False)
        print(f"  Saved cache: {target.cache_path}")

    n_active = int(clean_df["label"].sum())
    n_inactive = int((clean_df["label"] == 0).sum())
    print(f"  Compounds: {len(clean_df)}  |  Active: {n_active}  |  Inactive: {n_inactive}")
    print(f"  New compounds since last cache: {new_compounds}")

    if n_active < min_active or n_inactive < min_inactive:
        print(f"  SKIPPED — need >= {min_active} active and >= {min_inactive} inactive")
        return None

    X, y, valid_df = build_feature_matrix(clean_df)
    print(f"  Valid molecules: {len(valid_df)}")

    if y.sum() < min_active or (y == 0).sum() < min_inactive:
        print("  SKIPPED — not enough valid compounds after fingerprinting")
        return None

    previous_auc = load_previous_auc(target)
    model, metrics = train_and_evaluate(X, y)

    deploy = should_deploy_model(previous_auc, metrics)
    status = "deployed" if deploy else "rejected"

    print(f"  Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  CV ROC-AUC:   {metrics['cv_roc_auc']:.4f}")
    if previous_auc is not None:
        print(f"  Previous CV:  {previous_auc:.4f}")

    if deploy:
        save_model_bundle(target, model, metrics, data_source=source)
        update_manifest(target, metrics, source, new_compounds)
        print(f"  Model {status}: {target.model_path}")
    else:
        print(f"  Model {status} — new CV AUC worse than previous; kept old model")

    entry = {
        "timestamp": utc_now_iso(),
        "target_key": target.key,
        "condition": target.condition,
        "data_source": source,
        "compounds": metrics["n_compounds"],
        "new_compounds": new_compounds,
        "roc_auc": metrics["roc_auc"],
        "cv_roc_auc": metrics["cv_roc_auc"],
        "previous_cv_roc_auc": previous_auc,
        "status": status,
    }
    append_training_history(entry)

    metrics["status"] = status
    metrics["new_compounds"] = new_compounds
    metrics["data_source"] = source
    return metrics


def needs_update(target: MentalHealthTarget, force: bool = False) -> bool:
    if force:
        return True
    if not target.model_path.exists():
        return True
    return cache_is_stale(target, REFRESH_INTERVAL_DAYS)


def update_all_models(force: bool = False) -> dict[str, dict]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    targets = all_targets()
    results: dict[str, dict] = {}

    print(f"Checking {len(targets)} targets (refresh every {REFRESH_INTERVAL_DAYS} days)...")

    for target in targets:
        if not needs_update(target, force=force):
            print(f"\n[skip] {target.condition} — cache is fresh, model exists")
            continue

        try:
            metrics = train_one_target(
                target,
                force_refresh=force or cache_is_stale(target, REFRESH_INTERVAL_DAYS),
            )
            if metrics:
                results[target.key] = metrics
        except Exception as exc:
            print(f"  ERROR updating {target.key}: {exc}")

    summary_path = MODELS_DIR / "training_summary.txt"
    lines = [
        "Mental Health Bioactivity Models — Auto-Update Summary",
        "=" * 60,
        f"Last run: {utc_now_iso()}",
        "",
    ]
    for target in targets:
        if target.key in results:
            m = results[target.key]
            lines.append(
                f"{target.condition:30s}  CV-AUC={m['cv_roc_auc']:.3f}  "
                f"compounds={m['n_compounds']}  new={m['new_compounds']}  [{m['status']}]"
            )
        elif target.model_path.exists():
            lines.append(f"{target.condition:30s}  unchanged (fresh)")
        else:
            lines.append(f"{target.condition:30s}  SKIPPED")
    lines.append("")
    lines.append(f"Manifest: data/manifest.json")
    lines.append(f"History:  data/models/training_history.json")
    summary_path.write_text("\n".join(lines))

    print(f"\n{'=' * 60}")
    print("\n".join(lines))
    return results
