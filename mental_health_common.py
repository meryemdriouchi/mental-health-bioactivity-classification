"""Shared utilities for mental health bioactivity classification."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from targets_config import (
    ACTIVITY_TYPES,
    HISTORY_PATH,
    MANIFEST_PATH,
    MAX_CHEMBL_RECORDS,
    MentalHealthTarget,
)

ACTIVE_THRESHOLD_NM = 1000.0
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048
RANDOM_SEED = 42
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fetch_chembl_activities(
    target_id: str,
    standard_types: tuple[str, ...] = ACTIVITY_TYPES,
    max_records: int = MAX_CHEMBL_RECORDS,
) -> pd.DataFrame:
    """Download bioactivity records from ChEMBL for one or more assay types."""
    all_rows: list[dict] = []
    session = requests.Session()
    session.trust_env = False

    for standard_type in standard_types:
        print(f"  Fetching {standard_type} for {target_id}...")
        params = {"target_chembl_id": target_id, "standard_type": standard_type, "limit": 1000}
        rows: list[dict] = []
        url: str | None = CHEMBL_API

        while url and len(rows) < max_records:
            response = session.get(url, params=params if url == CHEMBL_API else None, timeout=90)
            response.raise_for_status()
            payload = response.json()

            for activity in payload.get("activities", []):
                rows.append(
                    {
                        "molecule_chembl_id": activity.get("molecule_chembl_id"),
                        "canonical_smiles": activity.get("canonical_smiles"),
                        "standard_value": activity.get("standard_value"),
                        "standard_units": activity.get("standard_units"),
                        "standard_type": activity.get("standard_type"),
                    }
                )

            url = payload.get("page_meta", {}).get("next")
            params = None

        print(f"    {standard_type} records: {len(rows)}")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"    Total raw records: {len(df)}")
    return df


def load_activity_data(
    target: MentalHealthTarget,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Load cached ChEMBL data, refresh from API, or fall back to sample data."""
    if not force_refresh and target.cache_path.exists():
        print(f"  Loading cached data from {target.cache_path}")
        return pd.read_csv(target.cache_path), "cached"

    try:
        df = fetch_chembl_activities(target.chembl_id)
        if df.empty:
            raise ValueError("ChEMBL returned no records")
        return df, "chembl"
    except (requests.RequestException, ValueError) as exc:
        if target.cache_path.exists():
            print(f"    ChEMBL failed ({exc}); using existing cache")
            return pd.read_csv(target.cache_path), "cached"
        print(f"    ChEMBL failed ({exc}); using sample data")
        return pd.read_csv(target.sample_path), "sample"


def clean_activity_data(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Filter, deduplicate, and label compounds."""
    if source in {"cached", "sample"} and "label" in df.columns:
        df = df.copy()
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        return df

    df = df.dropna(subset=["canonical_smiles", "standard_value"]).copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])
    df = df[df["standard_units"].str.upper() == "NM"]

    # Prefer IC50 > Ki > EC50 when the same compound appears in multiple assays
    type_rank = {"IC50": 0, "Ki": 1, "EC50": 2}
    if "standard_type" in df.columns:
        df["type_rank"] = df["standard_type"].map(type_rank).fillna(9)
        df = df.sort_values("type_rank")

    df = (
        df.groupby(["molecule_chembl_id", "canonical_smiles"], as_index=False)
        .agg(
            median_ic50_nm=("standard_value", "median"),
            n_assays=("standard_value", "count"),
            assay_types=("standard_type", lambda s: ",".join(sorted(set(s.dropna())))),
        )
    )
    df["label"] = (df["median_ic50_nm"] <= ACTIVE_THRESHOLD_NM).astype(int)
    df["label_name"] = df["label"].map({1: "active", 0: "inactive"})
    df["fetched_at"] = utc_now_iso()
    return df


def count_new_compounds(old_df: pd.DataFrame | None, new_df: pd.DataFrame) -> int:
    if old_df is None or old_df.empty:
        return len(new_df)
    old_ids = set(old_df["molecule_chembl_id"].astype(str))
    new_ids = set(new_df["molecule_chembl_id"].astype(str))
    return len(new_ids - old_ids)


def cache_is_stale(target: MentalHealthTarget, max_age_days: int) -> bool:
    if not target.cache_path.exists():
        return True
    mtime = datetime.fromtimestamp(target.cache_path.stat().st_mtime, tz=timezone.utc)
    age_days = (datetime.now(timezone.utc) - mtime).days
    return age_days >= max_age_days


def smiles_to_fingerprint(
    smiles: str,
    radius: int = FINGERPRINT_RADIUS,
    n_bits: int = FINGERPRINT_BITS,
) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    fps, labels, valid_rows = [], [], []

    for _, row in df.iterrows():
        fp = smiles_to_fingerprint(row["canonical_smiles"])
        if fp is not None:
            fps.append(fp)
            labels.append(row["label"])
            valid_rows.append(row)

    if not fps:
        raise ValueError("No valid molecules found.")

    return np.vstack(fps), np.array(labels), pd.DataFrame(valid_rows).reset_index(drop=True)


def train_and_evaluate(X: np.ndarray, y: np.ndarray) -> tuple[RandomForestClassifier, dict]:
    if len(np.unique(y)) < 2:
        raise ValueError("Need both active and inactive compounds to train.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    test_auc = float(roc_auc_score(y_test, y_prob))

    min_class = min(int(y.sum()), int((y == 0).sum()))
    n_splits = min(5, min_class)
    if n_splits >= 2:
        cv_auc = float(
            cross_val_score(
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
                X,
                y,
                cv=n_splits,
                scoring="roc_auc",
                n_jobs=1,
            ).mean()
        )
    else:
        cv_auc = test_auc

    return model, {
        "roc_auc": test_auc,
        "cv_roc_auc": cv_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["inactive", "active"], zero_division=0
        ),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "n_compounds": len(y),
        "n_active": int(y.sum()),
        "n_inactive": int((y == 0).sum()),
    }


def predict_single(model: RandomForestClassifier, smiles: str) -> dict:
    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return {"error": "Invalid SMILES"}
    prob = model.predict_proba(fp.reshape(1, -1))[0, 1]
    return {
        "probability_active": round(float(prob), 4),
        "prediction": "active" if prob >= 0.5 else "inactive",
    }


def load_previous_auc(target: MentalHealthTarget) -> float | None:
    if not target.model_path.exists():
        return None
    history = load_training_history()
    entries = [e for e in history if e.get("target_key") == target.key]
    if entries:
        return float(entries[-1].get("cv_roc_auc", entries[-1].get("roc_auc", 0)))
    return None


def should_deploy_model(previous_auc: float | None, new_metrics: dict, tolerance: float = 0.02) -> bool:
    """Keep new model unless cross-validated AUC drops meaningfully."""
    new_auc = new_metrics["cv_roc_auc"]
    if previous_auc is None:
        return True
    return new_auc >= (previous_auc - tolerance)


def save_model_bundle(
    target: MentalHealthTarget,
    model: RandomForestClassifier,
    metrics: dict,
    data_source: str,
) -> None:
    target.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target.model_path)

    lines = [
        f"{target.condition} — {target.target_name} ({target.chembl_id})",
        "=" * 60,
        target.description,
        "",
        f"Updated: {utc_now_iso()}",
        f"Data source: {data_source}",
        f"Compounds: {metrics['n_compounds']}  (active: {metrics['n_active']}, inactive: {metrics['n_inactive']})",
        f"Train: {metrics['train_size']}  |  Test: {metrics['test_size']}",
        f"Test ROC-AUC: {metrics['roc_auc']:.4f}",
        f"CV ROC-AUC:   {metrics['cv_roc_auc']:.4f}",
        "",
        metrics["classification_report"],
    ]
    target.metrics_path.write_text("\n".join(lines))


def load_training_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    return json.loads(HISTORY_PATH.read_text())


def append_training_history(entry: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = load_training_history()
    history.append(entry)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))


def update_manifest(target: MentalHealthTarget, metrics: dict, data_source: str, new_compounds: int) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())

    manifest["last_update"] = utc_now_iso()
    manifest.setdefault("targets", {})
    manifest["targets"][target.key] = {
        "condition": target.condition,
        "chembl_id": target.chembl_id,
        "data_source": data_source,
        "compounds": metrics["n_compounds"],
        "active": metrics["n_active"],
        "inactive": metrics["n_inactive"],
        "test_roc_auc": round(metrics["roc_auc"], 4),
        "cv_roc_auc": round(metrics["cv_roc_auc"], 4),
        "new_compounds_added": new_compounds,
        "model_path": str(target.model_path),
        "cache_path": str(target.cache_path),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def load_model(target: MentalHealthTarget) -> RandomForestClassifier:
    if not target.model_path.exists():
        raise FileNotFoundError(
            f"No model for {target.condition}. Run: python update_models.py"
        )
    return joblib.load(target.model_path)
