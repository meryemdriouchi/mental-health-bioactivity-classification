"""Shared utilities for mental health bioactivity classification."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from targets_config import MentalHealthTarget

ACTIVE_THRESHOLD_NM = 1000.0
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048
RANDOM_SEED = 42
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"


def fetch_chembl_activities(
    target_id: str,
    standard_type: str = "IC50",
    max_records: int = 2000,
) -> pd.DataFrame:
    print(f"  Fetching {standard_type} data for {target_id} from ChEMBL...")

    params = {"target_chembl_id": target_id, "standard_type": standard_type, "limit": 1000}
    rows: list[dict] = []
    url: str | None = CHEMBL_API
    session = requests.Session()
    session.trust_env = False

    while url and len(rows) < max_records:
        response = session.get(url, params=params if url == CHEMBL_API else None, timeout=60)
        response.raise_for_status()
        payload = response.json()

        for activity in payload.get("activities", []):
            rows.append(
                {
                    "molecule_chembl_id": activity.get("molecule_chembl_id"),
                    "canonical_smiles": activity.get("canonical_smiles"),
                    "standard_value": activity.get("standard_value"),
                    "standard_units": activity.get("standard_units"),
                }
            )

        url = payload.get("page_meta", {}).get("next")
        params = None

    print(f"    Raw records: {len(rows)}")
    return pd.DataFrame(rows)


def load_activity_data(target: MentalHealthTarget) -> tuple[pd.DataFrame, str]:
    if target.cache_path.exists():
        print(f"  Loading cached data from {target.cache_path}")
        return pd.read_csv(target.cache_path), "cached"

    try:
        return fetch_chembl_activities(target.chembl_id), "chembl"
    except requests.RequestException as exc:
        print(f"    ChEMBL download failed: {exc}")
        print(f"    Using sample data: {target.sample_path}")
        return pd.read_csv(target.sample_path), "sample"


def clean_activity_data(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if source in {"cached", "sample"} and "label" in df.columns:
        df = df.copy()
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        return df

    df = df.dropna(subset=["canonical_smiles", "standard_value"]).copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])
    df = df[df["standard_units"].str.upper() == "NM"]

    df = (
        df.groupby(["molecule_chembl_id", "canonical_smiles"], as_index=False)
        .agg(median_ic50_nm=("standard_value", "median"), n_assays=("standard_value", "count"))
    )
    df["label"] = (df["median_ic50_nm"] <= ACTIVE_THRESHOLD_NM).astype(int)
    df["label_name"] = df["label"].map({1: "active", 0: "inactive"})
    return df


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
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    return model, {
        "roc_auc": roc_auc_score(y_test, y_prob),
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


def save_model_bundle(target: MentalHealthTarget, model: RandomForestClassifier, metrics: dict) -> None:
    target.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target.model_path)

    lines = [
        f"{target.condition} — {target.target_name} ({target.chembl_id})",
        "=" * 60,
        target.description,
        "",
        f"Compounds: {metrics['n_compounds']}  (active: {metrics['n_active']}, inactive: {metrics['n_inactive']})",
        f"Train: {metrics['train_size']}  |  Test: {metrics['test_size']}",
        f"ROC-AUC: {metrics['roc_auc']:.4f}",
        "",
        metrics["classification_report"],
    ]
    target.metrics_path.write_text("\n".join(lines))


def load_model(target: MentalHealthTarget) -> RandomForestClassifier:
    if not target.model_path.exists():
        raise FileNotFoundError(
            f"No model for {target.condition}. Run: python train_mental_health_models.py"
        )
    return joblib.load(target.model_path)
