"""
Dementia bioactivity classifier — beginner starter project.

Downloads acetylcholinesterase (AChE) inhibition data from ChEMBL,
labels compounds as active/inactive, trains a Random Forest on
Morgan fingerprints, and reports evaluation metrics.

Target: CHEMBL220 (human acetylcholinesterase)
Related to symptomatic Alzheimer's treatment (e.g. donepezil).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# --- Configuration ---
TARGET_CHEBL_ID = "CHEMBL220"  # Human acetylcholinesterase
ACTIVE_THRESHOLD_NM = 1000.0    # IC50 <= 1 µM → active (common HTS cutoff)
FINGERPRINT_RADIUS = 2          # ECFP4
FINGERPRINT_BITS = 2048
RANDOM_SEED = 42

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_CSV = Path(__file__).parent / "sample_data" / "ache_sample.csv"
OUTPUT_CSV = DATA_DIR / "ache_activities.csv"
MODEL_PATH = DATA_DIR / "ache_model.joblib"
METRICS_PATH = DATA_DIR / "model_metrics.txt"

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"


def fetch_chembl_activities(
    target_id: str,
    standard_type: str = "IC50",
    max_records: int = 2000,
) -> pd.DataFrame:
    """Download bioactivity records from ChEMBL REST API."""
    print(f"Fetching {standard_type} data for {target_id} from ChEMBL...")

    params = {
        "target_chembl_id": target_id,
        "standard_type": standard_type,
        "limit": 1000,
    }

    rows: list[dict] = []
    url: str | None = CHEMBL_API
    session = requests.Session()
    session.trust_env = False  # avoid broken system proxy settings

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
                    "pchembl_value": activity.get("pchembl_value"),
                    "assay_description": activity.get("assay_description"),
                }
            )

        url = payload.get("page_meta", {}).get("next")
        params = None  # ChEMBL next URLs already include query params

    df = pd.DataFrame(rows)
    print(f"  Raw records downloaded: {len(df)}")
    return df


def load_sample_data() -> pd.DataFrame:
    """Load bundled sample data when ChEMBL is unreachable."""
    if not SAMPLE_CSV.exists():
        raise FileNotFoundError(f"Sample data not found at {SAMPLE_CSV}")

    print(f"Using bundled sample data from {SAMPLE_CSV}")
    df = pd.read_csv(SAMPLE_CSV)
    print(f"  Sample records loaded: {len(df)}")
    return df


def load_activity_data(target_id: str) -> tuple[pd.DataFrame, str]:
    """Try ChEMBL first; fall back to local sample CSV."""
    if OUTPUT_CSV.exists():
        print(f"Loading cached data from {OUTPUT_CSV}")
        return pd.read_csv(OUTPUT_CSV), "cached"

    try:
        return fetch_chembl_activities(target_id), "chembl"
    except requests.RequestException as exc:
        print(f"  ChEMBL download failed: {exc}")
        print("  Falling back to bundled sample data for offline learning.")
        return load_sample_data(), "sample"


def clean_activity_data(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Filter, deduplicate, and label compounds."""
    if source in {"cached", "sample"} and "label" in df.columns:
        df = df.copy()
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64").fillna(0).astype(int)
        print(f"  Using pre-labeled data: {len(df)} compounds")
        print(f"  Active (IC50 <= {ACTIVE_THRESHOLD_NM} nM): {df['label'].sum()}")
        print(f"  Inactive: {(df['label'] == 0).sum()}")
        return df

    df = df.dropna(subset=["canonical_smiles", "standard_value"]).copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])

    # Keep nM measurements (ChEMBL standard_value for IC50 is usually nM)
    df = df[df["standard_units"].str.upper() == "NM"]

    # One row per compound: median IC50 across assays reduces noise
    df = (
        df.groupby(["molecule_chembl_id", "canonical_smiles"], as_index=False)
        .agg(median_ic50_nm=("standard_value", "median"), n_assays=("standard_value", "count"))
    )

    df["label"] = (df["median_ic50_nm"] <= ACTIVE_THRESHOLD_NM).astype(int)
    df["label_name"] = df["label"].map({1: "active", 0: "inactive"})

    print(f"  Unique compounds after cleaning: {len(df)}")
    print(f"  Active (IC50 <= {ACTIVE_THRESHOLD_NM} nM): {df['label'].sum()}")
    print(f"  Inactive: {(df['label'] == 0).sum()}")
    return df


def smiles_to_fingerprint(smiles: str, radius: int = FINGERPRINT_RADIUS, n_bits: int = FINGERPRINT_BITS):
    """Convert SMILES to Morgan fingerprint bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Featurize all valid SMILES."""
    fps = []
    labels = []
    valid_rows = []

    for _, row in df.iterrows():
        fp = smiles_to_fingerprint(row["canonical_smiles"])
        if fp is not None:
            fps.append(fp)
            labels.append(row["label"])
            valid_rows.append(row)

    if not fps:
        raise ValueError("No valid molecules found. Check SMILES parsing.")

    X = np.vstack(fps)
    y = np.array(labels)
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)

    print(f"  Valid molecules for training: {len(valid_df)}")
    return X, y, valid_df


def train_and_evaluate(X: np.ndarray, y: np.ndarray) -> tuple[RandomForestClassifier, dict]:
    """Train Random Forest and compute metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=["inactive", "active"]),
        "train_size": len(y_train),
        "test_size": len(y_test),
    }

    return model, metrics


def predict_single(model: RandomForestClassifier, smiles: str) -> dict:
    """Predict activity for one SMILES string."""
    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return {"error": "Invalid SMILES"}
    prob = model.predict_proba(fp.reshape(1, -1))[0, 1]
    return {
        "smiles": smiles,
        "probability_active": round(float(prob), 4),
        "prediction": "active" if prob >= 0.5 else "inactive",
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data (ChEMBL, cache, or bundled sample)
    raw_df, source = load_activity_data(TARGET_CHEBL_ID)
    clean_df = clean_activity_data(raw_df, source)
    if source == "chembl":
        clean_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved cleaned data to {OUTPUT_CSV}")

    if clean_df["label"].sum() < 10 or (clean_df["label"] == 0).sum() < 10:
        print("ERROR: Too few active or inactive compounds to train reliably.")
        sys.exit(1)

    # 2. Featurize
    print("Generating Morgan fingerprints...")
    X, y, valid_df = build_feature_matrix(clean_df)

    # 3. Train
    model, metrics = train_and_evaluate(X, y)

    # 4. Save
    joblib.dump(model, MODEL_PATH)

    report_lines = [
        "Dementia Bioactivity Classifier — AChE (CHEMBL220)",
        "=" * 50,
        f"Active threshold: IC50 <= {ACTIVE_THRESHOLD_NM} nM",
        f"Compounds used: {len(valid_df)}",
        f"Train size: {metrics['train_size']}",
        f"Test size: {metrics['test_size']}",
        "",
        f"ROC-AUC: {metrics['roc_auc']:.4f}",
        "",
        "Confusion matrix (rows=true, cols=predicted):",
        "              inactive  active",
        f"  inactive    {metrics['confusion_matrix'][0][0]:>8}  {metrics['confusion_matrix'][0][1]:>6}",
        f"  active      {metrics['confusion_matrix'][1][0]:>8}  {metrics['confusion_matrix'][1][1]:>6}",
        "",
        metrics["classification_report"],
    ]
    METRICS_PATH.write_text("\n".join(report_lines))

    print("\n" + "\n".join(report_lines))
    print(f"\nModel saved to {MODEL_PATH}")

    # 5. Demo predictions on known dementia drugs
    print("\n--- Demo: known dementia-related molecules ---")
    demo_smiles = {
        "Donepezil (AChE inhibitor)": "CN1C(=O)CCc2ccccc2N(CCN3CCCCC3)C1=O",
        "Galantamine (AChE inhibitor)": "C=C[C@H]1CN2CC[C@H]2C[C@@H]1Oc3ccc(OC)cc3OC",
        "Aspirin (not a dementia drug)": "CC(=O)Oc1ccccc1C(=O)O",
    }
    for name, smi in demo_smiles.items():
        result = predict_single(model, smi)
        print(f"  {name}: {result}")


if __name__ == "__main__":
    main()
