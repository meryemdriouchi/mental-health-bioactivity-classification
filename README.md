# Mental Health Bioactivity Classification

Predict whether chemical compounds are **active** against protein targets linked to **multiple mental health conditions** — not just dementia.

```
Molecule (SMILES)  →  6 target models  →  Activity profile across conditions
```

**Research learning project only — not medical advice.**

---

## Conditions Covered

| Condition | Protein Target | Example Drugs | ChEMBL ID |
|-----------|---------------|-----------------|-----------|
| Alzheimer's / Dementia | Acetylcholinesterase (AChE) | Donepezil, Galantamine | CHEMBL220 |
| Depression | Serotonin transporter (SERT) | Fluoxetine, Sertraline | CHEMBL2284 |
| Schizophrenia | Dopamine D2 receptor | Haloperidol, Risperidone | CHEMBL217 |
| Anxiety | GABA-A receptor | Diazepam, Lorazepam | CHEMBL209386 |
| ADHD | Dopamine transporter (DAT) | Methylphenidate | CHEMBL238 |
| Bipolar / Depression | Monoamine oxidase A (MAO-A) | Phenelzine | CHEMBL1951 |

Each condition has its **own trained model**.

---

## Project Structure

```
dementia-bioactivity-classification/
├── targets_config.py              # Condition → target mapping
├── mental_health_common.py        # Shared ML utilities
├── train_mental_health_models.py  # Train all 6 models  ← START HERE
├── predict_mental_health.py       # Test molecules across all conditions
├── train_ache_classifier.py       # Original single-target script (legacy)
├── sample_data/                   # Offline training data per target
│   ├── ache_sample.csv
│   ├── sert_sample.csv
│   ├── d2_sample.csv
│   ├── gaba_sample.csv
│   ├── dat_sample.csv
│   └── maoa_sample.csv
└── data/
    ├── targets/                   # Cached ChEMBL data (after download)
    └── models/                    # Trained models + metrics
```

---

## Quick Start

```bash
cd ~/Documents/dementia-bioactivity-classification
pip install -r requirements.txt

# Train all 6 models
python train_mental_health_models.py

# Predict across all conditions (demo: fluoxetine, donepezil, aspirin)
python predict_mental_health.py

# Test a specific known drug
python predict_mental_health.py --drug fluoxetine
python predict_mental_health.py --drug haloperidol
python predict_mental_health.py --drug diazepam

# Test your own SMILES
python predict_mental_health.py "CCN(CC)CC"

# Test all reference drugs
python predict_mental_health.py --all-drugs
```

---

## How It Works

1. **One model per target** — each mental health condition maps to a protein with public bioactivity data
2. **ChEMBL download** — when online, pulls real IC50 measurements per target
3. **Offline fallback** — uses `sample_data/` if ChEMBL is unreachable
4. **Morgan fingerprints** — converts SMILES to 2048-bit structural vectors
5. **Random Forest** — binary classifier (active vs inactive) per target
6. **Unified prediction** — one SMILES tested against all 6 models at once

### Active / inactive rule

IC50 ≤ **1000 nM** (1 µM) → **active**

---

## Improve Accuracy

The sample datasets are small (~40 compounds each). For better models:

```bash
# Delete cached data and re-download from ChEMBL
rm -rf data/targets/*.csv
python train_mental_health_models.py
```

Expect ROC-AUC around **0.75–0.90** with full ChEMBL data (vs misleading 100% on tiny samples).

---

## Known Drugs You Can Test

| Drug | Condition | Command |
|------|-----------|---------|
| Fluoxetine | Depression | `--drug fluoxetine` |
| Donepezil | Dementia | `--drug donepezil` |
| Haloperidol | Schizophrenia | `--drug haloperidol` |
| Diazepam | Anxiety | `--drug diazepam` |
| Methylphenidate | ADHD | `--drug methylphenidate` |
| Phenelzine | Bipolar/Depression | `--drug phenelzine` |
| Aspirin | None (control) | `--drug aspirin` |

---

## Important Limitations

- Predicts **molecular bioactivity**, not patient diagnosis or treatment
- A molecule active on SERT does not automatically mean it is a safe antidepressant
- Small sample data produces unreliable metrics — use ChEMBL for real work
- Mental health is complex; real drugs often hit multiple targets

---

## Next Steps

1. Train on full ChEMBL data
2. Add more targets (5-HT2A, NET, BChE, BACE1)
3. Build a multi-label model (one model, many outputs)
4. Add a simple web UI or Jupyter notebook for predictions
