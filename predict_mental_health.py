"""
Predict molecule activity across multiple mental health conditions.

Usage:
  python predict_mental_health.py
  python predict_mental_health.py "CN1C(=O)CCc2ccccc2N(CCN3CCCCC3)C1=O"
  python predict_mental_health.py --drug fluoxetine
"""

from __future__ import annotations

import argparse
import sys

from mental_health_common import load_model, predict_single
from targets_config import MENTAL_HEALTH_TARGETS, all_targets

# Known drugs for quick testing (canonical SMILES)
KNOWN_DRUGS: dict[str, tuple[str, str]] = {
    "donepezil": ("Donepezil", "CN1C(=O)CCc2ccccc2N(CCN3CCCCC3)C1=O"),
    "galantamine": ("Galantamine", "C=C[C@H]1CN2CC[C@H]2C[C@@H]1Oc3ccc(OC)cc3OC"),
    "fluoxetine": ("Fluoxetine", "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1"),
    "sertraline": ("Sertraline", "CN[C@H]1CC[C@@H](c2ccc(Cl)cc2)c2ccccc12"),
    "haloperidol": ("Haloperidol", "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c1ccc(Cl)cc1"),
    "risperidone": ("Risperidone", "CC1=C(CCN2CCC(CC2)c2noc3cc(F)ccc23)C(=O)N2CCCCC2=N1"),
    "diazepam": ("Diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"),
    "lorazepam": ("Lorazepam", "OC1N=C(c2ccccc2Cl)c2cc(Cl)ccc2N1C(=O)C1CC1"),
    "methylphenidate": ("Methylphenidate", "COC(=O)C(c1ccccc1)C1CCCCN1"),
    "phenelzine": ("Phenelzine", "CCc1ccccc1OCC(N)N"),
    "aspirin": ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
}


def predict_all_conditions(smiles: str, molecule_name: str = "Molecule") -> list[dict]:
    results = []
    for target in all_targets():
        try:
            model = load_model(target)
            pred = predict_single(model, smiles)
        except FileNotFoundError as exc:
            pred = {"error": str(exc)}

        results.append(
            {
                "condition": target.condition,
                "target": target.target_name,
                "example_drugs": target.example_drugs,
                **pred,
            }
        )
    return results


def print_report(name: str, smiles: str, results: list[dict]) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"  SMILES: {smiles}")
    print("=" * 70)

    active_hits = []
    for r in results:
        if "error" in r:
            status = f"ERROR — {r['error']}"
        else:
            prob = r["probability_active"]
            status = f"{r['prediction'].upper():8s}  ({prob:.1%} probability)"
            if r["prediction"] == "active":
                active_hits.append(r["condition"])

        print(f"\n  {r['condition']}")
        print(f"    Target: {r['target']}")
        print(f"    Result: {status}")
        print(f"    Known drugs: {', '.join(r['example_drugs'])}")

    print(f"\n{'-' * 70}")
    if active_hits:
        print(f"  Predicted potentially active for: {', '.join(active_hits)}")
    else:
        print("  No active predictions across mental health targets.")
    print("  (Research use only — not medical advice.)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict mental health bioactivity")
    parser.add_argument("smiles", nargs="?", help="SMILES string to predict")
    parser.add_argument("--drug", choices=list(KNOWN_DRUGS.keys()), help="Test a known drug")
    parser.add_argument("--all-drugs", action="store_true", help="Test all known reference drugs")
    args = parser.parse_args()

    if args.all_drugs:
        for key, (name, smiles) in KNOWN_DRUGS.items():
            results = predict_all_conditions(smiles, name)
            print_report(name, smiles, results)
        return

    if args.drug:
        name, smiles = KNOWN_DRUGS[args.drug]
    elif args.smiles:
        name, smiles = "Custom molecule", args.smiles
    else:
        # Default demo: test 3 drugs across all conditions
        print("No input given — running demo on 3 reference drugs.\n")
        for key in ("fluoxetine", "donepezil", "aspirin"):
            name, smiles = KNOWN_DRUGS[key]
            results = predict_all_conditions(smiles, name)
            print_report(name, smiles, results)
        print("\nTip: python predict_mental_health.py --drug haloperidol")
        print("     python predict_mental_health.py --all-drugs")
        return

    results = predict_all_conditions(smiles, name)
    print_report(name, smiles, results)


if __name__ == "__main__":
    main()
