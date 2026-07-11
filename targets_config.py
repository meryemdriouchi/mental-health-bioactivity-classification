"""
Mental health drug targets mapped to ChEMBL bioactivity data.

Each target is a protein researchers screen molecules against.
Activity against a target is linked to potential relevance for a condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / "sample_data"
DATA_DIR = Path(__file__).parent / "data" / "targets"
MODELS_DIR = Path(__file__).parent / "data" / "models"


@dataclass(frozen=True)
class MentalHealthTarget:
    key: str
    condition: str
    target_name: str
    chembl_id: str
    example_drugs: tuple[str, ...]
    sample_file: str
    description: str

    @property
    def sample_path(self) -> Path:
        return SAMPLE_DIR / self.sample_file

    @property
    def cache_path(self) -> Path:
        return DATA_DIR / f"{self.key}_activities.csv"

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / f"{self.key}_model.joblib"

    @property
    def metrics_path(self) -> Path:
        return MODELS_DIR / f"{self.key}_metrics.txt"


MENTAL_HEALTH_TARGETS: dict[str, MentalHealthTarget] = {
    "alzheimers_dementia": MentalHealthTarget(
        key="alzheimers_dementia",
        condition="Alzheimer's / Dementia",
        target_name="Acetylcholinesterase (AChE)",
        chembl_id="CHEMBL220",
        example_drugs=("Donepezil", "Galantamine", "Rivastigmine"),
        sample_file="ache_sample.csv",
        description="Symptomatic cognitive decline; AChE inhibitors are standard care.",
    ),
    "depression": MentalHealthTarget(
        key="depression",
        condition="Depression",
        target_name="Serotonin transporter (SERT)",
        chembl_id="CHEMBL2284",
        example_drugs=("Fluoxetine", "Sertraline", "Escitalopram"),
        sample_file="sert_sample.csv",
        description="SSRIs block serotonin reuptake via SERT.",
    ),
    "schizophrenia": MentalHealthTarget(
        key="schizophrenia",
        condition="Schizophrenia",
        target_name="Dopamine D2 receptor",
        chembl_id="CHEMBL217",
        example_drugs=("Haloperidol", "Risperidone", "Olanzapine"),
        sample_file="d2_sample.csv",
        description="Antipsychotics often act on dopamine D2 receptors.",
    ),
    "anxiety": MentalHealthTarget(
        key="anxiety",
        condition="Anxiety",
        target_name="GABA-A receptor (benzodiazepine site)",
        chembl_id="CHEMBL209386",
        example_drugs=("Diazepam", "Lorazepam", "Alprazolam"),
        sample_file="gaba_sample.csv",
        description="Benzodiazepines enhance GABA-A signaling.",
    ),
    "adhd": MentalHealthTarget(
        key="adhd",
        condition="ADHD",
        target_name="Dopamine transporter (DAT)",
        chembl_id="CHEMBL238",
        example_drugs=("Methylphenidate", "Amphetamine"),
        sample_file="dat_sample.csv",
        description="Stimulants block dopamine reuptake via DAT.",
    ),
    "bipolar_depression": MentalHealthTarget(
        key="bipolar_depression",
        condition="Bipolar disorder / Depression",
        target_name="Monoamine oxidase A (MAO-A)",
        chembl_id="CHEMBL1951",
        example_drugs=("Phenelzine", "Tranylcypromine"),
        sample_file="maoa_sample.csv",
        description="MAO-A inhibitors raise monoamine levels (older antidepressant class).",
    ),
}


def all_targets() -> list[MentalHealthTarget]:
    return list(MENTAL_HEALTH_TARGETS.values())


def get_target(key: str) -> MentalHealthTarget:
    if key not in MENTAL_HEALTH_TARGETS:
        known = ", ".join(MENTAL_HEALTH_TARGETS)
        raise KeyError(f"Unknown target '{key}'. Known: {known}")
    return MENTAL_HEALTH_TARGETS[key]
