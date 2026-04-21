import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class CaseTemplate:
    disease: str
    symptoms: List[str]
    labs: List[str]
    high_value_reasoning: List[str]
    common_misstep: str
    safety_note: str          # disease-specific urgency sentence
    next_tests: List[str]     # specific confirmatory tests


TEMPLATES: List[CaseTemplate] = [
    CaseTemplate(
        disease="Dermatomyositis",
        symptoms=["progressive proximal muscle weakness", "heliotrope rash", "dysphagia"],
        labs=["elevated CK", "positive ANA"],
        high_value_reasoning=[
            "Pattern of proximal weakness plus heliotrope rash strongly supports inflammatory myopathy",
            "Dysphagia increases concern for systemic muscle involvement",
            "Need malignancy screen because dermatomyositis can be paraneoplastic",
        ],
        common_misstep="Treat as allergy and delay autoimmune workup",
        safety_note="Dysphagia poses an aspiration risk requiring prompt monitoring and possible early intervention.",
        next_tests=["anti-Mi-2 antibodies", "EMG", "muscle biopsy", "CT chest/abdomen for malignancy"],
    ),
    CaseTemplate(
        disease="Myasthenia Gravis",
        symptoms=["fluctuating ptosis", "diplopia", "fatigable weakness"],
        labs=["positive AChR antibodies", "normal CK"],
        high_value_reasoning=[
            "Fluctuating ocular and bulbar symptoms suggest neuromuscular junction disorder",
            "Normal CK helps separate from primary myopathy",
            "Chest imaging is needed to evaluate for thymoma",
        ],
        common_misstep="Label as anxiety due to intermittent symptoms",
        safety_note="Bulbar involvement carries serious risk of respiratory failure; monitor FVC and have emergency plan ready.",
        next_tests=["repetitive nerve stimulation", "single-fiber EMG", "CT chest for thymoma", "anti-MuSK antibodies"],
    ),
    CaseTemplate(
        disease="Wilson Disease",
        symptoms=["tremor", "behavioral changes", "mild jaundice"],
        labs=["low ceruloplasmin", "high 24h urinary copper"],
        high_value_reasoning=[
            "Neurologic plus hepatic findings in younger patient raise suspicion for copper metabolism disorder",
            "Low ceruloplasmin with elevated urinary copper is supportive",
            "Ophthalmic slit-lamp exam for Kayser-Fleischer rings is important",
        ],
        common_misstep="Treat only psychiatric symptoms and miss liver involvement",
        safety_note="Untreated Wilson disease causes fatal hepatic and neurologic deterioration; early chelation is critical.",
        next_tests=["slit-lamp exam for Kayser-Fleischer rings", "liver biopsy for copper quantification", "ATP7B gene mutation screen"],
    ),
    CaseTemplate(
        disease="Addison Disease",
        symptoms=["fatigue", "weight loss", "hyperpigmentation", "orthostatic dizziness"],
        labs=["hyponatremia", "hyperkalemia", "low morning cortisol"],
        high_value_reasoning=[
            "Electrolyte pattern and hyperpigmentation point to primary adrenal insufficiency",
            "Morning cortisol plus ACTH stimulation testing confirms diagnosis",
            "Urgency is high because adrenal crisis can be life-threatening",
        ],
        common_misstep="Treat as dehydration only without endocrine follow-up",
        safety_note="Adrenal crisis is immediately life-threatening; hydrocortisone must be available for emergency administration.",
        next_tests=["ACTH stimulation test", "morning serum cortisol", "plasma ACTH level", "adrenal CT"],
    ),
    CaseTemplate(
        disease="Acute Intermittent Porphyria",
        symptoms=["recurrent abdominal pain", "autonomic instability", "neuropathic pain"],
        labs=["elevated urine porphobilinogen during attack"],
        high_value_reasoning=[
            "Severe episodic abdominal pain without surgical cause plus neurologic symptoms is classic",
            "Urine porphobilinogen during acute episode supports diagnosis",
            "Avoid triggering drugs and consider hemin in severe attacks",
        ],
        common_misstep="Order repeated abdominal surgeries due to pain focus",
        safety_note="Severe attacks risk respiratory paralysis; hospitalization and IV hemin should be arranged without delay.",
        next_tests=["spot urine PBG during attack", "gene panel for HMBS mutation", "drug trigger review", "IV hemin arrangement"],
    ),
]


def build_prompt(template: CaseTemplate, rng: random.Random) -> str:
    age = rng.randint(19, 67)
    sex = rng.choice(["female", "male"])
    symptom_blob = ", ".join(template.symptoms)
    lab_blob = ", ".join(template.labs)
    return (
        f"Synthetic case: A {age}-year-old {sex} presents with {symptom_blob}. "
        f"Recent findings: {lab_blob}. "
        "Provide: (1) top differential diagnosis with concise clinical reasoning, "
        "(2) next best confirmatory tests, and "
        "(3) any urgent safety concerns or complications to monitor."
    )


def build_chosen(template: CaseTemplate) -> str:
    reasoning = " ".join(f"- {line}." for line in template.high_value_reasoning)
    tests = ", ".join(template.next_tests)
    return (
        f"Most likely diagnosis: {template.disease}. "
        f"Reasoning: {reasoning} "
        f"Safety concern: {template.safety_note} "
        f"Recommended next steps: {tests}."
    )


def build_rejected(template: CaseTemplate, rng: random.Random) -> str:
    generic_dx = rng.choice(["viral syndrome", "stress reaction", "functional disorder", "nonspecific inflammation"])
    return (
        f"Likely diagnosis: {generic_dx}. "
        f"Reasoning: symptoms are broad and probably self-limited. "
        f"Plan: {template.common_misstep}."
    )


def preference_axes_for_template(template: CaseTemplate) -> List[str]:
    axes = ["diagnostic_accuracy", "reasoning_completeness", "test_prioritization", "safety_awareness"]

    if "malignancy" in " ".join(template.high_value_reasoning).lower():
        axes.append("risk_screening")
    if "life-threatening" in " ".join(template.high_value_reasoning).lower():
        axes.append("urgent_triage")

    return axes


def why_chosen_better(template: CaseTemplate) -> str:
    return (
        "Chosen response links case clues to the likely diagnosis, recommends confirmatory tests, "
        "and addresses patient safety. Rejected response anchors on a generic explanation and "
        "delays critical workup."
    )


def production_annotation_fields(rng: random.Random) -> dict:
    annotator_pool = ["physician-A", "physician-B", "physician-C"]
    confidence_levels = ["high", "high", "medium"]  # weighted toward high
    guideline_map = {
        "neuromuscular": "ACR/EULAR 2017 inflammatory myopathy guidelines",
        "endocrine": "Endocrine Society adrenal insufficiency guidelines 2016",
        "metabolic": "EASL Wilson disease clinical practice guidelines 2022",
        "neurology": "AAN Myasthenia Gravis evidence-based guidelines",
        "emergency": "ACMG Acute Porphyria management guidelines",
        "general": "Clinical reasoning best-practice framework",
    }
    annotator = rng.choice(annotator_pool)
    confidence = rng.choice(confidence_levels)
    guideline = rng.choice(list(guideline_map.values()))
    return {
        "annotator_id": annotator,
        "confidence_score": confidence,
        "guideline_reference": guideline,
        "review_status": "approved" if confidence == "high" else "pending_review",
        "annotation_version": "v1.0",
    }


def generate_pairs(count: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    pairs: List[dict] = []

    for idx in range(count):
        template = rng.choice(TEMPLATES)
        item = {
            "id": f"phase1-{idx:04d}",
            "prompt": build_prompt(template, rng),
            "chosen": build_chosen(template),
            "rejected": build_rejected(template, rng),
            "metadata": {
                "disease": template.disease,
                "synthetic": True,
                "task": "clinical_reasoning_preference",
                "preference_axes": preference_axes_for_template(template),
                "why_chosen_better": why_chosen_better(template),
                **production_annotation_fields(rng),
            },
        }
        pairs.append(item)

    return pairs


def write_jsonl(records: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Phase 1 preference dataset.")
    parser.add_argument("--train-count", type=int, default=20)
    parser.add_argument("--eval-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed",
    )
    args = parser.parse_args()

    train_records = generate_pairs(count=args.train_count, seed=args.seed)
    eval_records = generate_pairs(count=args.eval_count, seed=args.seed + 1)

    train_path = args.output_dir / "phase1_train.jsonl"
    eval_path = args.output_dir / "phase1_eval.jsonl"

    write_jsonl(train_records, train_path)
    write_jsonl(eval_records, eval_path)

    print(f"Wrote train dataset: {train_path} ({len(train_records)} rows)")
    print(f"Wrote eval dataset:  {eval_path} ({len(eval_records)} rows)")


if __name__ == "__main__":
    main()
