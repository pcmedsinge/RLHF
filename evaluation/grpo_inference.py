"""
GRPO Inference Script — Phase 3 Evaluation
===========================================
Purpose: Load the GRPO-trained LoRA adapter on top of Mistral-7B and run
all 40 eval cases. Produces a 3-way comparison: Baseline vs DPO vs GRPO.

Analogy: The resident who learned from scored practice exams (GRPO) now
sits the same final exam as the resident who studied answer keys (DPO)
and the untrained resident (Baseline). Same cases, same scoring rubric.
"""
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Must match the keywords used in the reward function during training
# so that what the model was rewarded for is also what we measure.
DISEASE_SAFETY_TERMS: dict[str, list[str]] = {
    "dermatomyositis":              ["aspiration", "malignancy", "paraneoplastic", "malignancy screen"],
    "myasthenia gravis":            ["respiratory failure", "myasthenic crisis", "fvc", "bulbar"],
    "wilson disease":               ["chelation", "hepatic failure", "copper", "irreversible"],
    "addison disease":              ["adrenal crisis", "hydrocortisone", "adrenal insufficiency", "life-threatening"],
    "acute intermittent porphyria": ["respiratory paralysis", "hemin", "trigger", "hospitaliz"],
}

GENERIC_SAFETY_TERMS: list[str] = [
    "urgent", "emergenc", "life-threaten", "critical", "immediate",
    "without delay", "monitor", "hospitaliz", "crisis", "fatal", "serious complication",
    "safety", "prompt", "timely", "risk", "hazard", "admission", "serious",
]

TEST_KEYWORDS: list[str] = [
    "test", "biopsy", "imaging", "antibod", "screen", "lab",
    "ct", "mri", "exam", "workup", "confirm", "diagnos",
]

HALLUCINATION_TERMS: list[str] = [
    "fibromyalgia", "allergy", "viral syndrome", "stress reaction", "anxiety",
]


def load_grpo_model(base_model_name: str, adapter_path: Path):
    print(f"Loading tokenizer from adapter: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading base model (4-bit)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"Attaching GRPO LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    print(f"GRPO model loaded on {base_model.device}")
    return model, tokenizer


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 384) -> tuple:
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - start
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), elapsed


def score_response(response: str, case: dict) -> dict:
    disease = case["metadata"]["disease"].lower()
    resp_lower = response.lower()

    specific_terms = DISEASE_SAFETY_TERMS.get(disease, [])
    mentions_safety = (
        any(t in resp_lower for t in specific_terms)
        or any(t in resp_lower for t in GENERIC_SAFETY_TERMS)
    )

    return {
        "correct_diagnosis": disease in resp_lower,
        "suggests_tests":    any(kw in resp_lower for kw in TEST_KEYWORDS),
        "mentions_safety":   mentions_safety,
        "has_hallucination": any(wd in resp_lower for wd in HALLUCINATION_TERMS),
    }


def load_prior_results(path: Path) -> dict:
    """Load saved baseline or DPO results, keyed by case id."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {c["id"]: c for c in data.get("cases", [])}


def main() -> None:
    root          = Path(__file__).resolve().parents[1]
    eval_path     = root / "data" / "processed" / "phase1_eval.jsonl"
    adapter_path  = root / "training" / "grpo_adapter"
    baseline_path = root / "evaluation" / "baseline_results.json"
    dpo_path      = root / "evaluation" / "dpo_results.json"
    results_path  = root / "evaluation" / "grpo_results.json"

    base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"GRPO adapter not found at {adapter_path}. "
            "Run training/train_grpo_phase3.py first."
        )

    model, tokenizer = load_grpo_model(base_model_name, adapter_path)

    # Load all 40 eval cases
    cases = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            cases.append(json.loads(line))
    print(f"\nEvaluating GRPO on all {len(cases)} eval cases...\n")

    # Load prior results for 3-way comparison
    baseline_map = load_prior_results(baseline_path)
    dpo_map      = load_prior_results(dpo_path)

    results = []
    for i, case in enumerate(cases):
        print(f"\n{'='*80}")
        print(f"CASE {i+1}/{len(cases)}: {case['metadata']['disease']}")
        print(f"{'='*80}")

        response, elapsed = run_inference(model, tokenizer, case["prompt"])
        grpo_scores = score_response(response, case)

        baseline_scores = baseline_map.get(case["id"], {}).get("scores", {})
        dpo_scores      = dpo_map.get(case["id"], {}).get("dpo_scores", {})

        print(f"GRPO RESPONSE ({elapsed:.1f}s):\n{response}\n")
        print(f"GRPO     : {grpo_scores}")
        print(f"DPO      : {dpo_scores}")
        print(f"BASELINE : {baseline_scores}")

        results.append({
            "id":               case["id"],
            "disease":          case["metadata"]["disease"],
            "grpo_scores":      grpo_scores,
            "dpo_scores":       dpo_scores,
            "baseline_scores":  baseline_scores,
            "grpo_response":    response,
            "inference_time_sec": round(elapsed, 2),
        })

    # --- 3-way summary ---
    n = len(results)
    metrics = ["correct_diagnosis", "suggests_tests", "mentions_safety", "has_hallucination"]

    grpo_summary = {m: sum(1 for r in results if r["grpo_scores"].get(m))     for m in metrics}
    dpo_summary  = {m: sum(1 for r in results if r["dpo_scores"].get(m))      for m in metrics}
    base_summary = {m: sum(1 for r in results if r["baseline_scores"].get(m)) for m in metrics}

    print(f"\n{'='*80}")
    print("FINAL COMPARISON: Baseline vs DPO vs GRPO")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'Baseline':>10} {'DPO':>10} {'GRPO':>10}")
    print("-" * 60)
    for m in metrics:
        b = base_summary[m]
        d = dpo_summary[m]
        g = grpo_summary[m]
        print(f"{m:<25} {b:>7}/{n}   {d:>4}/{n}   {g:>4}/{n}")

    output = {
        "grpo_summary":     grpo_summary,
        "dpo_summary":      dpo_summary,
        "baseline_summary": base_summary,
        "cases":            results,
    }
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
