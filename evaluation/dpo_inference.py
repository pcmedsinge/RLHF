"""
DPO Inference Script — Phase 2 Evaluation
==========================================
Purpose: Load the DPO-trained LoRA adapter on top of Mistral-7B and run
the same 3 eval cases used in Phase 1 baseline. Compare results side by side.

Analogy: Re-running the Day-1 exam on the resident after mentoring sessions.
We use the same cases so the comparison is fair.
"""
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_dpo_model(base_model_name: str, adapter_path: Path):
    """Load base model and attach the DPO LoRA adapter."""
    print(f"Loading tokenizer: {base_model_name}")
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

    print(f"Attaching DPO LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    print(f"DPO model loaded on {base_model.device}")
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
    test_keywords = ["test", "biopsy", "imaging", "antibod", "screen", "lab", "ct", "mri", "exam"]
    safety_keywords = ["urgent", "emergenc", "life-threaten", "critical", "immediate", "safety"]
    wrong_dx = ["fibromyalgia", "allergy", "viral syndrome", "stress reaction"]
    return {
        "correct_diagnosis": disease in resp_lower,
        "suggests_tests": any(kw in resp_lower for kw in test_keywords),
        "mentions_safety": any(kw in resp_lower for kw in safety_keywords),
        "has_hallucination": any(wd in resp_lower for wd in wrong_dx),
    }


def load_baseline_results(baseline_path: Path) -> list:
    if not baseline_path.exists():
        return []
    with baseline_path.open("r", encoding="utf-8") as f:
        return json.load(f).get("cases", [])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    eval_path    = root / "data" / "processed" / "phase1_eval.jsonl"
    adapter_path = root / "training" / "dpo_adapter"
    baseline_path = root / "evaluation" / "baseline_results.json"
    results_path  = root / "evaluation" / "dpo_results.json"

    base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"DPO adapter not found at {adapter_path}. Run training/train_dpo_phase2.py first."
        )

    # --- Load model ---
    model, tokenizer = load_dpo_model(base_model_name, adapter_path)

    # --- Load eval cases (same 3 as baseline) ---
    cases = []
    with eval_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            cases.append(json.loads(line))

    # --- Load baseline for comparison ---
    baseline_cases = load_baseline_results(baseline_path)
    baseline_map = {c["id"]: c for c in baseline_cases}

    print(f"\nRunning DPO evaluation on {len(cases)} cases...\n")

    results = []
    for i, case in enumerate(cases):
        print(f"\n{'='*80}")
        print(f"CASE {i+1}: {case['metadata']['disease']}")
        print(f"{'='*80}")

        response, elapsed = run_inference(model, tokenizer, case["prompt"])
        scores = score_response(response, case)

        # Compare to baseline
        baseline = baseline_map.get(case["id"], {})
        baseline_scores = baseline.get("scores", {})

        print(f"DPO RESPONSE ({elapsed:.1f}s):\n{response}\n")
        print(f"DPO  SCORES : {scores}")
        print(f"BASE SCORES : {baseline_scores}")

        results.append({
            "id": case["id"],
            "disease": case["metadata"]["disease"],
            "dpo_scores": scores,
            "baseline_scores": baseline_scores,
            "dpo_response": response,
            "inference_time_sec": round(elapsed, 2),
        })

    # --- Summary ---
    n = len(results)
    dpo_summary = {
        "correct_diagnosis": sum(1 for r in results if r["dpo_scores"]["correct_diagnosis"]),
        "suggests_tests":    sum(1 for r in results if r["dpo_scores"]["suggests_tests"]),
        "mentions_safety":   sum(1 for r in results if r["dpo_scores"]["mentions_safety"]),
        "has_hallucination": sum(1 for r in results if r["dpo_scores"]["has_hallucination"]),
    }
    base_summary = {
        "correct_diagnosis": sum(1 for r in results if r["baseline_scores"].get("correct_diagnosis")),
        "suggests_tests":    sum(1 for r in results if r["baseline_scores"].get("suggests_tests")),
        "mentions_safety":   sum(1 for r in results if r["baseline_scores"].get("mentions_safety")),
        "has_hallucination": sum(1 for r in results if r["baseline_scores"].get("has_hallucination")),
    }

    print(f"\n{'='*80}")
    print("COMPARISON: Baseline vs DPO (after RLHF)")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'Baseline':>10} {'DPO':>10} {'Change':>10}")
    print("-" * 55)
    for k in dpo_summary:
        b = base_summary.get(k, 0)
        d = dpo_summary[k]
        change = f"+{d-b}" if d >= b else str(d - b)
        print(f"{k:<25} {b:>7}/{n}     {d:>4}/{n}    {change:>6}")

    output = {"dpo_summary": dpo_summary, "baseline_summary": base_summary, "cases": results}
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
