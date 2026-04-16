"""
Baseline Inference Script — Phase 1
====================================
Purpose: Load Mistral-7B (4-bit) and run it on 3 eval cases BEFORE any RLHF.
This establishes the "before" snapshot so we can measure improvement later.

Analogy: Testing a new resident's diagnostic skills on Day 1 before any mentoring.
"""
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """Load Mistral-7B in 4-bit quantization (~8GB VRAM on L4)."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print(f"Model loaded on {model.device}, dtype={model.dtype}")
    return model, tokenizer


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 384) -> str:
    """Generate a response for a single clinical case prompt."""
    # Mistral-Instruct chat format
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

    # Decode only the new tokens (skip the input prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response, elapsed


def load_eval_cases(eval_path: Path, n: int = 3):
    """Load first n cases from the eval dataset."""
    cases = []
    with eval_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            cases.append(json.loads(line))
    return cases


def score_response(response: str, case: dict) -> dict:
    """Simple keyword-based scoring to measure baseline quality."""
    disease = case["metadata"]["disease"].lower()
    chosen = case["chosen"].lower()
    resp_lower = response.lower()

    # 1. Did it mention the correct disease?
    correct_dx = disease in resp_lower

    # 2. Did it suggest relevant tests? (check if response has test-related words)
    test_keywords = ["test", "biopsy", "imaging", "antibod", "screen", "lab", "ct", "mri", "exam"]
    suggests_tests = any(kw in resp_lower for kw in test_keywords)

    # 3. Did it mention safety/urgency?
    safety_keywords = ["urgent", "emergenc", "life-threaten", "critical", "immediate", "safety"]
    mentions_safety = any(kw in resp_lower for kw in safety_keywords)

    # 4. Hallucination check — mentions diseases that are clearly wrong
    wrong_dx = ["fibromyalgia", "allergy", "viral syndrome", "stress reaction"]
    has_hallucination = any(wd in resp_lower for wd in wrong_dx)

    return {
        "correct_diagnosis": correct_dx,
        "suggests_tests": suggests_tests,
        "mentions_safety": mentions_safety,
        "has_hallucination": has_hallucination,
    }


def main():
    # --- Paths ---
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    eval_path = project_root / "data" / "processed" / "phase1_eval.jsonl"
    results_path = script_dir / "baseline_results.json"

    if not eval_path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {eval_path}")

    # --- Load model ---
    model, tokenizer = load_model()

    # --- Load eval cases ---
    cases = load_eval_cases(eval_path, n=3)
    print(f"\nRunning baseline on {len(cases)} eval cases...\n")
    print("=" * 80)

    # --- Run inference ---
    results = []
    for i, case in enumerate(cases):
        print(f"\n{'='*80}")
        print(f"CASE {i+1}: {case['metadata']['disease']}")
        print(f"{'='*80}")
        print(f"PROMPT:\n{case['prompt']}\n")

        response, elapsed = run_inference(model, tokenizer, case["prompt"])
        scores = score_response(response, case)

        print(f"MODEL RESPONSE ({elapsed:.1f}s):\n{response}\n")
        print(f"EXPECTED (chosen):\n{case['chosen']}\n")
        print(f"SCORES: {scores}")

        results.append({
            "id": case["id"],
            "disease": case["metadata"]["disease"],
            "prompt": case["prompt"],
            "model_response": response,
            "expected_chosen": case["chosen"],
            "scores": scores,
            "inference_time_sec": round(elapsed, 2),
        })

    # --- Summary ---
    n = len(results)
    summary = {
        "total_cases": n,
        "correct_diagnosis": sum(1 for r in results if r["scores"]["correct_diagnosis"]),
        "suggests_tests": sum(1 for r in results if r["scores"]["suggests_tests"]),
        "mentions_safety": sum(1 for r in results if r["scores"]["mentions_safety"]),
        "has_hallucination": sum(1 for r in results if r["scores"]["has_hallucination"]),
    }

    print(f"\n{'='*80}")
    print("BASELINE SUMMARY (before RLHF)")
    print(f"{'='*80}")
    for k, v in summary.items():
        pct = f" ({v}/{n} = {v/n*100:.0f}%)" if k != "total_cases" else ""
        print(f"  {k}: {v}{pct}")

    # --- Save ---
    output = {"summary": summary, "cases": results}
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
