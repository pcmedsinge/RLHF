"""
GRPO Training Script — Phase 3
================================
Purpose: Fine-tune Mistral-7B using Group Relative Policy Optimization (GRPO).
Unlike DPO which learns from pre-labelled preference pairs, GRPO lets the model
generate its own responses, scores them with a reward function we define, and
reinforces the better ones.

Analogy: Instead of studying answer keys (DPO), the resident now takes practice
exams. For each case they write 4 answers, a supervisor scores each one on a
clear rubric, and the resident learns from the scores. The rubric is explicit:
  +1.0  if the correct disease name is in the answer
  +1.0  if a disease-specific safety concern is mentioned
  +0.5  if confirmatory tests are recommended
Maximum score per response: 2.5

Key difference from DPO:
  - No chosen/rejected pairs needed — only prompts
  - Reward signal is precise and auditable (we wrote the rules)
  - Model generates num_generations=4 completions per prompt,
    scores them all, then updates toward higher-scoring ones

VRAM note: generating 4 responses per step is heavier than DPO.
Expect ~25-35 min on Jarvis L4 (24GB).
"""
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer


# ---------------------------------------------------------------------------
# Disease-specific safety keywords used by the reward function.
# Each disease has terms that only appear in a response that genuinely
# understands the clinical urgency — not just generic "urgent" filler.
# ---------------------------------------------------------------------------
DISEASE_SAFETY_TERMS: dict[str, list[str]] = {
    "dermatomyositis":         ["aspiration", "malignancy", "paraneoplastic", "malignancy screen"],
    "myasthenia gravis":       ["respiratory failure", "myasthenic crisis", "fvc", "bulbar"],
    "wilson disease":          ["chelation", "hepatic failure", "copper", "irreversible"],
    "addison disease":         ["adrenal crisis", "hydrocortisone", "adrenal insufficiency", "life-threatening"],
    "acute intermittent porphyria": ["respiratory paralysis", "hemin", "trigger", "hospitaliz"],
}

# Generic safety fallback terms (partial credit if disease-specific not matched)
GENERIC_SAFETY_TERMS: list[str] = [
    "urgent", "emergenc", "life-threaten", "critical", "immediate",
    "without delay", "monitor", "hospitaliz", "crisis", "fatal", "serious complication",
]

TEST_KEYWORDS: list[str] = [
    "test", "biopsy", "imaging", "antibod", "screen", "lab",
    "ct", "mri", "exam", "workup", "confirm",
]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts_as_dataset(path: Path) -> Dataset:
    """Load JSONL — GRPO only needs prompts, not chosen/rejected pairs."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append({
                "prompt":  row["prompt"],
                "disease": row["metadata"]["disease"].lower(),
            })
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Reward function
# Called by GRPOTrainer once per generated response.
# Returns a list of float scores, one per response in the batch.
# ---------------------------------------------------------------------------
def build_reward_fn(reward_weights: dict):
    """
    Returns a reward function closed over the weight config.
    GRPOTrainer calls: reward_fn(prompts, completions, **kwargs)
      prompts:     list[str]  — the original prompts
      completions: list[str]  — the generated responses to score
    kwargs may include extra dataset columns (we pass 'disease').
    """
    w_dx    = reward_weights["correct_diagnosis"]
    w_safe  = reward_weights["mentions_safety"]
    w_tests = reward_weights["suggests_tests"]

    def reward_fn(prompts: list, completions: list, **kwargs) -> list[float]:
        diseases = kwargs.get("disease", [""] * len(completions))
        scores = []
        for resp, disease in zip(completions, diseases):
            resp_lower = resp.lower()
            score = 0.0

            # 1. Correct diagnosis named
            if disease and disease in resp_lower:
                score += w_dx

            # 2. Safety concern — prefer disease-specific, fall back to generic
            specific_terms = DISEASE_SAFETY_TERMS.get(disease, [])
            if any(t in resp_lower for t in specific_terms):
                score += w_safe
            elif any(t in resp_lower for t in GENERIC_SAFETY_TERMS):
                score += w_safe * 0.5   # partial credit for generic safety mention

            # 3. Confirmatory tests suggested
            if any(kw in resp_lower for kw in TEST_KEYWORDS):
                score += w_tests

            scores.append(score)
        return scores

    return reward_fn


def main() -> None:
    # --- Config ---
    root = Path(__file__).resolve().parents[1]
    cfg  = load_config(root / "config" / "run_config.yaml")

    model_name  = cfg["model"]["base_model"]
    train_path  = root / cfg["dataset"]["train_file"]
    eval_path   = root / cfg["dataset"]["eval_file"]
    g           = cfg["grpo"]
    output_dir  = root / g["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model  : {model_name}")
    print(f"Train  : {train_path}  ({sum(1 for _ in train_path.open())} rows)")
    print(f"Output : {output_dir}")

    # --- Dataset (prompts only — GRPO generates its own responses) ---
    train_dataset = load_prompts_as_dataset(train_path)
    eval_dataset  = load_prompts_as_dataset(eval_path)

    # --- Quantization ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # --- Tokenizer ---
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model ---
    print("Loading model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    # --- LoRA (same config as Phase 2 for fair comparison) ---
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- GRPO config ---
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=g["per_device_train_batch_size"],
        gradient_accumulation_steps=g["gradient_accumulation_steps"],
        learning_rate=g["learning_rate"],
        num_train_epochs=g["num_train_epochs"],
        num_generations=g["num_generations"],       # 4 responses per prompt
        max_completion_length=g["max_completion_length"],
        bf16=True,               # L4 supports BFloat16 natively; fp16 conflicts with BF16 model dtype
        gradient_checkpointing=True,
        logging_steps=5,
        report_to="none",
        remove_unused_columns=False,
    )

    # --- Reward function ---
    reward_fn = build_reward_fn(cfg["reward"])

    # --- GRPO Trainer ---
    # Unlike DPOTrainer, GRPOTrainer takes a reward_funcs argument instead of
    # chosen/rejected datasets. It handles generation internally.
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_fn,
        peft_config=peft_config,
    )

    print("\nStarting GRPO training...")
    print(f"  Epochs          : {g['num_train_epochs']}")
    print(f"  Generations/step: {g['num_generations']}")
    print(f"  Reward weights  : {cfg['reward']}")
    print()

    trainer.train()

    print("\nSaving GRPO adapter...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
