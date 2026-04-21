"""
DPO Training Script — Phase 2
===============================
Purpose: Fine-tune Mistral-7B using Direct Preference Optimization on
approved clinical preference pairs. Uses LoRA (Low-Rank Adaptation) so
training fits in 24GB VRAM on Jarvis L4.

Analogy: This is the mentoring phase. The resident (Mistral-7B) studies
approved case pairs (chosen > rejected). Over many examples, it learns
to prefer the senior consultant's reasoning style.

Key parameters:
  - beta=0.1: controls how close the trained model stays to the base model
    (lower = more conservative, higher = more aggressive preference shift)
  - LoRA r=16: how much "extra capacity" we add for learning preferences
    (like giving the resident a small notebook to write adaptation notes)
  - 4-bit quantization: keeps base model in memory efficiently
"""
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl_as_dataset(path: Path) -> Dataset:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append({
                "prompt":   row["prompt"],
                "chosen":   row["chosen"],
                "rejected": row["rejected"],
            })
    return Dataset.from_list(rows)


def main() -> None:
    # --- Load config ---
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config" / "run_config.yaml")

    model_name = cfg["model"]["base_model"]
    train_path  = root / cfg["dataset"]["train_file"]
    eval_path   = root / cfg["dataset"]["eval_file"]
    output_dir  = root / cfg["training"]["output_dir"]

    print(f"Model  : {model_name}")
    print(f"Train  : {train_path}  ({sum(1 for _ in train_path.open())} rows)")
    print(f"Eval   : {eval_path}   ({sum(1 for _ in eval_path.open())} rows)")
    print(f"Output : {output_dir}")

    # --- Load datasets ---
    train_dataset = load_jsonl_as_dataset(train_path)
    eval_dataset  = load_jsonl_as_dataset(eval_path)

    # --- Quantization config (4-bit to fit L4 24GB) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # --- Load base model + tokenizer ---
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # --- LoRA config ---
    # LoRA adds small trainable matrices to the model.
    # We only train those small matrices, not the full model.
    # Analogy: resident writes notes in a small notebook, not rewriting the textbook.
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- DPO Training config ---
    t = cfg["training"]
    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        beta=t["beta"],                                        # preference strength
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        num_train_epochs=t["num_train_epochs"],
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",                                      # disable wandb for simplicity
        remove_unused_columns=False,
        max_length=512,
        max_prompt_length=256,
    )

    # --- DPO Trainer ---
    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # None = TRL auto-creates reference from base model
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # --- Train ---
    print("\n=== Starting DPO Training ===")
    print("This will take ~15-30 min on L4 for 1 epoch over 139 examples.")
    print("Watch for 'loss' to decrease over steps — that means the model is learning.\n")

    trainer.train()

    # --- Save adapter ---
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\n=== DPO Training Complete ===")
    print(f"LoRA adapter saved to: {output_dir}")
    print("Next: run evaluation/dpo_inference.py to compare DPO model vs baseline")


if __name__ == "__main__":
    main()
