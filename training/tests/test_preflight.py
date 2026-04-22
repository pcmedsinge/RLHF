"""
Pre-flight Tests — Phase 2 DPO + Phase 3 GRPO
===============================================
Run BEFORE starting Jarvis to catch config, data, and type errors
without spending any GPU money.

Covers:
  - Config loads and all required fields exist
  - Learning rate and beta are float (not string)
  - GRPO-specific fields present and valid types
  - Dataset files exist and have correct schema
  - Approved rows count meets minimum threshold
  - LoRA target modules are valid strings
  - Inference params are sane ranges

Run with:
    python3 training/tests/test_preflight.py
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "run_config.yaml"

PASS = "[PASS]"
FAIL = "[FAIL]"
failures = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS} {name}")
    else:
        msg = f"  {FAIL} {name}" + (f" — {detail}" if detail else "")
        print(msg)
        failures.append(name)


# ── 1. Config loading ────────────────────────────────────────────────────────
print("\n[1] Config")
try:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    check("config loads without error", True)
except Exception as e:
    check("config loads without error", False, str(e))
    print("Cannot continue without valid config.")
    sys.exit(1)

required_sections = ["project", "model", "dataset", "training", "lora", "inference", "grpo", "reward"]
for section in required_sections:
    check(f"section '{section}' exists", section in cfg)

# ── 2. Type checks ───────────────────────────────────────────────────────────
print("\n[2] Type validation")
t = cfg.get("training", {})

lr = t.get("learning_rate")
check("learning_rate is float", isinstance(lr, float),
      f"got type={type(lr).__name__} value={lr!r} — use 0.000005 not 5e-6 in YAML")

beta = t.get("beta")
check("beta is float", isinstance(beta, float),
      f"got type={type(beta).__name__} value={beta!r}")

check("batch_size is int >= 1",
      isinstance(t.get("per_device_train_batch_size"), int) and t.get("per_device_train_batch_size", 0) >= 1)

check("gradient_accumulation_steps is int >= 1",
      isinstance(t.get("gradient_accumulation_steps"), int) and t.get("gradient_accumulation_steps", 0) >= 1)

check("num_train_epochs is int >= 1",
      isinstance(t.get("num_train_epochs"), int) and t.get("num_train_epochs", 0) >= 1)

inf = cfg.get("inference", {})
check("max_new_tokens is int > 0",
      isinstance(inf.get("max_new_tokens"), int) and inf.get("max_new_tokens", 0) > 0)
check("temperature is float in (0, 2)",
      isinstance(inf.get("temperature"), float) and 0 < inf.get("temperature", -1) < 2)
check("top_p is float in (0, 1]",
      isinstance(inf.get("top_p"), float) and 0 < inf.get("top_p", -1) <= 1)

# ── 3. LoRA config ───────────────────────────────────────────────────────────
print("\n[3] LoRA config")
lora = cfg.get("lora", {})
check("lora.r is int > 0",
      isinstance(lora.get("r"), int) and lora.get("r", 0) > 0)
check("lora.lora_alpha is int > 0",
      isinstance(lora.get("lora_alpha"), int) and lora.get("lora_alpha", 0) > 0)
check("lora.lora_dropout is float in [0, 1)",
      isinstance(lora.get("lora_dropout"), float) and 0 <= lora.get("lora_dropout", -1) < 1)
modules = lora.get("target_modules", [])
check("lora.target_modules is non-empty list of strings",
      isinstance(modules, list) and len(modules) > 0 and all(isinstance(m, str) for m in modules))

# ── 4. Dataset files ─────────────────────────────────────────────────────────
print("\n[4] Dataset files")
ds = cfg.get("dataset", {})
train_path = ROOT / ds.get("train_file", "")
eval_path  = ROOT / ds.get("eval_file", "")

check("train file exists", train_path.exists(), str(train_path))
check("eval file exists",  eval_path.exists(),  str(eval_path))

REQUIRED_KEYS = {"prompt", "chosen", "rejected", "metadata"}
MIN_APPROVED_TRAIN = 50

for split_name, path in [("train", train_path), ("eval", eval_path)]:
    if not path.exists():
        check(f"{split_name} schema valid", False, "file missing, skipping schema check")
        continue
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    schema_ok = all(REQUIRED_KEYS.issubset(r.keys()) for r in rows)
    no_empty  = all(r.get("prompt") and r.get("chosen") and r.get("rejected") for r in rows)
    check(f"{split_name} schema valid ({len(rows)} rows)", schema_ok)
    check(f"{split_name} no empty fields", no_empty)

    if split_name == "train":
        check(f"train has >= {MIN_APPROVED_TRAIN} rows", len(rows) >= MIN_APPROVED_TRAIN,
              f"got {len(rows)}")

# ── 5. GRPO config ───────────────────────────────────────────────────────────
print("\n[5] GRPO config")
g = cfg.get("grpo", {})
g_lr = g.get("learning_rate")
check("grpo.learning_rate is float",
      isinstance(g_lr, float),
      f"got type={type(g_lr).__name__} value={g_lr!r} — use 0.000005 not 5e-6")
check("grpo.num_train_epochs is int >= 1",
      isinstance(g.get("num_train_epochs"), int) and g.get("num_train_epochs", 0) >= 1)
check("grpo.num_generations is int >= 2",
      isinstance(g.get("num_generations"), int) and g.get("num_generations", 0) >= 2,
      "must be >= 2 for GRPO to compute group relative advantage")
check("grpo.max_new_tokens is int > 0",
      isinstance(g.get("max_new_tokens"), int) and g.get("max_new_tokens", 0) > 0)

rw = cfg.get("reward", {})
check("reward.correct_diagnosis is float",
      isinstance(rw.get("correct_diagnosis"), float))
check("reward.mentions_safety is float",
      isinstance(rw.get("mentions_safety"), float))
check("reward.suggests_tests is float",
      isinstance(rw.get("suggests_tests"), float))

# ── 6. Output dir writeable ──────────────────────────────────────────────────
print("\n[6] Output directories")
output_dir_dpo  = ROOT / t.get("output_dir", "training/dpo_adapter")
output_dir_grpo = ROOT / g.get("output_dir", "training/grpo_adapter")
for label, odir in [("dpo output_dir", output_dir_dpo), ("grpo output_dir", output_dir_grpo)]:
    try:
        odir.mkdir(parents=True, exist_ok=True)
        test_file = odir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        check(f"{label} is writeable", True)
    except Exception as e:
        check(f"{label} is writeable", False, str(e))

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
if failures:
    print(f"PRE-FLIGHT FAILED — {len(failures)} issue(s):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("PRE-FLIGHT PASSED — safe to run train_grpo_phase3.py on Jarvis.")
