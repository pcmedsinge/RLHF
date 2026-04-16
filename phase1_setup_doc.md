# Phase 1 Setup — Quick Reference

## 1. Jarvis Labs Instance

- Template: **Axolotl** (not VM, not PyTorch)
- GPU: **1 x L4** (24GB VRAM)
- Storage: **50GB**
- Always work in `/home/` — it is persistent across pause/resume
- `/root/` is wiped on pause or destroy

---

## 2. HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Create token → **User permission** → **Read** scope
3. On Jarvis terminal:
   ```bash
   huggingface-cli login
   # paste token, answer NO to git credential prompt
   ```
4. Accept Mistral model access at:
   https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

---

## 3. Clone Project on Jarvis

```bash
cd "$HOME"
rm -rf rlhf          # only if folder already exists
git clone https://github.com/pcmedsinge/RLHF.git rlhf
cd rlhf
```

---

## 4. Install Dependencies

```bash
pip install transformers>=4.41.0 trl>=0.9.0 peft>=0.11.0 \
    accelerate>=0.30.0 bitsandbytes>=0.43.0 datasets>=2.19.0 \
    evaluate>=0.4.2 scikit-learn sentencepiece pyyaml
```

---

## 5. Download Mistral-7B + Sanity Check

```bash
bash evaluation/jarvis_setup.sh
```

Expected last line: `=== Mistral-7B is ready! ===`

---

## 6. Run Baseline Inference

```bash
python3 evaluation/baseline_inference.py
```

Results saved to: `evaluation/baseline_results.json`

---

## 7. Pause Jarvis When Done

Pause instance from dashboard to stop charges. `/home/` data is safe.

---

## Phase 1 Baseline Results (reference)

| Metric | Score |
|--------|-------|
| correct_diagnosis | 2/3 (67%) |
| suggests_tests | 3/3 (100%) |
| mentions_safety | 0/3 (0%) |
| has_hallucination | 0/3 (0%) |
