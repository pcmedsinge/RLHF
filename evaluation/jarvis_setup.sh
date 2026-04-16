#!/bin/bash
# ============================================================
# Run this ONCE on your Jarvis Labs L4 instance.
# It installs missing dependencies and downloads Mistral-7B.
# ============================================================
set -e

echo "=== Step 1: Checking GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== Step 2: Installing Python dependencies ==="
pip install --upgrade pip
pip install transformers>=4.41.0 trl>=0.9.0 peft>=0.11.0 \
    accelerate>=0.30.0 bitsandbytes>=0.43.0 datasets>=2.19.0 \
    evaluate>=0.4.2 scikit-learn sentencepiece pyyaml

echo ""
echo "=== Step 3: Logging into HuggingFace ==="
echo "You need a HuggingFace token with READ access."
echo "Get one from: https://huggingface.co/settings/tokens"
echo ""

# Check if already logged in
if python3 -c "from huggingface_hub import HfFolder; assert HfFolder.get_token()" 2>/dev/null; then
    echo "Already logged in to HuggingFace."
else
    echo "Please run: huggingface-cli login"
    echo "Then re-run this script."
    exit 1
fi

echo ""
echo "=== Step 4: Downloading Mistral-7B-Instruct-v0.2 (4-bit) ==="
echo "This downloads ~4GB and caches under ~/.cache/huggingface/"
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = 'mistralai/Mistral-7B-Instruct-v0.2'

print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name)

print('Downloading model in 4-bit quantization...')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map='auto',
)

print('Model loaded successfully!')
print(f'Model device: {model.device}')
print(f'Model dtype: {model.dtype}')

# Quick sanity check
inputs = tokenizer('Hello, I am a', return_tensors='pt').to(model.device)
out = model.generate(**inputs, max_new_tokens=20)
print(f'Sanity check output: {tokenizer.decode(out[0], skip_special_tokens=True)}')
print()
print('=== Mistral-7B is ready! ===')
"

echo ""
echo "=== Setup complete! ==="
echo "Next: run baseline_inference.py"
