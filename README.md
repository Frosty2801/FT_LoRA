# Learn QLoRA Fine-Tuning With a Real Problem

This project teaches QLoRA by fine-tuning a small language model to solve a practical task:

> Convert messy IT helpdesk tickets into structured triage JSON.

The model learns to read a ticket like:

```text
I cannot connect to the VPN from home. It says MFA timeout and I have a client call in 20 minutes.
```

And produce:

```json
{
  "category": "vpn_access",
  "urgency": "high",
  "priority": "P2",
  "summary": "User cannot connect to VPN because MFA times out before a client call.",
  "first_response": "Ask the user to retry MFA, confirm device time sync, and provide temporary access steps if the issue continues."
}
```

The point is not to build a perfect helpdesk bot. The point is to learn the full fine-tuning loop:

1. Create a dataset.
2. Format it as instruction examples.
3. Fine-tune with QLoRA.
4. Run inference.
5. Evaluate whether the model returns valid structured output.

## Project Structure

```text
.
├── configs/
│   └── qlora_tinyllama.yaml
├── data/
│   └── README.md
├── scripts/
│   ├── build_dataset.py
│   ├── evaluate.py
│   ├── infer.py
│   └── train_qlora.py
├── src/
│   └── helpdesk_lora/
│       ├── __init__.py
│       ├── dataset.py
│       └── schema.py
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

QLoRA training works best on an NVIDIA GPU. The default config uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, which is intentionally small for learning.

## 1. Build the Dataset

```bash
python3 scripts/build_dataset.py --output-dir data/helpdesk --train-size 240 --eval-size 60
```

This creates:

```text
data/helpdesk/train.jsonl
data/helpdesk/eval.jsonl
data/helpdesk/label_map.json
```

Each row contains:

- `instruction`: what the model should do
- `input`: the raw ticket
- `output`: the expected JSON answer

## 2. Train With QLoRA

```bash
python3 scripts/train_qlora.py --config configs/qlora_tinyllama.yaml
```

For a much faster smoke test, especially on CPU, build a tiny dataset and use the fast config:

```bash
python3 scripts/build_dataset.py --output-dir data/helpdesk --train-size 24 --eval-size 6
python3 scripts/train_qlora.py --config configs/qlora_tinyllama_fast.yaml
```

The fast config only trains for 5 optimizer steps, uses shorter sequences, smaller LoRA adapters, and fewer target modules. It is for checking the pipeline, not for quality.

The adapter is saved to:

```text
outputs/helpdesk-qlora
```

QLoRA keeps the base model quantized in 4-bit precision and trains only small LoRA adapter weights. That makes this project possible on much smaller hardware than full fine-tuning.

## 3. Run Inference

```bash
python3 scripts/infer.py \
  --adapter outputs/helpdesk-qlora \
  --ticket "My laptop says disk almost full and Outlook keeps crashing when I attach files."
```

## 4. Evaluate

```bash
python3 scripts/evaluate.py \
  --adapter outputs/helpdesk-qlora \
  --eval-file data/helpdesk/eval.jsonl
```

The evaluator checks:

- Whether the model output is valid JSON.
- Whether required fields are present.
- Whether the predicted category matches the expected category.

## What You Should Experiment With

Try changing these values in `configs/qlora_tinyllama.yaml`:

- `lora_r`: LoRA rank. Higher can learn more, but uses more memory.
- `lora_alpha`: scaling factor for LoRA updates.
- `learning_rate`: too high can make the adapter unstable.
- `num_train_epochs`: more epochs can help or overfit.
- `max_seq_length`: longer examples cost more memory.
- `max_steps`: hard limit for training steps. Useful for quick tests.
- `target_modules`: fewer modules train faster, but usually learn less.

Good beginner experiments:

```text
lora_r: 8  -> 16
learning_rate: 2e-4 -> 1e-4
num_train_epochs: 1 -> 3
```

## Why This Is a Real Problem

Helpdesk teams often receive noisy tickets with missing details, emotional language, vague symptoms, and mixed urgency. A triage assistant can reduce manual sorting by producing consistent structured metadata.

This is exactly the kind of task where fine-tuning can help:

- The desired output format is strict.
- The domain vocabulary is specific.
- The data style differs from generic internet text.
- The model must follow a repeatable business rule.

## Notes

This project creates a synthetic dataset so you can learn the mechanics without private company data. In a real deployment, you would replace the generated examples with anonymized historical tickets reviewed by subject matter experts.
