#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from helpdesk_lora.dataset import format_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run helpdesk triage inference with a LoRA adapter.")
    parser.add_argument("--adapter", type=Path, default=Path("outputs/helpdesk-qlora"))
    parser.add_argument("--ticket", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    prompt = format_prompt(args.ticket)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    print(generated.strip())


if __name__ == "__main__":
    main()
