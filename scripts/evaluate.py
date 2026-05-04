#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from helpdesk_lora.dataset import format_prompt, read_jsonl
from helpdesk_lora.schema import has_required_fields, parse_json_object


def generate(model, tokenizer, ticket: str, max_new_tokens: int) -> str:
    inputs = tokenizer(format_prompt(ticket), return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JSON validity and category accuracy.")
    parser.add_argument("--adapter", type=Path, default=Path("outputs/helpdesk-qlora"))
    parser.add_argument("--eval-file", type=Path, default=Path("data/helpdesk/eval.jsonl"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    args = parser.parse_args()

    rows = read_jsonl(args.eval_file)[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    valid_json = 0
    required_fields = 0
    category_correct = 0

    for row in rows:
        expected = json.loads(row["output"])
        prediction_text = generate(model, tokenizer, row["input"], args.max_new_tokens)
        prediction = parse_json_object(prediction_text)

        if prediction is None:
            continue

        valid_json += 1
        if has_required_fields(prediction):
            required_fields += 1
        if prediction.get("category") == expected["category"]:
            category_correct += 1

    total = len(rows)
    print(json.dumps(
        {
            "examples": total,
            "valid_json_rate": valid_json / total if total else 0.0,
            "required_fields_rate": required_fields / total if total else 0.0,
            "category_accuracy": category_correct / total if total else 0.0,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
