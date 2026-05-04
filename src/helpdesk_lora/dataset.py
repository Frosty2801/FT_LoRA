from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are an IT helpdesk triage assistant. Return only valid JSON with these fields: "
    "category, urgency, priority, summary, first_response."
)


INSTRUCTION = (
    "Classify the helpdesk ticket and produce triage JSON. "
    "Use category, urgency, priority, summary, and first_response."
)


def format_prompt(ticket: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{INSTRUCTION}\n\nTicket:\n{ticket}</s>\n"
        "<|assistant|>\n"
    )


def format_training_text(example: dict[str, Any]) -> str:
    return f"{format_prompt(example['input'])}{example['output']}</s>"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
