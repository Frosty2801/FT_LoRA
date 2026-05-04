from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


REQUIRED_FIELDS = ("category", "urgency", "priority", "summary", "first_response")


@dataclass(frozen=True)
class TriageLabel:
    category: str
    urgency: str
    priority: str
    summary: str
    first_response: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "category": self.category,
                "urgency": self.urgency,
                "priority": self.priority,
                "summary": self.summary,
                "first_response": self.first_response,
            },
            ensure_ascii=True,
        )


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from model text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    return value if isinstance(value, dict) else None


def has_required_fields(value: dict[str, Any]) -> bool:
    return all(field in value for field in REQUIRED_FIELDS)
