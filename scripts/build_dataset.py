#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from helpdesk_lora.dataset import INSTRUCTION, write_jsonl
from helpdesk_lora.schema import TriageLabel


CATEGORIES = {
    "vpn_access": {
        "symptoms": [
            "VPN disconnects every few minutes",
            "MFA times out before the VPN connects",
            "the VPN client says authentication failed",
            "I cannot reach internal dashboards from home",
        ],
        "responses": [
            "Ask the user to retry MFA, confirm device time sync, and collect VPN client logs.",
            "Verify account status, MFA enrollment, and whether the issue happens on another network.",
        ],
    },
    "email_outlook": {
        "symptoms": [
            "Outlook crashes when I attach files",
            "mail search returns no results",
            "I cannot send email to external clients",
            "my mailbox says it is over quota",
        ],
        "responses": [
            "Check mailbox quota, Outlook profile health, and whether webmail has the same issue.",
            "Ask for the error message, affected recipients, and recent changes to Outlook add-ins.",
        ],
    },
    "hardware_laptop": {
        "symptoms": [
            "my laptop battery dies after one hour",
            "the screen flickers during video calls",
            "the keyboard stops responding",
            "the laptop overheats and shuts down",
        ],
        "responses": [
            "Collect asset tag, run hardware diagnostics, and arrange repair or replacement if needed.",
            "Ask for device model, operating system, and whether the problem happens while docked.",
        ],
    },
    "password_account": {
        "symptoms": [
            "I am locked out after too many login attempts",
            "my password reset link expired",
            "I changed my password and now Teams keeps asking me to sign in",
            "my account says disabled",
        ],
        "responses": [
            "Verify the user's identity, check account lockout status, and guide a secure password reset.",
            "Confirm identity, inspect sign-in logs, and re-enable access only after policy checks.",
        ],
    },
    "software_install": {
        "symptoms": [
            "I need permission to install Figma",
            "the finance app update fails",
            "I cannot install the printer driver",
            "the installer says administrator approval required",
        ],
        "responses": [
            "Confirm business justification, device name, software version, and approval requirements.",
            "Check software catalog availability and provide managed installation steps.",
        ],
    },
    "network_wifi": {
        "symptoms": [
            "office Wi-Fi drops during meetings",
            "the guest network never opens the login page",
            "my desk has very slow network speed",
            "I cannot connect to the corporate Wi-Fi",
        ],
        "responses": [
            "Ask for location, device type, network name, and run basic connectivity checks.",
            "Check access point health, signal strength, and whether nearby users are affected.",
        ],
    },
}

URGENCY_HINTS = {
    "low": [
        "when you have time",
        "no rush",
        "before next week",
        "this is annoying but I can work around it",
    ],
    "medium": [
        "today if possible",
        "it slows down my work",
        "I have a meeting later",
        "my team is waiting on this",
    ],
    "high": [
        "I am blocked",
        "client call in 20 minutes",
        "payroll is due today",
        "production release is waiting",
    ],
    "critical": [
        "entire department is blocked",
        "security incident",
        "customer outage",
        "executive presentation starts now",
    ],
}

PRIORITY_BY_URGENCY = {
    "low": "P4",
    "medium": "P3",
    "high": "P2",
    "critical": "P1",
}

NOISE = [
    "I already restarted twice.",
    "This started after the last update.",
    "Please do not close this ticket automatically.",
    "I am using a company laptop.",
    "It worked yesterday.",
    "My manager asked me to file this.",
]


def make_summary(category: str, symptom: str, urgency: str) -> str:
    clean = symptom.rstrip(".")
    return f"User reports {clean} in {category.replace('_', ' ')} with {urgency} urgency."


def clean_fragment(value: str) -> str:
    return value.strip().rstrip(".!?")


def make_example(rng: random.Random) -> dict[str, str]:
    category = rng.choice(list(CATEGORIES))
    category_data = CATEGORIES[category]
    symptom = rng.choice(category_data["symptoms"])
    urgency = rng.choice(list(URGENCY_HINTS))
    urgency_hint = rng.choice(URGENCY_HINTS[urgency])
    response = rng.choice(category_data["responses"])

    fragments = [
        symptom,
        urgency_hint,
        rng.choice(NOISE),
    ]
    rng.shuffle(fragments)
    ticket = ". ".join(clean_fragment(fragment) for fragment in fragments) + "."

    label = TriageLabel(
        category=category,
        urgency=urgency,
        priority=PRIORITY_BY_URGENCY[urgency],
        summary=make_summary(category, symptom, urgency),
        first_response=response,
    )

    return {
        "instruction": INSTRUCTION,
        "input": ticket,
        "output": label.to_json(),
    }


def build_rows(size: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    return [make_example(rng) for _ in range(size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic helpdesk triage dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/helpdesk"))
    parser.add_argument("--train-size", type=int, default=240)
    parser.add_argument("--eval-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows = build_rows(args.train_size, args.seed)
    eval_rows = build_rows(args.eval_size, args.seed + 10_000)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "eval.jsonl", eval_rows)

    label_map = {
        "categories": sorted(CATEGORIES),
        "urgencies": sorted(URGENCY_HINTS),
        "priorities": sorted(set(PRIORITY_BY_URGENCY.values())),
    }
    (args.output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(train_rows)} train rows and {len(eval_rows)} eval rows to {args.output_dir}")


if __name__ == "__main__":
    main()
