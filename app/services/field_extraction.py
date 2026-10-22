"""Conservative, review-assisted cheque field candidates from OCR lines."""

import re
from typing import Any


AMOUNT_PATTERN = re.compile(r"(?:RM\s*)?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")


def _normalize(value: str) -> str:
    """Normalize OCR text so labels still match despite punctuation or spaces."""
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _center_y(line: dict[str, Any]) -> float:
    return sum(point[1] for point in line["box"]) / 4


def _left_x(line: dict[str, Any]) -> float:
    return min(point[0] for point in line["box"])


def _right_x(line: dict[str, Any]) -> float:
    return max(point[0] for point in line["box"])


def _find_payee(lines: list[dict[str, Any]]) -> str:
    """Return the closest OCR line to the right of a PAY label, if present."""
    pay_label = next(
        (line for line in lines if "PAY" in _normalize(line["text"])),
        None,
    )
    if pay_label is None:
        return ""

    candidates = [
        line
        for line in lines
        if _left_x(line) > _right_x(pay_label)
        and abs(_center_y(line) - _center_y(pay_label)) < 70
    ]
    return min(candidates, key=_left_x)["text"] if candidates else ""


def _find_amount(lines: list[dict[str, Any]]) -> str:
    """Return the first plausible RM or comma-formatted amount candidate."""
    for line in lines:
        text = line["text"].upper()
        match = AMOUNT_PATTERN.search(text)
        if match and ("RM" in text or "," in match.group(0)):
            return match.group(0)
    return ""


def extract(lines: list[dict[str, Any]]) -> dict[str, str]:
    """Extract tentative payee and amount values; both require user review."""
    return {
        "payee_candidate": _find_payee(lines),
        "amount_candidate": _find_amount(lines),
        "status": "review_required",
    }