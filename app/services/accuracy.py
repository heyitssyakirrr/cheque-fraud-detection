"""Compare user-supplied ground truth against a batch's actual results.

Two things get scored per cheque:
- Signature count: does the number of accepted signature detections match
  what a human counted on the actual cheque?
- Date digits: compared position-by-position against the ground truth
  6-digit string (or, if the ground truth says there's no date at all,
  scored as correct only if the pipeline also produced nothing usable).
"""

import csv
from pathlib import Path
from typing import Any


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    """Load ground truth from a CSV with columns: filename, signature_count, date.

    `date` should be DDMMYY, or "no date" for a cheque with no date written
    at all. Keyed by lowercased filename stem so it matches regardless of
    the uploaded file's extension or case. Returns an empty dict (nothing
    scored) if the file doesn't exist yet -- there's no ground truth to
    compare against until you create it.
    """
    if not path.is_file():
        return {}

    ground_truth: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            filename = (row.get("filename") or "").strip()
            signature_count = (row.get("signature_count") or "").strip()
            date_raw = (row.get("date") or "").strip()
            if not filename or not signature_count.lstrip("-").isdigit():
                continue
            date_text = date_raw.lower()
            date_value = None if date_text in ("no date", "none", "", "-") else "".join(filter(str.isdigit, date_raw))
            ground_truth[Path(filename).stem.lower()] = {
                "signature_count": int(signature_count),
                "date": date_value or None,
            }
    return ground_truth


def _score_signature(predicted_count: int, gt_count: int) -> dict[str, Any]:
    return {
        "gt_count": gt_count,
        "predicted_count": predicted_count,
        "match": predicted_count == gt_count,
        "difference": predicted_count - gt_count,
    }


def _score_digits(predicted_raw: str, gt_date: str | None) -> dict[str, Any]:
    if gt_date is None:
        # Ground truth says the cheque has no date at all. Since there's
        # currently no dedicated "no date detected" status, this is scored
        # as correct only if the classifier didn't produce a full 6-digit
        # reading -- an empty/short raw_digits string.
        no_date_predicted = not predicted_raw or len(predicted_raw) != 6
        return {
            "gt_date": "no date",
            "predicted_raw": predicted_raw or "(none)",
            "total_digits": None,
            "correct_digits": None,
            "digit_accuracy_pct": None,
            "per_digit_correct": None,
            "exact_match": no_date_predicted,
        }

    predicted = (predicted_raw or "")[:6].ljust(6, "?")
    per_digit_correct = [p == g for p, g in zip(predicted, gt_date)]
    correct = sum(per_digit_correct)
    return {
        "gt_date": gt_date,
        "predicted_raw": predicted_raw or "(none)",
        "total_digits": 6,
        "correct_digits": correct,
        "digit_accuracy_pct": round(correct / 6 * 100, 1),
        "per_digit_correct": per_digit_correct,
        "exact_match": predicted_raw == gt_date,
    }


def build_accuracy_report(ground_truth: dict[str, dict[str, Any]], completed: list[dict]) -> dict[str, Any]:
    """Score every completed cheque that has matching ground truth.

    Cheques with no corresponding ground truth entry are silently skipped
    (not everything in a batch necessarily has known-correct answers yet).
    """
    rows: list[dict[str, Any]] = []
    for item in completed:
        stem = Path(item["filename"]).stem.lower()
        gt = ground_truth.get(stem)
        if gt is None:
            continue

        result = item["result"]
        signature = result["signature"]
        if "count" in signature:
            predicted_signature_count = signature["count"]
        else:  # batches processed before the switch to AI Foundry (YOLO boxes)
            predicted_signature_count = sum(1 for d in signature.get("detections", []) if d["accepted"])
        rows.append(
            {
                "filename": item["filename"],
                "run_id": item["run_id"],
                "signature": _score_signature(predicted_signature_count, gt["signature_count"]),
                "digit": _score_digits(result["date"]["raw_digits"], gt["date"]),
            }
        )

    signature_matches = sum(1 for row in rows if row["signature"]["match"])
    total_gt_signatures = sum(row["signature"]["gt_count"] for row in rows)
    total_predicted_signatures = sum(row["signature"]["predicted_count"] for row in rows)

    # Cheques that actually have at least one signature per ground truth --
    # "did we find *a* signature on cheques that have one" is a different,
    # more forgiving question than "did we get the exact count right".
    cheques_with_signatures = [row for row in rows if row["signature"]["gt_count"] > 0]
    cheques_with_at_least_one_detected = sum(
        1 for row in cheques_with_signatures if row["signature"]["predicted_count"] > 0
    )

    scored_for_digits = [row for row in rows if row["digit"]["total_digits"] is not None]
    total_digit_correct = sum(row["digit"]["correct_digits"] for row in scored_for_digits)
    total_digit_count = sum(row["digit"]["total_digits"] for row in scored_for_digits)
    exact_date_matches = sum(1 for row in rows if row["digit"]["exact_match"])

    return {
        "rows": rows,
        "summary": {
            "cheques_scored": len(rows),
            "signature_matches": signature_matches,
            "total_gt_signatures": total_gt_signatures,
            "total_predicted_signatures": total_predicted_signatures,
            "signature_detection_rate_pct": (
                round(total_predicted_signatures / total_gt_signatures * 100, 1) if total_gt_signatures else None
            ),
            "cheques_with_signatures": len(cheques_with_signatures),
            "cheques_with_at_least_one_detected": cheques_with_at_least_one_detected,
            "presence_detection_rate_pct": (
                round(cheques_with_at_least_one_detected / len(cheques_with_signatures) * 100, 1)
                if cheques_with_signatures
                else None
            ),
            "total_digit_correct": total_digit_correct,
            "total_digit_count": total_digit_count,
            "digit_accuracy_pct": round(total_digit_correct / total_digit_count * 100, 1) if total_digit_count else None,
            "exact_date_matches": exact_date_matches,
            "exact_date_match_pct": round(exact_date_matches / len(rows) * 100, 1) if rows else None,
        },
    }