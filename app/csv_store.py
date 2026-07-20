"""
Appends every cheque extraction result to a local CSV file, one row per
upload - a simple, human-readable audit trail alongside the JSON response
returned to the UI. Mirrors the LLM Gateway's own request_log pattern:
one row written per attempt, structured fields first, raw model output
kept alongside for anyone auditing a result later.
"""

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import ChequeExtractionResult

CSV_PATH = Path(__file__).parent.parent / "cheque_results.csv"

FIELDNAMES = [
    "timestamp",
    "filename",
    "mime_type",
    "date",
    "date_confidence",
    "payee_name",
    "amount_numeric",
    "amount_words",
    "amount_match",
    "signature_present",
    "signature_confidence",
    "notes",
    "raw_gateway_response",
]


@dataclass
class ChequeResultRow:
    timestamp: str
    filename: str
    mime_type: str
    date: str | None = None
    date_confidence: str | None = None
    payee_name: str | None = None
    amount_numeric: str | None = None
    amount_words: str | None = None
    amount_match: bool | None = None
    signature_present: bool | None = None
    signature_confidence: str | None = None
    notes: str | None = None
    raw_gateway_response: str = ""


def build_row(filename: str, mime_type: str, extraction: ChequeExtractionResult, raw_response: str) -> ChequeResultRow:
    """Build a ChequeResultRow from the parsed extraction result."""
    return ChequeResultRow(
        timestamp=datetime.now(timezone.utc).isoformat(),
        filename=filename,
        mime_type=mime_type,
        date=extraction.date,
        date_confidence=extraction.date_confidence,
        payee_name=extraction.payee_name,
        amount_numeric=extraction.amount_numeric,
        amount_words=extraction.amount_words,
        amount_match=extraction.amount_match,
        signature_present=extraction.signature_present,
        signature_confidence=extraction.signature_confidence,
        notes=extraction.notes,
        raw_gateway_response=raw_response,
    )


def append_result(row: ChequeResultRow) -> None:
    """Append one row to cheque_results.csv, writing the header first if
    the file doesn't exist yet."""
    file_exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(row))