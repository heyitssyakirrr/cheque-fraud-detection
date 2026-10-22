"""Orchestrate one cheque through all independent processing services."""

import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2

from app.services.date_extraction import extract as extract_date
from app.services.preprocessing import (
    PreprocessOptions,
    crop_date_zone,
    crop_signature_zone,
    prepare_signature_image,
    preprocess,
)
from app.services.signature_detection import detect as detect_signatures
from app.services.text_detection import detect as detect_text
from app.settings import settings


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", Path(value).stem.lower()).strip("-")[:48] or "cheque"


def _create_run_directory(batch_id: str, sequence: int, source_name: str) -> tuple[str, Path]:
    """Nest every image's folder inside its batch folder.

    Previously every run lived flat under data/runs/, distinguishable only
    by a timestamp embedded in the folder name -- hard to browse once more
    than one batch exists. Now each batch gets its own folder, and each
    image inside it gets a short, sequence-ordered subfolder, so
    data/batches/<batch_id>/ directly shows every image from that upload
    with nothing to disambiguate.
    """
    run_id = f"{sequence:03d}-{_safe_name(source_name)}"
    run_dir = settings.batches_dir / batch_id / run_id

    if settings.save_artifacts:
        for name in ("original", "preprocessed", "signature", "text", "date"):
            (run_dir / name).mkdir(parents=True, exist_ok=True)

    return run_id, run_dir


def _write_summary(run_dir: Path, result: dict[str, Any]) -> None:
    """Save the most useful result fields in a one-row CSV."""
    row = {
        "run_id": result["run_id"],
        "signature_exists": result["signature"]["exists"],
        "signature_count": result["signature"]["count"],
        "signature_status": result["signature"]["status"],
        "date": result["date"]["date"],
        "date_status": result["date"]["status"],
        "text_count": result["text"]["count"],
    }

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)


def process(
    source: Path,
    options: PreprocessOptions,
    batch_id: str,
    sequence: int,
    source_name: str,
) -> tuple[str, dict[str, Any]]:
    """Process one uploaded cheque and persist all intermediate artifacts.

    Only two crops are sent to AI Foundry -- the signature zone and the date
    boxes -- never the full cheque.
    """
    run_id, run_dir = _create_run_directory(batch_id, sequence, source_name)
    timings: dict[str, float] = {}
    mark = time.perf_counter()

    original = cv2.imread(str(source))
    if original is None:
        raise ValueError("Unsupported or corrupt image")
    timings["read"] = time.perf_counter() - mark
    mark = time.perf_counter()

    save_artifacts = settings.save_artifacts
    if save_artifacts:
        cv2.imwrite(str(run_dir / "original" / "cheque.png"), original)

    preprocessed, preprocessing_metadata = preprocess(original, options)
    if save_artifacts:
        cv2.imwrite(str(run_dir / "preprocessed" / "cheque.png"), preprocessed)
    timings["preprocess"] = time.perf_counter() - mark
    mark = time.perf_counter()

    signature_image, signature_scale = prepare_signature_image(original, options)
    preprocessing_metadata["signature_scale"] = round(signature_scale, 4)

    signature_zone, signature_offset = crop_signature_zone(signature_image)
    signature_result = detect_signatures(
        signature_zone,
        run_dir / "signature",
        crop_offset=signature_offset,
        full_image=signature_image,
    )
    timings["signature"] = time.perf_counter() - mark
    mark = time.perf_counter()

    if settings.date_use_template_zone:
        # extract_date falls back to fixed template geometry when no labels
        # are supplied, so the OCR stage can be skipped outright.
        text_lines = []
    else:
        ocr_zone, (ocr_x, ocr_y) = crop_date_zone(preprocessed)
        text_lines = detect_text(ocr_zone, run_dir / "text")
        for line in text_lines:
            line["box"] = [[x + ocr_x, y + ocr_y] for x, y in line["box"]]
    timings["ocr"] = time.perf_counter() - mark
    mark = time.perf_counter()
    date_result = extract_date(
        preprocessed,
        text_lines,
        run_dir / "date",
    )
    timings["date"] = time.perf_counter() - mark

    result: dict[str, Any] = {
        "run_id": run_id,
        "batch_id": batch_id,
        "sequence": sequence,
        "source_filename": source_name,
        "preprocessing": preprocessing_metadata,
        "pixels": f"{original.shape[1]}x{original.shape[0]}",
        "timings": {name: round(value, 2) for name, value in timings.items()},
        "signature": signature_result,
        "text": {
            "count": len(text_lines),
            "detections": text_lines,
        },
        "date": date_result,
    }

    if save_artifacts:
        result_path = run_dir / "result.json"
        temporary_result_path = result_path.with_suffix(".json.tmp")
        temporary_result_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_result_path, result_path)
        _write_summary(run_dir, result)
    return run_id, result
