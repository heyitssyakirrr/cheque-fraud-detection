"""PaddleOCR v2.10 text recognition with coordinate-rich output."""

import csv
import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.settings import settings

logger = logging.getLogger(__name__)


_ocr: Any | None = None
_ocr_lock = threading.Lock()


def _engine() -> Any:
    """Create PaddleOCR once because model loading is expensive."""
    global _ocr

    if _ocr is None:
        with _ocr_lock:
            if _ocr is None:
                from paddleocr import PaddleOCR

                # Use the shared FileReader models/ folder instead of
                # downloading (this environment has no internet access).
                local_kwargs = {}
                if settings.ocr_det_dir.exists() and settings.ocr_rec_dir.exists():
                    local_kwargs = {
                        "det_model_dir": str(settings.ocr_det_dir),
                        "rec_model_dir": str(settings.ocr_rec_dir),
                    }
                    if settings.ocr_cls_dir.exists():
                        local_kwargs["cls_model_dir"] = str(settings.ocr_cls_dir)
                    else:
                        logger.warning(
                            "cls model not found at %s; continuing without it",
                            settings.ocr_cls_dir,
                        )
                else:
                    logger.warning(
                        "OCR models not found at %s; PaddleOCR will attempt "
                        "to download (will fail without internet access)",
                        settings.ocr_det_dir.parent,
                    )

                # Paddle defaults this to 10 and ignores OMP_NUM_THREADS, so
                # it needs telling separately.
                if settings.threads_per_worker:
                    local_kwargs["cpu_threads"] = settings.threads_per_worker

                _ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    ocr_version="PP-OCRv4",
                    show_log=False,
                    **local_kwargs,
                )
    return _ocr


def _write_csv(output_dir: Path, records: list[dict[str, Any]]) -> None:
    """Save flattened box coordinates for convenient Excel inspection."""
    csv_path = output_dir / "detections.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["id", "text", "confidence", "box"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **record,
                    "box": ";".join(
                        f"{x},{y}" for x, y in record["box"]
                    ),
                }
            )


def detect(image: np.ndarray, output_dir: Path) -> list[dict[str, Any]]:
    """Run OCR, annotate every text box, and return structured line data."""
    save_artifacts = settings.save_artifacts
    if save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)

    raw_result = _engine().ocr(image, cls=True)
    raw_lines = raw_result[0] if raw_result and raw_result[0] else []

    records: list[dict[str, Any]] = []
    annotated = image.copy() if save_artifacts else None

    for index, (box, text_pair) in enumerate(raw_lines, start=1):
        text, confidence = text_pair
        points = [[round(float(x), 2), round(float(y), 2)] for x, y in box]
        record = {
            "id": index,
            "text": str(text),
            "confidence": round(float(confidence), 4),
            "box": points,
        }
        records.append(record)

        if not save_artifacts:
            continue

        contour = np.array(points, dtype=np.int32)
        cv2.polylines(annotated, [contour], True, (0, 0, 220), 2)
        x, y = contour[0]
        label = f"{text} ({confidence:.2f})"
        cv2.putText(
            annotated,
            label,
            (int(x), max(15, int(y) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 180),
            1,
            cv2.LINE_AA,
        )

    if save_artifacts:
        cv2.imwrite(str(output_dir / "annotated.png"), annotated)
        _write_csv(output_dir, records)
    return records