"""Find the printed date label with PaddleOCR, crop the handwritten date
boxes beside it, and have AI Foundry read the digits in that crop.

PaddleOCR only locates the printed TARIKH/DATE labels; it runs locally and
nothing it reads leaves the server. Only the date-box crop is sent."""

import calendar
import csv
import json
from difflib import SequenceMatcher
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services import foundry_client, prompts
from app.settings import settings


def _normalise(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _bounds(box: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return min(xs), min(ys), max(xs), max(ys)


def _score_candidates(
    lines: list[dict[str, Any]],
    aliases: tuple[str, ...],
    image_shape: tuple[int, ...],
    allow_weak_wn: bool = False,
) -> dict[str, Any] | None:
    """Find the best-scoring line matching any of `aliases`."""
    normalised_aliases = [_normalise(alias) for alias in aliases]
    height, width = image_shape[:2]
    best: tuple[float, dict[str, Any]] | None = None
    for line in lines:
        text = _normalise(str(line["text"]))
        if len(text) < 2:
            continue
        direct = [alias for alias in normalised_aliases if alias in text or text in alias]
        similarity = max(
            (SequenceMatcher(None, text, alias).ratio() for alias in normalised_aliases if len(alias) >= 3),
            default=0.0,
        )
        x0, y0, x1, y1 = _bounds(line["box"])
        weak_wn = (
            allow_weak_wn
            and text == "WN"
            and (x0 + x1) / 2 >= width * 0.55
            and (y0 + y1) / 2 <= height * 0.50
        )
        if not direct and similarity < 0.66 and not weak_wn:
            continue
        score = float(line["confidence"]) + similarity + (2.0 if text in direct else 0.0)
        if best is None or score > best[0]:
            best = (score, line)
    return best[1] if best else None


def _find_label_lines(
    lines: list[dict[str, Any]], image_shape: tuple[int, ...]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Locate TARIKH and DATE as the two independent, stacked printed lines
    they actually are on a Public Bank cheque (TARIKH above, DATE below)."""
    tarikh = _score_candidates(lines, settings.date_label_aliases_tarikh, image_shape)
    date = _score_candidates(
        lines,
        settings.date_label_aliases_date + settings.date_label_aliases_ambiguous,
        image_shape,
        allow_weak_wn=True,
    )

    if tarikh is not None and date is not None and tarikh is not date:
        tx0, ty0, tx1, ty1 = _bounds(tarikh["box"])
        dx0, dy0, dx1, dy1 = _bounds(date["box"])
        x_overlap = min(tx1, dx1) - max(tx0, dx0)
        vertically_ordered = dy0 >= ty0
        if x_overlap <= 0 or not vertically_ordered:
            if float(tarikh["confidence"]) >= float(date["confidence"]):
                date = None
            else:
                tarikh = None

    return tarikh, date


def _vertical_bounds_from_labels(
    tarikh: dict[str, Any] | None,
    date: dict[str, Any] | None,
    lines: list[dict[str, Any]],
) -> tuple[float, float, bool]:
    """Compute the crop's top/bottom Y directly from the label line(s) found.

    Returns (top_y, bottom_y, was_estimated).
    """
    typical_height = _typical_line_height(lines) or 20.0
    line_gap = typical_height * settings.date_label_line_gap_ratio

    if tarikh is not None and date is not None:
        _, top_y, _, _ = _bounds(tarikh["box"])
        _, _, _, bottom_y = _bounds(date["box"])
        return top_y, bottom_y, False

    if tarikh is not None:
        _, top_y, _, tarikh_bottom = _bounds(tarikh["box"])
        estimated_bottom = tarikh_bottom + line_gap + typical_height
        return top_y, estimated_bottom, True

    _, date_top, _, bottom_y = _bounds(date["box"])  # date is not None here
    estimated_top = date_top - line_gap - typical_height
    return estimated_top, bottom_y, True


def _typical_line_height(lines: list[dict[str, Any]]) -> float | None:
    """Robust reference height for one printed line of text on this cheque
    -- the median height across all detected lines, used as the sizing
    reference for crop width and padding (see module docstrings in prior
    versions for the full rationale)."""
    heights = []
    for line in lines:
        text = str(line["text"])
        if len(_normalise(text)) < 2:
            continue
        x0, y0, x1, y1 = _bounds(line["box"])
        h = y1 - y0
        if h > 0:
            heights.append(h)
    if len(heights) < 3:
        return None
    return float(np.median(heights))


def _right_of_labels(
    tarikh: dict[str, Any] | None,
    date: dict[str, Any] | None,
    lines: list[dict[str, Any]],
    image_shape: tuple[int, ...],
) -> tuple[tuple[int, int, int, int], bool]:
    """Crop the date row: from the TARIKH/DATE labels to the cheque's right edge.

    The labels stay in the crop so the model can see it is a date field, and
    the date boxes are the last thing on that row, so running to the right
    edge always keeps all six boxes regardless of how wide the writing is.
    """
    height, width = image_shape[:2]
    top_y, bottom_y, was_estimated = _vertical_bounds_from_labels(tarikh, date, lines)
    typical_height = _typical_line_height(lines) or max(1.0, bottom_y - top_y)

    left_edges = [_bounds(line["box"])[0] for line in (tarikh, date) if line is not None]
    label_x0 = min(left_edges)

    safety_pad = typical_height * settings.date_field_vertical_safety_pad_ratio
    bottom_extra_pad = typical_height * settings.date_field_bottom_extra_pad_ratio
    crop_y0 = top_y - safety_pad
    crop_y1 = bottom_y + safety_pad + bottom_extra_pad
    crop_x0 = label_x0 - typical_height * 0.5
    crop_x1 = width * settings.date_field_right_edge_fraction

    max_width = width * settings.date_field_max_width_fraction
    max_height = height * settings.date_field_max_height_fraction
    crop_x1 = min(crop_x1, crop_x0 + max_width)
    if crop_y1 - crop_y0 > max_height:
        crop_y1 = crop_y0 + max_height

    crop_box = (
        max(0, int(crop_x0)),
        max(0, int(crop_y0)),
        min(width, int(crop_x1)),
        min(height, int(crop_y1)),
    )
    return crop_box, was_estimated


def _template_date_zone(image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Fallback for a Public Bank image whose printed label is unreadable."""
    height, width = image_shape[:2]
    return int(width * 0.60), int(height * 0.10), int(width * 0.99), int(height * 0.38)


def _calendar_valid(digits: str) -> bool:
    """Validate a concatenated DDMMYY string (no separators, always 6 digits)."""
    if len(digits) != 6 or not digits.isdigit():
        return False
    day, month, year = int(digits[0:2]), int(digits[2:4]), int(digits[4:6])
    if not 1 <= month <= 12:
        return False
    return 1 <= day <= calendar.monthrange(2000 + year, month)[1]


def _write_csv(output_dir: Path, result: dict[str, Any]) -> None:
    if not settings.save_artifacts:
        return
    with (output_dir / "result.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow({key: str(value) if isinstance(value, list) else value for key, value in result.items()})


def _clean_digits(value: Any) -> str:
    """Keep digits and '?' placeholders only (drops spaces, slashes, dashes)."""
    return re.sub(r"[^0-9?]", "", str(value or ""))


def extract(
    image: np.ndarray,
    lines: list[dict[str, Any]],
    output_dir: Path,
    native: np.ndarray | None = None,
    native_scale: float = 1.0,
) -> dict[str, Any]:
    """PaddleOCR label -> right-side crop -> AI Foundry reads the digits.

    The box is located on `image` (the preprocessed OCR image). When
    `native` is given, the crop itself is cut from it instead -- the
    original pixels -- with the box divided by `native_scale`.
    """
    if settings.save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
    tarikh, date = _find_label_lines(lines, image.shape)
    bounds_estimated = False
    if tarikh is not None or date is not None:
        crop_box, bounds_estimated = _right_of_labels(tarikh, date, lines, image.shape)
        found_texts = [str(line["text"]) for line in (tarikh, date) if line is not None]
        anchor_text = " / ".join(dict.fromkeys(found_texts))
        found_labels = [name for name, line in (("TARIKH", tarikh), ("DATE", date)) if line is not None]
        anchor_mode = "OCR/fuzzy " + "+".join(found_labels) + " anchor"
        if bounds_estimated:
            anchor_mode += " (one line estimated from the other)"
    else:
        crop_box = _template_date_zone(image.shape)
        anchor_text = ""
        anchor_mode = "Public Bank upper-right fallback"

    if native is not None:
        height, width = native.shape[:2]
        crop_box = (
            max(0, int(crop_box[0] / native_scale)),
            max(0, int(crop_box[1] / native_scale)),
            min(width, round(crop_box[2] / native_scale)),
            min(height, round(crop_box[3] / native_scale)),
        )
        image = native
    x0, y0, x1, y1 = crop_box
    crop = image[y0:y1, x0:x1]
    result: dict[str, Any] = {
        "status": "needs_review",
        "date": "",
        "date_source": "foundry",
        "validation": f"crop source: {anchor_mode}",
        "raw_digits": "",
        "anchor_text": anchor_text,
        "confidence": "",
        "note": "",
        "crop_box": [x0, y0, x1, y1],
        "label_bounds_estimated": bounds_estimated,
        "llm": {},
    }
    if crop.size == 0:
        result["status"] = "empty_crop"
        _write_csv(output_dir, result)
        return result

    # Cut from the original pixels, then resized once to the upload size.
    crop = foundry_client.fit_for_upload(crop)
    if settings.save_artifacts:
        cv2.imwrite(str(output_dir / "crop.png"), crop)

    digit_count = settings.digit_count
    llm = foundry_client.ask_about_image("date", prompts.SYSTEM_PROMPT, prompts.date_prompt(digit_count), crop)
    result["llm"] = {key: value for key, value in llm.items() if key != "data"}
    result["note"] = llm["note"]

    if llm["status"] == "blocked":
        result.update(status="blocked", validation=f"{llm['note']} -- read the date manually; crop source: {anchor_mode}")
    elif llm["status"] == "unreadable":
        result.update(validation=f"{llm['note']} -- read the date manually; crop source: {anchor_mode}")
    else:
        data = llm["data"]
        raw_digits = _clean_digits(data.get("digits"))
        confidence = str(data.get("confidence") or "").strip().lower()
        result.update(
            raw_digits=raw_digits,
            confidence=confidence if confidence in ("high", "medium", "low") else "",
            note=str(data.get("note") or "")[:200],
        )
        formatted_date = (
            f"{raw_digits[0:2]}/{raw_digits[2:4]}/{raw_digits[4:6]}" if len(raw_digits) == 6 else raw_digits
        )
        if not raw_digits:
            result.update(status="no_date", validation=f"model found no handwritten date; crop source: {anchor_mode}")
        elif _calendar_valid(raw_digits):
            result.update(
                status="date_candidate",
                date=formatted_date,
                validation=f"format and calendar valid; crop source: {anchor_mode}",
            )
        else:
            result.update(
                status="needs_review",
                date=formatted_date,
                validation=f"reading is not a valid {digit_count}-digit calendar date -- verify manually; crop source: {anchor_mode}",
            )

    if settings.save_artifacts:
        (output_dir / "llm_response.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_csv(output_dir, result)
    return result