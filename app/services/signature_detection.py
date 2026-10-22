"""Signature presence check: send the signature-zone crop to AI Foundry."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services import foundry_client, prompts
from app.settings import settings


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "yes"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no"):
        return False
    return None


def _as_count(value: Any) -> int | None:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _confidence(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in ("high", "medium", "low") else ""


def detect(
    image: np.ndarray,
    output_dir: Path,
    crop_offset: tuple[int, int] = (0, 0),
    full_image: np.ndarray | None = None,
) -> dict[str, Any]:
    """Ask Foundry whether the signature-zone crop holds a handwritten signature.

    `image` is the zone crop (see preprocessing.crop_signature_zone) and is
    the only thing sent. `crop_offset`/`full_image` are used only to draw
    the zone on the saved annotated image.

    Returns {"status", "exists", "count", "confidence", "note", "zone_box", "llm"}.
    status is "ok", or "blocked"/"unreadable" when there is no usable answer
    -- exists is then False and the cheque needs a human look.
    """
    save_artifacts = settings.save_artifacts
    if save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)

    offset_x, offset_y = crop_offset
    zone_h, zone_w = image.shape[:2]
    zone_box = [offset_x, offset_y, offset_x + zone_w, offset_y + zone_h]

    llm = foundry_client.ask_about_image("signature", prompts.SYSTEM_PROMPT, prompts.SIGNATURE_PROMPT, image)

    result: dict[str, Any] = {
        "status": llm["status"],
        "exists": False,
        "count": 0,
        "confidence": "",
        "note": llm["note"],
        "zone_box": zone_box,
        "llm": {key: value for key, value in llm.items() if key != "data"},
    }

    if llm["status"] == "ok":
        data = llm["data"]
        present = _as_bool(data.get("signature_present"))
        count = _as_count(data.get("signature_count"))
        if present is None:
            result.update(status="unreadable", note="reply had no true/false signature_present")
        else:
            # Keep presence and count consistent with each other.
            if present and not count:
                count = 1
            if not present:
                count = 0
            result.update(
                exists=present,
                count=count,
                confidence=_confidence(data.get("confidence")),
                note=str(data.get("note") or "")[:200],
            )

    if save_artifacts:
        cv2.imwrite(str(output_dir / "crop_sent.png"), image)
        canvas = (full_image if full_image is not None else image).copy()
        colour = (0, 170, 0) if result["exists"] else (0, 140, 255)
        cv2.rectangle(canvas, (zone_box[0], zone_box[1]), (zone_box[2], zone_box[3]), colour, 3)
        label = {
            "ok": f"signature: {'yes' if result['exists'] else 'no'} ({result['count']}) {result['confidence']}",
            "blocked": "signature: blocked by guardrail",
            "unreadable": "signature: unreadable reply",
        }[result["status"]]
        cv2.putText(canvas, label, (zone_box[0] + 4, max(22, zone_box[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
        cv2.imwrite(str(output_dir / "annotated.png"), canvas)
        (output_dir / "llm_response.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result
