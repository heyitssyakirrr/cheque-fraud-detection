"""Conversion of warehouse ``.img`` members to the JPEG expected by the pipeline."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image


def convert_img_bytes_to_jpg(data: bytes, output_path: Path, quality: int = 95) -> None:
    """Write page/frame zero from an image byte stream as a JPEG.

    This deliberately mirrors the supplied ``imgreader.py`` logic, except it
    accepts bytes so the source ZIP member never needs to be extracted first.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(data)) as image:
        if getattr(image, "n_frames", 1) > 1:
            image.seek(0)
        if image.mode in ("RGBA", "LA", "P", "PA"):
            image = image.convert("RGB")
        image.save(output_path, "JPEG", quality=quality)
