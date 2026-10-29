"""Conversion of warehouse ``.img`` members to the PNG read by the pipeline."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image


def convert_img_bytes_to_png(data: bytes, output_path: Path) -> None:
    """Write page/frame zero from an image byte stream as a lossless PNG.

    PNG rather than JPEG: JPEG compression smears thin pen strokes and adds
    ringing around black-and-white scans.

    This deliberately mirrors the supplied ``imgreader.py`` logic, except it
    accepts bytes so the source ZIP member never needs to be extracted first.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(data)) as image:
        if getattr(image, "n_frames", 1) > 1:
            image.seek(0)
        if image.mode in ("RGBA", "LA", "P", "PA"):
            image = image.convert("RGB")
        image.save(output_path, "PNG", compress_level=1)
