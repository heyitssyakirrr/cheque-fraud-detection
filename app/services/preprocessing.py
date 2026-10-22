"""Safe, image-wide preprocessing for cheque scans and photographs."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.settings import settings


@dataclass(frozen=True)
class PreprocessOptions:
    """User-adjustable preprocessing settings."""

    max_long_edge: int = 2200
    min_short_edge: int = 1000
    max_upscale_factor: float = 4.0
    contrast: float = 1.15
    denoise: bool = True
    deskew: bool = True


def _deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a conservatively deskewed image and its applied rotation."""
    mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    points = np.column_stack(np.where(mask > 0))

    if len(points) < 100:
        return gray, 0.0

    angle = cv2.minAreaRect(points[:, ::-1].astype(np.float32))[2]
    angle = angle - 90 if angle > 45 else angle

    # A cheque should only need a small correction. Reject unreliable angles.
    if abs(angle) > 8:
        return gray, 0.0

    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated = cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, float(angle)


def preprocess(
    image_bgr: np.ndarray,
    options: PreprocessOptions,
) -> tuple[np.ndarray, dict[str, float | list[int]]]:
    """Normalize a cheque image while keeping its layout coordinates intact."""
    height, width = image_bgr.shape[:2]
    short_edge, long_edge = min(height, width), max(height, width)
    # Normalise small scans before OCR/YOLO.  The prior pipeline only ever
    # downscaled, which left low-resolution cheque photos at scale 1.0.
    scale = 1.0
    if short_edge < options.min_short_edge:
        scale = min(options.min_short_edge / short_edge, options.max_upscale_factor)
    if long_edge * scale > options.max_long_edge:
        scale = options.max_long_edge / long_edge

    if abs(scale - 1.0) > 1e-6:
        image_bgr = cv2.resize(
            image_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if options.denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)

    deskew_angle = 0.0
    if options.deskew:
        gray, deskew_angle = _deskew(gray)

    enhanced = cv2.convertScaleAbs(gray, alpha=options.contrast, beta=0)
    output_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    metadata = {
        "scale": round(scale, 4),
        "original_size": [int(width), int(height)],
        "deskew_degrees": round(deskew_angle, 3),
        "output_size": [int(enhanced.shape[1]), int(enhanced.shape[0])],
    }
    return output_bgr, metadata


def prepare_signature_image(
    image_bgr: np.ndarray,
    options: PreprocessOptions,
) -> tuple[np.ndarray, float]:
    """Resize-only image for signature YOLO.

    The dedicated signature project does not denoise, increase contrast, or
    deskew before inference.  Those transforms can erase light pen strokes,
    so OCR receives its enhanced image while YOLO receives this colour copy.

    Uses `settings.signature_max_long_edge` rather than
    `options.max_long_edge` -- the latter is what date_extraction.py's crop
    geometry is calibrated against, so it can't move. Signature detection
    has no such dependency and benefits from more resolution (matches the
    max_dim=3000 already validated in script.py), hence its own cap.
    """
    height, width = image_bgr.shape[:2]
    short_edge, long_edge = min(height, width), max(height, width)
    scale = 1.0
    if short_edge < options.min_short_edge:
        scale = min(options.min_short_edge / short_edge, options.max_upscale_factor)
    if long_edge * scale > settings.signature_max_long_edge:
        scale = settings.signature_max_long_edge / long_edge
    if abs(scale - 1.0) <= 1e-6:
        return image_bgr.copy(), 1.0
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=interpolation), float(scale)


def crop_date_zone(image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop to the corner holding the printed TARIKH/DATE labels before OCR.

    OCR cost scales with pixel count, and the labels only ever appear in the
    upper right of this template, so scanning the whole page to find them is
    the single most expensive avoidable step in the pipeline.

    Returns (cropped_image, (x_offset, y_offset)) so the detected boxes can be
    translated back into the full image's coordinate frame, leaving all
    downstream date geometry working in unchanged coordinates.
    """
    height, width = image_bgr.shape[:2]
    x0 = int(width * settings.ocr_zone_left_fraction)
    y0 = int(height * settings.ocr_zone_top_fraction)
    y1 = int(height * settings.ocr_zone_bottom_fraction)
    return image_bgr[y0:y1, x0:width], (x0, y0)


def crop_signature_zone(image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop to the region a signature reliably occupies on this cheque
    template, before running YOLO -- entirely separate from resolution/
    format tuning. Forcing the model to search a smaller, targeted area
    instead of the full cheque means genuine signature ink occupies far
    more of the input pixels at any given imgsz, rather than competing
    against the dense printed background/security pattern across the
    whole page. Mirrors the same fixed-fraction-of-page approach already
    used for the date fallback zone (see date_extraction._template_date_zone),
    just calibrated to where a signature sits instead of where the date does.

    Returns (cropped_image, (x_offset, y_offset)) so detection boxes can be
    translated back into the full image's coordinate frame afterward.
    """
    height, width = image_bgr.shape[:2]
    x0 = int(width * settings.signature_zone_left_fraction)
    y0 = int(height * settings.signature_zone_top_fraction)
    x1 = width
    y1 = int(height * settings.signature_zone_bottom_fraction)
    return image_bgr[y0:y1, x0:x1], (x0, y0)