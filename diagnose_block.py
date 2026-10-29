"""Find out what makes the AI Foundry guardrail block a crop.

Run after a batch, on the same machine:

    python diagnose_block.py                      # today's batch
    python diagnose_block.py --batch-id 20261029

For every cheque whose date or signature crop came back "blocked", the saved
crop (exactly what was sent) is sent again in a few variants, one change at a
time, and the outcome of each is printed:

  same            the identical request again (is the block consistent?)
  neutral prompt  same image, a one-line "what is in this image" question
                  (is it the image, or the image + our prompt together?)
  blank image     our prompt with a plain white image (is it the prompt alone?)
  ink only        background security pattern removed, dark pen/print kept
  no labels       date only: the printed TARIKH/DATE labels cut off the left

Every variant is a normal, single request with the same kind of content the
pipeline already sends; nothing here is meant to get around the guardrail.
The point is to give the platform team a precise report (which images, which
filter category) so a false positive can be fixed on their side, and to tell
us whether a cleaner crop is enough.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime

import cv2
import numpy as np

from app.services import foundry_client, prompts
from app.settings import settings

NEUTRAL_PROMPT = 'What is in this image? Reply with this JSON object: {"description": "one short sentence"}'


def _ink_only(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 25)
    return cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR)


def _blank_like(image: np.ndarray) -> np.ndarray:
    return np.full_like(image, 255)


def _no_labels(image: np.ndarray) -> np.ndarray:
    width = image.shape[1]
    return image[:, int(width * 0.22):]


def _describe(record: dict) -> str:
    if record["status"] != "ok":
        details = record.get("filter_details")
        return f"{record['status'].upper()} ({record['note']})" + (f"\n        filter: {json.dumps(details)[:600]}" if details else "")
    return "passed -> " + json.dumps(record["data"])[:150]


def _try(purpose: str, label: str, prompt: str, image: np.ndarray) -> None:
    try:
        record = foundry_client.ask_about_image(f"diag-{purpose}", prompts.SYSTEM_PROMPT, prompt, image)
        print(f"    {label:<15} {_describe(record)}")
    except foundry_client.FoundryError as error:
        print(f"    {label:<15} ERROR {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch-id", default=f"{datetime.now():%Y%m%d}")
    parser.add_argument("--limit", type=int, default=5, help="max blocked crops to test (default 5)")
    args = parser.parse_args()
    # The outcome of each variant is printed below; skip the duplicate log warnings.
    logging.getLogger(foundry_client.logger.name).setLevel(logging.ERROR)

    batch_dir = settings.batch_dir(args.batch_id)
    if not batch_dir.is_dir():
        print(f"No batch folder: {batch_dir}")
        return 1

    tested = 0
    for run_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir() and (path / "result.json").is_file()):
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        checks = [
            ("signature", result["signature"]["status"], run_dir / "signature" / "crop_sent.png", prompts.SIGNATURE_PROMPT),
            ("date", result["date"]["status"], run_dir / "date" / "crop.png", prompts.date_prompt(settings.digit_count)),
        ]
        for purpose, status, crop_path, prompt in checks:
            if status != "blocked" or tested >= args.limit:
                continue
            image = cv2.imread(str(crop_path))
            if image is None:
                print(f"{run_dir.name} {purpose}: crop image missing ({crop_path})")
                continue
            tested += 1
            print(f"\n{run_dir.name}  {purpose}  ({crop_path.name}, {image.shape[1]}x{image.shape[0]} px)")
            _try(purpose, "same", prompt, image)
            _try(purpose, "neutral prompt", NEUTRAL_PROMPT, image)
            _try(purpose, "blank image", prompt, _blank_like(image))
            ink = _ink_only(image)
            cv2.imwrite(str(crop_path.with_name(crop_path.stem + "_ink_only.png")), ink)
            _try(purpose, "ink only", prompt, ink)
            if purpose == "date":
                _try(purpose, "no labels", prompt, _no_labels(image))

    if not tested:
        print(f"No blocked crops in {batch_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
