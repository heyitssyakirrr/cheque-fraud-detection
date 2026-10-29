"""Resumable batch runner for a ZIP of .img files, or a folder of images.

Production: put ``20260916_cheques.zip`` in ``data/incoming`` and run
``python run.py --batch-id 20260916``.

UAT: point it at a folder of PNG/JPEG/TIFF files instead, read in place:
``python run.py --batch-id 20260916 --folder /path/to/pngs``.

The live output is written to ``data/batches/20260916/20260916_output.csv``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path

from app.settings import settings

if settings.threads_per_worker:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, str(settings.threads_per_worker))

from app.ingestion import (  # noqa: E402
    batch_lock, inventory_folder, inventory_zip, sha256_file, stage_local_zip,
)
from app.manifest import Manifest  # noqa: E402
from app.pipeline import process  # noqa: E402
from app.services import text_detection  # noqa: E402
from app.services.img_conversion import convert_img_bytes_to_png  # noqa: E402
from app.services.preprocessing import PreprocessOptions  # noqa: E402


def _warm_up() -> float:
    """Load PaddleOCR (used only to locate the date labels). Signature and
    date reading are AI Foundry calls, so there is no other model to load."""
    started = time.perf_counter()
    if not settings.date_use_template_zone:
        text_detection._engine()
    return time.perf_counter() - started


def _init_worker() -> None:
    import cv2
    if settings.threads_per_worker:
        cv2.setNumThreads(settings.threads_per_worker)
    _warm_up()


def _completion_path(directory: Path, cheque_id: str) -> Path:
    import hashlib
    return directory / f"{hashlib.sha256(cheque_id.encode()).hexdigest()}.json"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, separators=(",", ":"))
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _process_one(task: dict) -> dict:
    """Worker function: success is made durable before the parent sees it."""
    started, counter = datetime.now(), time.perf_counter()
    temp_png = Path(task["tmp_dir"]) / f"{uuid.uuid4().hex}.png"
    try:
        if task["source_dir"]:
            # Folder mode: the pipeline reads the image file directly.
            source = Path(task["source_dir"]) / task["zip_member"]
        else:
            with zipfile.ZipFile(task["zip_path"]) as archive:
                raw = archive.read(task["zip_member"])
            convert_img_bytes_to_png(raw, temp_png)
            source = temp_png
        _, result = process(source, task["options"], task["batch_id"], task["sequence"],
                            Path(task["zip_member"]).name)
        timings = result.get("timings", {})
        signature = result["signature"]
        # REVIEW: the AI guardrail blocked the crop or the reply was unusable,
        # so a person must look -- never report NO for a cheque nobody checked.
        if signature["status"] != "ok":
            signature_detected = "REVIEW"
        else:
            signature_detected = "YES" if signature["exists"] else "NO"
        row = {
            "batch_id": task["batch_id"], "cheque_id": task["cheque_id"], "bank_code": task["bank_code"],
            "sequence": task["sequence"], "filename": Path(task["zip_member"]).name,
            "signature_detected": signature_detected,
            "date": result["date"]["date"], "start_time": started.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(time.perf_counter() - counter, 2), "pixels": result.get("pixels", ""),
            "t_read": timings.get("read", ""), "t_preprocess": timings.get("preprocess", ""),
            "t_signature": timings.get("signature", ""), "t_ocr": timings.get("ocr", ""),
            "t_date": timings.get("date", ""), "error": "",
            "signature_count": signature["count"], "signature_status": signature["status"],
            "date_status": result["date"]["status"], "date_confidence": result["date"]["confidence"],
        }
        _atomic_json(_completion_path(Path(task["completion_dir"]), task["cheque_id"]), row)
        return {"cheque_id": task["cheque_id"], "ok": True}
    except Exception as error:  # A bad scan must not end the whole batch.
        return {"cheque_id": task["cheque_id"], "ok": False, "error": str(error)}
    finally:
        temp_png.unlink(missing_ok=True)


def _import_completions(manifest: Manifest, directory: Path, output_csv: Path) -> int:
    imported = 0
    for path in directory.glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            manifest.mark_done(row["cheque_id"], row)
            manifest.append_output(output_csv, row["cheque_id"])
            imported += 1
        except (OSError, ValueError, KeyError) as error:
            raise RuntimeError(f"Invalid completion record {path}: {error}") from error
    return imported


def _task(row, batch_id: str, zip_path: Path | None, source_dir: Path | None, options: PreprocessOptions,
          tmp_dir: Path, completion_dir: Path) -> dict:
    return {"cheque_id": row["cheque_id"], "bank_code": row["bank_code"], "zip_member": row["zip_member"],
            "sequence": row["sequence"], "batch_id": batch_id, "zip_path": str(zip_path or ""),
            "source_dir": str(source_dir or ""), "options": options,
            "tmp_dir": str(tmp_dir), "completion_dir": str(completion_dir)}


def _handle_result(manifest: Manifest, result: dict, completion_dir: Path, output_csv: Path) -> None:
    cheque_id = result["cheque_id"]
    if result["ok"]:
        row = json.loads(_completion_path(completion_dir, cheque_id).read_text(encoding="utf-8"))
        manifest.mark_done(cheque_id, row)
        manifest.append_output(output_csv, cheque_id)
    else:
        manifest.mark_failure(cheque_id, result["error"])


def _run_tasks(manifest: Manifest, tasks: list[dict], workers: int, completion_dir: Path, output_csv: Path) -> None:
    if not tasks:
        return
    if workers == 1:
        print(f"Warming up models... ({_warm_up():.2f}s)", flush=True)
        for index, task in enumerate(tasks, 1):
            manifest.mark_in_progress(task["cheque_id"])
            result = _process_one(task)
            _handle_result(manifest, result, completion_dir, output_csv)
            print(f"[{index}/{len(tasks)}] {task['zip_member']} -> {'DONE' if result['ok'] else 'FAILED'}", flush=True)
        return

    capacity, iterator, completed = workers * settings.batch_inflight_multiplier, iter(tasks), 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
        pending = {}
        def submit() -> bool:
            try:
                task = next(iterator)
            except StopIteration:
                return False
            manifest.mark_in_progress(task["cheque_id"])
            pending[executor.submit(_process_one, task)] = task
            return True
        while len(pending) < capacity and submit():
            pass
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = {"cheque_id": task["cheque_id"], "ok": False, "error": str(error)}
                _handle_result(manifest, result, completion_dir, output_csv)
                completed += 1
                print(f"[{completed}/{len(tasks)}] {task['zip_member']} -> {'DONE' if result['ok'] else 'FAILED'}", flush=True)
                submit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process one daily cheque ZIP (or a UAT image folder) resumably.")
    parser.add_argument("--batch-id", default=f"{datetime.now():%Y%m%d}", help="Date in YYYYMMDD form")
    parser.add_argument("--folder", type=Path, default=None,
                        help="UAT: read PNG/JPEG/TIFF files from this folder instead of the daily ZIP")
    args = parser.parse_args(argv)
    batch_id = args.batch_id
    if not (len(batch_id) == 8 and batch_id.isdigit()):
        parser.error("--batch-id must be YYYYMMDD")
    source_zip = settings.incoming_dir / settings.batch_zip_name_pattern.format(date=batch_id)
    batch_dir, completion_dir, tmp_dir = settings.batch_dir(batch_id), settings.batch_dir(batch_id) / "completion", settings.batch_dir(batch_id) / "tmp"
    output_csv, failed_csv = batch_dir / f"{batch_id}_output.csv", batch_dir / f"{batch_id}_failed.csv"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with batch_lock(batch_dir / ".lock"):
        for leftover in [*tmp_dir.glob("*.jpg"), *tmp_dir.glob("*.png")]:
            leftover.unlink(missing_ok=True)
        folder = args.folder or settings.uat_image_dir
        if folder is not None:
            source_dir, zip_path = folder.resolve(), None
            inventory = inventory_folder(source_dir)
            if not inventory:
                raise RuntimeError(f"Folder contains no PNG/JPEG/TIFF files: {source_dir}")
            # Keyed on the folder path, not its contents, so PNGs added later
            # join this batch instead of needing a new batch ID.
            fingerprint = f"folder:{source_dir}"
            print(f"Reading images directly from folder {source_dir}", flush=True)
        else:
            source_dir = None
            zip_path = stage_local_zip(source_zip, settings.batch_zip_path(batch_id))
            inventory = inventory_zip(zip_path)
            if not inventory:
                raise RuntimeError("ZIP contains no .img files under PBB/ or PIBB/")
            fingerprint = sha256_file(zip_path)
        manifest = Manifest(batch_dir / "manifest.db", batch_id, fingerprint, settings.max_attempts)
        try:
            manifest.seed(inventory, allow_additions=source_dir is not None)
            manifest.recover_interrupted()
            manifest.rebuild_output(output_csv)  # repairs crash ambiguity before new appends
            recovered = _import_completions(manifest, completion_dir, output_csv)
            todo = manifest.todo()
            workers = min(settings.recommend_workers(os.cpu_count() or 1, len(todo)), len(todo)) if todo else 1
            print(f"Batch {batch_id}: {len(inventory)} cheques; {len(todo)} to process; {workers} worker(s).", flush=True)
            if recovered:
                print(f"Recovered {recovered} durable completion(s).", flush=True)
            options = PreprocessOptions(
                max_long_edge=settings.preprocess_max_long_edge,
                contrast=settings.preprocess_contrast,
                denoise=settings.preprocess_denoise,
                deskew=settings.preprocess_deskew,
            )
            _run_tasks(manifest, [_task(row, batch_id, zip_path, source_dir, options, tmp_dir, completion_dir) for row in todo], workers, completion_dir, output_csv)
            manifest.write_failures(failed_csv)
            counts = manifest.counts()
            print(f"Output: {output_csv}\nStatus: {counts}", flush=True)
            if zip_path and not manifest.todo() and not manifest.failures() and settings.cleanup_zip_after_batch:
                zip_path.unlink(missing_ok=True)
            return 1 if manifest.todo() else (2 if manifest.failures() else 0)
        finally:
            manifest.close()


if __name__ == "__main__":
    raise SystemExit(main())
