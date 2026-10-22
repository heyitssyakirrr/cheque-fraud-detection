"""FastAPI web application for batch cheque processing."""

import asyncio
import csv
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from app.pipeline import process
from app.services.accuracy import build_accuracy_report, load_ground_truth
from app.services.preprocessing import PreprocessOptions
from app.settings import settings


app = FastAPI(title="Public Bank Berhad Cheque Processor")
templates = Jinja2Templates(directory=str(settings.root / "app" / "templates"))
app.mount("/static", StaticFiles(directory=str(settings.root / "app" / "static")), name="static")

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/tiff"}
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,80}")

# Single source of truth for what the batch CSV contains -- keeps the
# downloadable summary and the on-screen evidence panel showing the same
# underlying fields, rather than two hand-maintained lists drifting apart.
BATCH_SUMMARY_FIELDS = [
    "sequence", "filename", "run_id",
    "signature_detected", "signature_count", "signature_confidence", "signature_status",
    "date", "date_status", "date_confidence", "raw_digits", "date_validation",
    "text_region_count", "preprocessing_scale", "preprocessing_deskew_degrees",
]

# One SSE queue per batch currently being processed in the background.
# A batch_id present here means "still processing, connect to the stream";
# absent means "either finished (read the finished result.json files from
# disk) or never existed (404)".
_batch_queues: dict[str, "asyncio.Queue[str | None]"] = {}


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    defaults = {"max_edge": 3000, "contrast": 1.15}
    return templates.TemplateResponse(request, "index.html", {"defaults": defaults})


def _validate_settings(max_edge: int, contrast: float) -> None:
    valid = 800 <= max_edge <= 5000 and 0.5 <= contrast <= 2.5
    if not valid:
        raise HTTPException(422, "One or more processing settings are outside the permitted range.")


async def _save_valid_upload(upload: UploadFile) -> Path:
    """Validate one image and save it to a unique temporary upload path."""
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, f"{upload.filename}: upload PNG, JPEG, or TIFF only.")

    incoming_dir = settings.root / "data" / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    path = incoming_dir / f"{uuid.uuid4().hex}.upload"
    byte_count = 0

    try:
        with path.open("wb") as destination:
            while chunk := await upload.read(1024 * 1024):
                byte_count += len(chunk)
                if byte_count > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(
                        413,
                        f"{upload.filename}: maximum upload size is {settings.max_upload_mb} MB.",
                    )
                destination.write(chunk)

        try:
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as error:
            raise HTTPException(422, f"{upload.filename}: file is not a valid image.") from error

        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _batch_summary_row(item: dict) -> dict:
    """Flatten one cheque's result into the detailed batch CSV's columns."""
    result = item["result"]
    signature = result["signature"]
    return {
        "sequence": result["sequence"],
        "filename": item["filename"],
        "run_id": item["run_id"],
        "signature_detected": signature["exists"],
        "signature_count": signature.get("count", ""),
        "signature_confidence": signature.get("confidence", ""),
        "signature_status": signature.get("status", ""),
        "date": result["date"]["date"],
        "date_status": result["date"]["status"],
        "date_confidence": result["date"]["confidence"],
        "raw_digits": result["date"]["raw_digits"],
        "date_validation": result["date"]["validation"],
        "text_region_count": result["text"]["count"],
        "preprocessing_scale": result["preprocessing"]["scale"],
        "preprocessing_deskew_degrees": result["preprocessing"]["deskew_degrees"],
    }


def _write_batch_summary(batch_id: str, completed: list[dict]) -> None:
    batch_dir = settings.batches_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    with (batch_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BATCH_SUMMARY_FIELDS)
        writer.writeheader()
        for item in completed:
            writer.writerow(_batch_summary_row(item))


def _load_batch_results(batch_id: str) -> list[dict]:
    """Read every completed image's result.json from a finished batch's folder."""
    completed = []
    for run_dir in sorted(path for path in (settings.batches_dir / batch_id).iterdir() if path.is_dir()):
        result_path = run_dir / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        completed.append(
            {"filename": result.get("source_filename", run_dir.name), "run_id": run_dir.name, "result": result}
        )
    completed.sort(key=lambda item: item["result"].get("sequence", 0))
    return completed


def _write_accuracy_csv(batch_dir: Path, report: dict) -> None:
    fieldnames = [
        "filename", "run_id",
        "gt_signature_count", "predicted_signature_count", "signature_match",
        "gt_date", "predicted_raw", "correct_digits", "digit_accuracy_pct", "exact_match",
    ]
    with (batch_dir / "accuracy_report.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow(
                {
                    "filename": row["filename"],
                    "run_id": row["run_id"],
                    "gt_signature_count": row["signature"]["gt_count"],
                    "predicted_signature_count": row["signature"]["predicted_count"],
                    "signature_match": row["signature"]["match"],
                    "gt_date": row["digit"]["gt_date"],
                    "predicted_raw": row["digit"]["predicted_raw"],
                    "correct_digits": row["digit"]["correct_digits"],
                    "digit_accuracy_pct": row["digit"]["digit_accuracy_pct"],
                    "exact_match": row["digit"]["exact_match"],
                }
            )


def _load_accuracy_report(batch_id: str) -> dict | None:
    report_path = settings.batches_dir / batch_id / "accuracy_report.json"
    if not report_path.is_file():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def _score_and_save_accuracy(batch_id: str, completed: list[dict]) -> None:
    """Score a batch against data/ground_truth.csv, if it exists.

    Runs automatically at the end of every batch (see _run_batch) so the
    report is already there with zero clicks for cheques you process
    repeatedly -- the "rescore" button exists only to re-run this after
    you've edited the CSV, without reprocessing the images themselves.
    """
    ground_truth = load_ground_truth(settings.root / settings.ground_truth_path)
    if not ground_truth:
        return
    report = build_accuracy_report(ground_truth, completed)
    batch_dir = settings.batches_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "accuracy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_accuracy_csv(batch_dir, report)


def _sse(event_name: str, payload: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"


async def _run_batch(
    batch_id: str,
    saved: list[tuple[int, str, Path]],
    options: PreprocessOptions,
) -> None:
    """Process every cheque in the background, pushing an SSE event per
    image onto this batch's queue as it finishes.

    Each image's actual processing (`process()`) is synchronous, CPU-bound
    work (OpenCV, PaddleOCR, YOLO, torch) -- run via `run_in_executor`
    rather than awaited directly, so it doesn't block the event loop and
    starve the /batches/{batch_id}/stream request that's meanwhile trying
    to flush each event to the browser as soon as it's queued.
    """
    queue = _batch_queues[batch_id]
    loop = asyncio.get_event_loop()
    completed: list[dict] = []

    try:
        await queue.put(_sse("batch_start", {"batch_id": batch_id, "total": len(saved)}))
        for sequence, filename, path in saved:
            try:
                run_id, result = await loop.run_in_executor(
                    None, process, path, options, batch_id, sequence, filename,
                )
                item = {"filename": filename, "run_id": run_id, "result": result}
                completed.append(item)
                card_html = templates.get_template("_cheque_card.html").render(
                    item=item, batch_id=batch_id, loop_index=sequence,
                )
                await queue.put(
                    _sse(
                        "cheque",
                        {
                            "html": card_html,
                            "signature_detected": result["signature"]["exists"],
                            "date_status": result["date"]["status"],
                        },
                    )
                )
            except Exception as error:  # noqa: BLE001 -- one bad image shouldn't abort the rest of the batch
                await queue.put(_sse("cheque_error", {"filename": filename, "sequence": sequence, "error": str(error)}))
            finally:
                path.unlink(missing_ok=True)

        _write_batch_summary(batch_id, completed)
        _score_and_save_accuracy(batch_id, completed)
        await queue.put(
            _sse(
                "batch_done",
                {"batch_id": batch_id, "completed": len(completed), "summary_url": f"/batches/{batch_id}/summary.csv"},
            )
        )
    finally:
        await queue.put(None)  # sentinel: always sent, even if something above raised


@app.post("/process")
async def start_process(
    cheques: list[UploadFile] = File(...),
    max_edge: int = Form(2200),
    contrast: float = Form(1.15),
    denoise: bool = Form(False),
    deskew: bool = Form(False),
) -> RedirectResponse:
    """Save uploads, start processing in the background, and redirect
    immediately to this batch's own results page -- that page is what
    opens the live stream, so upload and results stay on separate pages/
    URLs instead of one page swapping its own content underneath the user.
    """
    if not cheques:
        raise HTTPException(422, "Select at least one cheque image.")
    _validate_settings(max_edge, contrast)

    options = PreprocessOptions(max_long_edge=max_edge, contrast=contrast, denoise=denoise, deskew=deskew)
    batch_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"

    saved: list[tuple[int, str, Path]] = []
    for sequence, upload in enumerate(cheques, start=1):
        path = await _save_valid_upload(upload)
        saved.append((sequence, upload.filename or f"cheque-{sequence}", path))

    _batch_queues[batch_id] = asyncio.Queue()
    asyncio.create_task(_run_batch(batch_id, saved, options))

    return RedirectResponse(url=f"/batches/{batch_id}", status_code=303)


@app.get("/batches/{batch_id}/stream")
async def batch_stream(batch_id: str) -> StreamingResponse:
    """SSE endpoint the results page connects to while a batch is still
    processing. Only exists while `_run_batch` for this batch is active."""
    queue = _batch_queues.get(batch_id)
    if queue is None:
        raise HTTPException(404, "This batch isn't currently processing.")

    async def generate():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
        _batch_queues.pop(batch_id, None)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/batches/{batch_id}", response_class=HTMLResponse)
def view_batch(request: Request, batch_id: str) -> HTMLResponse:
    """Show a batch: live (still processing, connects to the SSE stream)
    or finished (rendered directly from the saved result.json files, no
    reprocessing, works even after a server restart)."""
    if not ID_PATTERN.fullmatch(batch_id):
        raise HTTPException(404)

    if batch_id in _batch_queues:
        return templates.TemplateResponse(
            request, "results.html", {"batch_id": batch_id, "live": True, "results": []},
        )

    batch_dir = settings.batches_dir / batch_id
    if not batch_dir.is_dir():
        raise HTTPException(404)

    completed = _load_batch_results(batch_id)

    return templates.TemplateResponse(
        request,
        "results.html",
        {
            "batch_id": batch_id,
            "live": False,
            "results": completed,
            "accuracy_report": _load_accuracy_report(batch_id),
        },
    )


@app.post("/batches/{batch_id}/ground-truth")
async def rescore_batch(batch_id: str) -> RedirectResponse:
    """Re-run scoring for an already-finished batch -- use after editing
    data/ground_truth.csv, without needing to reprocess any images."""
    if not ID_PATTERN.fullmatch(batch_id):
        raise HTTPException(404)
    if not (settings.batches_dir / batch_id).is_dir():
        raise HTTPException(404)

    completed = _load_batch_results(batch_id)
    _score_and_save_accuracy(batch_id, completed)

    return RedirectResponse(url=f"/batches/{batch_id}", status_code=303)


@app.get("/batches/{batch_id}/summary.csv")
def batch_summary(batch_id: str) -> FileResponse:
    if not ID_PATTERN.fullmatch(batch_id):
        raise HTTPException(404)
    target = (settings.batches_dir / batch_id / "summary.csv").resolve()
    if not target.is_file():
        raise HTTPException(404)
    return FileResponse(target, filename=f"{batch_id}-summary.csv")


@app.get("/batches/{batch_id}/accuracy_report.csv")
def accuracy_report_csv(batch_id: str) -> FileResponse:
    if not ID_PATTERN.fullmatch(batch_id):
        raise HTTPException(404)
    target = (settings.batches_dir / batch_id / "accuracy_report.csv").resolve()
    if not target.is_file():
        raise HTTPException(404)
    return FileResponse(target, filename=f"{batch_id}-accuracy-report.csv")


@app.get("/batches/{batch_id}/{run_id}/{artifact:path}")
def artifact(batch_id: str, run_id: str, artifact: str) -> FileResponse:
    """Serve only files belonging to the requested image's own folder."""
    if not ID_PATTERN.fullmatch(batch_id) or not ID_PATTERN.fullmatch(run_id):
        raise HTTPException(404)

    run_dir = (settings.batches_dir / batch_id / run_id).resolve()
    target = (run_dir / artifact).resolve()
    if run_dir not in target.parents or not target.is_file():
        raise HTTPException(404)

    return FileResponse(target)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=5000, reload=True)