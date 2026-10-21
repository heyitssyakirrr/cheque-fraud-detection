import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.audit import LOG_PATH, audit_log
from app.config import get_settings
from app.llm_client import LLMError, extract_cheque_data
from app.parsing import parse_extraction_response
from app.schemas import ChequeUploadResponse
from app.csv_store import CSV_PATH, append_result, build_row


@asynccontextmanager
async def lifespan(_app: FastAPI):
    s = get_settings()
    audit_log.info(
        f"STARTUP  endpoint={s.llm_base_url or '<LLM_BASE_URL NOT SET>'}{s.llm_extract_path} model={s.llm_model} "
        f"helper_id={'set' if s.helper_id else '<NOT SET>'}"
    )
    audit_log.info(f"STARTUP  audit log: {LOG_PATH} | results CSV: {CSV_PATH}")
    yield


app = FastAPI(title="Cheque Fraud Detection", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "application/pdf"}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/cheques/upload", response_model=ChequeUploadResponse)
async def upload_cheque(file: UploadFile = File(...)) -> ChequeUploadResponse:
    request_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()

    if file.content_type not in ALLOWED_MIME_TYPES:
        audit_log.warning(f"[{request_id}] REJECTED file={file.filename!r} type={file.content_type} (unsupported type)")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {ALLOWED_MIME_TYPES}",
        )

    image_bytes = await file.read()
    audit_log.info(f"[{request_id}] UPLOAD   file={file.filename!r} type={file.content_type} size={len(image_bytes) / 1024:.1f} KB")

    try:
        raw_response = await extract_cheque_data(image_bytes, file.content_type, request_id)
    except LLMError as e:
        audit_log.error(f"[{request_id}] FAILED   after {time.perf_counter() - started:.1f}s: {e}")
        raise HTTPException(status_code=502, detail=f"{e} (request id {request_id})") from e

    extraction, parse_error = parse_extraction_response(raw_response)
    if parse_error:
        audit_log.warning(f"[{request_id}] PARSE    {parse_error}")
    else:
        audit_log.info(f"[{request_id}] PARSE    ok")

    row = build_row(
        request_id=request_id,
        filename=file.filename or "unknown",
        mime_type=file.content_type or "unknown",
        extraction=extraction,
        raw_response=raw_response,
        parse_error=parse_error,
    )
    append_result(row)

    audit_log.info(
        f"[{request_id}] RESULT   date={extraction.date} ({extraction.date_confidence}) | payee={extraction.payee_name!r} | "
        f"amount={extraction.amount_numeric!r} / {extraction.amount_words!r} match={extraction.amount_match} | "
        f"signature={extraction.signature_present} ({extraction.signature_confidence}) | notes={extraction.notes!r}"
    )
    audit_log.info(
        f"[{request_id}] DONE     {'with PARSE ERROR' if parse_error else 'ok'} in {time.perf_counter() - started:.1f}s"
    )

    return ChequeUploadResponse(
        request_id=request_id,
        extraction=extraction,
        parse_error=parse_error,
        raw_gateway_response=raw_response,
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8001, reload=True)
