from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.gateway_client import extract_cheque_data
from app.parsing import parse_extraction_response
from app.schemas import ChequeUploadResponse
from app.csv_store import append_result, build_row

app = FastAPI(title="Cheque Fraud Detection")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "application/pdf"}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/cheques/upload", response_model=ChequeUploadResponse)
async def upload_cheque(file: UploadFile = File(...)) -> ChequeUploadResponse:
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {ALLOWED_MIME_TYPES}",
        )

    image_bytes = await file.read()

    try:
        raw_response = await extract_cheque_data(image_bytes, file.content_type)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gateway call failed: {e}") from e

    extraction = parse_extraction_response(raw_response)

    row = build_row(
        filename=file.filename or "unknown",
        mime_type=file.content_type or "unknown",
        extraction=extraction,
        raw_response=raw_response,
    )
    append_result(row)

    return ChequeUploadResponse(extraction=extraction, raw_gateway_response=raw_response)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8001, reload=True)