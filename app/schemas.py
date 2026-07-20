from pydantic import BaseModel


class ChequeExtractionResult(BaseModel):
    date: str | None = None
    date_confidence: str | None = None
    payee_name: str | None = None
    amount_numeric: str | None = None
    amount_words: str | None = None
    amount_match: bool | None = None
    signature_present: bool | None = None
    signature_confidence: str | None = None
    notes: str | None = None


class ChequeUploadResponse(BaseModel):
    extraction: ChequeExtractionResult
    raw_gateway_response: str  # keep the raw text around for debugging early on