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
    request_id: str  # matches the [id] in logs/audit.log and the CSV row
    extraction: ChequeExtractionResult
    parse_error: str | None = None  # set when the model's reply couldn't be read
    raw_gateway_response: str  # keep the raw text around for debugging early on