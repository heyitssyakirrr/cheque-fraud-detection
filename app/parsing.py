import json
import re

from app.schemas import ChequeExtractionResult


def parse_extraction_response(raw_text: str) -> ChequeExtractionResult:
    """Gemini is asked for raw JSON, but often still wraps it in markdown
    code fences despite instructions - strip those defensively before
    parsing, and never let a malformed response crash the request."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Malformed JSON - return an empty result rather than crashing,
        # so the caller can see raw_gateway_response and decide what to do.
        return ChequeExtractionResult()

    return ChequeExtractionResult.model_validate(data)