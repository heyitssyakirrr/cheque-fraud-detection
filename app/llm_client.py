import base64

import httpx

from app.config import get_settings

EXTRACTION_PROMPT = """You are analyzing a bank cheque - it may be provided as an image or a PDF page. Look carefully and respond with ONLY a JSON object, no markdown, no code fences, no extra text - just the raw JSON, matching exactly this shape:

{
  "date": "<the date written on the cheque, in YYYY-MM-DD if possible, else as written>",
  "date_confidence": "<high, medium, or low>",
  "payee_name": "<the name written after 'Pay to' or similar, exactly as written>",
  "amount_numeric": "<the amount written in numeric/digit form, including currency symbol if present, exactly as written>",
  "amount_words": "<the amount written in words, exactly as written>",
  "amount_match": <true or false - do the numeric and word amounts represent the same value>,
  "signature_present": <true or false - is a signature-like mark visible in the signature area>,
  "signature_confidence": "<high, medium, or low>",
  "notes": "<one short sentence on anything relevant - e.g. faded, skewed, unclear, amount mismatch>"
}

If any field is not clearly visible or legible, set it to null rather than guessing. This is a visual presence/extraction check only - you are NOT verifying the signature against any reference, just noting whether one is visibly present."""


async def extract_cheque_data(image_bytes: bytes, mime_type: str) -> str:
    """Calls the LLM Gateway's /v1/generate with the cheque image, returns
    Gemini's raw text response (parsing happens separately, in parsing.py)."""
    settings = get_settings()
    image_base64 = base64.b64encode(image_bytes).decode("ascii")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.gateway_url}/v1/generate",
            headers={"X-API-Key": settings.gateway_api_key},
            json={
                "prompt": EXTRACTION_PROMPT,
                "image_base64": image_base64,
                "image_mime_type": mime_type,
                "temperature": 0.1,
                "max_tokens": 2048,
            },
        )
    response.raise_for_status()
    return response.json()["data"]