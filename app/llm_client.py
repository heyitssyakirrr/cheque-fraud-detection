import base64
import time

import httpx

from app.audit import audit_log, shorten
from app.config import get_settings


class LLMError(Exception):
    """The Foundry call failed or returned nothing usable. The message is
    meant to be shown to the user as-is."""

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


def _build_file_part(image_bytes: bytes, mime_type: str) -> dict:
    """Wrap the cheque as an OpenAI-style content part: images go in as an
    image_url data URL, PDFs as a file part."""
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    if mime_type == "application/pdf":
        return {"type": "file", "file": {"filename": "cheque.pdf", "file_data": data_url}}
    return {"type": "image_url", "image_url": {"url": data_url}}


def _message_text(content) -> str | None:
    """content is normally a string, but some deployments return a list of
    parts - join their text."""
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if isinstance(content, str) and content.strip():
        return content
    return None


async def extract_cheque_data(image_bytes: bytes, mime_type: str, request_id: str) -> str:
    """Calls AI Foundry's chat completions endpoint with the cheque, returns
    the model's raw text response (parsing happens separately, in parsing.py).
    Raises LLMError with a readable reason on any failure."""
    settings = get_settings()
    url = f"{settings.llm_base_url}{settings.llm_extract_path}"
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the fields from this cheque."},
                    _build_file_part(image_bytes, mime_type),
                ],
            },
        ],
        "max_tokens": settings.llm_max_tokens,
        "stream": False,
    }

    audit_log.info(
        f"[{request_id}] FOUNDRY  calling POST {url} model={settings.llm_model} "
        f"max_tokens={settings.llm_max_tokens} helper_id={settings.helper_id or '<not set>'} "
        f"api_key={'set' if settings.llm_api_key else '<NOT SET>'}"
    )
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.llm_api_key}",
                    "X-PBAI-Helper-Id": settings.helper_id,
                },
                json=payload,
            )
    except httpx.HTTPError as e:
        audit_log.error(f"[{request_id}] FOUNDRY  network error after {time.perf_counter() - started:.1f}s: {type(e).__name__}: {e}")
        raise LLMError(f"Could not reach AI Foundry at {url}: {type(e).__name__}: {e}") from e

    elapsed = time.perf_counter() - started
    if response.is_error:
        audit_log.error(f"[{request_id}] FOUNDRY  HTTP {response.status_code} in {elapsed:.1f}s, body: {shorten(response.text)}")
        raise LLMError(f"AI Foundry returned HTTP {response.status_code}: {response.text[:1000]}")

    try:
        body = response.json()
        choice = body["choices"][0]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        audit_log.error(f"[{request_id}] FOUNDRY  HTTP {response.status_code} but unexpected body: {shorten(response.text)}")
        raise LLMError(f"AI Foundry returned an unexpected response shape: {response.text[:1000]}") from e

    finish_reason = choice.get("finish_reason")
    usage = body.get("usage") or {}
    audit_log.info(
        f"[{request_id}] FOUNDRY  HTTP {response.status_code} in {elapsed:.1f}s finish_reason={finish_reason} "
        f"tokens(prompt={usage.get('prompt_tokens', '?')}, completion={usage.get('completion_tokens', '?')})"
    )

    text = _message_text((choice.get("message") or {}).get("content"))
    if text is None:
        reason = "the model hit max_tokens" if finish_reason == "length" else f"finish_reason={finish_reason}"
        audit_log.error(f"[{request_id}] FOUNDRY  reply had no text ({reason}), body: {shorten(response.text)}")
        raise LLMError(f"AI Foundry returned an empty reply ({reason}).")

    audit_log.info(f"[{request_id}] FOUNDRY  model reply: {shorten(text)}")
    return text