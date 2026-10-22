"""AI Foundry chat-completions client for small image crops.

Only crops ever leave this server (the signature zone and the date boxes),
never the full cheque: less personal data is sent, and the platform's
guardrail is far less likely to block a request that carries none.

One synchronous httpx client per process, because the pipeline runs inside
both the web app's thread pool and run.py's worker processes.
"""

import base64
import json
import logging
import re
import threading
import time
from typing import Any

import cv2
import httpx
import numpy as np

from app.settings import settings

logger = logging.getLogger(__name__)

_client: httpx.Client | None = None
_client_lock = threading.Lock()

# Worth retrying: rate limiting and server-side failures. A content-filter
# block is deliberately NOT retried -- that would just be trying to get
# around the platform's guardrail.
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class FoundryError(Exception):
    """Foundry could not be reached or kept failing after retries. Raised so
    the caller fails this cheque (run.py retries it on a later attempt)."""


def _http() -> httpx.Client:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if not settings.llm_base_url:
                    raise FoundryError(
                        "LLM_BASE_URL is not set. Add it to .env in the project root."
                    )
                _client = httpx.Client(
                    timeout=settings.llm_timeout_seconds,
                    verify=settings.llm_verify_ssl,
                )
    return _client


def encode_png(image: np.ndarray) -> str:
    """Encode a crop as a PNG data URL, capped at `llm_image_max_long_edge`."""
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge > settings.llm_image_max_long_edge:
        scale = settings.llm_image_max_long_edge / long_edge
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode crop as PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """The model is asked for one JSON object; tolerate code fences or a
    sentence around it."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
    for candidate in (cleaned, cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _message_text(content: Any) -> str:
    if isinstance(content, list):  # some deployments return content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else ""


def ask_about_image(purpose: str, system_prompt: str, user_prompt: str, image: np.ndarray) -> dict[str, Any]:
    """Send one crop with its prompt and return a result record.

    `status` is one of:
      ok        -- the model answered and `data` holds its JSON object
      blocked   -- the platform guardrail refused the request or the answer
      unreadable -- the model answered but not with a JSON object
    Network/server failures that persist through the retries raise
    FoundryError instead, so the cheque is failed rather than given a wrong
    answer.
    """
    body = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": encode_png(image)}},
                ],
            },
        ],
        "max_tokens": settings.llm_max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if settings.helper_id:
        headers["X-PBAI-Helper-Id"] = settings.helper_id
    url = f"{settings.llm_base_url.rstrip('/')}{settings.llm_chat_path}"

    last_error = ""
    for attempt in range(1, settings.llm_max_attempts + 1):
        started = time.perf_counter()
        try:
            response = _http().post(url, headers=headers, json=body)
        except httpx.HTTPError as error:
            last_error = f"{type(error).__name__}: {error}"
            logger.warning("Foundry %s: network error on attempt %d/%d: %s",
                           purpose, attempt, settings.llm_max_attempts, last_error)
        else:
            elapsed = round(time.perf_counter() - started, 2)
            if response.status_code in _RETRYABLE_STATUS:
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                logger.warning("Foundry %s: %s on attempt %d/%d",
                               purpose, last_error, attempt, settings.llm_max_attempts)
            elif response.is_error:
                # 400/401/403/404: retrying the same request will not help.
                raise FoundryError(f"Foundry {purpose}: HTTP {response.status_code}: {response.text[:1000]}")
            else:
                return _result_from_response(purpose, response, elapsed)
        if attempt < settings.llm_max_attempts:
            time.sleep(settings.llm_retry_backoff_seconds * attempt)

    raise FoundryError(f"Foundry {purpose}: gave up after {settings.llm_max_attempts} attempts ({last_error})")


def _result_from_response(purpose: str, response: httpx.Response, elapsed: float) -> dict[str, Any]:
    try:
        payload = response.json()
        choice = payload["choices"][0]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise FoundryError(f"Foundry {purpose}: unexpected response: {response.text[:1000]}") from error

    usage = payload.get("usage") or {}
    finish_reason = choice.get("finish_reason")
    text = _message_text((choice.get("message") or {}).get("content"))
    record: dict[str, Any] = {
        "status": "ok",
        "data": None,
        "finish_reason": finish_reason,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "seconds": elapsed,
        "raw_text": text,
        "note": "",
    }

    if finish_reason == "content_filter":
        # prompt_tokens == 0: blocked before the model read the request.
        where = "request" if usage.get("prompt_tokens") == 0 else "answer"
        record.update(status="blocked", note=f"blocked by the platform guardrail on the {where}")
        logger.warning("Foundry %s: %s (tokens prompt=%s completion=%s)",
                       purpose, record["note"], usage.get("prompt_tokens"), usage.get("completion_tokens"))
        return record

    data = _parse_json_object(text)
    if data is None:
        note = "model hit max_tokens before answering" if finish_reason == "length" else "reply was not a JSON object"
        record.update(status="unreadable", note=note)
        logger.warning("Foundry %s: %s: %r", purpose, note, text[:300])
        return record

    record["data"] = data
    return record
