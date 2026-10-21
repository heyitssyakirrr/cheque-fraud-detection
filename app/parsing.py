import json
import re

from pydantic import ValidationError

from app.schemas import ChequeExtractionResult


def _load_json_object(text: str) -> dict:
    """The model is asked for raw JSON, but often wraps it in markdown code
    fences or a sentence like "Here is the result:" - try the cleaned text
    first, then the outermost {...} inside it."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("reply contains no JSON object") from None
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"reply contains malformed JSON ({e.msg} at char {e.pos})") from None
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    return data


def parse_extraction_response(raw_text: str) -> tuple[ChequeExtractionResult, str | None]:
    """Returns (result, parse_error). Never raises: on failure the result is
    empty and parse_error says why, so the UI can show a warning instead of
    an empty result that looks like "nothing detected"."""
    try:
        data = _load_json_object(raw_text)
    except ValueError as e:
        return ChequeExtractionResult(), f"Could not read the model's reply: {e}"

    try:
        return ChequeExtractionResult.model_validate(data), None
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        return ChequeExtractionResult(), f"Model reply had invalid fields: {problems}"
