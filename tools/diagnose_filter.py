"""
Find out what makes the AI Foundry content filter block the cheque request.

Sends a few variants of the request, from harmless to the full real one, and
prints for each whether it was blocked and which filter category fired. The
variant where blocking starts tells you the trigger: the prompt wording, any
image at all, or the cheque image itself.

Usage (from the project folder, with .env filled in):
    python -m tools.diagnose_filter path/to/cheque.png
"""

import base64
import json
import struct
import sys
import zlib
from pathlib import Path

import httpx

from app.config import get_settings
from app.llm_client import EXTRACTION_PROMPT, _build_file_part, _flagged


def blank_png(size: int = 64) -> bytes:
    """A plain white PNG, built without any imaging library."""
    row = b"\x00" + b"\xff\xff\xff" * size
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * size)) + chunk(b"IEND", b""))


def user_with_image(text: str, image: bytes, mime: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}, _build_file_part(image, mime)]}


# The field lines of EXTRACTION_PROMPT, so round 2 can ask for subsets of them.
FIELDS = {
    "date": '"date": "<the date written on the cheque, in YYYY-MM-DD if possible, else as written>"',
    "date_confidence": '"date_confidence": "<high, medium, or low>"',
    "payee_name": "\"payee_name\": \"<the name written after 'Pay to' or similar, exactly as written>\"",
    "amount_numeric": '"amount_numeric": "<the amount written in numeric/digit form, including currency symbol if present, exactly as written>"',
    "amount_words": '"amount_words": "<the amount written in words, exactly as written>"',
    "amount_match": '"amount_match": <true or false - do the numeric and word amounts represent the same value>',
    "signature_present": '"signature_present": <true or false - is a signature-like mark visible in the signature area>',
    "signature_confidence": '"signature_confidence": "<high, medium, or low>"',
    "notes": '"notes": "<one short sentence on anything relevant - e.g. faded, skewed, unclear, amount mismatch>"',
}
HEADER = ("You are analyzing a bank cheque - it may be provided as an image or a PDF page. Look carefully and respond "
          "with ONLY a JSON object, no markdown, no code fences, no extra text - just the raw JSON, matching exactly this shape:")
FOOTER = ("If any field is not clearly visible or legible, set it to null rather than guessing. This is a visual "
          "presence/extraction check only - you are NOT verifying the signature against any reference, just noting "
          "whether one is visibly present.")
SOFT_HEADER = "Read the attached document image and fill in this JSON object. Reply with the JSON only:"
SOFT_FOOTER = "Use null for anything you cannot read clearly."


def make_prompt(keys: list[str], header: str = HEADER, footer: str = FOOTER) -> str:
    body = ",\n".join(f"  {FIELDS[k]}" for k in keys)
    return f"{header}\n\n{{\n{body}\n}}\n\n{footer}"


def run(name: str, messages: list[dict]) -> None:
    s = get_settings()
    body = {"model": s.llm_model, "messages": messages, "max_tokens": s.llm_max_tokens, "stream": False}
    headers = {"Content-Type": "application/json", "X-PBAI-Helper-Id": s.helper_id}
    print(f"\n=== {name}")
    try:
        with httpx.Client(timeout=120.0, verify=False) as client:
            r = client.post(f"{s.llm_base_url}{s.llm_extract_path}", headers=headers, json=body)
    except httpx.HTTPError as e:
        print(f"    network error: {type(e).__name__}: {e}")
        return
    try:
        payload = r.json()
    except ValueError:
        print(f"    HTTP {r.status_code}, non-JSON body: {r.text[:500]}")
        return
    if r.is_error:
        print(f"    HTTP {r.status_code}: {json.dumps(payload)[:1500]}")
        return
    choice = (payload.get("choices") or [{}])[0]
    usage = payload.get("usage") or {}
    finish = choice.get("finish_reason")
    prompt_flags = [f for item in payload.get("prompt_filter_results") or []
                    if isinstance(item, dict) for f in _flagged(item.get("content_filter_results"))]
    answer_flags = _flagged(choice.get("content_filter_results"))
    verdict = "BLOCKED" if finish == "content_filter" else "passed"
    print(f"    {verdict}: finish_reason={finish} tokens(prompt={usage.get('prompt_tokens')}, "
          f"completion={usage.get('completion_tokens')})")
    print(f"    flagged on prompt: {prompt_flags or '-'} | flagged on answer: {answer_flags or '-'}")
    print(f"    prompt_filter_results: {json.dumps(payload.get('prompt_filter_results'))[:1500]}")
    print(f"    answer_filter_results: {json.dumps(choice.get('content_filter_results'))[:800]}")
    content = (choice.get("message") or {}).get("content")
    print(f"    reply: {str(content)[:300]!r}")


def round_basic(cheque: bytes) -> None:
    blank = blank_png()
    run("1. plain text, no image, no system prompt",
        [{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}])
    run("2. our system prompt, text only (no image)",
        [{"role": "system", "content": EXTRACTION_PROMPT},
         {"role": "user", "content": "No cheque attached for this test; return every field as null."}])
    run("3. blank white image, neutral question",
        [user_with_image("What colour is this image? One word.", blank, "image/png")])
    run("4. cheque image, neutral question (no system prompt)",
        [user_with_image("Describe this image in one short sentence.", cheque, "image/png")])
    run("5. full real request (system prompt + cheque image)",
        [{"role": "system", "content": EXTRACTION_PROMPT},
         user_with_image("Extract the fields from this cheque.", cheque, "image/png")])

    print("\nHow to read this:")
    print("  2 blocked            -> the prompt wording is the trigger")
    print("  3 blocked            -> any image is blocked (images may not be allowed on this deployment)")
    print("  4 blocked, 3 passed  -> the cheque image's content is the trigger (e.g. PII: names, account numbers)")
    print("  only 5 blocked       -> the combination of prompt + cheque is the trigger")


def round_fields(cheque: bytes, repeat: int = 2) -> None:
    """Round 2: the cheque with variations of the prompt, to find which part of the request
    trips the filter. Each runs `repeat` times because the filter is not fully consistent."""
    all_keys = list(FIELDS)
    no_payee = [k for k in all_keys if k != "payee_name"]
    variants = [
        ("A. full prompt as system message (same as test 5)", make_prompt(all_keys), True),
        ("B. full prompt sent as the user message instead", make_prompt(all_keys), False),
        ("C. all fields EXCEPT payee_name", make_prompt(no_payee), True),
        ("D. only date", make_prompt(["date", "date_confidence"]), True),
        ("E. only amounts", make_prompt(["amount_numeric", "amount_words", "amount_match"]), True),
        ("F. only signature", make_prompt(["signature_present", "signature_confidence"]), True),
        ("G. only payee_name", make_prompt(["payee_name"]), True),
        ("H. all fields, softer wording (no 'bank cheque', no signature sentence)",
         make_prompt(all_keys, SOFT_HEADER, SOFT_FOOTER), True),
    ]
    for name, prompt, as_system in variants:
        for i in range(1, repeat + 1):
            if as_system:
                messages = [{"role": "system", "content": prompt},
                            user_with_image("Extract the fields from this cheque.", cheque, "image/png")]
            else:
                messages = [user_with_image(prompt, cheque, "image/png")]
            run(f"{name}  [try {i}/{repeat}]", messages)

    print("\nHow to read this: compare which letters are BLOCKED vs passed.")
    print("  only A, B, G (and anything with payee_name) blocked -> asking for the payee's name is the trigger")
    print("  F blocked                                         -> asking about the signature is the trigger")
    print("  A blocked but B passed                            -> sending the prompt as a system message is the trigger")
    print("  H passed but A blocked                            -> the wording ('bank cheque', 'signature' sentence) is the trigger")
    print("  everything blocked                                -> any field extraction from this image is blocked")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit("usage: python -m tools.diagnose_filter path/to/cheque.png [--basic]")
    cheque = Path(args[0]).read_bytes()
    if "--basic" in sys.argv:
        round_basic(cheque)
    else:
        round_fields(cheque)


if __name__ == "__main__":
    main()
