"""Prompts sent to AI Foundry with each crop.

How these are written, and why:

* Each request carries only a small crop and asks one narrow, visual
  question about it -- is there a handwritten signature, which digits are in
  the date boxes. Nothing asks the model to identify a person, read names or
  account numbers, judge authenticity, or reproduce what it sees.
* They describe only what is actually in the crop. Words such as "bank",
  "cheque", "account", "extract", "verify", "fraud" or "forgery" are left out:
  they add nothing the model needs for a crop of a date box or a signing
  area, and tests showed requests framed as pulling data out of a bank
  document were the ones the platform's guardrail refused.
* The signature prompt tells the model not to describe or copy the
  signature. One earlier test was blocked on the model's *answer*, so the
  answer is kept to a yes/no and a count.
* Plain instructions, no "ONLY"/"MUST"/"ignore" style wording, and a short
  fixed JSON shape so the reply is small and easy to check.

If the guardrail still blocks a request, that is the platform telling us
the use case needs its approval -- raise it with the platform team rather
than rewording to get around it.
"""

SYSTEM_PROMPT = (
    "You look at small image crops of paper documents and answer one simple "
    "visual question about each crop. Reply with a single JSON object and no other text."
)

SIGNATURE_PROMPT = """This image is a crop of the signing area of a paper document.

Is there a handwritten signature in it? A signature here means pen strokes written by hand, such as a signed name, a scribble or initials. Printed text, printed lines, stamps and background patterns do not count.

Please do not describe, read out or copy the signature - only say whether it is there.

Reply with this JSON object:
{"signature_present": true or false, "signature_count": number of separate handwritten signatures (0 if none), "confidence": "high", "medium" or "low", "note": "a few words if the image is unclear, otherwise an empty string"}"""

DATE_PROMPT = """This image is a crop of a date field on a paper form. The date is handwritten as {digit_count} digits, normally one digit in each printed box, in day, month, year order (DD MM YY).

Please read the handwritten digits from left to right.

Reply with this JSON object:
{{"digits": "the {digit_count} digits in order with no spaces or separators, using ? for any digit you cannot read, or an empty string if nothing is written", "confidence": "high", "medium" or "low", "note": "a few words if something is unclear, otherwise an empty string"}}"""


def date_prompt(digit_count: int) -> str:
    return DATE_PROMPT.format(digit_count=digit_count)
