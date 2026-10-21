"""
Human-readable audit log of every cheque upload: what came in, what was sent
to AI Foundry, what came back, and what the app concluded. Written to
logs/audit.log (rotated at 5 MB, 10 files kept) and echoed to the console.

Every line carries a short request id, so one upload can be followed from
start to finish with e.g. `findstr a1b2c3d4 logs\\audit.log`. API keys and
image data are never logged.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_PATH = LOG_DIR / "audit.log"

# Long model replies are cut to this many characters in the log; the full
# text is always kept in cheque_results.csv.
MAX_LOGGED_CHARS = 2000


def _build_logger() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("cheque.audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=10, encoding="utf-8")
        file_handler.setFormatter(fmt)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


audit_log = _build_logger()


def shorten(text: str | None) -> str:
    """Trim long text for the log and keep it on one line."""
    if text is None:
        return "<none>"
    text = text.replace("\r", "").replace("\n", "\\n")
    if len(text) > MAX_LOGGED_CHARS:
        return text[:MAX_LOGGED_CHARS] + f"... [{len(text) - MAX_LOGGED_CHARS} more chars]"
    return text
