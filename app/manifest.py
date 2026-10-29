"""The SQLite source of truth for one resumable batch."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from app.ingestion import InventoryItem

OUTPUT_FIELDS = [
    "batch_id", "cheque_id", "bank_code", "sequence", "filename",
    "signature_detected", "date", "start_time", "end_time",
    "duration_seconds", "pixels", "t_read", "t_preprocess",
    "t_signature", "t_ocr", "t_date", "error",
    # Added with the switch to AI Foundry; appended so existing columns keep their positions.
    "signature_count", "signature_status", "date_status", "date_confidence",
]


class Manifest:
    def __init__(self, path: Path, batch_id: str, fingerprint: str, max_attempts: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.batch_id, self.max_attempts = path, batch_id, max_attempts
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cheques (
              cheque_id TEXT PRIMARY KEY, bank_code TEXT NOT NULL, zip_member TEXT NOT NULL,
              sequence INTEGER NOT NULL, crc INTEGER NOT NULL, size INTEGER NOT NULL,
              status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
              last_error TEXT, result_json TEXT, output_emitted INTEGER NOT NULL DEFAULT 0,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)
        found = self._meta("fingerprint")
        if found and found != fingerprint:
            raise RuntimeError("Existing manifest belongs to a different ZIP or folder; use a new batch ID.")
        if not found:
            self._set_meta("batch_id", batch_id)
            self._set_meta("fingerprint", fingerprint)
            self.db.commit()

    def close(self) -> None:
        self.db.close()

    def _meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", (key, value))

    def seed(self, items: Iterable[InventoryItem], allow_additions: bool = False) -> None:
        """Register the batch inventory. A ZIP must match the manifest exactly;
        with `allow_additions` (folder mode) new files are appended after the
        existing ones and files already recorded keep their results."""
        if allow_additions:
            self._add_new(list(items))
            return
        rows = [(i.cheque_id, i.bank_code, i.zip_member, i.sequence, i.crc, i.size, "PENDING") for i in items]
        with self.db:
            self.db.executemany("""
                INSERT OR IGNORE INTO cheques
                (cheque_id, bank_code, zip_member, sequence, crc, size, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)
        count = self.db.execute("SELECT COUNT(*) FROM cheques").fetchone()[0]
        if count != len(rows):
            raise RuntimeError("ZIP inventory differs from the existing manifest; refusing unsafe resume.")

    def _add_new(self, items: list[InventoryItem]) -> None:
        known = {row[0] for row in self.db.execute("SELECT cheque_id FROM cheques")}
        next_sequence = self.db.execute("SELECT COALESCE(MAX(sequence), 0) FROM cheques").fetchone()[0] + 1
        rows = []
        for item in items:
            if item.cheque_id in known:
                continue
            rows.append((item.cheque_id, item.bank_code, item.zip_member, next_sequence, item.crc, item.size, "PENDING"))
            next_sequence += 1
        with self.db:
            self.db.executemany("""
                INSERT INTO cheques (cheque_id, bank_code, zip_member, sequence, crc, size, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)

    def recover_interrupted(self) -> None:
        with self.db:
            self.db.execute("UPDATE cheques SET status='PENDING', updated_at=CURRENT_TIMESTAMP WHERE status='IN_PROGRESS'")

    def todo(self) -> list[sqlite3.Row]:
        return self.db.execute("""
            SELECT * FROM cheques WHERE status IN ('PENDING', 'RETRYABLE_FAILED')
            ORDER BY sequence
        """).fetchall()

    def mark_in_progress(self, cheque_id: str) -> None:
        with self.db:
            self.db.execute("""
                UPDATE cheques SET status='IN_PROGRESS', updated_at=CURRENT_TIMESTAMP
                WHERE cheque_id=? AND status IN ('PENDING', 'RETRYABLE_FAILED')
            """, (cheque_id,))

    def mark_done(self, cheque_id: str, result: dict[str, Any]) -> None:
        with self.db:
            self.db.execute("""
                UPDATE cheques SET status='DONE', result_json=?, last_error=NULL,
                updated_at=CURRENT_TIMESTAMP WHERE cheque_id=?
            """, (json.dumps(result, separators=(",", ":")), cheque_id))

    def mark_failure(self, cheque_id: str, error: str) -> None:
        row = self.db.execute("SELECT attempts FROM cheques WHERE cheque_id=?", (cheque_id,)).fetchone()
        attempts = int(row[0]) + 1
        status = "PERMANENT_FAILED" if attempts >= self.max_attempts else "RETRYABLE_FAILED"
        with self.db:
            self.db.execute("""
                UPDATE cheques SET status=?, attempts=?, last_error=?, updated_at=CURRENT_TIMESTAMP
                WHERE cheque_id=?
            """, (status, attempts, error[:4000], cheque_id))

    def done_not_emitted(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM cheques WHERE status='DONE' AND output_emitted=0 ORDER BY sequence").fetchall()

    def mark_emitted(self, cheque_id: str) -> None:
        with self.db:
            self.db.execute("UPDATE cheques SET output_emitted=1 WHERE cheque_id=?", (cheque_id,))

    def done_rows(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM cheques WHERE status='DONE' ORDER BY sequence").fetchall()

    def failures(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM cheques WHERE status='PERMANENT_FAILED' ORDER BY sequence").fetchall()

    def counts(self) -> dict[str, int]:
        return dict(self.db.execute("SELECT status, COUNT(*) FROM cheques GROUP BY status").fetchall())

    def _output_row(self, row: sqlite3.Row) -> dict[str, Any]:
        result = json.loads(row["result_json"])
        return {field: result.get(field, "") for field in OUTPUT_FIELDS}

    def rebuild_output(self, path: Path) -> None:
        _atomic_csv(path, OUTPUT_FIELDS, (self._output_row(row) for row in self.done_rows()))
        with self.db:
            self.db.execute("UPDATE cheques SET output_emitted=1 WHERE status='DONE'")

    def append_output(self, path: Path, cheque_id: str) -> None:
        row = self.db.execute("SELECT * FROM cheques WHERE cheque_id=? AND status='DONE'", (cheque_id,)).fetchone()
        if not row or row["output_emitted"]:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(self._output_row(row))
            file.flush()
            os.fsync(file.fileno())
        self.mark_emitted(cheque_id)

    def write_failures(self, path: Path) -> None:
        fields = ["batch_id", "cheque_id", "bank_code", "sequence", "attempts", "last_error"]
        _atomic_csv(path, fields, ({field: (self.batch_id if field == "batch_id" else row[field]) for field in fields} for row in self.failures()))


def _atomic_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
