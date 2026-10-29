"""ZIP staging, validation, inventory and single-run locking for batch input."""

from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InventoryItem:
    cheque_id: str
    bank_code: str
    zip_member: str
    sequence: int
    crc: int
    size: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_local_zip(source: Path, destination: Path) -> Path:
    """Copy a source ZIP once, never exposing a partially copied destination."""
    if destination.exists():
        return destination
    if not source.is_file():
        raise FileNotFoundError(f"Batch ZIP not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with source.open("rb") as src, partial.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    if partial.stat().st_size != source.stat().st_size:
        partial.unlink(missing_ok=True)
        raise IOError("Staged ZIP size does not match source")
    os.replace(partial, destination)
    return destination


def inventory_zip(path: Path) -> list[InventoryItem]:
    """Return a stable, safe inventory without extracting ZIP members."""
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise zipfile.BadZipFile(f"ZIP CRC validation failed for {corrupt!r}")
        candidates: list[zipfile.ZipInfo] = []
        names: set[str] = set()
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            parts = [part for part in name.split("/") if part]
            if name.startswith("/") or ".." in parts:
                raise ValueError(f"Unsafe ZIP member path: {info.filename!r}")
            if info.is_dir() or not name.lower().endswith(".img"):
                continue
            if len(parts) < 2 or parts[0] not in {"PBB", "PIBB"}:
                raise ValueError(f"Cheque must be under PBB/ or PIBB/: {info.filename!r}")
            if name in names:
                raise ValueError(f"Duplicate ZIP member: {info.filename!r}")
            names.add(name)
            candidates.append(info)
    candidates.sort(key=lambda item: item.filename.replace("\\", "/"))
    return [
        InventoryItem(
            cheque_id=info.filename.replace("\\", "/"),
            bank_code=info.filename.replace("\\", "/").split("/", 1)[0],
            zip_member=info.filename,
            sequence=index,
            crc=info.CRC,
            size=info.file_size,
        )
        for index, info in enumerate(candidates, start=1)
    ]


FOLDER_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def inventory_folder(folder: Path) -> list[InventoryItem]:
    """Inventory plain image files in a folder (UAT input, instead of a ZIP).

    Subfolders are included. A file under PBB/ or PIBB/ keeps that bank code;
    anything else is labelled "UAT". Files are read in place, never copied.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    files = sorted(
        (path for path in folder.rglob("*")
         if path.is_file() and path.suffix.lower() in FOLDER_IMAGE_SUFFIXES),
        key=lambda path: path.relative_to(folder).as_posix(),
    )
    items = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(folder).as_posix()
        top = relative.split("/", 1)[0]
        items.append(InventoryItem(
            cheque_id=relative,
            bank_code=top if top in {"PBB", "PIBB"} and "/" in relative else "UAT",
            zip_member=relative,
            sequence=index,
            crc=0,
            size=path.stat().st_size,
        ))
    return items


@contextmanager
def batch_lock(path: Path):
    """Prevent two manual/scheduler processes from owning one batch.

    Uses msvcrt on Windows and fcntl on Linux/macOS. The OS releases either
    lock automatically if the process dies, so a crash never leaves a stale lock.
    """
    if os.name == "nt":
        import msvcrt

        def lock(fd: int) -> None:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

        def unlock(fd: int) -> None:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        def lock(fd: int) -> None:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def unlock(fd: int) -> None:
            fcntl.flock(fd, fcntl.LOCK_UN)

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    locked = False
    try:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            lock(handle.fileno())
        except OSError as error:
            raise RuntimeError(f"Batch is already running (lock: {path})") from error
        locked = True
        yield
    finally:
        if locked:
            try:
                handle.seek(0)
                unlock(handle.fileno())
            except OSError:
                pass
        handle.close()
