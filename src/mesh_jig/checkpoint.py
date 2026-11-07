"""Publish complete individual checkpoint files without truncating the previous copy."""
from collections.abc import Iterable
import os
from pathlib import Path
import tempfile


def atomic_write_text(path: Path, chunks: str | Iterable[str]) -> None:
    """Write UTF-8 to a sibling temporary file, close it, then replace ``path``.

    Iterables allow JSONL serialization while preparing the private file. Errors
    propagate and the temporary file is removed on ordinary exception unwinding.
    This is per-file replacement, not a multi-file transaction or an fsync-based
    power-loss guarantee. Abrupt process termination can leave a temporary file.
    """
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        stream = os.fdopen(fd, "w", encoding="utf-8")
        fd = None  # The stream now owns the descriptor, including on write failure.
        with stream:
            for chunk in (chunks,) if isinstance(chunks, str) else chunks:
                stream.write(chunk)
        os.replace(temporary, path)
    finally:
        if fd is not None:
            os.close(fd)
        temporary.unlink(missing_ok=True)
