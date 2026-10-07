"""Durable candidate lifecycle and bounded request history; no model calls here."""
from __future__ import annotations

import json
import os
from pathlib import Path


class Lifecycle:
    """One serial builder per project. State is owned by the harness, not its model."""

    def __init__(self, root: Path, max_previews: int = 2, require_evaluation: bool = True):
        if type(max_previews) is not int or max_previews < 0:
            raise ValueError("max_previews must be nonnegative (0 disables the cap)")
        if type(require_evaluation) is not bool:
            raise ValueError("require_evaluation must be a boolean")
        self.root, self.max_previews, self.require_evaluation = root.resolve(), max_previews, require_evaluation
        self.path = self.root / "agent" / "candidate-state.json"
        try:
            self.path.resolve().relative_to(self.root)
        except ValueError:
            raise ValueError("candidate-state.json path escapes the project; inspect agent/ before resuming") from None
        self.counts, self.pending = {}, None
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as e:
                raise ValueError("invalid candidate-state.json; inspect it before resuming") from e
            if (not isinstance(data, dict) or data.get("version") != 1
                    or not isinstance(data.get("previews"), dict)):
                raise ValueError("invalid candidate-state.json; inspect it before resuming")
            for name, count in data["previews"].items():
                key = self.validate(name)
                if type(count) is not int or count < 0:
                    raise ValueError("invalid persisted preview count")
                # On Windows the filesystem treats candidate names as case-insensitive. Merge aliases so
                # hand-edited or older state cannot grant another preview by changing capitalization.
                self.counts[key] = self.counts.get(key, 0) + count
            self.pending = data.get("pending")
            if self.pending is not None:
                self.pending = self.validate(self.pending)
        if self.pending and self.measured(self.pending):
            self.pending = None
            self.save()

    def validate(self, name: str) -> str:
        if not isinstance(name, str) or not name or name in (".", "..") or "/" in name or "\\" in name or ":" in name:
            raise ValueError("invalid persisted candidate name")
        attempts = (self.root / "attempts").resolve()
        try:
            attempts.relative_to(self.root)
        except ValueError:
            raise ValueError("invalid persisted candidate name: attempts/ escapes the project") from None
        resolved = (attempts / name).resolve()
        try:
            resolved.relative_to(attempts)
        except ValueError:
            raise ValueError("invalid persisted candidate name: path escapes attempts/") from None
        return os.path.normcase(resolved.name)

    def measured(self, name: str) -> bool:
        try:
            return bool(json.loads((self.root / "attempts" / name / "eval.json").read_text(encoding="utf-8")).get("ok"))
        except (OSError, ValueError):
            return False

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps({"version": 1, "previews": self.counts, "pending": self.pending}) + "\n", encoding="utf-8")
        temp.replace(self.path)

    def refuse(self, candidate: str, *, preview: bool = False) -> str:
        candidate = self.validate(candidate)
        if self.pending and self.measured(self.pending):
            self.pending = None
            self.save()
        if self.require_evaluation and self.pending and candidate != self.pending:
            return f"REFUSED: evaluate pending candidate attempts/{self.pending}/ before working on another candidate. Repair it if needed; a failed evaluation does not clear it."
        if preview and self.max_previews and self.counts.get(candidate, 0) >= self.max_previews:
            return f"REFUSED: {self.max_previews} previews used for {candidate}; evaluate this candidate even if it lost score."
        return ""

    def previewed(self, candidate: str) -> None:
        candidate = self.validate(candidate)
        self.counts[candidate] = self.counts.get(candidate, 0) + 1
        if not self.measured(candidate):
            self.pending = candidate
        self.save()


def estimate_tokens(messages: list[dict], schemas: list[dict]) -> int:
    """Conservative byte-based text estimate, plus 4096 per resized image.

    This is not an endpoint tokenizer. Callers must configure their context limit
    and reply reserve; vision accounting varies by model.
    """
    def size(value):
        if isinstance(value, dict):
            if value.get("type") == "image_url":
                return 4096
            return sum(size(k) + size(v) for k, v in value.items()) + 8
        if isinstance(value, list):
            return sum(size(v) for v in value) + 4
        return len(str(value).encode("utf-8")) + 4
    return size(messages) + size(schemas)


def compact(messages: list[dict], schemas: list[dict], checkpoint: str, limit: int, reserve: int) -> tuple[list[dict], bool]:
    """Keep the opening contract and references, state, and whole recent tool rounds."""
    if not limit:
        return messages, False
    if limit <= 0 or reserve < 0 or reserve >= limit:
        raise ValueError("context limit must exceed the nonnegative reply reserve")
    budget = limit - reserve
    if estimate_tokens(messages, schemas) <= budget:
        return messages, False
    base = messages[:2] + [{"role": "user", "content": checkpoint}]
    if estimate_tokens(base, schemas) > budget:
        raise ValueError("context budget cannot fit the opening contract, references, tools and checkpoint; increase --context-tokens or reduce reference image input")
    # Assistant boundaries keep every tool call paired with all of its results.
    starts = [i for i in range(2, len(messages)) if messages[i].get("role") == "assistant"]
    for i in starts[-2:]:
        candidate = base + messages[i:]
        if estimate_tokens(candidate, schemas) <= budget:
            return candidate, True
    return base, True
