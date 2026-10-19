from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
from typing import Any


class LLMResponseCache:
    VALID_MODES = {"off", "read", "readwrite", "refresh"}

    def __init__(self, directory: Path | str, mode: str = "readwrite") -> None:
        self.directory = Path(directory)
        self.mode = mode.lower()
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported LLM cache mode: {mode}")
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if self.mode in {"off", "refresh"}:
            return None
        path = self.directory / f"{key}.json"
        if not path.exists():
            with self._lock:
                self.misses += 1
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        with self._lock:
            self.hits += 1
        return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        if self.mode not in {"readwrite", "refresh"}:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        temporary = self.directory / f"{key}.{threading.get_ident()}.tmp"
        temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        with self._lock:
            temporary.replace(path)
            self.writes += 1

    def require_hit(self, key: str) -> dict[str, Any]:
        value = self.get(key)
        if value is None:
            raise RuntimeError(f"LLM cache miss in read-only mode: {key}")
        return value

    def stats(self) -> dict[str, int | str]:
        return {"mode": self.mode, "hits": self.hits, "misses": self.misses, "writes": self.writes}
