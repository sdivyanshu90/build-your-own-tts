"""Content-addressed local storage that prevents path traversal."""

from __future__ import annotations

import hashlib
from pathlib import Path


class LocalStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes, media_type: str = "audio/wav") -> str:
        if media_type != "audio/wav":
            raise ValueError("local synthesis storage accepts only audio/wav")
        key = hashlib.sha256(data).hexdigest() + ".wav"
        target = (self.root / key).resolve()
        if self.root not in target.parents:
            raise ValueError("invalid storage key")
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
        return key
