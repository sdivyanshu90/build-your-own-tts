"""Optional espeak phonemization with deterministic grapheme fallback."""

from __future__ import annotations

import re
from typing import Protocol


class Backend(Protocol):
    def phonemize(self, text: str, language: str) -> list[str]: ...


class GraphemeBackend:
    """Offline fallback: characters are symbols and spaces become word boundaries."""

    def phonemize(self, text: str, language: str) -> list[str]:
        del language
        return ["|" if char.isspace() else char.lower() for char in text if char.isprintable()]


class EspeakBackend:
    def phonemize(self, text: str, language: str) -> list[str]:
        from phonemizer import phonemize  # type: ignore[import-not-found]

        rendered = phonemize(
            text,
            language=language.split("-")[0],
            backend="espeak",
            separator=None,
            strip=True,
            preserve_punctuation=True,
            njobs=1,
        )
        return [symbol for symbol in re.findall(r"\S| ", rendered) if symbol != " "]


class Phonemizer:
    def __init__(self, backend: str = "auto") -> None:
        self.name = backend
        self.backend: Backend = self._resolve(backend)
        self._cache: dict[tuple[str, str], tuple[str, ...]] = {}

    @staticmethod
    def _resolve(name: str) -> Backend:
        if name == "grapheme":
            return GraphemeBackend()
        if name not in {"auto", "espeak"}:
            raise ValueError(f"unknown phonemizer backend: {name}")
        try:
            import phonemizer  # noqa: F401
        except ImportError:
            if name == "espeak":
                raise
            return GraphemeBackend()
        return EspeakBackend()

    def phonemize(self, text: str, language: str) -> tuple[str, ...]:
        key = (text, language)
        if key in self._cache:
            return self._cache[key]
        try:
            result = tuple(self.backend.phonemize(text, language))
        except (RuntimeError, OSError):
            if self.name == "espeak":
                raise
            result = tuple(GraphemeBackend().phonemize(text, language))
        if len(self._cache) >= 4096:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = result
        return result
