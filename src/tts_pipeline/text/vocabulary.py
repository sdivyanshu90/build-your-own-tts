"""Serializable symbol vocabulary with content-addressed compatibility checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tts_pipeline.errors import CompatibilityError, ValidationError

SPECIALS = ("<pad>", "<unk>", "<bos>", "<eos>", "|")


@dataclass(frozen=True)
class Vocabulary:
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.symbols[: len(SPECIALS)] != SPECIALS:
            raise ValidationError(f"vocabulary must start with {SPECIALS}")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValidationError("vocabulary contains duplicate symbols")

    @classmethod
    def build(cls, sequences: Iterable[Iterable[str]]) -> Vocabulary:
        observed = sorted({symbol for seq in sequences for symbol in seq if symbol not in SPECIALS})
        return cls(SPECIALS + tuple(observed))

    @classmethod
    def default_graphemes(cls) -> Vocabulary:
        symbols = tuple("abcdefghijklmnopqrstuvwxyz0123456789.,!?;:'\"-()")
        return cls(SPECIALS + symbols)

    @property
    def checksum(self) -> str:
        payload = "\n".join(self.symbols).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def encode(self, symbols: Iterable[str], boundaries: bool = True) -> list[int]:
        mapping = {symbol: index for index, symbol in enumerate(self.symbols)}
        ids = [mapping.get(symbol, mapping["<unk>"]) for symbol in symbols]
        return ([mapping["<bos>"]] + ids + [mapping["<eos>"]]) if boundaries else ids

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> list[str]:
        result = []
        for token_id in ids:
            if not 0 <= token_id < len(self.symbols):
                raise ValidationError(f"token id out of range: {token_id}")
            symbol = self.symbols[token_id]
            if not (skip_special and symbol.startswith("<")):
                result.append(symbol)
        return result

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "symbols": self.symbols, "checksum": self.checksum}
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path, expected_checksum: str | None = None) -> Vocabulary:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = cls(tuple(payload["symbols"]))
        if payload.get("checksum") != vocab.checksum:
            raise CompatibilityError("vocabulary checksum does not match its symbols")
        if expected_checksum and vocab.checksum != expected_checksum:
            raise CompatibilityError("vocabulary is incompatible with model artifact")
        return vocab
