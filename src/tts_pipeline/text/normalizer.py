"""English text normalization with an inspectable transformation trace."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from tts_pipeline.errors import ValidationError

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
    20: "twentieth",
    30: "thirtieth",
}
_ABBREVIATIONS = {
    "mr.": "mister",
    "mrs.": "misses",
    "dr.": "doctor",
    "st.": "saint",
    "vs.": "versus",
    "etc.": "et cetera",
    "e.g.": "for example",
    "i.e.": "that is",
}


@dataclass(frozen=True)
class NormalizationResult:
    original: str
    normalized: str
    stages: tuple[tuple[str, str], ...]


def number_words(value: int) -> str:
    """Convert a bounded integer to deterministic US-English words."""
    if value < 0:
        return "minus " + number_words(-value)
    if value < 20:
        return _ONES[value]
    if value < 100:
        return _TENS[value // 10] + (" " + _ONES[value % 10] if value % 10 else "")
    if value < 1000:
        return (
            number_words(value // 100)
            + " hundred"
            + (" " + number_words(value % 100) if value % 100 else "")
        )
    for scale, label in ((1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")):
        if value >= scale:
            return (
                number_words(value // scale)
                + f" {label}"
                + (" " + number_words(value % scale) if value % scale else "")
            )
    raise ValueError("number is outside supported range")


def _decimal(match: re.Match[str]) -> str:
    whole, fractional = match.group(1), match.group(2)
    return f"{number_words(int(whole))} point {' '.join(_ONES[int(x)] for x in fractional)}"


def _currency(match: re.Match[str]) -> str:
    symbol, raw = match.group(1), match.group(2).replace(",", "")
    names = {"$": ("dollar", "cent"), "£": ("pound", "pence"), "€": ("euro", "cent")}
    major_name, minor_name = names[symbol]
    amount = Decimal(raw)
    major = int(amount)
    minor = int((amount - major) * 100)
    text = f"{number_words(major)} {major_name}{'' if major == 1 else 's'}"
    if minor:
        text += f" and {number_words(minor)} {minor_name}{'' if minor == 1 else 's'}"
    return text


class TextNormalizer:
    """Apply ordered, deterministic normalization rules.

    NFKC intentionally folds compatibility characters (full-width letters, ligatures) so the
    tokenizer sees a small stable alphabet. Original text and every stage remain available.
    """

    def __init__(
        self, unicode_form: Literal["NFC", "NFKC"] = "NFKC", max_characters: int = 5000
    ) -> None:
        self.unicode_form = unicode_form
        self.max_characters = max_characters

    def normalize(self, text: str, trace: bool = False) -> NormalizationResult:
        if not isinstance(text, str) or not text.strip():
            raise ValidationError("text must contain at least one non-whitespace character")
        if len(text) > self.max_characters:
            raise ValidationError(f"text exceeds {self.max_characters} character limit")
        bad = [c for c in text if unicodedata.category(c) == "Cc" and c not in "\n\t\r"]
        if bad:
            raise ValidationError("text contains unsupported control characters")
        stages: list[tuple[str, str]] = []

        def apply(name: str, value: str) -> str:
            stages.append((name, value))
            return value

        value = apply("unicode", unicodedata.normalize(self.unicode_form, text))
        value = apply(
            "punctuation",
            value.translate(
                {
                    ord(key): replacement
                    for key, replacement in {
                        "“": '"',
                        "”": '"',
                        "’": "'",
                        "—": "-",
                        "–": "-",
                        "…": "...",
                    }.items()
                }
            ),
        )
        value = apply(
            "urls",
            re.sub(
                r"https?://([^\s/]+)(?:/\S*)?",
                lambda m: " website " + m.group(1).replace(".", " dot "),
                value,
                flags=re.I,
            ),
        )
        value = apply(
            "emails",
            re.sub(
                r"\b([\w.+-]+)@([\w.-]+)\b",
                lambda m: (
                    m.group(1).replace(".", " dot ") + " at " + m.group(2).replace(".", " dot ")
                ),
                value,
            ),
        )
        for short, long in _ABBREVIATIONS.items():
            value = re.sub(rf"(?i)(?<!\w){re.escape(short)}(?!\w)", long, value)
        value = apply("abbreviations", value)
        value = apply(
            "currency", re.sub(r"([$£€])\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", _currency, value)
        )
        value = apply(
            "percent",
            re.sub(
                r"(-?\d+(?:\.\d+)?)\s*%", lambda m: self._numeric(m.group(1)) + " percent", value
            ),
        )
        value = apply("dates", re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", self._date, value))
        value = apply("times", re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", self._time, value))
        value = apply(
            "ordinals",
            re.sub(
                r"\b(\d+)(st|nd|rd|th)\b",
                lambda m: self._ordinal(int(m.group(1))),
                value,
                flags=re.I,
            ),
        )
        value = apply("decimals", re.sub(r"\b(-?\d+)\.(\d+)\b", _decimal, value))
        value = apply(
            "numbers",
            re.sub(
                r"(?<![\w.])-?\d[\d,]*(?![\w.])",
                lambda m: number_words(int(m.group().replace(",", ""))),
                value,
            ),
        )
        value = apply("acronyms", re.sub(r"\b[A-Z]{2,}\b", lambda m: " ".join(m.group()), value))
        value = apply("whitespace", re.sub(r"\s+", " ", value).strip())
        return NormalizationResult(text, value, tuple(stages) if trace else ())

    def _numeric(self, raw: str) -> str:
        if "." in raw:
            return _decimal(re.match(r"(-?\d+)\.(\d+)", raw))  # type: ignore[arg-type]
        return number_words(int(raw))

    @staticmethod
    def _date(match: re.Match[str]) -> str:
        try:
            date = datetime.strptime(match.group(0), "%Y-%m-%d")
        except ValueError:
            return match.group(0)
        return (
            f"{date.strftime('%B')} {TextNormalizer._ordinal(date.day)}, {number_words(date.year)}"
        )

    @staticmethod
    def _time(match: re.Match[str]) -> str:
        hour, minute = int(match.group(1)), int(match.group(2))
        suffix = "a m" if hour < 12 else "p m"
        spoken_hour = hour % 12 or 12
        spoken_minute = "o'clock" if minute == 0 else number_words(minute)
        return f"{number_words(spoken_hour)} {spoken_minute} {suffix}"

    @staticmethod
    def _ordinal(value: int) -> str:
        if value in _ORDINALS:
            return _ORDINALS[value]
        if value < 100 and value % 10 in _ORDINALS:
            return f"{_TENS[value // 10]} {_ORDINALS[value % 10]}"
        return number_words(value) + "th"


def segment_sentences(text: str) -> list[str]:
    """Split at safe sentence boundaries, retaining terminal punctuation."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
