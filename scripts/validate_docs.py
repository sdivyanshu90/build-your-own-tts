#!/usr/bin/env python3
"""Validate Markdown hierarchy, local links, code fences, and Mermaid block declarations."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_FILES = sorted(
    [path for path in ROOT.glob("*.md") if path.name != "PROMPT.md"]
    + list((ROOT / "docs").rglob("*.md"))
)
LINK_PATTERN = re.compile(r"\[[^]]*\]\(([^)]+)\)")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
NUMBERED_H2_PATTERN = re.compile(r"^(\d+)\.\s+")
MERMAID_DECLARATIONS = {
    "flowchart",
    "graph",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
    "stateDiagram-v2",
    "erDiagram",
    "journey",
    "gantt",
    "pie",
    "gitGraph",
    "mindmap",
    "timeline",
}


def validate_document(path: Path) -> list[str]:
    relative = path.relative_to(ROOT)
    lines = path.read_text(encoding="utf-8").splitlines()
    errors: list[str] = []
    if not lines or not lines[0].startswith("# "):
        errors.append(f"{relative}: first line must be the document H1")

    headings: list[tuple[int, str, int]] = []
    fence_language: str | None = None
    fence_start = 0
    mermaid_content: list[str] = []

    for line_number, line in enumerate(lines, 1):
        if line.startswith("```"):
            if fence_language is None:
                fence_language = line[3:].strip()
                fence_start = line_number
                mermaid_content = []
            else:
                if fence_language == "mermaid":
                    declaration = next(
                        (
                            item.strip().split(maxsplit=1)[0]
                            for item in mermaid_content
                            if item.strip()
                        ),
                        None,
                    )
                    if declaration not in MERMAID_DECLARATIONS:
                        errors.append(
                            f"{relative}:{fence_start}: Mermaid block has no supported declaration"
                        )
                fence_language = None
                mermaid_content = []
            continue
        if fence_language is not None:
            if fence_language == "mermaid":
                mermaid_content.append(line)
            continue
        match = HEADING_PATTERN.match(line)
        if match:
            headings.append((len(match.group(1)), match.group(2), line_number))

    if fence_language is not None:
        errors.append(f"{relative}:{fence_start}: unclosed fenced code block")

    h1_count = sum(level == 1 for level, _, _ in headings)
    if h1_count != 1:
        errors.append(f"{relative}: expected exactly one H1, found {h1_count}")
    previous_level = 0
    for level, title, line_number in headings:
        if previous_level and level > previous_level + 1:
            errors.append(
                f"{relative}:{line_number}: heading level jumps "
                f"from H{previous_level} to H{level}: {title}"
            )
        previous_level = level

    numbered_h2 = [
        (int(match.group(1)), line_number)
        for level, title, line_number in headings
        if level == 2 and (match := NUMBERED_H2_PATTERN.match(title))
    ]
    if numbered_h2:
        values = [value for value, _ in numbered_h2]
        expected = list(range(1, len(values) + 1))
        if values != expected:
            errors.append(
                f"{relative}: numbered H2 sequence is {values}, expected consecutive {expected}"
            )

    for match in LINK_PATTERN.finditer("\n".join(lines)):
        target = match.group(1).split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            errors.append(f"{relative}: broken local link: {match.group(1)}")
    return errors


def main() -> int:
    errors = [error for path in MARKDOWN_FILES for error in validate_document(path)]
    if errors:
        print("Documentation validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    mermaid_count = sum(
        path.read_text(encoding="utf-8").count("```mermaid") for path in MARKDOWN_FILES
    )
    print(f"Validated {len(MARKDOWN_FILES)} Markdown files and {mermaid_count} Mermaid diagrams.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
