#!/usr/bin/env python3
"""Download a user-supplied public dataset archive with checksum and safe extraction."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="HTTPS archive URL reviewed by the operator")
    parser.add_argument("sha256", help="Expected publisher-provided SHA-256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.url.startswith("https://"):
        raise SystemExit("only HTTPS URLs are accepted")
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "dataset.tar"
    with urllib.request.urlopen(args.url, timeout=60) as response, archive.open("wb") as stream:  # noqa: S310
        while block := response.read(1024 * 1024):
            stream.write(block)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != args.sha256.lower():
        archive.unlink()
        raise SystemExit(f"checksum mismatch: received {digest}")
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            target = (args.output / member.name).resolve()
            if args.output.resolve() not in target.parents and target != args.output.resolve():
                raise SystemExit(f"unsafe archive path: {member.name}")
        bundle.extractall(args.output, filter="data")
    print(args.output)


if __name__ == "__main__":
    main()
