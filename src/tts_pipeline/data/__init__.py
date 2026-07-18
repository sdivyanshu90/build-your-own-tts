"""Manifest ingestion, validation, splitting, and dataset loading."""

from tts_pipeline.data.manifest import (
    ManifestRecord,
    load_manifest,
    split_manifest,
    validate_manifest,
)

__all__ = ["ManifestRecord", "load_manifest", "split_manifest", "validate_manifest"]
