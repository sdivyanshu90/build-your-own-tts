# Changelog

This project follows a human-readable changelog. Future releases should separate Added, Changed, Fixed,
Security, Deprecated, and Removed items and explicitly call out artifact/schema compatibility.

## 0.1.0 - 2026-07-18

### Added

- Typed YAML configuration with inheritance, environment overrides, full-run and artifact fingerprints.
- Deterministic English normalization trace, optional espeak phonemizer, grapheme fallback, and
  checksum-protected vocabulary.
- JSONL manifest loading, validation/reporting, deterministic speaker-stratified splits, synthetic fixture,
  PyTorch mel extraction, feature caching, and explicit alignment artifact contract.
- FastSpeech2-style acoustic model with speaker/language conditioning, duration/pitch/energy predictors,
  length regulation, masking, post-net, and finite mask-aware losses.
- HiFi-GAN-style generator, multi-period/multi-scale discriminators, adversarial/feature/mel objectives,
  and fully resumable vocoder training.
- Reproducible training, local/optional MLflow tracking, atomic integrity-checked checkpoints, inference
  bundles, long-text synthesis, WAV encoding, tracing, storage, and watermark interfaces.
- FastAPI raw/JSON/storage/chunked endpoints, probes, metrics, request IDs, shared-key hook, rate limit,
  bounded concurrency, timeout, idempotency, and privacy-oriented JSON logging.
- CLI lifecycle, CPU/GPU containers, Compose hardening, CI/security/image workflows, and unit/integration/
  e2e/performance/regression tests.
- Engineering handbook, architecture decision, model/data cards, security and responsible-use policy.

### Known limitations

- No trained weights or production forced-aligner/F0 backend are distributed.
- Fixture alignment/pitch are not suitable for quality training.
- Streaming is chunked completed-WAV delivery rather than incremental neural synthesis.
- Rate/idempotency state is process-local and request timeout is not hard worker cancellation.
