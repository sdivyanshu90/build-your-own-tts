# Model card: modular FastSpeech2-style and HiFi-GAN-style TTS

## Card status

This is a template and implementation-level card for repository version 0.1.0. The repository does not
ship trained production weights. Every trained/exported bundle must copy this card, fill in dataset/run/
evaluation sections, identify its exact model version, and undergo responsible-use approval.

## Model overview

The platform defines two learned components:

- a FastSpeech2-style acoustic model that maps vocabulary token IDs plus speaker/language embeddings and
  variance controls to log-mel spectrograms; and
- a HiFi-GAN-style generator that maps compatible log-mel spectrograms to mono waveform.

The acoustic model includes sinusoidal positions, encoder/decoder self-attention-convolution blocks,
duration/pitch/energy predictors, length regulation, mel projection, and convolutional post-net. The
vocoder includes transposed-convolution upsampling and multi-kernel dilated residual blocks. Vocoder
training uses multi-period and multi-scale discriminators; discriminators are not deployed.

Reference audio is 22,050 Hz with 80 natural-log mel bins, FFT/window 1024, hop 256, 0–8 kHz range,
Slaney filter normalization, and magnitude power 1. Bundle compatibility verifies exact configuration and
vocabulary checksum.

## Intended uses

Allowed intended uses are authorized synthetic voices for accessibility, assistive reading, consented
product narration, research/education, and controlled speech interfaces. A deployment must have explicit
speaker/data rights for its actual purpose and must enforce access appropriate to risk.

The random bundle created by `tts export-model` without checkpoints is intended only to verify package,
artifact, inference, API, and WAV interfaces. It produces noise-like audio and must not be evaluated as a
trained speech model.

## Prohibited and out-of-scope uses

Prohibited uses include non-consensual voice cloning, deceptive impersonation, fraud/social engineering,
harassment, fabricated evidence, biometric authentication bypass, undisclosed political persuasion,
evasion of required disclosure, training on scraped/private voices without rights, and any unlawful or
harmful use.

Out of scope for the reference implementation: zero-shot voice cloning, verified speaker enrollment,
emotion/style transfer, conversational dialogue policy, true incremental neural streaming, and safety
guarantees from watermark detection.

## Users and affected people

Direct users are ML/backend engineers and authorized API clients. Affected people include enrolled
speakers, dataset contributors, listeners who may interpret generated speech, people referenced in
content, operators/reviewers, and groups impacted by language/accent quality disparities. Release review
must consider impacts beyond the API caller.

## Training data

No training data is distributed. A concrete model card must record:

- manifest/dataset fingerprint and immutable version;
- source, license, consent scope, collection and transcript method;
- speakers, hours, utterances, languages, and session distribution;
- train/validation/test split policy and leakage controls;
- preprocessing, normalizer, phonemizer, vocabulary, aligner, F0/energy versions;
- exclusions, quality filters, known bias, retention, and revocation lineage; and
- whether any pretrained component was used and under what license/provenance.

See [DATA_CARD.md](DATA_CARD.md) and the [data pipeline](docs/data-pipeline.md).

## Training procedure

A concrete release must record acoustic/vocoder configurations, optimizer/schedulers, effective batch,
precision, epochs/steps, early stopping, hardware/software/container, seed/determinism warnings, parent
checkpoints, experiment tracker URI, and final checkpoint hashes. It must distinguish teacher-target
validation from predicted-mode synthesis evaluation.

Duration targets must come from a documented forced-alignment backend. The uniform fixture backend and
zero fixture pitch targets are not valid evidence of a production training pipeline.

## Evaluation requirements

At minimum, report on a frozen held-out set:

| Dimension | Required evidence |
|---|---|
| Intelligibility | ASR WER/CER plus manual critical-error review |
| Naturalness | blinded MOS with confidence interval and/or AB preference |
| Prosody | duration and voiced-F0/energy metrics plus human phrasing review |
| Speaker | authorized similarity/consistency by speaker; leakage/confusion review |
| Signal | duration, peak, RMS, clipping, silence, artifact listening |
| Robustness | normalization/Unicode/long/OOD/adversarial text suites |
| Performance | cold/warm latency, RTF, throughput, memory, concurrency percentiles |
| Fairness | quality/intelligibility slices for supported speaker/language/accent coverage |
| Safety | authorization, enumeration, abuse, disclosure/watermark, revocation red team |

Report sample counts, confidence intervals, tools/models/versions, hardware, and release thresholds chosen
before viewing candidate results. Objective metrics do not replace listening.

## Known implementation limitations

- English normalization is reference-complete only for a bounded set of formats; accepting another
  language code does not create language support.
- Automatic phonemizer fallback can produce different symbols across environments unless backend is
  explicitly pinned.
- Fixture preprocessing uses uniform durations and zero pitch.
- Raw pitch/energy targets lack learned normalization statistics in the compact reference.
- Long text uses simple punctuation segmentation and fixed token-window fallback.
- Streaming endpoint chunks completed WAV, not model generation.
- In-memory rate/idempotency state does not coordinate replicas.
- Thread timeout is not guaranteed to stop model execution.
- No-op watermarker provides no disclosure or provenance signal.
- `weights_only=True` reduces deserialization risk but does not make untrusted artifacts safe.

## Model-specific risks

Generated speech can be misleading, biased, unintelligible, offensive through supplied content, or
mistaken for a real speaker. Speaker embeddings may memorize identity. Poor data balance can degrade or
stereotype accents. Wrong normalization can change financial/date meaning. Alignment errors can omit or
repeat content. A valid WAV is not evidence of truth or consent.

## Safeguards and deployment requirements

Use explicit identity and per-speaker authorization, strong rate/output limits, immutable verified
bundles, private logs/metrics, text/audio minimization, consent lineage and revocation, restricted speaker
controls, abuse monitoring/reporting, disclosure appropriate to context, and a tested watermark adapter
only when claimed. Follow [responsible use](docs/responsible-use.md) and [security](docs/security.md).

## Concrete release appendix template

```text
Model version:
Bundle SHA-256/signature:
Code revision/image digest:
Artifact fingerprint/vocabulary checksum:
Approved speakers/languages/purposes:
Dataset and split fingerprints:
Training run IDs/checkpoints:
Evaluation report and thresholds:
Known slice failures:
Deployment limits and hardware:
Consent expiration/revocation owner:
Security/responsible-use reviewers and date:
```

