# ADR 0001: FastSpeech2-style acoustic model with a HiFi-GAN-style vocoder

- Status: accepted
- Date: 2026-07-18
- Decision owners: project maintainers

## Context

The repository needs a model architecture that is trainable at modest scale, understandable to an
engineer, controllable during inference, and separable into replaceable components. It must support
multi-speaker conditioning, variable-length batching, low-latency parallel inference, explicit duration
handling, and an offline smoke-test path. It is not intended to reproduce a frontier model or conceal
the data/alignment work needed for natural speech.

## Options considered

### Autoregressive attention model plus neural vocoder

Tacotron-style models are conceptually direct and can learn soft alignment, but inference is sequential.
They are vulnerable to attention failures such as skipped, repeated, or never-ending words—especially
for long or out-of-domain text. Rate control is indirect, and production latency is less predictable.

### VITS-style end-to-end model

VITS can produce excellent quality and jointly learn alignment, acoustic representation, and waveform
generation. The tradeoff is much tighter coupling among alignment, latent representation, flow,
discriminator, and waveform losses. That coupling makes isolated component replacement, debugging, and
educational inspection harder. End-to-end training is also sensitive to data and optimization choices.

### FastSpeech2-style acoustic model plus HiFi-GAN-style vocoder

FastSpeech2 performs parallel token encoding and explicit length regulation. Duration, pitch, and energy
are first-class supervised signals and inference controls. HiFi-GAN performs parallel waveform synthesis
and is trained independently. The boundary is a configured log-mel spectrogram.

## Decision

Use a compact FastSpeech2-style acoustic model and a separate HiFi-GAN-style vocoder. Require explicit
duration artifacts rather than implying they arise automatically. Preserve interfaces so either model
can later be replaced if compatibility checks are updated.

## Rationale

1. Explicit durations make alignment provenance inspectable and long-text behavior more predictable.
2. Non-autoregressive inference exposes parallelism and avoids decoder stop-token failures.
3. Pitch, energy, and speaking-rate controls map to explicit parts of the model.
4. The mel boundary allows acoustic and vocoder training to use different batch strategies and even
   different compatible datasets.
5. Failures can be localized: pronunciation/text, alignment/duration, mel prediction, or vocoding.
6. Tensor shapes and masks are teachable without hiding important production assumptions.

## Consequences

Positive consequences include modularity, controllability, batched inference, predictable upper bounds,
and the ability to use a compatible pretrained vocoder. Negative consequences include maintaining two
training systems, requiring forced-alignment infrastructure, possible train/inference pitch-duration
mismatch, and a strict mel compatibility contract.

Model quality will depend at least as much on transcript accuracy, alignment quality, speaker consented
data coverage, and vocoder training as on architecture dimensions. This ADR does not claim state-of-the-
art naturalness.

## Compatibility commitments

A model bundle is compatible only when vocabulary checksum, mel bins, sample rate, FFT/window/hop,
frequency range, acoustic dimensions, speaker/language counts, and vocoder topology match. Serving paths,
training batch size, and learning-rate schedules do not invalidate inference weights.

## Revisit criteria

Reconsider this decision if measured requirements demand true incremental streaming, substantially
better multilingual prosody, end-to-end speaker adaptation, or quality that cannot be met without a
different representation. A replacement requires a new ADR, migration strategy, artifact version, and
side-by-side quality/latency/security evaluation.
