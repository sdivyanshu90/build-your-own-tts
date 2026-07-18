# Command-line interface reference

The `tts` entry point is implemented with Typer in `src/tts_pipeline/cli.py`. Commands return nonzero on
validation or uncaught domain failures, making them suitable for scripts. Run `tts <command> --help` for
the installed version; this chapter explains intent, inputs, artifacts, and production caveats.

## Global configuration pattern

Commands that depend on model/audio/training settings accept `--config` or `-c` and default to
`configs/development.yaml`. Always pass a pinned configuration in automation. Relative paths resolve
from current working directory except manifest audio paths, which resolve from manifest location.

## `validate-config`

```bash
tts validate-config --config configs/base.yaml
```

Loads inheritance/environment overrides, applies all Pydantic and cross-component checks, and prints
resolved JSON plus full configuration fingerprint. Run in CI and at deployment startup. It does not
verify that a model directory or dataset exists.

## `validate-dataset`

```bash
tts validate-dataset data/corpus/manifest.jsonl --config configs/base.yaml
```

Loads JSONL and reports valid/total records, errors, warnings, speaker distribution, total duration, and
dataset identity fingerprint. Expected sample rate comes from config. Exits status 2 if report has errors.
It reads audio metadata but does not preprocess whole waveforms or verify consent metadata.

## `build-vocabulary`

```bash
tts build-vocabulary manifest.jsonl --output artifacts/vocabulary.json
```

Normalizes and phonemizes every transcript, then writes deterministic ordered symbols/checksum. The
command currently constructs default normalizer/automatic phonemizer rather than reading text config;
for production, pin backend/environment and confirm it matches training/serving. Never rebuild vocabulary
for existing acoustic weights.

## `preprocess`

```bash
tts preprocess manifest.jsonl processed vocabulary.json \
  --config configs/base.yaml --fixture-alignment
```

Creates `features/*.npz`, `alignments/*.json`, and `index.jsonl`. The explicit fixture flag is required in
the reference because no production aligner backend is configured. Without it, command fails rather than
inventing labels. A production integration should consume approved alignment artifacts and extract real
F0 targets.

## `align`

```bash
tts align manifest.jsonl aligned vocabulary.json --fixture \
  --config configs/development.yaml
```

Uses the same reference preprocessing/alignment path and is present to expose lifecycle intent. It is
not an MFA launcher. When adding a production aligner, separate raw alignment execution from feature
conversion and preserve backend outputs/version/confidence.

## `train-acoustic`

```bash
tts train-acoustic processed/index.jsonl vocabulary.json \
  --run-dir runs/acoustic --config configs/base.yaml
```

Loads preprocessed tensors, derives a small seeded validation split, prints startup report, and runs the
acoustic trainer. `--resume checkpoint.pt` restores full state. Outputs include tracker events and
`best.pt`, `latest.pt`, periodic step checkpoints plus integrity sidecars. For production, provide frozen
train/validation indexes rather than the compact CLI’s derived split.

## `train-vocoder`

```bash
tts train-vocoder manifest.jsonl --run-dir runs/vocoder \
  --config configs/base.yaml --resume runs/vocoder/latest.pt
```

Builds waveform/mel segment dataset and runs alternating HiFi-GAN updates. It is computationally
expensive with reference model and not a quality recipe. Resume restores generator/discriminators, both
optimizers/schedulers, scaler, epoch, and step.

## `export-model`

```bash
tts export-model artifacts/english-v1 --vocabulary vocabulary.json \
  --acoustic-checkpoint runs/acoustic/best.pt --version english-v1 \
  --config configs/base.yaml
```

Creates vocabulary, acoustic/vocoder tensor files, and integrity manifest. If vocabulary omitted, a
default grapheme vocabulary is used. If acoustic checkpoint omitted, acoustic weights are random. The
current command constructs random vocoder weights; a real release workflow must load/promote the approved
trained generator before export. Therefore this command is complete for bundle/interface smoke testing
but requires extension for a natural full model release.

## `synthesize`

```bash
tts synthesize "Hello." --output output.wav --speaker default --language en-US \
  --rate 1.0 --pitch 1.0 --energy 1.0 --seed 7 --config configs/base.yaml
```

Loads bundle from resolved `serving.model_dir`, synthesizes, writes WAV, and prints metadata. Set
`TTS_MODEL_DIR` to relocate bundle. Library validation rejects unknown speaker/language and invalid
controls. The CLI writes to the operator-supplied output path; do not expose it directly to untrusted
users.

## `evaluate`

```bash
tts evaluate output.wav --latency 0.42
```

Reports duration, peak, RMS, clipping, silence, and optional RTF. It is signal sanity, not a complete TTS
quality score. See evaluation chapter.

## `inspect-checkpoint`

```bash
tts inspect-checkpoint runs/acoustic/latest.pt
```

Uses `weights_only=True` to print format, state metadata, and model tensor count. It does not verify a
sidecar in this inspection command or prove trust. Never inspect arbitrary files in a privileged process.

## `benchmark`

```bash
tts benchmark --text "The quick brown fox jumps over the lazy dog." \
  --iterations 10 --config configs/base.yaml
```

Loads the configured bundle, runs deterministic synthesis repeatedly, and prints first (“cold” synthesis)
versus later warm calls, duration, RTF, device/Python, and peak CUDA allocation. Bundle load itself occurs
before iteration timing; measure process cold start separately.

## `serve`

```bash
tts serve --config configs/base.yaml
```

Configures JSON logging and starts Uvicorn at serving host/port. Use a production gateway/process manager,
TLS, identity, distributed limits, and capacity-tested process count. One process per GPU is the default.

## Automation practices

Use explicit paths/config, capture stdout/stderr and exit code, retain resolved config/fingerprints, write
to immutable versioned output directories, and do not parse human help text as a stable machine schema.
For workflow orchestration, call typed Python functions where richer error/report structures are needed.

