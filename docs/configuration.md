# Configuration reference

## Configuration philosophy

All signal-processing and model-shape decisions must be explicit and reproducible. YAML is the human
interface; Pydantic models in `src/tts_pipeline/config.py` are the executable schema. Unknown keys are
rejected (`extra="forbid"`) so a misspelling cannot silently fall back to a default.

`configs/base.yaml` defines the reference system. `development.yaml` and `test.yaml` use `extends` to
override nested fields. Resolution is a recursive dictionary merge: a child mapping replaces only its
specified keys, while a scalar or list replaces the parent value entirely. There is one inheritance
level in the current loader; do not assume recursive `extends` chains.

Environment overrides currently exist for:

| Variable | Resolved field | Typical use |
|---|---|---|
| `TTS_DEVICE` | `runtime.device` | select `cpu`, `cuda`, or another PyTorch device |
| `TTS_MAX_CONCURRENCY` | `serving.max_concurrency` | deployment-specific memory limit |
| `TTS_MODEL_DIR` | `serving.model_dir` | relocate an immutable model bundle |
| `TTS_API_KEY` | read indirectly through `serving.api_key_env` | enable API-key authentication |
| `TTS_LOG_LEVEL` | logging setup | operational verbosity |

The value of `TTS_CONFIG` is used by the environment app factory, while the CLI generally accepts
`--config` explicitly.

## Runtime section

- `seed` controls Python, NumPy, CPU PyTorch, and CUDA RNGs. A seed makes a run reproducible only when
  inputs, code, libraries, hardware-sensitive operations, and worker seeding are also controlled.
- `device` is passed to `torch.device`. A CUDA value must correspond to an available device.
- `deterministic` requests deterministic PyTorch algorithms with warnings. Some GPU/library combinations
  can still vary; record the environment and use regression tolerances.

## Audio section

`sample_rate` is the canonical waveform rate. Every training waveform is resampled to it, the vocoder
generates at it, and requested output rates are derived from it. Changing it invalidates features and
models.

`n_fft` controls frequency-bin count (`n_fft/2 + 1` one-sided bins). `win_length` is the Hann analysis
window and cannot exceed `n_fft`. `hop_length` controls frame spacing and must equal the product of
vocoder upsample rates.

`n_mels`, `f_min`, and `f_max` define the mel representation. `f_max` cannot exceed Nyquist
(`sample_rate / 2`). `center` controls STFT padding and therefore frame counts at boundaries. `power=1`
uses magnitude; changing it to `2` uses power and changes feature scale. `log_floor` prevents `log(0)`.

`peak_normalize`, `trim_silence`, and `trim_db` control deterministic waveform conditioning. These values
affect duration and features and therefore belong in compatibility fingerprints.

## Text section

`language` is the default language code, not proof of multilingual support. `unicode_form` is `NFKC` or
`NFC`; the reference uses NFKC to fold compatibility characters. `max_characters` limits total request
work before phonemization. `max_tokens_per_chunk` limits each model call after segmentation.

`unknown_policy` records intended unknown-character behavior, although the current vocabulary encoder
maps unknown symbols to `<unk>`; applications requiring reject/drop behavior should enforce that policy
before encoding and add tests.

## Acoustic-model section

`vocabulary_path` is a training convenience. Inference bundles carry their own vocabulary and checksum.
`hidden_dim` must divide evenly by `attention_heads`. Encoder and decoder layer counts determine depth;
`conv_filter_size` controls FFT-block feed-forward capacity; `dropout` applies during training.

`max_positions` bounds both token/mel positional encoding. Inference also rescales excessive predicted
durations to prevent allocation beyond this bound. `speaker_count` and `language_count` size conditioning
tables; changing either invalidates weights. Variance predictor filter/kernel settings and post-net depth
are weight-shape parameters.

## Vocoder section

`channels` is the generator’s initial channel width. Each upsample stage halves it. Lists
`upsample_rates` and `upsample_kernel_sizes` must have the same length. Their rate product must equal
`hop_length`; with `[8,8,2,2]`, the product is 256. Kernel/padding combinations are selected so each
stage produces exactly the desired length. `resblock_kernel_sizes` creates parallel receptive fields at
every stage.

## Training section

Batch size and worker count influence memory and throughput, not inference compatibility. `epochs` and
`patience` provide upper and early-stopping limits. `learning_rate` and `weight_decay` configure AdamW.
`gradient_accumulation` divides loss and delays optimizer updates; the effective batch is approximately
batch size times accumulation times distributed world size. `gradient_clip` bounds global gradient norm.

`mixed_precision` enables CUDA AMP when the device is CUDA. `checkpoint_every` counts optimizer updates,
not raw batches. With accumulation greater than one, global step increments less often than batch index.

## Serving section

`max_concurrency` is a hard bound on simultaneous model work inside one process. Queue and request
timeouts have different meanings: the first limits waiting to acquire the semaphore; the second limits
the awaited synthesis operation. Python worker-thread cancellation is cooperative, so a timed-out model
call may finish in its worker even though the response is abandoned; production GPU isolation may
require process-level execution for hard cancellation.

`max_request_bytes` is checked from `Content-Length`; an ingress proxy should independently enforce an
actual body limit, including chunked bodies. `expose_normalized_text` is a privacy decision. Base64
responses increase memory and payload size and should normally be disabled. `api_key_env` names the
secret variable; it does not contain the secret. `model_dir` can change across deployments without
invalidating the artifact.

## Fingerprints

`Settings.fingerprint()` hashes the complete resolved configuration for experiment reproduction.
`Settings.artifact_fingerprint()` hashes only audio, acoustic-model, and vocoder sections. This
distinction is essential: changing a serving directory must not invalidate weights, but changing mel
bins must. Vocabulary compatibility is checked separately because the vocabulary is a versioned file.

## Safe change procedure

1. Modify a child config rather than the base when experimenting.
2. Run `tts validate-config --config <file>` and retain the printed resolved configuration.
3. Decide whether the change affects raw data, cached features, acoustic weights, vocoder weights, or
   serving only.
4. Regenerate every affected downstream artifact; never reuse a cache based only on matching filenames.
5. Add a test for new cross-field invariants.
6. Export and load a bundle using the deployment’s relocated path to verify that only meaningful fields
   participate in compatibility.

