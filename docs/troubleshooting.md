# Troubleshooting and incident runbook

## Method

Diagnose from earliest deterministic boundary to latest learned boundary. Record request ID, model/
config/vocabulary versions, device, input size, and numeric summaries. Use authorized or synthetic text;
do not paste private production transcripts into issues or shared chat.

## Configuration fails validation

Run `tts validate-config --config <path>` and read the full Pydantic field path. Common causes are
`f_max` above Nyquist, window larger than FFT, hidden dimension not divisible by heads, unequal vocoder
list lengths, or upsample product not equal to hop. Inspect resolved parent/child merge rather than only
the child YAML. Environment overrides may change device/concurrency/model directory.

## Bundle configuration fingerprint mismatch

Compare `artifact_fingerprint`, not complete serving configuration. Audio/model/vocoder tensor parameters
must match export. A relocated `TTS_MODEL_DIR` should not change compatibility. If meaningful fields
differ, use the original matching config or retrain/re-export; never edit the manifest hash manually.

## Bundle integrity or vocabulary checksum failure

Stop rollout. Verify immutable source/version, transfer completeness, filesystem, and expected manifest
trust. Recopy from approved registry to a new path and verify again. Do not disable hashing. A vocabulary
checksum failure means token IDs may map to wrong embeddings.

## `/health` is 200 but `/ready` is 503

This is expected after model-load failure. Inspect startup JSON exception for missing manifest/file,
fingerprint/hash/state-dict shape/device failure. Confirm mount path/permissions and that config has been
resolved inside the container. Liveness should not be changed to mask readiness.

## Dataset validation errors

For missing files, remember relative paths resolve from manifest directory. Corrupt audio may be an
unsupported codec in the SciPy fallback; install SoundFile/libsndfile or standardize WAV. Duplicate errors
can indicate repeated records. Duration errors require checking segmentation, sample-rate headers, and
silence. Quarantine with reason; do not blindly raise limits.

## Preprocessing refuses to align

The repository has no production aligner configured. For software smoke tests only, pass
`--fixture-alignment`. For training, implement/import precomputed forced-alignment artifacts following
the documented contract. If frames are fewer than tokens, inspect transcript/audio mismatch, very short
clip, hop/centering, and symbol granularity.

## Alignment sum mismatch

Compare token count/order, mel frame count, sample-rate/hop/center convention, time-to-frame rounding, and
silence/boundary token mapping. Visualize cumulative duration boundaries on mel. Do not pad target arrays
until sums happen to match.

## Acoustic loss is NaN or infinite

Identify batch/sample IDs. Check finite mel/pitch/energy, non-negative durations, non-empty masks, target
scale, AMP behavior, learning rate, and recent config/cache changes. Reproduce in full precision on one
batch. Resume from last clean checkpoint only after root cause is fixed.

## Training does not improve

Attempt one-sample overfit. Confirm gradients/nonzero masks, teacher targets, vocabulary/checksum,
alignment listening, and optimizer steps with accumulation. Separate duration/pitch/energy/mel losses.
If one sample cannot overfit, scaling data or model will not solve pipeline correctness.

## Generated audio is empty or extremely short

Inspect normalized text and token count, then predicted log durations and selected durations. Inference
ensures at least one frame per valid token, so truly empty output suggests no speakable chunks or later
encoding failure. Very short output suggests untrained/collapsed durations or excessive rate.

## Audio is noise

Random `tts export-model` bundles intentionally contain untrained weights and generate noise-like output.
For trained systems, test vocoder with ground-truth mel. If that is noisy, check vocoder training and mel
compatibility. If ground-truth mel is clear but predicted mel is noisy, inspect acoustic model.

## Audio is metallic, buzzing, or muffled

Compare sample rate, hop, mel bins/range, magnitude/power, Slaney normalization, natural-log floor, and
centering between feature extraction and vocoder. Inspect adversarial balance and training duration.
Muffled ground-truth-mel reconstruction indicates vocoder/feature issue; clear ground truth but muffled
prediction indicates acoustic over-smoothing.

## Words are skipped, repeated, or mispronounced

Print normalization trace and phoneme symbols using a secure local reproduction. Verify backend is
consistent across vocabulary/training/serving. Inspect unknown IDs and token durations. Mispronunciation
with correct phonemes is acoustic/data; wrong phonemes are text/G2P; correct mel timing with bad waveform
is vocoder.

## Chunk boundary clicks or strange pauses

Inspect sentence segmentation and each chunk waveform level. Crossfade is 15 ms and may be too short for
a discontinuity or too long for consonants. Ensure model learned punctuation pauses. Do not concatenate
encoded WAV files. Consider clause-aware segmentation and vocoder overlap context.

## CPU synthesis is slow

Measure stage spans and distinguish import/model load, phonemizer, acoustic, vocoder, resampling, and
encoding. Warm model, use smaller trained config, control PyTorch/OpenMP threads, avoid excess workers,
and consider GPU/exported graph only after profiling. RTF and short-request latency are separate.

## CUDA out of memory

Record free/resident/peak memory, tokens, predicted frames, concurrency, and worker count. Lower batch or
online concurrency, chunk tokens, dimensions, or precision after testing. Ensure one process per GPU and
no overlapping rollout models. Duration caps protect but do not guarantee capacity for every config.

## Requests time out but GPU remains busy

Route timeout cancels the await, not necessarily worker/native/GPU execution. Reduce allowed work,
increase capacity, and use process/queue isolation for hard cancellation. Watch active versus actual GPU
load during timeout storms.

## Rate limiting seems inconsistent

Limiter is process-local and keyed by observed client host. Multiple replicas and reverse proxies make it
inconsistent. Configure trusted client identity and use distributed gateway/Redis limiting with tenant
keys. Do not trust arbitrary forwarded headers.

## Incident closure

After mitigation, document impact/timeline/root cause, add the smallest reproducing fixture/test, update
this runbook and relevant design chapter, evaluate whether artifacts/data need retirement, and verify the
fix in canary plus rollback path.
