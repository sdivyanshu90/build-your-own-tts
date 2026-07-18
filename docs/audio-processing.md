# Audio I/O, conditioning, and mel feature extraction

## 1. Representation and units

Waveforms are one-dimensional float32 tensors with nominal range `[-1,1]`. Sample rate is samples per
second; it does not describe bit depth. Duration is `samples / sample_rate`. The project’s canonical rate
is 22,050 Hz, so one second contains 22,050 samples.

Training uses log-mel frames rather than raw waveform. The acoustic model predicts one vector of `M=80`
mel values per frame. The vocoder expands each frame into `HOP=256` waveform samples.

```mermaid
flowchart LR
    File[("Audio file")]
    Decode["Decode to float32<br/>[samples, channels]"]
    Mono["Channel mean → mono"]
    Resample["Polyphase resample<br/>to canonical rate"]
    Finite["Finite/range validation"]
    Trim["Conservative silence trim"]
    Peak["Peak normalize to 0.95"]
    STFT["Hann-window STFT<br/>complex [N/2+1,F]"]
    Magnitude["Magnitude or power"]
    Mel["Slaney mel filter bank<br/>[M,F]"]
    Log["log(max(mel, floor))"]
    Cache[("Versioned feature cache")]

    File --> Decode --> Mono --> Resample --> Finite --> Trim --> Peak
    Peak --> STFT --> Magnitude --> Mel --> Log --> Cache
```

## 2. Audio loading

`audio.io` prefers SoundFile/libsndfile for common audio formats and exact float32 conversion. A SciPy
fallback supports WAV in minimal environments. Loading always returns `[samples, channels]`; channels are
averaged to mono.

The loader rejects read errors and later rejects NaN/infinity. Integer WAV fallback values are scaled by
the maximum representable magnitude. Production ingestion should standardize accepted MIME/container/
codec combinations at upload and never trust a filename extension alone.

## 3. Resampling

When source rate differs, SciPy polyphase resampling is used. For source `R_s` and target `R_t`, the
greatest common divisor reduces up/down factors:

```text
g = gcd(R_s, R_t)
up = R_t / g
down = R_s / g
```

Polyphase filtering avoids the severe aliasing caused by naive sample dropping or linear interpolation.
Resampling still changes phase/boundaries and must be identical between training and inference feature
validation. Do not repeatedly resample the same file through multiple rates.

## 4. Conditioning order

`process_waveform` converts to float, replaces non-finite values defensively, clamps to `[-1,1]`, trims
silence if configured, and peak-normalizes. File loading performs an earlier explicit non-finite check;
the `nan_to_num` protects direct tensor callers.

Silence trimming computes the waveform peak and a threshold `peak * 10^(-trim_db/20)`. It retains the
first-to-last active samples plus one hop of boundary padding. This is conservative compared with
aggressive voice activity detection. Fully silent audio is retained and should be rejected by higher-
level dataset quality rules.

Peak normalization scales non-negligible waveforms to peak 0.95. It is deterministic and prevents
clipping, but it does not equalize perceived loudness. RMS/LUFS normalization is a potential adapter for
datasets with inconsistent gain; changing strategy changes model inputs and must be versioned.

## 5. STFT from first principles

Speech changes over time, so one Fourier transform of the whole utterance loses timing. The short-time
Fourier transform analyzes overlapping, windowed regions:

`X[m,k] = Σ_n x[n] w[n - mH] exp(-j 2πkn/N)`

where `m` is frame, `k` is frequency bin, `N` is FFT size, `H` is hop length, and `w` is the Hann window.
For real input, PyTorch returns `N/2 + 1` unique complex bins.

Reference values:

| Parameter | Value | Consequence |
|---|---:|---|
| sample rate | 22,050 | Nyquist is 11,025 Hz |
| FFT/window | 1024 | ~46.4 ms analysis window |
| hop | 256 | ~11.61 ms between frames |
| center | true | reflection padding aligns a frame around boundary samples |

Longer windows improve frequency resolution but blur fast timing. Smaller hops improve time resolution
and increase frame count, training memory, and vocoder work. Centering changes the frame-count formula
and makes true online streaming require buffered lookahead.

## 6. Mel filter bank

The magnitude spectrum is raised to configured `power` and multiplied by triangular filters spaced on
the mel scale:

```text
mel(f) = 2595 log10(1 + f/700)
f(mel) = 700 (10^(mel/2595) - 1)
```

The implementation creates `M+2` boundary frequencies between `f_min` and `f_max`, forms overlapping
triangles, and applies area normalization so wider high-frequency filters do not automatically collect
more energy. Matrix multiplication maps `[..., N/2+1, F]` to `[..., M, F]`.

The final representation is `log(max(mel, log_floor))`. The floor prevents negative infinity for empty
bins and limits dynamic range. This is natural logarithm. A pretrained vocoder expecting log10,
different mel normalization, magnitude versus power, or different floor is incompatible even if tensor
shape is 80 bins.

## 7. Numerical and shape expectations

For mono input `[S]`, mel output is `[M,F]`. Batched waveform `[B,S]` produces `[B,M,F]`. Values must be
finite. Very short waveforms can fail reflection padding; dataset duration validation prevents normal
training inputs from reaching that case.

The tests verify mel bin count, deterministic equality on repeated CPU calls, finiteness, WAV RIFF/WAVE
headers, and crossfade length. A fuller DSP validation program should compare the custom filter bank to
a pinned reference, test known sine-bin localization, resampling passband/alias rejection, and feature
statistics on a frozen fixture corpus.

## 8. Feature caching

`cache_features` hashes resolved source path, modification time, and audio-config fingerprint. It writes
a compressed temporary `.npz` then atomically renames it. Stored metadata includes the feature
fingerprint. Modification time is convenient but not content integrity; long-lived production caches
should hash source bytes and preprocessing code/container version.

Never reuse caches after sample rate, FFT/window/hop, mel bins/range, power, floor, trim, normalization,
or centering changes. A cache miss is cheaper than silently training on mixed feature definitions.

## 9. Augmentation policy

The current core does not inject augmentation. A training-only adapter may add seeded gain, noise,
room response, band limitation, speed perturbation, or spectrogram masks. It must never affect
validation/test, must preserve transcript truth, and should model expected deployment conditions rather
than merely increase variety. Speed changes require corresponding duration/F0 handling.

## 10. Post-processing and encoding

Chunk waveforms are joined with linear overlap crossfades. For overlap width `W`, one chunk fades from 1
to 0 while the next fades from 0 to 1; total length decreases by `W` per boundary. The combined waveform
is peak-limited to 0.95, passed through the watermark hook, optionally polyphase-resampled, clamped, and
encoded as PCM-16 WAV.

PCM-16 quantization maps float samples to signed integers near `[-32767,32767]`. It introduces small
quantization noise but is widely compatible. If a future API offers float WAV or compressed audio, it
must state MIME type, subtype, sample rate, and any encoder delay.

## 11. Debugging audio mismatches

When acoustic and vocoder outputs are incompatible, compare in order: sample rate, hop product, mel bin
count, f-min/f-max, window/FFT, centering, magnitude/power, mel scale/normalization, log base/floor, and
training normalization statistics. Matching shape alone is insufficient.
