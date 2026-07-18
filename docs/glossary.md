# Glossary and notation

## Speech and signal terms

**Acoustic model** converts linguistic tokens into an intermediate acoustic representation. In this
project the representation is a log-mel spectrogram, not a waveform.

**Alignment** maps each input token to the time region—or number of mel frames—in which it is spoken.
FastSpeech2 needs explicit duration targets during supervised training.

**Frame** is one time step of a spectrogram. With sample rate 22,050 and hop length 256, adjacent frames
are approximately 11.61 ms apart: `256 / 22050` seconds.

**F0 (fundamental frequency)** is the lowest periodic frequency of voiced speech and is perceived mainly
as pitch. Unvoiced sounds have no stable F0, so pitch estimators need an explicit unvoiced convention.

**Grapheme** is a written symbol such as a letter. A grapheme tokenizer sees spelling; a phoneme
tokenizer attempts to represent pronunciation.

**HiFi-GAN** is a family of parallel neural vocoders trained with adversarial, feature-matching, and
mel-reconstruction objectives.

**Mel scale** is a nonlinear frequency scale intended to approximate human pitch discrimination. Mel
spectrograms compress high-frequency resolution compared with linear FFT bins.

**Phoneme** is an abstract contrastive speech sound. Phoneme symbols are language and backend dependent;
they do not fully encode prosody, coarticulation, or speaker style.

**Real-time factor (RTF)** is processing time divided by output audio duration. `RTF < 1` means synthesis
is faster than playback; it is not a latency percentile or throughput measure.

**STFT** applies a windowed Fourier transform at overlapping waveform positions, producing complex
time-frequency coefficients.

**Vocoder** converts acoustic features into waveform samples. It is distinct from the acoustic model.

## Model terms

**Duration predictor** estimates `log(1 + frames)` for every input token. Inference converts predictions
back to non-negative integer frame counts.

**FFT block** in FastSpeech terminology means a feed-forward Transformer block; it is unrelated to the
Fast Fourier Transform used in audio preprocessing.

**Length regulator** repeats each encoded token state according to its duration, changing token time
`T` into acoustic-frame time `F`.

**Post-net** is a convolutional residual network that predicts a correction to the initial mel output.

**Teacher target** is a supervised value supplied during training—duration, pitch, or energy—rather than
the model’s own prediction.

**Variance adaptor/control** refers to duration, pitch, and energy prediction and conditioning. The word
“variance” here means sources of speech variation, not statistical variance.

## Engineering terms

**Artifact fingerprint** is a SHA-256 digest of compatibility-relevant audio, acoustic-model, and vocoder
configuration. It deliberately excludes serving paths and training schedules.

**Checkpoint** is resumable training state: weights plus optimizer, scheduler, scaler, epoch, and global
step. A model bundle is a separately exported inference artifact.

**Fixture alignment** is the uniform development backend. It proves schemas and tensor flows but does
not estimate speech timing and must not be presented as a production alignment method.

**Manifest** is a JSON Lines index of audio, transcript, speaker, language, and metadata.

**Model bundle** is an inference directory containing acoustic weights, vocoder weights, vocabulary,
and a versioned integrity manifest.

**Readiness** means the process has loaded and verified a compatible model. **Liveness** means the process
can answer at all. They intentionally have different endpoints.

**Safe loading** in this repository means integrity checking and `torch.load(weights_only=True)`. It
reduces but does not eliminate the risks of accepting untrusted model files.

