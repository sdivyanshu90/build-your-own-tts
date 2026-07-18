# Data card: speech corpus manifest and derived features

## Card status and purpose

This is the repository’s corpus-card template. It documents the expected schema and review questions but
does not describe a bundled real-speaker dataset. Every training corpus or merged version must have its
own completed card linked to an immutable manifest fingerprint.

## Data unit and schema

One JSONL record represents one audio utterance and transcript:

```json
{
  "audio_path": "wavs/000001.wav",
  "transcript": "What was actually spoken.",
  "normalized_transcript": null,
  "speaker_id": "speaker_001",
  "language": "en-US",
  "duration": 2.31,
  "metadata": {
    "session_id": "session_04",
    "source": "consented-studio-collection-v2",
    "consent_record": "controlled-reference"
  }
}
```

Audio paths resolve relative to manifest. `speaker_id` should be a pseudonym, not a public legal name.
Sensitive identity/consent evidence belongs in a controlled system linked by opaque reference.

## Required concrete-corpus description

A completed card must state:

- why data was collected and intended model/product use;
- who funded, collected, transcribed, cleaned, and approved it;
- acquisition date/location/method and recording equipment/environment;
- source license/contract and whether redistribution/derivatives are permitted;
- speaker count, utterance count, total hours, duration distribution, and file formats/rates;
- languages, dialects, accents, styles, domains, and code-switching;
- demographic composition where collection/reporting is lawful, consented, necessary, and privacy-safe;
- transcript conventions, quality review, disagreement/error estimates;
- known sensitive content and handling;
- access, retention, deletion, incident, and revocation process; and
- changes from previous corpus version.

## Consent and provenance

For each real speaker, preserve verified authority and explicit informed consent covering model training
and synthetic output purpose, audience, duration, territory, languages, redistribution, disclosure,
adaptation, and revocation. “Found online,” “public,” or a general recording license may not grant rights
to build a recognizable synthetic voice. Obtain legal/ethical review appropriate to context.

Provenance must survive derivation. Source record identity must map to feature cache, alignment,
checkpoint, and released bundles so revocation can be executed and evidenced.

Repository fixtures are deterministic sine waves labelled synthetic and contain no person’s voice.

## Collection and preprocessing

A concrete card must specify channel/bit depth/rate, mono conversion, resampling, amplitude checks,
silence trim/normalization, segmentation, exclusions, mel parameters, normalizer/phonemizer/vocabulary,
forced aligner/model/dictionary, F0 estimator/unvoiced policy, energy definition, and every artifact
version/fingerprint.

The reference preprocessing’s uniform fixture alignment and zero pitch are for software tests only and
must not be recorded as a valid production method.

## Validation and quality report

Report missing/corrupt/duplicate files, empty/invalid transcripts, duration/sample-rate/channel warnings,
speaker/language/session distribution, total accepted/rejected duration, amplitude/clipping/silence,
language/transcript agreement, alignment confidence/failure, OOV symbols, and quarantine reasons.

Include counts before and after every filter. Inspect whether filtering disproportionately removes a
speaker/language/demographic slice. Listen to stratified samples and document reviewer protocol.

## Splitting and leakage

State exact train/validation/test manifests, seed, fractions/counts, and grouping. The reference helper is
speaker-stratified by record identity. Production should normally group by recording session/source to
avoid near-duplicate acoustic leakage. Explain whether speakers overlap splits and what evaluation claim
that supports. Never use test data for early stopping, vocabulary/rule tuning, or subjective candidate
selection.

## Distribution and bias

Describe imbalances in hours, utterance length, speaker, language/accent, recording condition, speaking
style, age/gender where lawful, and text domain. Explain likely model behavior: dominant speakers may have
better quality; unseen names/phonemes/domains may fail; studio-only audio may not generalize to noisy
adaptation data. Avoid claiming population representativeness without sampling evidence.

## Privacy and sensitive data

Speech can reveal identity, health, location, emotion, background speakers, and environment. Transcripts
can contain personal data. Minimize collection, redact/remove unintended third-party speech, restrict
raw access, encrypt, audit, set retention, and establish data-subject contact. Do not place private
transcripts/audio in test fixtures, Git, experiment dashboards, or public issue reports.

## Public dataset note

LJSpeech is referenced in documentation only as a familiar manifest conversion example. Operators must
review current source, license, speaker expectations, and suitability at download/use time. The repository
does not endorse every public dataset for voice-product deployment.

## Maintenance and revocation

Assign a data owner and review date. New/changed files require a new immutable version and report, not an
in-place edit under the same fingerprint. On correction/revocation, identify all derived features,
alignments, runs, checkpoints, and bundles; block active access; delete/retain according to consent,
contract, law, and documented backup policy; record completion.

## Concrete corpus appendix template

```text
Corpus name/version/fingerprint:
Owner/contact/review date:
Purpose and prohibited use:
Sources/licenses/consent scope:
Speakers/languages/hours/utterances:
Recording/transcription method:
Validation and rejection report:
Split manifests/grouping/seed:
Preprocessing/alignment/F0 versions:
Known bias, gaps, and sensitive content:
Access/retention/revocation process:
Derived approved model versions:
Reviewers and approval date:
```

