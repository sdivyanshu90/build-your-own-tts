# Contributing guide

## Before starting

Use an issue or design discussion for architectural changes, public schemas, artifact formats, model
topology, data/voice additions, external services, or security/responsible-use behavior. Small bug,
documentation, and test fixes can proceed directly. Never attach private recordings, transcripts,
credentials, or unapproved model files.

By contributing, you agree to the [code of conduct](CODE_OF_CONDUCT.md), responsible-use policy, and
project license requirements.

## Development environment

Use Python 3.11+ and system libsndfile; espeak-ng is optional unless testing explicit phonemization.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pre-commit install
```

Run before submitting:

```bash
ruff format --check .
ruff check .
mypy
pytest --cov=tts_pipeline
```

Use the synthetic fixture for end-to-end changes. Do not download a real corpus during tests.

## Change design expectations

Maintain package boundaries described in [architecture](docs/architecture.md). Prefer typed, injected
interfaces over global state. Validate assumptions at the earliest boundary and use domain-specific
errors with context. Preserve unrelated user changes. Avoid runtime downloads in serving and untrusted
pickle loading.

If changing configuration, document default/range/compatibility and add invariant tests. If changing
tensor shapes, update model documentation, shape tests, checkpoint/bundle compatibility, and migration.
If changing normalization, add table and regression cases for ordering/boundaries. If changing DSP,
regenerate feature compatibility and numerical tests.

## Testing expectations

- Unit tests cover pure rules, shapes, masks, numerical finiteness, and error cases.
- Integration tests cover artifact export/load, filesystem contracts, and HTTP behavior.
- E2E tests traverse text through WAV using synthetic/random smoke fixtures.
- Regression tests freeze intentional semantics with documented update rationale.
- Performance tests state hardware, warm-up, input sizes, and thresholds; avoid flaky wall-clock gates on
  shared CI.

A bug fix should include a failing-before/passing-after test. Avoid assertions so broad that incorrect
audio or security behavior passes.

## Documentation expectations

Update the handbook when changing public commands/endpoints/schemas/configuration, tensor contracts,
artifact formats, failure modes, security boundaries, or operational procedures. Code docstrings explain
local responsibility; handbook chapters explain system rationale and workflows. Ensure examples match
actual CLI help and clearly label fixture versus production behavior.

Architectural decisions require a new numbered ADR with context, options, decision, consequences,
compatibility, and revisit criteria. Do not rewrite historical accepted ADRs to hide past reasoning;
supersede them.

## Data, voice, and model contributions

Any real voice/data/model contribution requires source/license, explicit consent scope, provenance,
dataset/model card, access/retention/revocation, dependency/weight license, evaluation, and responsible-
use review. Public availability is insufficient. Maintainers may reject technically valid contributions
whose rights, consent, safety, or lineage are unclear.

Fixtures must be synthetic, public-domain, licensed for repository redistribution, or explicitly
consented for this exact purpose. Prefer deterministic generated signals and invented text.

## Security-sensitive contributions

Privately report exploitable findings rather than first submitting a public test. Changes to auth,
authorization, paths, archives, deserialization, secrets, logs, storage, model sources, or rate/
concurrency controls need negative tests and threat-model updates. Avoid logging public request objects.

## Pull request checklist

- Scope and rationale are clear; major tradeoffs are documented.
- Code is typed/formatted/linted and imports are consistent.
- Tests cover success, invalid input, boundaries, and regression.
- Documentation/cards/ADR/changelog are updated as applicable.
- No secret/private data/weights/generated artifacts are committed.
- Dependency additions justify need, license, optionality, pinning, and security impact.
- Backward compatibility and migration/rollback are explained.
- Voice/data rights and responsible-use review are complete where relevant.

## Review and release

Reviewers evaluate correctness, maintainability, numerical behavior, compatibility, security/privacy,
responsible use, tests, and docs. Approval does not automatically promote a model/data artifact. Release
owners separately perform evaluation, artifact signing/versioning, cards, deployment canary, and rollback.

