# Security policy

## Supported versions

The latest release on the default branch receives security fixes. Older commits, locally modified
forks, third-party models, and deployment configurations are not guaranteed support. A future stable
release program should publish an explicit version/end-of-support table.

## Reporting a vulnerability

Report suspected vulnerabilities privately to repository maintainers through the repository host’s
private security-advisory mechanism. If unavailable, contact the organization security owner through a
non-public channel. Do not open a public issue containing exploit details, credentials, private text,
speaker data, model artifacts, or a working denial-of-service payload.

Include:

- affected revision/version and deployment topology;
- component and prerequisite access;
- reproducible steps using synthetic/non-sensitive data;
- impact to confidentiality, integrity, availability, authorization, or voice safety;
- whether exploitation is active or data may be exposed; and
- suggested mitigation if known.

Maintainers should acknowledge receipt, establish a private coordinator, assess severity/scope, develop
and test a fix, notify affected operators when necessary, publish a sanitized advisory, and credit the
reporter if desired. Timelines depend on severity and coordinated disclosure requirements.

## Security scope

In scope includes input validation, authentication/rate/concurrency bypass, path traversal, unsafe
archive/model loading, artifact integrity, secret/text/audio leakage, cross-tenant speaker access,
container/deployment defaults, dependency supply chain, and abuse-control bypass with concrete technical
impact.

General model quality, expected noise from random weights, feature requests, and policy disagreements
without a security bypass belong in ordinary issues. Non-consensual voice misuse reports should still be
handled urgently through the responsible-use/takedown process even when no software vulnerability exists.

## Operator baseline

Set API secrets through a secret manager; terminate TLS at a trusted gateway; enforce authenticated
tenant and speaker authorization; apply distributed rate/output limits; mount approved model bundles
read-only; separate writable output storage; run non-root/read-only containers; restrict network egress;
protect metrics; scan dependencies/images/bundles; and minimize text/audio retention.

The implementation bounds schema/text/control/duration/concurrency, generates storage names, uses
constant-time shared-key comparison, emits privacy-oriented JSON logs, verifies bundle/checkpoint hashes,
and calls `torch.load(weights_only=True)`. These reduce risk but do not replace a trusted artifact source,
signature/provenance, identity provider, distributed control plane, or privacy program.

## PyTorch and model-file warning

PyTorch serialization is a broad and evolving format. `weights_only=True` constrains object construction
but should not be treated as a sandbox. Accept artifacts only from trusted builders, verify signed
provenance and expected digest, scan them, enforce size, and load in a restricted environment. Prefer a
reviewed tensor-only format when importing third-party weights.

## Full threat model

See [docs/security.md](docs/security.md) for assets, trust boundaries, threats, residual risks, and the
security verification matrix. See [docs/responsible-use.md](docs/responsible-use.md) for consent,
impersonation, disclosure, abuse monitoring, and revocation.

