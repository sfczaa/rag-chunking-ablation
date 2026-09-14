# Security

Do not post credentials, private data, or exploitable details in public issues.
Use GitHub's private vulnerability reporting when it is available.

Install dependencies from the repository requirements in an
isolated environment. Keep credentials in environment variables or the hosting
provider's secret store. Never commit local data, access tokens, or credentials.

Corpus preparation uses an isolated copy of the Apache-2.0 Punkt inference
code, not the NLTK package. Training, pickle loading and model import/export
APIs are excluded. English parameters come from a fixed official NLTK data
revision, with the archive SHA-256 checked before parsing or caching; no
archive paths are extracted to disk. See
[`rag_chunk/_vendor/README.md`](rag_chunk/_vendor/README.md) for provenance.

This removes the NLTK package dependency associated with
[GHSA-8mgp-746c-j5xp](https://github.com/nltk/nltk/security/advisories/GHSA-8mgp-746c-j5xp).
The upstream package advisory still has no published fix as of 2026-09-14.
Existing environments may retain an earlier NLTK installation; use a fresh
environment for the dependency set declared here. The inference copy does
not receive automatic Dependabot updates: review upstream Punkt security
changes when updating dependencies, then rerun sentence-boundary regression
checks before replacing its code or parameter archive. Run
`python scripts/check_sentence_tokenizer.py` for the real English-table cases;
the first uncached run requires network access.

Dataset repository code stays disabled unless the operator explicitly opts in
with RAG_TRUST_REMOTE_CODE. Only use trusted datasets and model artifacts.
