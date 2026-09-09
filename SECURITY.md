# Security

Do not post credentials, private data, or exploitable details in public issues.
Use GitHub's private vulnerability reporting when it is available.

Install dependencies from the repository requirements or lockfile in an
isolated environment. Keep credentials in environment variables or the hosting
provider's secret store. Never commit local data, access tokens, or credentials.

NLTK is used for sentence tokenization. GHSA-8mgp-746c-j5xp has no published
fix in the audit database as of 2026-09-09. It concerns model import/export
APIs with caller-controlled paths; this project does not call those APIs or
expose model paths to demo users. Do not add such endpoints without a separate
containment review. A dependency scan can still report this advisory.

Dataset repository code stays disabled unless the operator explicitly opts in
with RAG_TRUST_REMOTE_CODE. Only use trusted datasets and model artifacts.
