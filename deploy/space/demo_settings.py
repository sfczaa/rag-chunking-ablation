"""Pinned demo artifacts and request validation."""

import os
from pathlib import Path
import re

MAX_QUERY_CHARS = 2000
ARTIFACTS = {
    "assets": ("sfczaa/rag-chunking-ablation-demo-assets",
               "9aff2c5ef8cc04413c6837229d07908fdb0780b8"),
    "embedding": ("BAAI/bge-base-en-v1.5",
                  "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"),
    "reranker": ("BAAI/bge-reranker-base",
                 "2cfc18c9415c912f9d8155881c133215df768a70"),
    "finetuned": ("sfczaa/bge-reranker-base-nq-ft",
                  "962cdfa7380f19cdf37db8f242d14e610de8ba71"),
}


def artifact_location(name, repo_env=None, revision_env=None):
    default_repo, default_revision = ARTIFACTS[name]
    repo = os.environ.get(repo_env, default_repo) if repo_env else default_repo
    # The local smoke runner supplies a staged fine-tuned model directory.
    if name == "finetuned" and Path(repo).is_dir():
        return repo, None
    revision = os.environ.get(revision_env) if revision_env else None
    if repo != default_repo and not revision:
        raise ValueError(f"{revision_env} is required for a custom {repo_env}.")
    revision = revision or default_revision
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Artifact revisions must be full commit SHAs.")
    return repo, revision


def validate_query(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        raise ValueError("Question must be text.")
    if len(text) > MAX_QUERY_CHARS:
        raise ValueError(f"Question must be {MAX_QUERY_CHARS:,} characters or fewer.")
    return text.strip()
