"""Guards for a run that several Colab accounts continue in one shared Drive folder.

- Root identity: the shared folder holds RUN_ROOT_ID.json. A missing or different
  id means the mount is not the shared project folder (for example a private folder
  with the same name), so the run stops without creating anything.
- Write probe: write, read back, replace and delete a file in each directory the
  run writes to, from the process that will do the writing.
- Run lock: one runtime at a time. A lock left by a runtime that died is never
  removed automatically; clearing it takes an explicit flag.
- Run identity: code hash, data hash and recipe are recorded on the first run and
  compared on every later one, so a resume never mixes two different runs.

Drive propagates changes between runtimes with a delay, so the lock guards against
starting a second runtime minutes after the first, not against two starts in the
same instant.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import socket
import time
import uuid

ROOT_ID_FILE = "RUN_ROOT_ID.json"


def check_root_identity(data_root, expected_id: str) -> str:
    path = pathlib.Path(data_root) / ROOT_ID_FILE
    if not path.exists():
        raise SystemExit(f"[run-guard] no {ROOT_ID_FILE} under {data_root}: this mount is "
                         "not the shared project folder. Check that the shared folder is "
                         "added to My Drive under the expected name. Nothing was created.")
    found = json.loads(path.read_text(encoding="utf-8")).get("root_id")
    if found != expected_id:
        raise SystemExit(f"[run-guard] {ROOT_ID_FILE} holds {found}, expected {expected_id}: "
                         "this is a different folder with the same name. Nothing was written.")
    return found


def probe_directory(path) -> None:
    """Write, read back, replace and delete a file in ``path``."""
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    tmp, final = path / f".probe_{token}.tmp", path / f".probe_{token}.json"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"token": token}))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    if json.loads(final.read_text(encoding="utf-8")).get("token") != token:
        raise SystemExit(f"[run-guard] {path}: the probe file did not read back")
    final.unlink()
    if final.exists() or tmp.exists():
        raise SystemExit(f"[run-guard] {path}: the probe file could not be deleted")


def file_sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_lock(lock_path):
    lock_path = pathlib.Path(lock_path)
    if not lock_path.exists():
        return None
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"unreadable": True}


def acquire_lock(lock_path, *, run_version: str, account: str, task: str,
                 clear_stale: bool = False) -> dict:
    lock_path = pathlib.Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    held = read_lock(lock_path)
    if held is not None:
        who = (f"account {held.get('account')} on {held.get('host')}, task "
               f"{held.get('task')}, since {held.get('started_at')}")
        if not clear_stale:
            raise SystemExit(f"[run-guard] {lock_path.name} is held by {who}. If that runtime "
                             "is stopped, rerun with --clear-stale-lock. The lock is never "
                             "removed automatically.")
        print(f"[run-guard] clearing a stale lock on request: {who}", flush=True)
        lock_path.unlink()
    record = {"token": uuid.uuid4().hex, "run_version": run_version, "account": account,
              "task": task, "host": socket.gethostname(), "pid": os.getpid(),
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(record, indent=2))
    if (read_lock(lock_path) or {}).get("token") != record["token"]:
        raise SystemExit("[run-guard] another runtime took the lock at the same moment")
    print(f"[run-guard] lock taken: account {account}, task {task}", flush=True)
    return record


def release_lock(lock_path, record: dict) -> None:
    lock_path = pathlib.Path(lock_path)
    held = read_lock(lock_path)
    if held is not None and held.get("token") == record["token"]:
        lock_path.unlink()
        print("[run-guard] lock released", flush=True)
    else:
        print("[run-guard] WARN: the lock is no longer ours; left in place", flush=True)


@contextlib.contextmanager
def held_lock(lock_path, **kwargs):
    record = acquire_lock(lock_path, **kwargs)
    try:
        yield record
    finally:
        release_lock(lock_path, record)


def check_run_identity(path, identity: dict) -> str:
    """Record ``identity`` on the first run; stop if a later run differs."""
    path = pathlib.Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
        return "recorded"
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded != identity:
        changed = sorted(k for k in set(recorded) | set(identity)
                         if recorded.get(k) != identity.get(k))
        raise SystemExit(f"[run-guard] run identity changed in {changed} ({path}). A resume "
                         "must use the same code, data and recipe; stopping.")
    return "matched"
