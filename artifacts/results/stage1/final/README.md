# results/stage1/final/

Archived snapshot of the **Stage 1** chunking sweep optimizer.

## Archive status (recorded 2026-07-06): original artifacts lost

The Stage 1 run's CSV/JSON/plots were **never archived here** — `results/latest/`
was overwritten by later sweeps before `save_stage_results.py --stage stage1`
was run. A recovery search on 2026-07-06 found no copy locally, in the root
`results/` Drive downloads, or anywhere on Drive (no `stage1` results folder
exists there). The original files are unrecoverable.

**The numbers themselves are not lost.** The pipeline is deterministic (frozen
MiniLM embeddings, fixed model weights, exact FAISS search), and Stage 2 re-ran
the identical fixed + BiLSTM grid unchanged while adding the Transformer arm.
The `method in {fixed, bilstm}` rows (20 of 30) of
[`../../stage2/final/sweep_results.csv`](../../stage2/final/sweep_results.csv)
match the Stage 1 headline numbers recorded in `README.md`, `PROGRESS.md` and
`日誌.docx` exactly (e.g. fixed 15/1 → R@5 0.8571; bilstm target 15/0 →
R@5 0.8670). Use that subset as the Stage 1 record; no reconstructed files are
placed here to avoid passing a derivation off as the original archive.

Populate it by running, after a sweep:

```bash
python scripts/save_stage_results.py --stage stage1
```

That copies everything from `results/latest/` here (copy only — `latest/`,
`runs/`, and `data/` are never moved or deleted). `latest/` keeps being
overwritten by new runs; this folder is the kept snapshot.

See [`docs/stage1_chunking_optimizer.md`](../../../../docs/stage1_chunking_optimizer.md).
