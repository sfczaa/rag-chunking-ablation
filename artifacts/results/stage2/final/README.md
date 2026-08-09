# results/stage2/final/

Reserved archive folder for **Stage 2** (Transformer boundary model) — **not
implemented yet**. Structure is prepared ahead of the work.

Once Stage 2 produces a sweep, populate it with:

```bash
python scripts/save_stage_results.py --stage stage2
```

That copies everything from `results/latest/` here (copy only). `latest/`,
`runs/`, and `data/` are never moved or deleted.

See [`docs/stage2_transformer_boundary.md`](../../../../docs/stage2_transformer_boundary.md).
