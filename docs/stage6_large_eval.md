# Stage 6 - Larger-Scale Robustness Evaluation

**Status: complete.** Run on Colab (T4) 2026-07-05/06: check mode reproduced
the archived Stage 5 rows exactly (35/35, all deltas 0.0000), the large eval
ran on **1000 docs / 1032 questions**, and **all 5 direction checks replicate**.
Archived to `artifacts/results/stage6/final/`. Results below.

Stage 6 asks one question: **do the Stage 1-5
conclusions replicate at ~5x the evaluation scale?** Every stage so far ran on
200 NQ docs / 203 questions, where one standard error is ~0.034 Recall — large
enough that several key conclusions had to be phrased "within noise". At ~1000
questions the SE halves, so the directions either firm up or get overturned.

## Scope

Changed (the *only* change):

- `N_NQ_DOCS`: 200 → `N_NQ_DOCS_LARGE` (~1000). The NQ stream stops at the
  N-th usable document, so the **actual** docs/questions counts are reported
  in every output, not assumed. The large corpus caches under a separate
  `nq/large_n<N>/` folder — the 200-doc cache used by Stages 1-5 (and by the
  Stage 6 check mode) is never overwritten.

Held fixed (identical to Stages 3-5):

- Dataset source (NQ validation stream), chunking logic, fixed / BiLSTM /
  Transformer grids: sizes `{6,8,10,12,15}` × overlap `{0,1}` = 30 configs.
- Boundary/chunking embeddings (`all-MiniLM-L6-v2`) and the existing Stage 2
  BiLSTM/Transformer weights, loaded as-is.
- Dense retrieval: `BAAI/bge-base-en-v1.5` with the BGE query instruction.
- Reranker: `BAAI/bge-reranker-base`, off-the-shelf, **top-20 only** — no
  rerank50 (Stage 5 showed the deeper pool strictly hurts), no fine-tuning.

Not included: BM25/RRF, RL, BGE-M3, new chunking objectives/models, reranker
training.

## Arms

- **`bge`** on the **full 30-config grid** — this is the bulk of the run and
  what direction claims 1-2 are tested on.
- **`rerank20`** only on `STAGE6_RERANK_CONFIGS` (5 configs: fixed 6/0,
  fixed 15/0, fixed 15/1, bilstm 15/0, transformer 15/0) — the small-chunk
  config plus the size-15 sweet-spot configs, which is exactly what direction
  claims 3-4 need. Reranking all 30 configs at n≈1000 would add hours of GPU
  time without changing what the claims test.

## Restart / resume safety

A large run takes hours on the Colab free tier, so the script checkpoints
per config: every finished config appends one JSON line to
`results/latest/stage6_checkpoint_<mode>.jsonl`. Re-running the same command
resumes — finished configs are skipped, and boundary probabilities are only
recomputed for model types that still have pending configs. `--fresh`
discards the checkpoint. The checkpoint records the run fingerprint (corpus
counts + models); a mismatched resume aborts instead of mixing rows.

The `results/latest/` cleanup runs only after all prechecks, model weights,
the reranker and the corpus have loaded; it never deletes a file without a
byte-identical copy in a stage archive, and it never touches `stage6_*`
working files. Stage 3/4/5 finals are read-only.

## Run

```bash
# 1. sanity mode first: N=200 on the same cached corpus as Stages 3/5;
#    every row must reproduce the archived Stage 5 rows exactly.
python scripts/12_large_eval.py --check

# 2. the large eval (~1000 docs; resume-safe — re-run the same command
#    after an interruption).
python scripts/12_large_eval.py

# 3. figures (small vs large eval side by side); writes 2 PNGs into latest/
#    so they get archived together with the CSVs.
python scripts/13_stage6_plots.py
```

The script refuses to start if `results/stage5/final/` is missing or was
built with different retrieval/reranker models.

## Validation

- **Check mode** compares all 35 rows (30 bge + 5 rerank20) against the
  archived Stage 5 rows (`stage6_check_vs_stage5.csv`): every
  `delta_n_chunks` must be 0 and every recall delta 0.0000 (the console
  prints a verdict). This proves the Stage 6 code path is the Stage 3/5
  pipeline before any conclusion is drawn from it.
- **Large mode** cannot check exact deltas — the eval set itself changes —
  so `stage6_direction_check.csv` re-tests the four small-eval direction
  claims with explicit rules (2 SE uses the actual question count):

| # | Claim | Rule |
|---|---|---|
| 1a | size > method | Pearson r(avg chunk size, R@5) ≥ 0.5 over the 30 bge configs |
| 1b | size > method | R@5 size effect (largest vs smallest size) > mean method spread at matched (size, overlap) |
| 2 | BGE-only remains strong | best bge R@5 ≥ 0.80 (loose heuristic — the corpus is ~5× larger, so some absolute drop vs 0.921 is expected) |
| 3 | rerank20 helps mainly at small chunks | ΔR@1 at fixed size 6 ≥ 2 SE **and** > mean ΔR@1 at the size-15 configs |
| 4 | rerank20 does not clearly improve the sweet spot | max ΔR@1 over the size-15 configs < 2 SE |

The CSV reports the observed value, the rule and the small-eval reference for
every claim, so borderline verdicts can be judged by a human rather than
trusted blindly. A claim failing to replicate is a *finding*, not an error.

## Outputs

Written to `artifacts/results/latest/`:

- `stage6_large_eval_results.csv` — one row per (arm × config): 30 bge +
  5 rerank20 = 35 rows, each with `n_docs` / `n_questions` (the actual
  loaded counts), `pool_recall@20` and `rerank_seconds` on rerank rows.
- `stage6_matched_summary.csv` — the 5 selected configs with bge / rerank20
  side by side and `rerank20_minus_bge@k` deltas.
- `stage6_direction_check.csv` — the table above with observed values and
  yes/no verdicts.
- `stage6_large_eval_summary.md` — run metadata (actual corpus counts), best
  config, deltas and direction verdicts.
- `stage6_check_vs_stage5.csv` — from the check-mode run.
- `stage6_size_vs_recall.png` — bge-only R@5 vs avg chunk size, small vs large
  eval side by side (claim 1 in one glance; `scripts/13_stage6_plots.py`).
- `stage6_rerank_delta.png` — rerank20−bge ΔR@1 on the 5 matched configs at
  both scales, with the ±2 SE band (claims 3–4).
- `stage6_checkpoint_{check,large}.jsonl` — resume checkpoints (kept for
  provenance).

After the check passes and the large run finishes, archive with
`python scripts/save_stage_results.py --stage stage6`.

## Results (Colab run, 2026-07-05/06)

Eval set: 1000 docs / 1032 questions (requested ~1000). 1 SE ≈ 0.015 at the
mean bge R@1, so 2 SE = 0.031 (vs 0.067 at n=203).

Direction checks — **5/5 replicate**:

| # | Claim | Observed at n=1032 | Rule | n=203 reference |
|---|---|---|---|---|
| 1a | size > method | Pearson r(size, R@5) = **0.9515** | ≥ 0.5 | r ≈ 0.77 |
| 1b | size > method | size effect on R@5 = **0.0523** > method spread | size > spread | 0.07 vs ≤ 0.014 |
| 2 | BGE-only strong | best bge R@5 = **0.8808** | ≥ 0.80 | 0.921 |
| 3 | rerank20 helps small chunks | ΔR@1 @ fixed 6/0 = **+0.0359** | ≥ 2 SE and > size-15 mean | +0.113 |
| 4 | no gain at size-15 | max size-15 ΔR@1 = **+0.0010** | < 2 SE | within 1 SE |

Best bge config (unchanged from Stage 3/5): **fixed size 15, overlap 0** —
R@1 0.6279 / R@3 0.8159 / R@5 0.8808 (avg chunk size 14.6, 19 507 chunks).
The absolute drop from 0.921 at 200 docs is expected: 5× more distractor
documents in the index.

rerank20 − bge on the 5 selected configs (ΔR@1 / ΔR@3 / ΔR@5):

| Config | pool@20 | ΔR@1 | ΔR@3 | ΔR@5 |
|---|---|---|---|---|
| fixed 6/0 | 0.9264 | **+0.0359** | +0.0174 | +0.0203 |
| fixed 15/0 | 0.9641 | +0.0010 | 0.0000 | −0.0048 |
| bilstm t15/0 | 0.9632 | −0.0029 | +0.0203 | +0.0116 |
| transformer t15/0 | 0.9574 | −0.0116 | +0.0039 | +0.0097 |
| fixed 15/1 | 0.9593 | +0.0010 | +0.0174 | +0.0194 |

Findings:

1. **The size effect gets cleaner with scale** (r 0.77 → 0.95): it was never
   noise, and the method colours stay intermixed along the trend
   (`stage6_size_vs_recall.png`).
2. **The reranker's small-chunk rescue shrinks at scale** (+0.113 → +0.036 R@1
   at fixed 6/0) but stays above 2 SE — real, just smaller when the pool
   ceiling is lower (0.926 vs 0.951 at n=203).
3. **Size-15 deltas are zero** (all within ±0.012 vs 2 SE = 0.031), while the
   size-15 pool ceilings remain ≥ 0.95 — the answer chunk is in the top-20 for
   ~96% of questions but the off-the-shelf cross-encoder cannot rank it first
   any better than BGE already does. That gap (R@1 0.63 vs pool 0.96) is the
   one place left where a *fine-tuned* reranker could plausibly pay off — noted
   as the conditional trigger for a possible later stage, not done here.
4. Cost at n=1032: ~600–730 s per rerank20 config (20 640 pairs), vs ~100 s at
   n=203 — reranking cost scales linearly with questions, which is exactly why
   the rerank arm was restricted to 5 configs.
