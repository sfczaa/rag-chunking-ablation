# Stage 12: one extra CE epoch vs the Stage 8 reranker

Status: pre-registered on 2026-09-15, run on 2026-09-16. Verdict: TIE
(CE - Stage 8 = +0.0019 R@1, 95% CI [-0.0120, +0.0159], n = 1032). Everything
above the Results section was written before any GPU run for this stage.

Question: on the Stage 6 bench, does the reranker trained for one more epoch of
listwise cross-entropy on the Stage 11 groups rank better than the Stage 8
weights it started from?

This stage does not use RL and does not rerun Stage 11. Stage 11 ended with
NO-GO at its dev gate, and that result stays as it is. This stage does not write
any Stage 11 verdict file.

## Background

The Stage 11 dev gate scored three rerankers on the Stage 8 dev bench (406
questions, fixed 15/0). The gate compared RL with CE. A side comparison in the
same run showed the CE arm ahead of the Stage 8 weights it started from:

| Dev comparison | R@1, 95% CI |
| --- | --- |
| CE - Stage 8 | +0.0222 [+0.0001, +0.0442] |

The dev run was not designed for this comparison, so it is treated as a
hypothesis. Stage 12 tests it once on the Stage 6 bench: NQ validation, 1000
documents, 1032 questions. Both models were trained only on NQ train documents.
The Stage 11 CE weights have not been scored on this bench before.

## Arms

No training. For each config, all arms rerank the same BGE top-20 pool.

| Arm | Weights | Role |
| --- | --- | --- |
| `bge` | dense BGE order | pipeline check |
| `rerank20` | `BAAI/bge-reranker-base`, off the shelf | pipeline check |
| `rerank20_ft` | `models/bge_reranker_ft/final` | reference: Stage 8 |
| `rerank20_s11_ce` | `models/bge_reranker_stage11/ce` | candidate: Stage 8 plus one CE epoch |

Stage 8 weights. The weights that produced the archived `stage8/final` numbers
were lost. The weights on Drive were retrained on 2026-07-25 with the Stage 8
recipe. The CE arm started from these weights, so CE minus Stage 8 is the effect
of the extra epoch. The Stage 8 row is compared with its archived value and the
difference is reported. It is not used as a validity check.

Starting point check. The CE arm's `training_meta.json` records an identity for
its starting weights: a hash of `config.json` plus the size of the weight file.
The script stops before scoring if this does not match the current Stage 8
directory. The check only catches a wrong directory; a retrain with the same file
size would pass it. The Stage 8 weight file was last written on 2026-07-25, and
the CE arm was trained on 2026-09-15.

CE arm training: 495 steps over 1977 groups (groups file SHA-1
`4f67bb1fd344b30915f30403ddeea1a064b064b6`), learning rate 2e-5 with 10% warmup,
seed 11, 7.9 minutes on a T4.

## Pre-registered criteria

1. Pipeline validity. At fixed 15/0, and at fixed 6/0 if it is run, the `bge` and
   `rerank20` rows must match
   `artifacts/results/stage8/final/stage8_ft_eval_results.csv` within 0.005 on
   R@1, R@3 and R@5, with the same chunk count and the same 1032 questions. If any
   of this fails, the verdict is `INVALID`.
2. Primary comparison. Stage 6 bench, fixed 15/0, R@1, `rerank20_s11_ce` minus
   `rerank20_ft`, paired over the 1032 questions. The estimate is the mean
   per-question hit difference, with a 95% interval of mean +/- 1.96 standard
   errors (the estimator Stage 11 used).
   - `CE-BETTER`: lower bound above 0 and mean at least 0.02.
   - `STAGE8-BETTER`: upper bound below 0.
   - `TIE`: everything else. This includes an interval above 0 with a mean below
     0.02; that case is reported with its numbers, and the verdict is still `TIE`.

   The verdict uses unrounded values. Tables show four decimals. The 0.02
   threshold is the one used by the Stage 8 gate and the Stage 11 claim. It was
   set before the dev result existed and is not adjusted because dev gave +0.022.
3. Also reported, outside the verdict: R@3 and R@5 for every arm, the paired R@5
   difference, and the gap between the Stage 8 row and its archived value. Fixed
   6/0 is optional and is reported the same way. It does not affect the verdict.
4. One run. Questions, configs, training and seeds are not changed to get a
   different verdict. If the run crashes, it resumes from the checkpoint and
   finished configs are not scored again. That counts as the same run.
5. Scope. Both models were trained on NQ and the bench is NQ, so the result
   applies to NQ only.

## Expected outcome (computed before the run)

Only the dev numbers are used here. The dev interval implies a paired standard
error of about 0.0112 at n = 406. If the two rerankers disagree at the same rate
on the Stage 6 bench, the standard error at n = 1032 is about 0.0070 and the
interval half-width is about 0.014. Any mean of 0.02 or more then has a lower
bound above 0, so in practice the 0.02 threshold decides `CE-BETTER`. If the true
gain were exactly the dev estimate of +0.022, the observed mean would reach 0.02
about 62% of the time. A difference picked out of a larger comparison also tends
to be smaller on new data. A `TIE` is therefore a likely result even if the extra
epoch helps slightly, and it would not show that the epoch has no effect.

## How to run

```bash
python scripts/29_eval_ce_epoch.py                       # the claim, fixed 15/0
python scripts/29_eval_ce_epoch.py --configs 15:0,6:0    # optional secondary
```

Or run all cells of `notebooks/RAG_chunk_optimize_stage12_colab.ipynb` on a GPU
runtime. Estimated cost on a T4, based on the reranking times in `stage8/final`:

| Step | Estimate |
| --- | --- |
| Fixed 15/0: 3 rerankers at 9.7 minutes each, plus the dense index and model loading | about 45 minutes |
| Optional fixed 6/0: 3 rerankers at 5.7 minutes each, plus the dense index | about 30 minutes |

If the runtime is lost, only the config in progress is lost. Each finished config
is written to the checkpoint and skipped on the next run.

## Outputs

Written to `artifacts/results/latest/`. They are not archived automatically,
because that folder also holds working files from other stages.

- `stage12_checkpoint_final.jsonl`: per-config rows with per-question hits
- `stage12_eval_results.csv`: R@1, R@3, R@5 and pool recall for every arm
- `stage12_paired_deltas.csv`: CE minus Stage 8, R@1 and R@5, paired 95% CI
- `stage12_check_vs_stage8.csv`: the reproduction check
- `stage12_summary.md` and `stage12_verdict.json`

## Results

### Run 1 (2026-09-16): TIE

Run on a Colab T4 from `notebooks/RAG_chunk_optimize_stage12_colab.ipynb`, fixed
15/0 only.

Pipeline validity: PASS. The `bge` and `rerank20` rows reproduce `stage8/final`
exactly (every recall delta 0.0000), with the same chunk count (19,507) and the
same 1032 questions. Pool recall@20 was 0.9641.

| Arm | R@1 | R@3 | R@5 |
| --- | --- | --- | --- |
| dense BGE | 0.6279 | 0.8159 | 0.8808 |
| off-the-shelf rerank20 | 0.6289 | 0.8159 | 0.8760 |
| Stage 8 weights | 0.7345 | 0.8750 | 0.9186 |
| + one CE epoch | 0.7364 | 0.8808 | 0.9254 |

| Paired comparison | mean | 95% CI |
| --- | --- | --- |
| CE - Stage 8, R@1 | +0.0019 | [-0.0120, +0.0159] |
| CE - Stage 8, R@5 | +0.0068 | [-0.0023, +0.0159] |

By criterion 2 the verdict is TIE: the interval includes 0 and the mean is far
below the 0.02 threshold.

The Stage 8 row scores 0.7345 against its archived 0.7355 at R@1 (-0.0010) and
0.9186 against 0.9215 at R@5 (-0.0029). Those weights were retrained, so this gap
is reported and does not affect validity.

Reading. The dev estimate of +0.0222 did not survive the Stage 6 bench: the same
comparison over 1032 questions gives +0.0019, under a tenth of it and well inside
noise. The pre-registered expectation covered this case, since a difference picked
out of a larger comparison tends to shrink on new data and the dev interval's
lower bound was only +0.0001. The R@5 estimate is also uncertain: it moves
+0.0068 with an interval that includes 0, and R@3 is +0.0058 on the arm table. One
more CE epoch on 1977 fresh groups showed no measurable gain at the
configuration the project deploys.

On this NQ bench, the extra CE epoch suggested by the Stage 11 dev gate did not
beat the Stage 8 weights by the project's 0.02 threshold. This applies to one
epoch over 1977 groups mined with the Stage 8 rule; it does not establish
whether more training helps under other conditions.

Cost: 555 s per reranker for 20,640 pairs (9.3 minutes each), three rerankers plus
the dense index.
