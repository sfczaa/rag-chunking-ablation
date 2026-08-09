# Stage 8 - Fine-tuning the cross-encoder reranker (Route C)

**Status: complete — executed on Colab 2026-07-08 and archived** (9 files in
`artifacts/results/stage8/final/`, read-only). Dev gate verdict: **GO**
(ΔR@1 +0.0468 ≥ +0.02). Final eval: built-in check vs `stage6/final` **OK
(exact)**; headline **ft − off-the-shelf ΔR@1 = +0.107 at fixed 15/0**
(2 SE = 0.030, n = 1032). Results below in "Results (executed)".

Stage 8 asks one question: **can a fine-tuned cross-encoder close part of the
gap between R@1 ≈ 0.63 and the pool ceiling ≈ 0.96 at the size-15 sweet
spot?** This is the only model change left with theoretical headroom.

## Trigger evidence (why this stage exists at all)

From the archived Stage 6 run (1000 docs / 1032 questions):

- At the size-15 configs, `pool_recall@20` = 0.957–0.964 — the answer chunk
  is in the BGE top-20 for ~96% of questions. The pool is not the bottleneck.
- Yet R@1 ≈ 0.63, and the **off-the-shelf** `BAAI/bge-reranker-base` adds
  ~nothing there (max size-15 ΔR@1 = +0.001). Ranking is the bottleneck.
- Stage 5's earlier verdict "203 questions is too small to justify
  fine-tuning" no longer binds: the eval bench is now 1032 questions and the
  NQ **train** split provides unlimited disjoint training data.

An honest possible outcome is "fine-tuning does not help either" — that is a
publishable negative result, and the go/no-go gate below is designed to reach
it cheaply.

## Training data (NQ train split — disjoint from every eval bench)

All evaluation benches (Stages 1–7) come from the NQ **validation** split (or
TriviaQA). Stage 8 trains on the NQ **train** split, so train/eval are
disjoint by construction. One streaming pass builds two things
(cached under `data/nq_train/`, never touching the eval caches):

- **Training corpus**: the first `STAGE8_N_TRAIN_DOCS` (default 2000) usable
  documents and their questions (~1 question/doc, same collection rule as
  `nq_data`).
- **Dev bench**: the **next** `STAGE8_N_DEV_DOCS` (default 400) documents and
  the questions whose gold doc lies in that window (questions pointing back
  into a training doc are dropped). The dev bench is therefore disjoint from
  the training docs *and* from the final eval — the Stage 6 bench is touched
  exactly once, at the end.

**Mining** (deployment-matched): chunk the training corpus with the
deployment config (**fixed size 15 / overlap 0**), embed with BGE, retrieve
each training question's top-20 pool — exactly what the reranker sees at eval
time. Then per question:

- **positive** = the highest-dense-ranked pool chunk that is answer-bearing
  *and* from the gold document (the metric's own hit rule);
- **hard negatives** = the top `STAGE8_NUM_NEGATIVES` (default 7) remaining
  pool chunks by dense rank — the exact distractors BGE currently ranks high;
- questions whose pool contains **no** positive are dropped (≈4% expected
  from the pool ceiling; reranking cannot rescue them at eval either, so
  training on them would optimize an unreachable case). Dropped counts are
  reported, not silent.

Groups are written to `data/nq_train/stage8_train_groups.jsonl` with
provenance (doc ids, dense ranks).

## Training

`scripts/17_train_reranker.py`, plain `transformers` + `torch` (no
sentence-transformers training API — its interface has churned across major
versions; inference still goes through `CrossEncoder`, which loads any HF
sequence-classification checkpoint directory):

- Base model `BAAI/bge-reranker-base`, listwise softmax cross-entropy over
  each (1 positive + 7 negatives) group — the standard reranker objective.
- `max_length` = `RERANK_MAX_LENGTH` (512), identical to eval-time truncation.
- Defaults: 2 epochs, lr 2e-5 with 10% linear warmup, weight decay 0.01,
  4 groups (= 32 pairs) per step, fp16 on GPU, fixed seed 42.
- Saves every epoch to `models/bge_reranker_ft/epoch<i>/` and the last one to
  `models/bge_reranker_ft/final/` plus a `training_meta.json` (base model,
  data fingerprint, hyperparameters) — the archive-grade provenance.

## Go/no-go gate (the cheap stop-loss)

`scripts/18_eval_reranker_ft.py --dev` scores the dev bench at **fixed 15/0**
(primary) and fixed 6/0 (context) with three arms sharing one identical BGE
top-20 pool: `bge`, `rerank20` (off-the-shelf) and `rerank20_ft`.

| Dev ΔR@1 (ft − off-the-shelf) at fixed 15/0 | Verdict |
|---|---|
| ≥ +0.02 (`STAGE8_GO_THRESHOLD`) | **GO** — run the final eval on the Stage 6 bench |
| ≤ 0 | **NO-GO** — stop, archive the dev results as an honest negative result |
| in between | judgment call: at most **one** retry with more data/epochs, then decide; no threshold-shopping |

The dev bench (~400 questions, 1 SE ≈ 0.024) is for the *decision*, not the
*claim* — final numbers only ever come from the Stage 6 bench.

## Final evaluation (only after GO)

`scripts/18_eval_reranker_ft.py` (default mode) re-uses the Stage 6 large
corpus cache (`nq/large_n1000/`, 1032 questions) and the 5
`STAGE6_RERANK_CONFIGS`. Per config it builds one BGE top-20 pool and scores
it with both cross-encoders, yielding three rows: `bge`, `rerank20`,
`rerank20_ft`.

**Built-in check discipline:** the `bge` and `rerank20` rows re-run the exact
Stage 6 pipeline on the same corpus, so they must reproduce the archived
`stage6/final` rows **exactly** (`stage8_check_vs_stage6.csv`, every recall
delta 0.0000). Only then does the `rerank20_ft` row mean anything — same
pattern as Stages 4–7, but the check rides along in the same run.

Headline test: **ft − off-the-shelf ΔR@1 at the size-15 configs vs
2 SE ≈ 0.031** (n=1032). Secondary: how much of the R@1 0.63 → pool 0.96 gap
closes; whether the fixed 6/0 small-chunk rescue grows; R@3/R@5 side effects.
If the dev gain does not survive the larger bench, that is the finding.

## Scope guard

Changed: reranker **weights** only (fine-tuned on NQ train split).
Held fixed: corpus/questions (Stage 6 cache), chunking configs, boundary
models, BGE retriever, depth top-20 only, metric.
Not included: rerank50, new reranker architectures, distillation, retriever
fine-tuning, TriviaQA training data.

## Run (Colab, two sessions)

```bash
# Session 1 — data + training + go/no-go (~2-3 h on a T4):
python scripts/16_build_rerank_train_data.py      # stream train split, mine groups
python scripts/17_train_reranker.py               # fine-tune (fp16, ~1-1.5 h)
python scripts/18_eval_reranker_ft.py --dev       # go/no-go verdict printed

# Session 2 — ONLY if the gate says GO (~2 h):
python scripts/18_eval_reranker_ft.py             # Stage 6 bench, built-in check
python scripts/save_stage_results.py --stage stage8
```

Both eval modes checkpoint per config to `results/latest/` and are
resume-safe; the final mode refuses to run if `stage6/final` is missing or
was built with different models.

## Outputs

Written to `artifacts/results/latest/`, archived to
`artifacts/results/stage8/final/` after review:

- `stage8_dev_results.csv` — dev bench, 3 arms × 2 configs, + gate verdict.
- `stage8_ft_eval_results.csv` — Stage 6 bench, 5 configs × 3 arms.
- `stage8_matched_summary.csv` — per config: pool@20, bge / ots / ft R@k side
  by side, ft−ots and ft−bge deltas.
- `stage8_check_vs_stage6.csv` — the built-in exact-reproduction check.
- `stage8_ft_delta.png` — ΔR@1 (ots vs ft) per config with the ±2 SE band.
- `stage8_summary.md` — data/training provenance, gate history, verdicts.
- `models/bge_reranker_ft/` — the fine-tuned weights + `training_meta.json`
  (model weights live under `models/`, not in the results archive).

## Results (executed 2026-07-08)

**Data build**: 2161 train questions → 2034 groups kept (108 dropped
no-positive = the expected pool-ceiling misses, 19 dropped too-few-negatives);
train `pool_recall@20` = 0.950; dev bench 400 docs / 406 questions.

**Training**: 2 epochs × 1018 steps, fp16 T4, 16.6 min. Mean epoch loss
0.904 → 0.513.

**Dev gate**: ΔR@1 (ft − ots) at fixed 15/0 = **+0.0468** ≥ +0.02 → **GO**
(no retry needed). Side observation, decision-only: on the dev bench (train
split docs) even the off-the-shelf reranker helped at 15/0 (+0.081), unlike
the validation-split bench — dev makes decisions, never claims.

**Final eval (Stage 6 bench, 1032 questions)**: check vs `stage6/final`
exact — all 10 bge/rerank20 rows reproduce with every delta 0.0000, so the
`rerank20_ft` rows are trustworthy.

| config | bge R@1 | ots R@1 | ft R@1 | ft−ots ΔR@1 | ft R@5 |
|---|---|---|---|---|---|
| fixed 6/0 | 0.5087 | 0.5446 | 0.6502 | +0.1056 | 0.8585 |
| fixed 15/0 | 0.6279 | 0.6289 | **0.7355** | **+0.1066** | 0.9215 |
| fixed 15/1 | 0.6260 | 0.6269 | 0.7200 | +0.0930 | 0.9215 |
| bilstm 15/0 | 0.6298 | 0.6269 | 0.7141 | +0.0872 | 0.9244 |
| transformer 15/0 | 0.6366 | 0.6250 | 0.7316 | +0.1066 | 0.9099 |

### Findings

1. **In-domain fine-tuning works — the first intervention that moves the
   size-15 sweet spot.** ΔR@1 (ft − ots) = +0.087…+0.107 across all 5
   configs, ~3.5× the 2 SE = 0.030 band, where the off-the-shelf reranker
   gave at most +0.001. R@3 (+0.05…+0.06) and R@5 (+0.03…+0.05) improve too —
   no metric trades down.
2. **The bottleneck diagnosis was correct**: at fixed 15/0 the gap between
   R@1 (0.628) and the pool ceiling (0.964) was a *ranking* problem; 2034
   in-domain groups and 17 minutes of GPU close about a third of it
   (0.628 → 0.736).
3. **The project's headline survives unchanged**: even with the fine-tuned
   reranker, size still dominates (ft R@1: size 6 = 0.650 vs size 15 =
   0.736) and the size-15 chunking methods still tie within noise
   (ft R@5 spread 0.9099–0.9244 ≈ 2 SE).
4. **Honest caveats**: (a) the gain is *in-domain* — trained on NQ train
   split, evaluated on NQ validation split; nothing here claims transfer to
   other datasets (a TriviaQA transfer arm was added later — see
   [`stage8_transfer.md`](stage8_transfer.md): the gain *partially* transfers,
   as damage control rather than net lift over the dense baseline). (b) Questions
   are disjoint by split, but popular Wikipedia pages can appear in both
   splits' corpora — standard for NQ, noted for transparency. (c) Reranking
   costs ~0.56 s/question on a T4 at depth 20, unchanged by fine-tuning.
