---
license: mit
base_model: BAAI/bge-reranker-base
pipeline_tag: text-classification
tags:
  - reranker
  - cross-encoder
  - retrieval
  - rag
language:
  - en
datasets:
  - natural_questions
---

# bge-reranker-base fine-tuned on Natural Questions (train split)

A cross-encoder reranker for RAG retrieval: `BAAI/bge-reranker-base` fine-tuned
on hard-negative groups mined from the **Natural Questions train split**. It was
produced as Stage 8 of a controlled ablation study of RAG chunking, where the
diagnosis was explicit — at the best chunk size the answer was already inside the
BGE top-20 pool for ~96% of questions while Recall@1 sat at ~0.63, so **ranking,
not pool recall, was the bottleneck**.

## Training

| | |
|---|---|
| Base model | `BAAI/bge-reranker-base` |
| Data | 2034 groups (1 positive + 7 hard negatives) mined from 2161 NQ **train** questions |
| Mining | deployment-matched chunking (fixed 15 sentences / overlap 0), BGE top-20 pool (`BAAI/bge-base-en-v1.5`) |
| Objective | listwise softmax cross-entropy over each group |
| Schedule | 2 epochs, lr 2e-5, warmup 10%, weight decay 0.01, 4 groups/step, fp16, seed 42 |
| Hardware | ~17 min on a single T4 |
| Loss | 0.904 → 0.513 |

Every evaluation below uses the NQ **validation** split, disjoint from training.

## Results

**In-domain (NQ, 1000 docs / 1032 questions).** Δ vs the off-the-shelf reranker,
with all arms reranking one identical BGE top-20 pool (2 SE = 0.030):

| chunking config | dense R@1 | off-the-shelf R@1 | **this model R@1** | Δ vs off-the-shelf |
|---|---|---|---|---|
| fixed 6/0 | 0.509 | 0.545 | 0.650 | +0.106 |
| **fixed 15/0** | 0.628 | 0.629 | **0.736** | **+0.107** |
| fixed 15/1 | 0.626 | 0.627 | 0.720 | +0.093 |
| bilstm 15/0 | 0.630 | 0.627 | 0.714 | +0.087 |
| transformer 15/0 | 0.637 | 0.625 | 0.732 | +0.107 |

At fixed 15/0 this closes about a third of the gap between dense R@1 (0.628) and
the pool ceiling (0.964). R@3 and R@5 rise too; no metric trades down.

**Cross-dataset (TriviaQA rc.wikipedia, 472 docs / 300 questions, 2 SE = 0.054).**
Reported because it is the honest limit of the result:

| | Δ vs off-the-shelf R@1 | **Δ vs no reranking (dense)** |
|---|---|---|
| in-domain (NQ), fixed 15/0 | +0.107 | **+0.107** |
| cross-dataset (TriviaQA), fixed 15/0 | +0.057 | **+0.003** |

Out of domain the *off-the-shelf* reranker actively hurts (−0.03…−0.07 R@1 vs
plain dense retrieval). This model recovers to **parity with the dense baseline**
— so its cross-dataset win over the off-the-shelf reranker is largely **damage
control, not net lift**. The direction is consistent (positive at 5/5 configs,
4/5 above 2 SE), but the magnitude is about half the in-domain gain.

**Use it in-domain, or expect parity rather than improvement.**

## Usage

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("sfczaa/bge-reranker-base-nq-ft", max_length=512)
scores = model.predict([(question, passage) for passage in candidates])
ranked = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
```

Drop-in replacement for `BAAI/bge-reranker-base` — same architecture, tokenizer
and 512-token truncation. Reranking costs ~0.56 s/question at depth 20 on a T4,
unchanged by fine-tuning.

## Caveats

- Trained and validated on Natural Questions; questions are split-disjoint, but
  popular Wikipedia pages can appear in both splits' corpora (standard for NQ).
- The cross-dataset evaluation uses distant-supervised TriviaQA gold (answer
  string match on entity pages), weaker than NQ's annotated gold; absolute recall
  is not comparable between the two benchmarks.
- Evaluated only on English Wikipedia-style passages at 6–15 sentence chunks.

## License

MIT, matching the `BAAI/bge-reranker-base` base model.
