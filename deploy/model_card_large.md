---
license: mit
base_model: BAAI/bge-reranker-large
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

# bge-reranker-large fine-tuned on Natural Questions (train split)

A cross-encoder reranker for RAG retrieval: `BAAI/bge-reranker-large` fine-tuned
on the same hard-negative groups and recipe as
[`sfczaa/bge-reranker-base-nq-ft`](https://huggingface.co/sfczaa/bge-reranker-base-nq-ft).
It was trained in Stage 15 of a controlled ablation study of RAG chunking to test
whether model size limits the fine-tuned reranker.

## Training

| | |
|---|---|
| Base model | `BAAI/bge-reranker-large` at revision `55611d7bca2a7133960a6d3b71e083071bbfc312` |
| Data | 2034 groups (1 positive + 7 hard negatives) mined from NQ train questions |
| Objective | listwise softmax cross-entropy over each group |
| Schedule | 2 epochs, lr 2e-5, warmup 10%, weight decay 0.01, 4 groups per step, fp16, seed 42 |
| Memory settings | 2 groups per forward/backward pass, gradient checkpointing |
| Final mean epoch loss | 0.3147 |

## Results

All evaluations use the NQ validation split, fixed 15-sentence chunks, and one
shared BGE top-20 candidate pool per question. The comparison is against the base
model fine-tuned with the same data and recipe.

| Bench | Questions | base fine-tuned R@1 | this model R@1 | Difference, 95% CI |
|---|---|---|---|---|
| Stage 6 bench | 1032 | 0.7345 | 0.7558 | +0.0213 [+0.0009, +0.0417] |
| Holdout bench, no overlap with the above | 3063 | 0.7231 | 0.7424 | +0.0193 [+0.0072, +0.0314] |

The gain held on questions that had not been used before. Its size is close to
the study's pre-registered 0.02 R@1 threshold on both benches. Without
fine-tuning, `BAAI/bge-reranker-large` ranked below `BAAI/bge-reranker-base` on
the Stage 6 bench (R@1 0.5940 against 0.6289). Reranking takes about three times
as long per question as the base model.

## Usage

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("sfczaa/bge-reranker-large-nq-ft", max_length=512)
scores = model.predict([(question, passage) for passage in candidates])
ranked = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
```

## Caveats

- Trained and evaluated on Natural Questions only. Questions are split-disjoint,
  but popular Wikipedia pages can appear in both splits (188 of the 2852 holdout
  titles also appear in the training documents).
- One training run with one seed.
- Evaluated on English Wikipedia passages chunked at 15 sentences.

## License

MIT, matching the `BAAI/bge-reranker-large` base model.
