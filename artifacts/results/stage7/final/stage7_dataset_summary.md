# Stage 7 - cross-dataset robustness check (TriviaQA rc.wikipedia)

- Dataset: `mandarjoshi/trivia_qa` / `rc.wikipedia` / `validation` — full Wikipedia entity pages bundled in the dataset (no fetching).
- Eval set: **472 docs / 300 questions** (requested 300 kept questions).
- Gold documents: the question's entity pages whose sentence-joined text contains the kept answer string (`answer.value` first, then aliases). **Distant supervision** — known to contain the answer, not human-verified to support it (weaker than NQ's annotated gold).
- Boundary/chunking embedding model: `sentence-transformers/all-MiniLM-L6-v2` (Stage 2 weights, unchanged)
- Dense retrieval embedding model: `BAAI/bge-base-en-v1.5`
- Arm: **bge only** (no BM25/RRF, no reranking, no fine-tuning)

## Loader statistics

- Scanned 302 validation rows to keep 300 questions.
- Dropped: 2 with no answer candidate in any entity page, 0 with no usable page; 1 pages under 2 sentences.
- Multi-gold questions (answer in 2 pages): 118.
- Document length (sentences): median 152, mean 209.9, range 4-833 (comparability guard: median >= 30 passed).

Best bge config: `bilstm`, `target=15,min=11,max=19,overlap=1` — R@1=0.6933 R@3=0.8667 R@5=0.9200

## Methods side by side (recall@5 per size x overlap)

| size | overlap | fixed | bilstm | transformer | spread |
|---|---|---|---|---|---|
| 6 | 0 | 0.8700 | 0.8567 | 0.8767 | 0.0200 |
| 6 | 1 | 0.8867 | 0.8733 | 0.8667 | 0.0200 |
| 8 | 0 | 0.8833 | 0.8933 | 0.8900 | 0.0100 |
| 8 | 1 | 0.8900 | 0.8900 | 0.8833 | 0.0067 |
| 10 | 0 | 0.8800 | 0.8800 | 0.9000 | 0.0200 |
| 10 | 1 | 0.9000 | 0.9100 | 0.8833 | 0.0267 |
| 12 | 0 | 0.9167 | 0.9100 | 0.9067 | 0.0100 |
| 12 | 1 | 0.9167 | 0.9100 | 0.9033 | 0.0133 |
| 15 | 0 | 0.9100 | 0.9033 | 0.9067 | 0.0067 |
| 15 | 1 | 0.9067 | 0.9200 | 0.9000 | 0.0200 |

## Direction checks (does the NQ headline transfer?)

- **yes** — 1. chunk size matters more than chunking method: pearson_r(avg_chunk_size, recall@5) over the 30 bge configs — observed 0.8005 (rule: r >= 0.5; NQ reference: NQ: r ~ 0.77 (n=203), 0.95 (n=1032))
- **yes** — 1. chunk size matters more than chunking method: recall@5 size effect (largest vs smallest size) minus mean method spread at matched (size, overlap) — observed 0.0208 (rule: size effect > method spread; NQ reference: NQ n=203: size ~ +0.07 vs method <= 0.014; n=1032: 0.052 > spread)
- **yes** — 2. methods tie at matched (large) chunk size: max method spread of recall@5 over the size-15 cells — observed 0.02 (rule: max spread < 2 SE (0.0355); NQ reference: NQ stage3: spread 0.020 at size 15/ov0 (~1 SE = 0.019))
- **yes** — 3. best chunk size is large: nominal size of the best-recall@5 config (project ranking) — observed 15.0 (rule: best nominal size in {12, 15}; NQ reference: NQ: best size 15 at n=203 and at n=1032)

**4/4 direction checks replicate.**

