# Stage 17 - answer accuracy with a reader on top of retrieval

Claim 1 (chunk size): SIZE-NOT-DETECTED. Claim 2 (boundary placement): INCONCLUSIVE.

- reader: `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554`, greedy, fp16, GPU Tesla T4
- libraries: torch 2.11.0+cu130, transformers 5.18.0
- Stage 6 bench: 1000 docs / 1032 questions, 1031 scored; gold chunk for 1028
- criterion 1 (weights and archive check): PASS
- criterion 2 (gold - closed book): +0.3716, 95% CI [+0.3407, +0.4025], PASS
- claim 1 R@5 gap over the scored questions: +0.0611; accuracy gap / R@5 gap = 0.25

| arm | accuracy | EM | F1 | R@1 | R@5 | acc, answer in top 5 | acc, answer outside top 5 | words | prompt tokens | cut | lost to cut | s per question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed_book | 0.1183 | 0.0582 | 0.1589 | - | - | - | - | 6.64 | 38 | 0 | 0 | 0.67 |
| gold | 0.4903 | 0.3356 | 0.5408 | - | - | - | - | 5.09 | 860 | 21 | 3 | 1.10 |
| fixed15_ft | 0.4471 | 0.2958 | 0.4895 | 0.7352 | 0.9185 | 0.4805 | 0.0714 | 5.58 | 3627 | 65 | 1 | 3.53 |
| fixed6_ft | 0.4316 | 0.2842 | 0.4726 | 0.6469 | 0.8574 | 0.4955 | 0.0476 | 5.45 | 1915 | 34 | 0 | 1.95 |
| bilstm15_ft | 0.4646 | 0.3123 | 0.5037 | 0.7148 | 0.9234 | 0.4989 | 0.0506 | 5.49 | 3569 | 64 | 0 | 3.45 |
| fixed15_large | 0.4636 | 0.3007 | 0.4977 | 0.7556 | 0.9292 | 0.4916 | 0.0959 | 5.81 | 3629 | 63 | 1 | 3.57 |

| role | comparison | n | mean | 95% CI | 90% CI |
| --- | --- | --- | --- | --- | --- |
| reader check | gold - closed_book | 1028 | +0.3716 | [+0.3407, +0.4025] | - |
| claim 1 | fixed15_ft - fixed6_ft | 1031 | +0.0155 | [-0.0040, +0.0351] | - |
| claim 2 | bilstm15_ft - fixed15_ft | 1031 | +0.0175 | [-0.0001, +0.0351] | [+0.0027, +0.0322] |
| reported | fixed15_large - fixed15_ft | 1031 | +0.0165 | [+0.0017, +0.0313] | - |

| arm | acc, answer <= 5 NQ tokens | n | acc, longer answer | n |
| --- | --- | --- | --- | --- |
| closed_book | 0.1536 | 768 | 0.0152 | 263 |
| gold | 0.6120 | 768 | 0.1308 | 260 |
| fixed15_ft | 0.5651 | 768 | 0.1027 | 263 |
| fixed6_ft | 0.5495 | 768 | 0.0875 | 263 |
| bilstm15_ft | 0.5964 | 768 | 0.0798 | 263 |
| fixed15_large | 0.5846 | 768 | 0.1103 | 263 |
