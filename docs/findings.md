# 研究結果與限制

本專案固定語料, 問題與 doc-constrained Recall@k, 逐階段比較切塊方式,
chunk 大小, embedding, hybrid retrieval 與 reranking. 結果來自
`artifacts/results/<stage>/final/` 的封存檔案.

## 切塊與檢索

在本次設定中, chunk 大小與 Recall@5 的關聯比切塊方法更強.
Stage 6 的相關係數為 0.95, TriviaQA 的跨資料集結果為 0.80;
相同大小下三種切塊方式的差異較小. 這些結果適用於本次資料,
切塊範圍與檢索指標, 不能推論所有 RAG 任務都不需要學習型切塊.

MiniLM 改為 BGE 後, 30 個配對設定的 Recall@5 都上升, 平均增加 0.054.
BM25 與 BGE 的等權 RRF 融合相較 BGE, 30 個設定中有 24 個下降,
平均差為 -0.021; 相較 BM25 則全部上升. 這表示此組合未改善強基線,
並不排除其他權重或資料集有不同結果.

來源: `stage3/final/stage2_vs_stage3_matched.csv`,
`stage4/final/hybrid_retriever_matched.csv`,
`stage6/final/stage6_direction_check.csv`,
`stage7/final/stage7_direction_check.csv`.

## Reranker

Stage 5 將候選深度由 20 增至 50, 候選池 recall ceiling 從 0.965 增至
0.985, 但最終 Recall@1 在 30 個設定中為 28 次下降, 2 次持平.
較多候選沒有直接轉為更好的排序結果.

Stage 8 使用 NQ train split 挖掘 hard negatives, 訓練 cross-encoder.
固定 15/0 設定的 Recall@1 從 0.629 升至 0.736. dev gate 與基線
重現檢查在正式評估前執行. 此提升屬於 NQ 域內結果; TriviaQA transfer
沒有呈現相同提升, 因此不宣稱跨資料集通用改善.

來源: `stage5/final/rerank_matched.csv`,
`stage8/final/stage8_matched_summary.csv`,
`stage8/final/stage8_dev_gate.md`,
`stage8/final/stage8_check_vs_stage6.csv`,
`stage8/final/stage8_transfer_summary.md`.

## 指標解讀

doc-constrained Recall@k 要求答案命中來自 gold 文件的 chunk, 降低常見
短答案在不相關文件中被誤算命中的情形. 這是檢索指標, 不是生成答案
品質, 事實正確性或使用者滿意度. 重現封存數字也不等於證明研究結果
能推廣至不同模型, 語料, 硬體或使用情境.
