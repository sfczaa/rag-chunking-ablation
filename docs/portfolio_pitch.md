# 作品集講稿 — elevator pitch 與面試追問

適用場合:履歷連結、面試自我介紹、被追問細節時的談話要點。
所有數字都出自唯讀封存 `artifacts/results/stage2~8/final/`,出處逐條標注。

---

## Elevator pitch(2–3 句)

**中文:**

> 我做了一個受控消融的 RAG chunking 研究:語料、問題、指標全部固定,先公平比較
> 固定切塊與兩種學習型切塊(BiLSTM / Transformer boundary model),再一次只改一個
> 變因地疊加 embedder 升級、BM25/RRF hybrid、cross-encoder reranking,最後在 5 倍
> 評測規模**和第二個資料集(TriviaQA)**上重驗。結論是 chunk **大小**主導 recall
> (r(size, R@5) = 0.95,跨資料集 0.80),切塊**方法**在噪音內打平;真正的兩個
> 槓桿是檢索 embedder(MiniLM→BGE,30/30 config 全面 +0.054 R@5)和 **in-domain
> fine-tune 的 reranker**(我用 NQ train split 挖 hard negatives 訓練
> cross-encoder,R@1 +0.107,是唯一在 size-15 甜蜜點有效的介入)。每個 stage 都先
> byte-exact 重現前一個封存 baseline 才下新結論,負結果照樣報告。

**English:**

> A controlled-ablation study of RAG chunking: with the corpus, questions and
> metric held fixed, I compared fixed-size chunking against two learned
> chunkers, then stacked an embedder upgrade, hybrid retrieval and
> cross-encoder reranking — one variable per stage — and re-validated
> everything at 5× evaluation scale **and on a second dataset (TriviaQA)**.
> Chunk **size** dominates recall (r = 0.95; 0.80 cross-dataset) while
> chunking **method** ties within noise; the two real levers are the retrieval
> embedder (+0.054 R@5 across all 30 matched configs) and a reranker I
> **fine-tuned on hard negatives mined from the NQ train split** (+0.107 R@1 —
> the only intervention that works at the size-15 sweet spot). Every stage
> reproduces the previous archived baseline exactly before adding its change,
> and negative results are reported as findings.

一句話版(電梯真的很短時):

> 「我用八個受控 stage 證明:RAG 裡值得調的是 chunk 大小、embedder 和 in-domain
> 的 reranker fine-tune,不是切塊演算法——而且在 5 倍規模和第二個資料集上都重現了
> 這個結論。」

---

## 追問 1:為什麼 hybrid(BM25 + RRF)沒有贏?

**30 秒版:** 因為融合的前提是兩個 ranking 各有互補的信號,而在這個 benchmark 上
BM25 比 BGE 弱太多(R@5 0.803 vs 0.921,差 ~0.12)。等權 RRF 把一個強 ranking 和
一個弱 ranking 平均,結果是稀釋而不是互補:rrf − bge 在 R@5 平均 −0.021,30 個
config 有 24 個是負的(sign test P ≈ 2×10⁻⁴)——效應小,但方向非常一致。

**追問「那 RRF 完全沒用嗎?」:** 不是。RRF 對 BM25 是 30/30 全贏(平均 +0.100
R@5),融合是「安全」的——它能把弱檢索器拉回接近 dense 的水準;只是疊在強 dense
retriever 上沒有增量。這區分很重要:RRF 沒壞,是這個組合裡沒有它能補的洞。

**追問「為什麼 BM25 在這裡這麼弱?」:** NQ 的問題是自然語言問句,答案常用不同措辭
出現在 Wikipedia 文中,lexical overlap 本來就低;而 doc-constrained 指標又要求命中
gold 文件,BM25 容易被其他文件的高頻詞騙走。語意向量在這種 paraphrase-heavy 的
設定下優勢最大。

**追問「有沒有可能是你 fusion 做得不好?」:** 有可能改善空間(weighted fusion、
query-dependent routing),但 24/30 一致負向已經回答了「等權 RRF 在這裡不值得」;
weighted 細掃被明確列為不建議方向——那是加小數點,不是改方向。

數字出處:`stage4/final/hybrid_retriever_matched.csv`、`hybrid_sweep_results.csv`。

---

## 追問 2:為什麼 rerank top-50 比 top-20 差?

**30 秒版:** 把候選池從 20 加深到 50,pool ceiling 只從 0.965 升到 0.985
(+0.02),但 reranker 要多看 30 個幾乎全是 distractor 的候選。cross-encoder 不是
完美 ranker——每多一個 distractor,就多一次「被誤排進 top-5」的機會。結果 rerank50
在 R@1 上 30 個 config 沒有一個贏過 rerank20(28 敗 2 平,P ≈ 7×10⁻⁹)。

**核心一句話:** 加深候選池只有在「pool recall 增益 > 排序誤差成本」時才划算;
這裡增益是 +0.02,誤差成本明顯更大。深度是 precision/recall trade-off,不是
「越多越好」。

**追問「成本呢?」:** T4 上約 24.7 ms/(question, chunk) pair,depth 20 ≈ 0.49 s/查詢,
depth 50 ≈ 1.18 s——2.4 倍延遲買到更差的 R@1。

**追問「這結論在大規模下還成立嗎?」:** Stage 6 沒有重跑 rerank50(Stage 5 已顯示
嚴格更差,重跑只是燒 GPU),但 rerank20 的行為在 n=1032 重現:增益集中在小 chunk
(+0.036 R@1 @ fixed 6/0,> 2 SE),在 size-15 甜蜜點增益是零(max +0.001)。

數字出處:`stage5/final/rerank_matched.csv`、`stage5_reranker_summary.md`、
`stage6/final/stage6_matched_summary.csv`。

---

## 追問 3:為什麼不 fine-tune boundary model?

**30 秒版:** 因為量測顯示瓶頸不在 boundary 網路。第一,輸入信號弱:frozen MiniLM
句向量的 topic-boundary 信號只有 base rate 的 ~1.4 倍(boundary precision ≈ 0.13),
Transformer boundary F1 只有 0.18——不是網路不夠深,是特徵裡沒有多少可學的。
第二,目標不對齊:訓練目標是 Wikipedia section pseudo-labels,評測目標是 retrieval
recall,把前者學得更好不保證後者變好。第三,也是決定性的:六個 stage 一致顯示
**切法在同 size 下打平、size 主導**(Stage 6 把 r 從 0.77 收斂到 0.95)——就算
boundary model 變完美,天花板仍由 chunk size 和 embedder 決定。

**核心一句話:** 這是「先量測、再決定投資」:不是做不到,是證據說這個方向沒有
槓桿。同樣的錢投 embedder 已證實 +0.054;投 reranker fine-tune 有明確的缺口可打
(size-15 的 pool recall@20 ≈ 0.96 但 R@1 只有 0.63,排序是瓶頸)——所以我把
fine-tune 資源排給 reranker 而不是 boundary model,而 Stage 8 證實了這個判斷:
reranker fine-tune 拿到 +0.107 R@1,是全專案唯一在甜蜜點有效的介入。

**追問「Transformer F1 那麼低是不是壞掉?」:** 一開始真的壞掉——sigmoid 輸出
collapse 到常數 ~0.55,診斷發現是 unit-amplitude positional encoding 把小尺度的
frozen MiniLM 向量淹掉(10–40 倍),加一層 input LayerNorm 後恢復判別力。這段
calibration/debug 是專案裡我最想講的工程故事之一:F1 = 0 的第一反應不該是加深
模型,而是先檢查機率分布。

數字出處:`stage2/final/transformer_boundary_diagnostics.json`、
`docs/stage2_transformer_boundary.md`、`stage6/final/stage6_direction_check.csv`。

---

## 備用:三個常被順帶問到的

**「這結論會不會只是 NQ / Wikipedia 的 artifact?」**
Stage 7 用 TriviaQA rc.wikipedia(題型不同、gold 定義不同、300 題 / 472 個完整
頁面)重跑同一組 30 configs:r(size, R@5) = 0.80、size-15 三法打平(spread 0.020
< 2 SE 0.036)、best size 仍是 15,4/4 方向檢驗重現。最有說服力的細節是「贏家」
從 fixed(NQ)換成 BiLSTM(TriviaQA)但差距都在噪音內——真正贏的方法不會跨資料集
換人,噪音才會。誠實註記:size 效應在 TriviaQA 較平(+0.036 vs +0.062),方向
不變、幅度視資料集而定。數字出處:`stage7/final/stage7_direction_check.csv`。

**「你的 headline 指標為什麼是 doc-constrained Recall@k?」**
短答案常見的 recall 算法會被騙:年份、人名這種短答案會 substring-match 到不相關
文件的 chunk,而且兩種切法被騙的程度不同,會汙染比較。所以每個 chunk 記住來源
文件,只有命中 gold 文件的才算分;unconstrained 數字照樣附上供參。

**「規模化之後最意外的是什麼?」**
最好的意外是 size 效應變得更乾淨(r 0.77 → 0.95)——它從來不是噪音;絕對值下降
(0.921 → 0.881)完全在預期內,因為 index 裡 distractor 文件多了 5 倍。reranker 的
小 chunk 救援縮小到 +0.036 但仍 > 2 SE,而甜蜜點增益維持是零——off-the-shelf
reranker 在 pool ceiling 0.96 下只能排到 R@1 0.63,這個 gap 就是 Stage 8
fine-tune reranker 的 case,後來也確實打中了(+0.107 R@1)。

**「off-the-shelf reranker 沒用,為什麼 fine-tune 就有用?」**(Stage 8)
因為瓶頸是「這個任務的相關性定義」而不是模型容量:bge-reranker-base 是在
MS MARCO 類通用資料上訓的,分不出 NQ 甜蜜點池子裡「提到答案的 chunk」和「來自
gold 文件、真正回答問題的 chunk」。我從 NQ **train** split(與所有評測 bench
不相交)挖了 2034 組 (1 正例 + 7 hard negatives)——negatives 就是 BGE 自己排很
前面的干擾項——用 listwise CE 訓 2 epochs(T4 上 17 分鐘)。流程上先設便宜的
go/no-go:dev bench ΔR@1 ≥ +0.02 才准跑正式評測(實測 +0.047 → GO);正式跑時
bge/rerank20 兩個 baseline 列必須 byte-exact 重現 stage6 封存(10/10 過)才讀
fine-tuned 列。結果 5 個 config 全部 +0.087~+0.107 R@1(2 SE = 0.030),
fixed 15/0 從 0.629 → 0.736,補掉到 pool ceiling 0.964 缺口的約三分之一。
誠實註記:這是 in-domain 增益(NQ train → NQ val),不宣稱跨資料集轉移;而且
headline 不變——fine-tune 之後 size 仍主導(size 6 的 0.650 < size 15 的
0.736)、三種切法仍打平。數字出處:`stage8/final/stage8_matched_summary.csv`、
`stage8_dev_gate.md`、`stage8_check_vs_stage6.csv`。
