"""Central configuration for the RAG semantic-chunking project.

Paths default to a project-local ``artifacts`` directory.  Override the
storage root by setting the environment variable ``RAG_DATA_ROOT`` (handy
for Colab, a local run, or a quick ephemeral test), or by calling
:func:`set_data_root`.

Every tunable number lives here so the notebook / scripts stay declarative.
The data-scale knobs are read at *call time* by the pipeline functions, so
:func:`apply` / :func:`use_smoke` can re-scale the whole pipeline at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Storage root + derived paths (recomputed by set_data_root)
# --------------------------------------------------------------------------- #
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "artifacts"

# These are assigned by set_data_root() below; declared here for clarity.
DATA_ROOT: Path
DATA_DIR: Path
WIKI_RAW_DIR: Path
WIKI_PROCESSED_DIR: Path
EMBEDDINGS_DIR: Path
NQ_DIR: Path
NQ_INDEX_DIR: Path
NQ_INDEX_MAIN_DIR: Path
NQ_INDEX_SWEEP_DIR: Path
MODELS_DIR: Path
RESULTS_DIR: Path
RESULTS_LATEST_DIR: Path
RESULTS_RUNS_DIR: Path
RESULTS_TABLE: Path
RESULTS_FIGURE: Path
ALL_DIRS: list


def set_data_root(root) -> None:
    """Point every artifact path at ``root`` (recomputes derived paths)."""
    global DATA_ROOT, DATA_DIR, WIKI_RAW_DIR, WIKI_PROCESSED_DIR, EMBEDDINGS_DIR
    global NQ_DIR, NQ_INDEX_DIR, NQ_INDEX_MAIN_DIR, NQ_INDEX_SWEEP_DIR
    global MODELS_DIR, RESULTS_DIR, RESULTS_LATEST_DIR, RESULTS_RUNS_DIR
    global RESULTS_TABLE, RESULTS_FIGURE, ALL_DIRS
    DATA_ROOT = Path(root)
    DATA_DIR = DATA_ROOT / "data"
    WIKI_RAW_DIR = DATA_DIR / "wiki_raw"              # cached raw wikitext (.txt)
    WIKI_PROCESSED_DIR = DATA_DIR / "wiki_processed"  # (sentences, labels) .jsonl
    EMBEDDINGS_DIR = DATA_DIR / "embeddings"          # offline sentence embeddings
    NQ_DIR = DATA_DIR / "nq"                           # cached NQ docs / questions
    # FAISS indices live under nq/indices/ so per-config sweep indices never
    # scatter loose files next to the corpus cache.
    NQ_INDEX_DIR = NQ_DIR / "indices"
    NQ_INDEX_MAIN_DIR = NQ_INDEX_DIR / "main"          # the Phase 5 evaluation pair
    NQ_INDEX_SWEEP_DIR = NQ_INDEX_DIR / "sweep"        # opt-in temp sweep indices
    MODELS_DIR = DATA_ROOT / "models"                  # trained weights
    RESULTS_DIR = DATA_ROOT / "results"                # figures + comparison tables
    RESULTS_LATEST_DIR = RESULTS_DIR / "latest"        # newest sweep artifacts
    RESULTS_RUNS_DIR = RESULTS_DIR / "runs"            # opt-in timestamped snapshots
    RESULTS_TABLE = RESULTS_DIR / "recall_comparison.csv"
    RESULTS_FIGURE = RESULTS_DIR / "recall_comparison.png"
    ALL_DIRS = [
        DATA_DIR, WIKI_RAW_DIR, WIKI_PROCESSED_DIR, EMBEDDINGS_DIR, NQ_DIR,
        NQ_INDEX_DIR, NQ_INDEX_MAIN_DIR, MODELS_DIR, RESULTS_DIR,
        RESULTS_LATEST_DIR,
    ]


set_data_root(os.environ.get("RAG_DATA_ROOT", DEFAULT_DATA_ROOT))


def ensure_dirs() -> None:
    """Create every artifact directory (idempotent)."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Phase 1 — Wikipedia data preparation
# --------------------------------------------------------------------------- #
WIKI_DUMP = "wikimedia/wikipedia"
WIKI_DUMP_CONFIG = "20231101.en"     # used (streamed) only to pick article titles
N_WIKI_ARTICLES = 5000               # how many articles to build the dataset from
WIKI_API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
# A descriptive User-Agent is required by the Wikimedia API etiquette.
WIKI_API_USER_AGENT = (
    "RAG-chunk-optimize/1.0 "
    "(https://github.com/sfczaa/rag-chunking-ablation)"
)
WIKI_API_BATCH = 50                  # titles per API request (API max for anon)

# Sections that are not real article content -> dropped before labelling.
WIKI_SKIP_SECTIONS = {
    "references", "see also", "external links", "notes", "further reading",
    "bibliography", "sources", "footnotes", "citations", "external link",
}
MIN_SENTENCES_PER_ARTICLE = 8        # discard stubs / too-short articles
MIN_SENTENCES_PER_SECTION = 1        # keep sections with at least this many sents

# Reproducible train / val / test split (by article).
SPLIT_RATIOS = (0.8, 0.1, 0.1)
SEED = 42

# --------------------------------------------------------------------------- #
# Phase 2 — Offline embedding
# --------------------------------------------------------------------------- #
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
EMBED_BATCH = 256                    # sentences per encode() call

# Stage 3 decouples boundary/chunking embeddings from retrieval embeddings.
# EMBED_MODEL remains the historical MiniLM setting and is still the boundary
# default, so existing Stage 1/2 weights and no-arg callers stay compatible.
BOUNDARY_EMBED_MODEL = EMBED_MODEL
BOUNDARY_EMBED_DIM = EMBED_DIM

# Retrieval embeddings can be swapped independently. Stage 3 sets this to BGE,
# while the default keeps the old MiniLM retrieval path unchanged.
RETRIEVAL_EMBED_MODEL = os.environ.get("RAG_RETRIEVAL_EMBED_MODEL", EMBED_MODEL)
RETRIEVAL_EMBED_DIM = None            # None -> infer from SentenceTransformer
RETRIEVAL_EMBED_BATCH = EMBED_BATCH
RETRIEVAL_EMBED_NORMALIZE = True      # cosine/IP retrieval; required for BGE
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# None means auto: apply BGE_QUERY_INSTRUCTION for BGE retrieval models only.
RETRIEVAL_QUERY_INSTRUCTION = None

# --------------------------------------------------------------------------- #
# Phase 3 — BiLSTM training
# --------------------------------------------------------------------------- #
# Boundary model family. Both "bilstm" (Stage 1) and "transformer" (Stage 2) are
# implemented and share one interface + the target-size cutting policy. This stays
# "bilstm" so Phase 4/5 and the Stage 1 sweep are unchanged; the Stage 2 sweep
# selects the transformer per-method instead of via this default.
MODEL_TYPE = "bilstm"                 # "bilstm" | "transformer"
HIDDEN_SIZE = 128                    # H ; BiLSTM output per step = 2H
NUM_LAYERS = 1
DROPOUT = 0.0
LEARNING_RATE = 1e-3
MAX_EPOCHS = 20
EARLY_STOP_PATIENCE = 3              # epochs without val-loss improvement
# pos_weight for the weighted BCE.  None -> computed from the training split
# (= #neg / #pos, ~15 per the spec).  Set a float to override.
POS_WEIGHT = None
# -- Learned chunking policy ------------------------------------------------- #
# "threshold": cut wherever boundary prob >= BOUNDARY_THRESHOLD (min/max clamp).
#              This is the original Phase 4/5 behaviour, kept for back-compat.
# "target":    target-size semantic cutting — within a [min,max] window the model
#              picks its most confident boundary, so learned chunks stay size-
#              comparable to the fixed-size baseline (used by the Phase 6 sweep).
SEMANTIC_CHUNK_POLICY = "target"     # "target" | "threshold"
SEMANTIC_TARGET_CHUNK_SIZE = 10      # preferred learned chunk length (sentences)
SEMANTIC_MIN_CHUNK_SIZE = 6          # earliest a target-size chunk may be cut
SEMANTIC_MAX_CHUNK_SIZE = 12         # latest a target-size chunk may be cut
SEMANTIC_OVERLAP = 1                 # sentence overlap between target-size chunks

# Threshold-policy knobs (back-compat; used by Phase 4/5 build_indexes).
BOUNDARY_THRESHOLD = 0.8             # probability cut for "this is a boundary"
BILSTM_MIN_CHUNK_SIZE = 4            # do not cut learned chunks before this many sentences
BILSTM_MAX_CHUNK_SIZE = 12           # force a learned chunk cut after this many sentences
BILSTM_OVERLAP = 1                   # sentence overlap between learned chunks

# -- Transformer boundary model (Stage 2) ------------------------------------ #
# A second learned boundary model: a small Transformer encoder over the same
# MiniLM sentence embeddings, with the same predict_proba/predict_boundaries
# interface and the same target-size cutting policy as the BiLSTM.
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4                 # must divide EMBED_DIM (384 % 4 == 0)
TRANSFORMER_FF_DIM = 512
TRANSFORMER_DROPOUT = 0.1

# -- Boundary-threshold calibration (Stage 2.1) ------------------------------ #
# The Transformer's sigmoid outputs occupy a different range than the BiLSTM's,
# so the shared BOUNDARY_THRESHOLD (0.8, tuned for the BiLSTM) can score its
# Boundary F1 at ~0 even when the learned boundaries are fine. This *separate*
# threshold is calibrated on the validation split (max boundary F1) by Phase 7
# training. It is used ONLY for the Boundary F1 diagnostic — the retrieval sweep
# uses the target-size (argmax) policy and ignores any threshold, so Stage 1 /
# Stage 2 retrieval behaviour is unchanged.
TRANSFORMER_BOUNDARY_THRESHOLD = 0.5     # fallback; overwritten by val calibration
# Thresholds tried by the validation F1 sweep (0.01 .. 0.99 inclusive).
BOUNDARY_THRESHOLD_GRID = [round(0.01 * i, 2) for i in range(1, 100)]
# Calibration diagnostic artifacts (written to RESULTS_LATEST_DIR by Phase 7).
TRANSFORMER_THRESHOLD_CSV = "transformer_threshold_f1.csv"
TRANSFORMER_DIAGNOSTICS_JSON = "transformer_boundary_diagnostics.json"
TRANSFORMER_THRESHOLD_PNG = "transformer_boundary_threshold_f1.png"

# Per-model weight files, kept separate so BiLSTM and Transformer coexist.
MODEL_FILENAME = "bilstm_best.pt"     # back-compat default (BiLSTM)
MODEL_FILENAMES = {
    "bilstm": "bilstm_best.pt",
    "transformer": "transformer_best.pt",
}

# --------------------------------------------------------------------------- #
# Phase 4 — RAG retrieval pipeline (Natural Questions)
# --------------------------------------------------------------------------- #
NQ_DATASET = "google-research-datasets/natural_questions"
NQ_CONFIG = "default"
NQ_SPLIT = "validation"
N_NQ_DOCS = 200                      # distinct documents to index (corpus size)
FIXED_CHUNK_SIZE = 10                # baseline: sentences per fixed chunk
FIXED_CHUNK_OVERLAP = 1              # sentence overlap between fixed chunks

# --------------------------------------------------------------------------- #
# Phase 5 — Evaluation
# --------------------------------------------------------------------------- #
RECALL_KS = (1, 3, 5)

# --------------------------------------------------------------------------- #
# Phase 6 — Chunking sweep optimizer
# --------------------------------------------------------------------------- #
# Full grid: every (size, overlap) for fixed-size and (target_size, overlap) for
# learned target-size cutting. The learned min/max window is derived per target
# size in the sweep (min = max(2, target-4), max = target+4).
FIXED_SIZE_GRID = [6, 8, 10, 12, 15]
TARGET_SIZE_GRID = [6, 8, 10, 12, 15]
OVERLAP_GRID = [0, 1]

# Smaller grid for `--quick` (a fast sanity sweep, still both methods + overlaps).
QUICK_FIXED_SIZE_GRID = [8, 10, 12]
QUICK_TARGET_SIZE_GRID = [8, 10, 12]
QUICK_OVERLAP_GRID = [0, 1]

# --------------------------------------------------------------------------- #
# Stage 4 — hybrid retrieval ablation (BM25 + BGE + RRF)
# --------------------------------------------------------------------------- #
# Okapi BM25 over the exact same chunks the dense index embeds (pure-numpy
# implementation in rag_chunk/hybrid.py — no extra Colab dependency).
BM25_K1 = 1.5
BM25_B = 0.75
# Reciprocal Rank Fusion: fused score(d) = sum over retrievers of 1/(RRF_K + rank).
RRF_K = 60                       # the standard constant from the RRF paper
HYBRID_FUSE_DEPTH = 50           # top-N candidates taken from each retriever
# Stage 4 output filenames (written under RESULTS_LATEST_DIR).
HYBRID_SWEEP_CSV = "hybrid_sweep_results.csv"
HYBRID_BEST_JSON = "hybrid_best_config.json"
HYBRID_MATCHED_CSV = "hybrid_retriever_matched.csv"
HYBRID_SCATTER_PNG = "hybrid_recall_vs_chunk_size.png"
HYBRID_RETRIEVER_PLOT_PNG = "hybrid_retriever_comparison.png"

# --------------------------------------------------------------------------- #
# Stage 5 — cross-encoder reranking (BGE top-k + off-the-shelf reranker)
# --------------------------------------------------------------------------- #
# BGE dense retrieval fetches a candidate pool per question; a pretrained
# cross-encoder (no training / fine-tuning) rescores the (question, chunk)
# pairs and reorders the pool. One rerank arm per pool depth.
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_DEPTHS = (20, 50)         # BGE candidate-pool sizes to rerank
RERANK_BATCH_SIZE = 32           # (question, chunk) pairs per cross-encoder batch
RERANK_MAX_LENGTH = 512          # cross-encoder token cap (long chunks truncate)
# Stage 5 output filenames (written under RESULTS_LATEST_DIR).
RERANK_SWEEP_CSV = "rerank_sweep_results.csv"
RERANK_BEST_JSON = "rerank_best_config.json"
RERANK_MATCHED_CSV = "rerank_matched.csv"
RERANK_SCATTER_PNG = "rerank_recall_vs_chunk_size.png"
RERANK_COMPARISON_PNG = "rerank_comparison.png"

# --------------------------------------------------------------------------- #
# Stage 6 — larger-scale robustness evaluation
# --------------------------------------------------------------------------- #
# Same dataset source / chunking / models as Stages 3-5; the only change is the
# corpus scale. The large corpus caches under a separate nq/large_n<N>/ folder
# so the default 200-doc cache (used by Stages 1-5 and the Stage 6 check mode)
# is never overwritten.
N_NQ_DOCS_LARGE = 1000           # ~1000 docs/questions; actual counts reported
# Chunking configs that also get the rerank20 arm at large scale (the bge arm
# covers the full 30-config grid). (method, size-or-target, overlap).
STAGE6_RERANK_CONFIGS = (
    ("fixed", 6, 0),
    ("fixed", 15, 0),
    ("fixed", 15, 1),
    ("bilstm", 15, 0),
    ("transformer", 15, 0),
)
# Stage 6 output filenames (written under RESULTS_LATEST_DIR).
STAGE6_RESULTS_CSV = "stage6_large_eval_results.csv"
STAGE6_MATCHED_CSV = "stage6_matched_summary.csv"
STAGE6_DIRECTION_CSV = "stage6_direction_check.csv"
STAGE6_SUMMARY_MD = "stage6_large_eval_summary.md"
# Stage 6 figures (scripts/13_stage6_plots.py; run after the large eval, before
# archiving, so the PNGs land in stage6/final together with the CSVs).
STAGE6_SIZE_PLOT_PNG = "stage6_size_vs_recall.png"
STAGE6_DELTA_PLOT_PNG = "stage6_rerank_delta.png"

# --------------------------------------------------------------------------- #
# Stage 7 — cross-dataset robustness check (TriviaQA rc.wikipedia, bge-only)
# --------------------------------------------------------------------------- #
# Same pipeline, grids, boundary weights and BGE retriever as Stage 3; the only
# change is the QA dataset. TriviaQA rc.wikipedia bundles full Wikipedia pages
# per question (entity_pages.wiki_context), so the 6-15 sentence chunk grids
# stay meaningful with no article fetching. See docs/stage7_cross_dataset.md.
STAGE7_DATASET = "mandarjoshi/trivia_qa"
STAGE7_DATASET_CONFIG = "rc.wikipedia"
STAGE7_SPLIT = "validation"
STAGE7_N_QUESTIONS = 300         # kept questions; the stream stops when reached
# Stage 7 output filenames (written under RESULTS_LATEST_DIR).
STAGE7_RESULTS_CSV = "stage7_cross_dataset_results.csv"
STAGE7_MATCHED_CSV = "stage7_matched_summary.csv"
STAGE7_DIRECTION_CSV = "stage7_direction_check.csv"
STAGE7_SUMMARY_MD = "stage7_dataset_summary.md"
STAGE7_SCATTER_PNG = "stage7_recall_vs_chunk_size.png"

# --------------------------------------------------------------------------- #
# Stage 8 — fine-tune the cross-encoder reranker (Route C)
# --------------------------------------------------------------------------- #
# Trigger (Stage 6): at the size-15 sweet spot pool_recall@20 ~ 0.96 but
# R@1 ~ 0.63 and the off-the-shelf reranker adds ~0 — ranking, not pool
# recall, is the remaining bottleneck. Training data comes from the NQ *train*
# split; every eval bench uses the validation split, so they stay disjoint.
# See docs/stage8_reranker_finetune.md.
STAGE8_N_TRAIN_DOCS = 2000       # training-corpus documents (~1 question/doc)
STAGE8_N_DEV_DOCS = 400          # dev bench: the NEXT window of the train split
STAGE8_NUM_NEGATIVES = 7         # hard negatives per question (group = 1 + 7)
STAGE8_TRAIN_CHUNK_SIZE = 15     # deployment chunking: fixed size 15, overlap 0
STAGE8_TRAIN_CHUNK_OVERLAP = 0
STAGE8_MINE_DEPTH = 20           # mine positives/negatives from the BGE top-20
# Training hyperparameters (scripts/17_train_reranker.py).
STAGE8_EPOCHS = 2
STAGE8_LR = 2e-5
STAGE8_WARMUP_FRAC = 0.1
STAGE8_WEIGHT_DECAY = 0.01
STAGE8_GROUPS_PER_STEP = 4       # batch = groups x (1 + negatives) pairs
STAGE8_SEED = 42
STAGE8_FT_MODEL_DIRNAME = "bge_reranker_ft"   # under MODELS_DIR
# Go/no-go gate on the dev bench at fixed 15/0: dev ΔR@1 (ft - off-the-shelf)
# >= threshold -> GO; <= 0 -> NO-GO (honest negative result); in between ->
# at most one retry, no threshold-shopping.
STAGE8_GO_THRESHOLD = 0.02
# Stage 8 output filenames (written under RESULTS_LATEST_DIR).
STAGE8_DEV_RESULTS_CSV = "stage8_dev_results.csv"
STAGE8_RESULTS_CSV = "stage8_ft_eval_results.csv"
STAGE8_MATCHED_CSV = "stage8_matched_summary.csv"
STAGE8_CHECK_CSV = "stage8_check_vs_stage6.csv"
STAGE8_SUMMARY_MD = "stage8_summary.md"
STAGE8_DELTA_PNG = "stage8_ft_delta.png"
# Training-data filenames (written under DATA_DIR / "nq_train").
STAGE8_TRAIN_GROUPS_JSONL = "stage8_train_groups.jsonl"

# Sweep output filenames (written under RESULTS_LATEST_DIR, snapshotted under
# RESULTS_RUNS_DIR only with --save-run).
SWEEP_RESULTS_CSV = "sweep_results.csv"
BEST_CONFIG_JSON = "best_config.json"
FAIR_TABLE_CSV = "fair_comparison_table.csv"
RECALL_PLOT_PNG = "recall_vs_chunk_size.png"
MODEL_PLOT_PNG = "model_comparison.png"
# Scatter of Recall@k vs avg chunk size, coloured by method — shows that recall
# tracks chunk *size*, not chunk *method* (the Stage 2 headline finding).
SIZE_SCATTER_PNG = "recall_vs_size_scatter.png"

# Cross-stage portfolio figure (scripts/14_evolution_plot.py): best R@5 per
# stage, read from the archived stage finals only. Written under
# RESULTS_DIR / "portfolio" — not latest/, because it is derived from the
# archives rather than produced by a new experiment.
PORTFOLIO_DIRNAME = "portfolio"
EVOLUTION_PLOT_PNG = "best_r5_evolution.png"


def model_path(model_type: str = "bilstm") -> Path:
    """Weight file for a boundary model. Defaults to 'bilstm' so existing
    no-arg callers (Phase 4/5) keep loading the BiLSTM weights unchanged."""
    return MODELS_DIR / MODEL_FILENAMES.get(model_type, MODEL_FILENAME)


def is_bge_retrieval_model(model_name: str | None = None) -> bool:
    """Whether a retrieval model should use BGE query-prefix semantics."""
    name = (model_name or RETRIEVAL_EMBED_MODEL).lower()
    return "bge-" in name or "/bge" in name


def retrieval_query_instruction(model_name: str | None = None) -> str:
    """Instruction prefix for retrieval queries.

    ``RETRIEVAL_QUERY_INSTRUCTION=None`` keeps MiniLM unchanged and applies the
    standard BGE query instruction only when the retrieval model is BGE.
    """
    if RETRIEVAL_QUERY_INSTRUCTION is not None:
        return RETRIEVAL_QUERY_INSTRUCTION
    return BGE_QUERY_INSTRUCTION if is_bge_retrieval_model(model_name) else ""


def _safe_model_slug(model_name: str) -> str:
    """Filesystem-safe model namespace for alternate retrieval indices."""
    chars = []
    for ch in model_name.lower():
        chars.append(ch if ch.isalnum() else "-")
    slug = "-".join(part for part in "".join(chars).split("-") if part)
    return slug or "model"


def retrieval_index_base_dir(model_name: str | None = None) -> Path:
    """Index namespace for the active retrieval embedding model.

    MiniLM keeps the historical ``nq/indices/main`` and ``nq/indices/sweep``
    paths. Alternate retrieval models (BGE) live under a model-specific folder
    so their FAISS files never collide with MiniLM.
    """
    model_name = model_name or RETRIEVAL_EMBED_MODEL
    if model_name == EMBED_MODEL:
        return NQ_INDEX_DIR
    return NQ_INDEX_DIR / f"retrieval_{_safe_model_slug(model_name)}"


def retrieval_index_main_dir(model_name: str | None = None) -> Path:
    return retrieval_index_base_dir(model_name) / "main"


def retrieval_index_sweep_dir(model_name: str | None = None) -> Path:
    return retrieval_index_base_dir(model_name) / "sweep"


# --------------------------------------------------------------------------- #
# Runtime overrides
# --------------------------------------------------------------------------- #
def apply(**overrides) -> None:
    """Override config constants at runtime (e.g. ``apply(N_NQ_DOCS=10)``)."""
    g = globals()
    for key, val in overrides.items():
        if key not in g:
            raise KeyError(f"unknown config key: {key!r}")
        g[key] = val


def use_smoke(root=None) -> None:
    """Tiny end-to-end settings for a fast plumbing check.

    Writes to a separate ``<root>_smoke`` folder so it never clobbers the real
    artifacts.  ~minutes on a GPU runtime.
    """
    apply(
        N_WIKI_ARTICLES=30,
        MAX_EPOCHS=1,
        EARLY_STOP_PATIENCE=1,
        N_NQ_DOCS=10,
        MIN_SENTENCES_PER_ARTICLE=4,
    )
    if root is None:
        base = str(DATA_ROOT)
        root = base if base.endswith("_smoke") else base + "_smoke"
    set_data_root(root)
    ensure_dirs()
    print(f"[smoke] tiny settings active; artifacts -> {DATA_ROOT}")


def summary() -> str:
    """One-glance dump of the active configuration (printed by the notebook)."""
    return (
        f"DATA_ROOT          = {DATA_ROOT}\n"
        f"N_WIKI_ARTICLES    = {N_WIKI_ARTICLES}\n"
        f"EMBED_MODEL        = {EMBED_MODEL} (d={EMBED_DIM})\n"
        f"BOUNDARY_EMBED_MODEL = {BOUNDARY_EMBED_MODEL} (d={BOUNDARY_EMBED_DIM})\n"
        f"RETRIEVAL_EMBED_MODEL = {RETRIEVAL_EMBED_MODEL}\n"
        f"HIDDEN_SIZE (H)    = {HIDDEN_SIZE}\n"
        f"MAX_EPOCHS         = {MAX_EPOCHS} (early stop patience {EARLY_STOP_PATIENCE})\n"
        f"N_NQ_DOCS          = {N_NQ_DOCS}\n"
        f"FIXED_CHUNK_SIZE   = {FIXED_CHUNK_SIZE}\n"
        f"FIXED_CHUNK_OVERLAP = {FIXED_CHUNK_OVERLAP}\n"
        f"RECALL_KS          = {RECALL_KS}\n"
        f"MODEL_TYPE         = {MODEL_TYPE}\n"
        f"SEMANTIC_CHUNK_POLICY = {SEMANTIC_CHUNK_POLICY} "
        f"(target={SEMANTIC_TARGET_CHUNK_SIZE}, min={SEMANTIC_MIN_CHUNK_SIZE}, "
        f"max={SEMANTIC_MAX_CHUNK_SIZE}, overlap={SEMANTIC_OVERLAP})\n"
        f"BOUNDARY_THRESHOLD = {BOUNDARY_THRESHOLD} "
        f"(min={BILSTM_MIN_CHUNK_SIZE}, max={BILSTM_MAX_CHUNK_SIZE}, "
        f"overlap={BILSTM_OVERLAP})\n"
        f"TRANSFORMER_BOUNDARY_THRESHOLD = {TRANSFORMER_BOUNDARY_THRESHOLD} "
        f"(calibrated on val by Phase 7; diagnostic only)\n"
    )
