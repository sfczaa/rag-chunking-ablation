"""Phase 3 — train the BiLSTM boundary detector.

One article per optimisation step (variable length, no padding).  The loss is
a weighted BCE (``pos_weight = #neg/#pos`` from the training split, ~15) to
counter the ~6% positive-boundary rate.  Early stopping watches validation loss
and the best weights are saved to Drive.
"""

from __future__ import annotations

import random

import config as C


def _set_seed(seed: int = C.SEED) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(model_type: str, device):
    """Instantiate a boundary model of ``model_type`` on ``device``.

    Both models share the same (n,d) -> (n-1,) interface, so everything
    downstream (chunking, retrieval, sweep) treats them identically.
    """
    if model_type == "bilstm":
        from models import BiLSTMBoundary

        return BiLSTMBoundary(C.EMBED_DIM, C.HIDDEN_SIZE, C.NUM_LAYERS, C.DROPOUT).to(device)
    if model_type == "transformer":
        from models import TransformerBoundary

        return TransformerBoundary(
            C.EMBED_DIM, C.TRANSFORMER_LAYERS, C.TRANSFORMER_HEADS,
            C.TRANSFORMER_FF_DIM, C.TRANSFORMER_DROPOUT).to(device)
    raise ValueError(
        f"unknown model_type {model_type!r} (expected 'bilstm' or 'transformer')")


def load_model(model_type: str = "bilstm", device: str | None = None):
    """Rebuild the architecture and load the trained weights for inference.

    ``model_type`` defaults to 'bilstm', so existing no-arg callers (Phase 4/5,
    the Stage 1 sweep) keep loading the BiLSTM weights unchanged.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_type, device)
    state = torch.load(C.model_path(model_type), map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def train_model(model_type: str = "bilstm", max_epochs: int | None = None) -> dict:
    """Train, early-stop on val loss, save best weights. Returns history+stats.

    ``model_type`` selects the boundary model ('bilstm' default keeps the old
    behaviour); 'transformer' trains the Stage 2 model on the *same* cached
    MiniLM embeddings + Wikipedia labels (no Phase 1/2 rerun needed).
    """
    import torch
    import torch.nn as nn

    if max_epochs is None:
        max_epochs = C.MAX_EPOCHS

    from rag_chunk import embedding, metrics, wiki_data

    _set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] model_type={model_type}  device={device}")

    train_recs = wiki_data.load_split("train")
    val_recs = wiki_data.load_split("val")

    # Preload every used article's embeddings into RAM once. Otherwise each epoch
    # re-reads thousands of tiny .npy files from Google Drive, whose per-file
    # latency dominates Phase 3 on Colab. (~0.5-1 GB for the default scale.)
    emb_cache: dict[str, "object"] = {}
    for rec in (*train_recs, *val_recs):
        if len(rec["labels"]) == 0:
            continue
        emb_cache.setdefault(rec["id"], embedding.load_article_embeddings(rec["id"]))
    print(f"[train] preloaded {len(emb_cache)} article embeddings into RAM")

    pw = C.POS_WEIGHT if C.POS_WEIGHT is not None else metrics.class_balance("train")["pos_weight"]
    import math
    if not math.isfinite(pw):
        print(f"[train] WARN pos_weight={pw} not finite (no positives in train?); using 1.0")
        pw = 1.0
    print(f"[train] pos_weight = {pw:.3f}")
    pos_weight = torch.tensor([float(pw)], device=device)

    model = _build_model(model_type, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def run_epoch(records: list[dict], train_mode: bool) -> float:
        model.train(train_mode)
        order = list(range(len(records)))
        if train_mode:
            random.shuffle(order)
        total, count = 0.0, 0
        grad_ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with grad_ctx:
            for i in order:
                rec = records[i]
                if len(rec["labels"]) == 0:
                    continue
                x = torch.as_tensor(
                    emb_cache[rec["id"]],
                    dtype=torch.float32, device=device,
                )
                y = torch.as_tensor(rec["labels"], dtype=torch.float32, device=device)
                logits = model(x)
                loss = loss_fn(logits, y)
                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total += loss.item() * len(rec["labels"])
                count += len(rec["labels"])
        return total / max(count, 1)

    C.ensure_dirs()
    history = []
    best_val, patience = float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        tr = run_epoch(train_recs, True)
        va = run_epoch(val_recs, False)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        marker = ""
        if va < best_val - 1e-4:
            best_val, patience = va, 0
            torch.save(model.state_dict(), C.model_path(model_type))
            marker = "  <- saved best"
        else:
            patience += 1
        print(f"[train] epoch {epoch:02d}  train={tr:.4f}  val={va:.4f}{marker}")
        if patience >= C.EARLY_STOP_PATIENCE:
            print(f"[train] early stopping (no val improvement for {patience} epochs)")
            break

    # reload best weights for the held-out report.
    model.load_state_dict(torch.load(C.model_path(model_type), map_location=device,
                                     weights_only=True))

    result = {"model_type": model_type, "history": history, "best_val_loss": best_val,
              "pos_weight": float(pw), "model_path": str(C.model_path(model_type))}

    if model_type == "transformer":
        # The shared BOUNDARY_THRESHOLD (0.8) is BiLSTM-tuned and mis-scores the
        # Transformer's differently-ranged probabilities — at 0.8 it can read F1=0
        # even when the learned boundaries are fine. Calibrate the threshold on the
        # validation split (max F1), report test F1 there, and write diagnostics.
        # This is the Boundary F1 *diagnostic* only; the retrieval sweep uses the
        # target-size (argmax) policy and ignores the threshold, so it is unchanged.
        from rag_chunk import calibration

        cal = calibration.calibrate_boundary_threshold(model, model_type="transformer")
        C.apply(TRANSFORMER_BOUNDARY_THRESHOLD=cal["threshold"])
        result["test_boundary_f1"] = cal["test_boundary_f1"]
        result["calibration"] = cal
        print(f"[train] best val loss = {best_val:.4f}; "
              f"calibrated threshold={cal['threshold']:.2f}; "
              f"test boundary F1 = {cal['test_boundary_f1']['f1']:.4f}")
    else:
        test_f1 = metrics.boundary_f1(model, split="test")
        result["test_boundary_f1"] = test_f1
        print(f"[train] best val loss = {best_val:.4f}; "
              f"test boundary F1 = {test_f1['f1']:.4f}")

    return result
