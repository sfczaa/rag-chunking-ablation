"""Transformer semantic-boundary detector (Stage 2).

Same job and interface as ``models/bilstm.py``: given the sentence embeddings of
ONE document ``E = [e_1, ..., e_n]`` (``e_i in R^d``), score every inter-sentence
boundary

    position i  (between sentence i and i+1),   i = 0 .. n-2

with the probability that a topic change happens there.

Architecture::

    E (n, d)
      -> LayerNorm(E)                          (rescale the *frozen* MiniLM vectors
                                               so content and PE are comparable —
                                               see note below)
      + sinusoidal positional encoding        (self-attention is order-agnostic,
                                               so positions must be injected)
      -> TransformerEncoder(num_layers=2)      -> H_seq (n, d)
      -> for each boundary i, pairwise features of the two adjacent states
         [h_i ; h_{i+1} ; |h_i - h_{i+1}| ; h_i * h_{i+1}]      (4d,)
      -> Linear(4d -> 1)                       -> logit_i
      (sigmoid is applied at inference / inside the loss for stability)

The absolute-difference and element-wise product terms give the head an explicit
view of how *dissimilar* two adjacent sentence states are — the natural signal
for a topic change — instead of leaving it to recover that from the raw concat.

Why the input LayerNorm matters
-------------------------------
The MiniLM sentence vectors are *frozen* and small per component (RMS ~0.05),
while the sinusoidal PE has amplitude ~1 per component. Adding raw PE swamps the
content by ~10-40x, so adjacent-state differences become driven by *position*
rather than *topic* and the encoder collapses to a near-constant predictor (the
first cut of this model reported Boundary F1 = 0 / a degenerate "cut-everywhere"
baseline). LayerNorm rescales each sentence vector to unit per-component variance
first, so content and PE sit at a comparable scale and the boundary signal
survives — independent of the embedder's output magnitude.

Like the BiLSTM, the module is single-document (no padding): one article per
training step, which sidesteps variable-length padding. It is a **drop-in**
boundary model — ``forward`` / ``predict_proba`` / ``predict_boundaries`` match
``BiLSTMBoundary`` exactly, so chunking, retrieval and the sweep treat both the
same way.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TransformerBoundary(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible by num_heads "
                f"({num_heads}) for multi-head attention")
        self.input_dim = input_dim
        self.num_heads = num_heads
        # Rescale the frozen MiniLM vectors to unit per-component variance before
        # the (unit-amplitude) positional encoding is added, so content is not
        # swamped by position. Without this the encoder learns a constant.
        self.input_norm = nn.LayerNorm(input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # pairwise features of two adjacent encoded states:
        # [h_i ; h_{i+1} ; |h_i - h_{i+1}| ; h_i * h_{i+1}] -> 4d
        self.boundary_head = nn.Linear(4 * input_dim, 1)

    def _positional_encoding(self, n: int, device, dtype) -> torch.Tensor:
        """Standard sinusoidal positional encoding ``(n, d)`` (no parameters)."""
        pe = torch.zeros(n, self.input_dim, device=device, dtype=dtype)
        position = torch.arange(0, n, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.input_dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Score the boundaries of a single document.

        Parameters
        ----------
        emb : (n, d) float tensor  -- sentence embeddings of one article.

        Returns
        -------
        (n-1,) float tensor of raw logits, one per inter-sentence boundary.
        Returns an empty (0,) tensor when the article has < 2 sentences.
        """
        if emb.dim() != 2:
            raise ValueError(f"expected (n, d) tensor, got shape {tuple(emb.shape)}")
        n = emb.size(0)
        if n < 2:
            return emb.new_zeros((0,))

        x = self.input_norm(emb) + self._positional_encoding(n, emb.device, emb.dtype)
        seq = self.encoder(x.unsqueeze(0)).squeeze(0)   # (n, d)
        h_left, h_right = seq[:-1], seq[1:]              # (n-1, d) each
        pair = torch.cat(                                # (n-1, 4d)
            [h_left, h_right, (h_left - h_right).abs(), h_left * h_right], dim=-1)
        logits = self.boundary_head(pair).squeeze(-1)    # (n-1,)
        return logits

    @torch.no_grad()
    def predict_proba(self, emb: torch.Tensor) -> torch.Tensor:
        """Boundary probabilities (n-1,) for a single document."""
        return torch.sigmoid(self.forward(emb))

    @torch.no_grad()
    def predict_boundaries(self, emb: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Boolean (n-1,) mask: True where a cut should be made."""
        return self.predict_proba(emb) > threshold
